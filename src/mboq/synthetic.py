"""Synthetic CME-style FIFO exchange emitting Databento-format MBO records with known truth.

The generator runs its own price-time-priority matching engine and writes the message stream a
Databento GLBX.MDP3 MBO file would contain (SPEC.md section 2 conventions): one action per event,
market orders as T (aggressor) + F (each resting fill) + C/M (each resting order's update), F_LAST
on the last record of every event, and optional Clear + snapshot blocks.

`cancel_mode` controls *where in the queue* cancels happen, which is the quantity the MBP
estimators have to guess. That gives known-answer tests:
  "tail"           cancels hit the last order at a level      -> risk_averse is exact
  "front"          cancels hit the first order                -> front_optimistic is exact
  "volume"         order chosen with probability proportional to size -> prob_power1 ~ unbiased
  "order"          order chosen uniformly                     -> no estimator is exact

Two switches (off by default; when off, the stream is identical to earlier versions) add the CME
behaviours SPEC v0.3 handles:
  p_iceberg        share of new adds that are native icebergs: a displayed tranche plus a hidden
                   reserve. When a market order empties the display, the exchange refills it from
                   the reserve under the same order id and gives the refill new time priority (back
                   of the level); the stream shows the fills, then one `M` with the new display.
  p_fill_shuffle   share of (market order, price level) pairs with at least two fills whose `F`
                   records are listed in a random order instead of priority order (the matching,
                   and so the book, is unchanged).
The generator's books are the ground truth (displayed sizes in priority order); `self.counts`
records how often each behaviour happened.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

F_LAST, F_SNAPSHOT = 128, 32


class SyntheticExchange:
    def __init__(self, seed=0, tick=10_000_000, mid_ticks=9_000, cancel_mode="volume",
                 p_add=0.50, p_cancel=0.30, p_modify=0.08, p_market=0.12, p_clear=0.0, p_implicit_snap=0.0,
                 mean_gap_ns=2_000_000, max_mkt=12, near_levels=4, p_iceberg=0.0, p_fill_shuffle=0.0):
        self.rng = np.random.default_rng(seed)
        self.tick, self.cancel_mode = tick, cancel_mode
        self.p = np.array([p_add, p_cancel, p_modify, p_market, p_clear, p_implicit_snap], dtype=float)
        self.p = self.p / self.p.sum()
        self.gap, self.max_mkt, self.near = mean_gap_ns, max_mkt, near_levels
        self.p_iceberg, self.p_fill_shuffle = p_iceberg, p_fill_shuffle
        self.books = {"B": {}, "A": {}}          # side -> price_ticks -> OrderedDict(oid -> displayed size)
        self.where = {}                          # oid -> (side, price_ticks)
        self.peak, self.reserve = {}, {}         # icebergs: oid -> display (peak) size, hidden quantity left
        # refills: every iceberg refill; refill_not_larger: refills whose M size is <= the size the stream
        # showed before the event (v0.2 kept their priority); shuffled: levels whose F records were permuted;
        # out_of_order: of those, the ones whose listing differs from priority order
        self.counts = dict(icebergs=0, refills=0, refill_not_larger=0, shuffled=0, out_of_order=0)
        self.next_oid = 1_000_001
        self.ts = 1_790_000_000_000_000_000
        self.recs = []                           # tuples (ts, oid, price_ns, size, flags, action, side)
        for k in range(1, 6):                    # seed the book
            for _ in range(3):
                self._add("B", mid_ticks - k, int(self.rng.integers(1, 10)))
                self._add("A", mid_ticks + k, int(self.rng.integers(1, 10)))
        self._flag_last()

    # ---------------------------------------------------------------- emit helpers
    def _emit(self, oid, px_ticks, size, action, side, flags=0):
        self.recs.append((self.ts, oid, px_ticks * self.tick, size, flags, action, side))

    def _flag_last(self):
        ts, oid, px, sz, fl, a, s = self.recs[-1]
        self.recs[-1] = (ts, oid, px, sz, fl | F_LAST, a, s)

    def best(self, side):
        b = self.books[side]
        if not b:
            return None
        return max(b) if side == "B" else min(b)

    def _add(self, side, px, size):
        oid = self.next_oid
        self.next_oid += 1
        self.books[side].setdefault(px, OrderedDict())[oid] = size
        self.where[oid] = (side, px)
        self._emit(oid, px, size, "A", side)
        return oid

    def _drop(self, oid, keep_reserve=False):
        side, px = self.where.pop(oid)
        lv = self.books[side][px]
        del lv[oid]
        if not lv:
            del self.books[side][px]
        if not keep_reserve:
            self.peak.pop(oid, None)
            self.reserve.pop(oid, None)

    # ---------------------------------------------------------------- actions
    def _do_add(self):
        side = "B" if self.rng.uniform() < 0.5 else "A"
        bb, ba = self.best("B"), self.best("A")
        size = int(self.rng.integers(1, 10))
        if bb is None or ba is None:
            ref = (bb if bb is not None else (ba - 1 if ba is not None else 9_000))
            px = ref if side == "B" else (ba if ba is not None else ref + 1)
            if side == "A" and bb is not None and px <= bb:
                px = bb + 1
            if side == "B" and ba is not None and px >= ba:
                px = ba - 1
        else:
            u = self.rng.uniform()
            if u < 0.10 and ba - bb > 1:         # improve inside the spread
                px = bb + 1 if side == "B" else ba - 1
            else:
                k = int(self.rng.integers(0, self.near))
                px = bb - k if side == "B" else ba + k
        oid = self._add(side, px, size)
        if self.p_iceberg > 0 and self.rng.uniform() < self.p_iceberg:   # native iceberg: hidden reserve
            self.peak[oid] = size
            self.reserve[oid] = int(self.rng.integers(1, 4 * size + 1))
            self.counts["icebergs"] += 1

    def _pick_resting(self):
        side = "B" if self.rng.uniform() < 0.5 else "A"
        b = self.books[side]
        if not b:
            return None
        best = self.best(side)
        # mostly near the touch, where queues matter
        cands = sorted(b, reverse=(side == "B"))[: self.near]
        px = cands[int(self.rng.integers(0, len(cands)))] if self.rng.uniform() > 0.5 else best
        lv = b[px]
        ids = list(lv)
        if self.cancel_mode == "tail":
            oid = ids[-1]
        elif self.cancel_mode == "front":
            oid = ids[0]
        elif self.cancel_mode == "volume":
            w = np.array([lv[i] for i in ids], dtype=float)
            oid = ids[int(self.rng.choice(len(ids), p=w / w.sum()))]
        else:
            oid = ids[int(self.rng.integers(0, len(ids)))]
        return oid

    def _do_cancel(self):
        oid = self._pick_resting()
        if oid is None:
            return False
        side, px = self.where[oid]
        size = self.books[side][px][oid]
        part = int(self.rng.integers(1, size + 1)) if self.rng.uniform() < 0.25 and size > 1 else size
        if part >= size:
            self._drop(oid)
            self._emit(oid, px, size, "C", side)
        else:
            self.books[side][px][oid] = size - part
            self._emit(oid, px, part, "C", side)
        return True

    def _do_modify(self):
        oid = self._pick_resting()
        if oid is None:
            return False
        side, px = self.where[oid]
        lv = self.books[side][px]
        size = lv[oid]
        u = self.rng.uniform()
        if u < 0.4 and size > 1:                 # size down: keeps priority
            new = int(self.rng.integers(1, size))
            lv[oid] = new
            self._emit(oid, px, new, "M", side)
        elif u < 0.75:                           # size up: loses priority (iceberg-like refresh)
            new = size + int(self.rng.integers(1, 6))
            del lv[oid]
            lv[oid] = new                        # re-append at the back
            self._emit(oid, px, new, "M", side)
        else:                                    # price change: back of the new level
            opp_best = self.best("A" if side == "B" else "B")
            step = int(self.rng.integers(1, 3)) * (-1 if self.rng.uniform() < 0.5 else 1)
            npx = px + step
            if opp_best is not None and ((side == "B" and npx >= opp_best) or (side == "A" and npx <= opp_best)):
                npx = px - step
            self._drop(oid, keep_reserve=True)   # an iceberg stays an iceberg
            self.books[side].setdefault(npx, OrderedDict())[oid] = size
            self.where[oid] = (side, npx)
            self._emit(oid, npx, size, "M", side)
        return True

    def _do_market(self):
        ag = "B" if self.rng.uniform() < 0.5 else "A"       # aggressor side; it hits the other side
        rest = "A" if ag == "B" else "B"
        qty = int(self.rng.integers(1, self.max_mkt + 1))
        if not self.books[rest]:
            return False
        updates = []
        while qty > 0 and self.books[rest]:
            px = self.best(rest)
            lv = self.books[rest][px]
            fills, shown, gone = [], {}, []      # (oid, take) in matching order; size the stream last showed; removed
            refilled = set()
            while qty > 0 and lv:                # price-time priority: always the front of the queue
                oid = next(iter(lv))
                shown.setdefault(oid, lv[oid])
                take = min(qty, lv[oid])
                fills.append((oid, take))
                qty -= take
                if take < lv[oid]:
                    lv[oid] -= take              # partial fill keeps its place (and ends the order: qty is 0)
                elif self.reserve.get(oid, 0) > 0:
                    # native iceberg: refill the display from the reserve; the refill gets new time
                    # priority (back of the level) and can be reached again by this same order
                    new = min(self.peak[oid], self.reserve[oid])
                    self.reserve[oid] -= new
                    del lv[oid]
                    lv[oid] = new
                    refilled.add(oid)
                    self.counts["refills"] += 1
                else:
                    del lv[oid]
                    gone.append(oid)
            if not lv:
                del self.books[rest][px]
            for oid in gone:
                del self.where[oid]
                self.peak.pop(oid, None)
                self.reserve.pop(oid, None)
            listed = fills
            if self.p_fill_shuffle > 0 and len(fills) > 1 and self.rng.uniform() < self.p_fill_shuffle:
                listed = [fills[k] for k in self.rng.permutation(len(fills))]
                self.counts["shuffled"] += 1
                self.counts["out_of_order"] += listed != fills
            self._emit(0, px, sum(t for _, t in fills), "T", ag)
            for oid, take in listed:
                self._emit(oid, px, take, "F", rest)
            # the C/M that apply the fills: removed orders (a C of the size last shown), then every
            # surviving order this market order touched, in its final priority order, with its new
            # displayed size (a refilled iceberg's M is a requeue under SPEC v0.3, sent in the order
            # in which the refills joined the back, so the rebuild's queue equals this one)
            updates += [(oid, px, shown[oid], "C") for oid in gone]
            for oid in lv:                       # empty if the order swept the level
                if oid in shown:
                    if oid in refilled and lv[oid] <= shown[oid]:
                        self.counts["refill_not_larger"] += 1
                    updates.append((oid, px, lv[oid], "M"))
        for oid, px, sz, a in updates:
            self._emit(oid, px, sz, a, rest)
        return True

    def _do_snapshot(self, with_clear: bool):
        if with_clear:
            self._emit(0, 0, 0, "R", "N")
        n = 0
        for side in ("B", "A"):
            for px in sorted(self.books[side], reverse=(side == "B")):
                for oid, size in self.books[side][px].items():
                    self._emit(oid, px, size, "A", side, F_SNAPSHOT)
                    n += 1
        return n > 0 or with_clear

    # ---------------------------------------------------------------- driver
    def step(self):
        self.ts += int(self.rng.exponential(self.gap)) + 1
        kind = int(self.rng.choice(6, p=self.p))
        ok = [self._do_add, self._do_cancel, self._do_modify, self._do_market,
              lambda: self._do_snapshot(True), lambda: self._do_snapshot(False)][kind]()
        if ok is False:
            self._do_add()
        self._flag_last()

    def truth_sample(self, levels=10):
        row = []
        for side in ("B", "A"):
            prices = sorted(self.books[side], reverse=(side == "B"))[:levels]
            for i in range(levels):
                if i < len(prices):
                    lv = self.books[side][prices[i]]
                    row += [prices[i] * self.tick, sum(lv.values()), len(lv)]
                else:
                    row += [0, 0, 0]
        return row

    def records(self) -> dict:
        r = self.recs
        return {
            "ts_event": np.array([x[0] for x in r], dtype=np.uint64),
            "order_id": np.array([x[1] for x in r], dtype=np.uint64),
            "price": np.array([x[2] for x in r], dtype=np.int64),
            "size": np.array([x[3] for x in r], dtype=np.uint32),
            "flags": np.array([x[4] for x in r], dtype=np.uint8),
            "action": np.array([x[5].encode() for x in r], dtype="S1"),
            "side": np.array([x[6].encode() for x in r], dtype="S1"),
        }


def generate(n_events=20_000, sample_every=50, **kw):
    """Run the exchange; return (records, sample_times, truth_samples).

    Truth samples are the generator's own book after event j, sampled at that event's ts_event
    (so `Reference` sampled at the same times must reproduce them exactly).
    """
    ex = SyntheticExchange(**kw)
    times, truth = [], []
    for j in range(n_events):
        ex.step()
        if j % sample_every == sample_every - 1:
            times.append(ex.ts)
            truth.append(ex.truth_sample())
    return ex.records(), np.array(times, dtype=np.int64), np.array(truth, dtype=np.int64), ex

"""Pure-Python reference implementation of SPEC.md.

Written for clarity, not speed: every rule in SPEC.md maps to a few lines here. The C++ core
(`mboq._core`) implements the same spec with different data structures (priority stamps and
intrusive lists instead of explicit ahead sets), and tests assert both give identical output.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

F_LAST, F_SNAPSHOT, F_MAYBE_BAD_BOOK = 128, 32, 4
ESTIMATORS = ["risk_averse", "prob_power1", "prob_power2", "prob_power3", "prob_log", "front_optimistic"]
OUTCOMES = ["censored", "filled", "cancelled", "moved_price", "requeued", "cleared"]
# columns of results["queue"] (SPEC section 8), in order
QUEUE_COLUMNS = ["queue_time_index", "order_id", "side", "price", "level_index", "rank", "size", "size_ahead",
                 "level_order_count", "level_total_size"]
# keys of results["tracks"] (SPEC section 9): one row per track; the last four are (tracks, K) arrays
TRACK_COLUMNS = ["label_index", "order_id", "side", "price", "t_start", "size0", "size_ahead0", "outcome", "t_end",
                 "true_first_fill_t", "true_filled_qty",
                 "est_first_fill_t", "est_filled_qty", "v_est_at_true_first_fill", "v_est_at_end"]
# SPEC 9 (v0.5): two more keys, only with track_horizon_ns > 0; "e0" is a (tracks, K) array
TRACK_HORIZON_COLUMNS = ["est_stop_t", "e0"]


def check_queue_args(queue_times, queue_levels) -> np.ndarray:
    """Validate the SPEC section 8 inputs (shared by both engines); return queue_times as int64."""
    qt = np.asarray(queue_times, dtype=np.int64).reshape(-1)
    if int(queue_levels) < 0:
        raise ValueError("queue_levels must be >= 0")
    if len(qt) > 1 and bool((np.diff(qt) < 0).any()):
        raise ValueError("queue_times must be sorted (non-decreasing)")
    return qt


def check_track_horizon(track_horizon_ns) -> int:
    """Validate the SPEC 9 (v0.5) track horizon (shared by both engines): an int >= 0, 0 = v0.4."""
    h = int(track_horizon_ns)
    if h < 0:
        raise ValueError("track_horizon_ns must be >= 0")
    return h


def phi(k: int, front: float, back: float) -> float:
    """Front share of an unexplained queue decrease (SPEC section 5)."""
    if k == 0:
        return 0.0
    if k == 5:
        return 1.0
    if k in (1, 2, 3):
        a, b = front ** k, back ** k
    else:
        a, b = math.log1p(front), math.log1p(back)
    return a / (a + b) if a + b > 0 else 0.0


class _Shadow:
    __slots__ = ("sid", "side", "price", "t_place", "t_end", "size", "q0", "ahead", "filled",
                 "t_fill", "end", "est_v", "est_filled", "est_t", "err", "abserr", "nobs")

    def __init__(self, sid, side, price, t_place, horizon, size, level: OrderedDict):
        self.sid, self.side, self.price, self.t_place = sid, side, price, t_place
        self.t_end, self.size = t_place + horizon, size
        self.ahead = {oid: o["size"] for oid, o in level.items()}     # explicit ahead set
        self.q0 = sum(self.ahead.values())
        self.filled, self.t_fill, self.end = 0, -1, "horizon"
        self.est_v = [float(self.q0)] * len(ESTIMATORS)
        self.est_filled = [0.0] * len(ESTIMATORS)
        self.est_t = [-1] * len(ESTIMATORS)
        self.err = [0.0] * len(ESTIMATORS)
        self.abserr = [0.0] * len(ESTIMATORS)
        self.nobs = 0

    def v_true(self) -> int:
        return sum(self.ahead.values())

    def worse(self, px: int) -> bool:
        return px < self.price if self.side == "B" else px > self.price


class _Track:
    """SPEC 9: one resting order's life, followed by the section 5 estimators with X excluded from Q."""
    __slots__ = ("li", "oid", "side", "price", "t_start", "size0", "ahead0", "outcome", "t_end", "tff", "tfq",
                 "v", "ef", "et", "vff", "need_vff", "q0ex", "tr_from", "post", "stop", "stop_t")

    def __init__(self, li, oid, side, price, ts, size, ahead_q, tr_from):
        self.li, self.oid, self.side, self.price, self.t_start = li, oid, side, price, ts
        self.size0, self.ahead0 = size, ahead_q
        self.outcome, self.t_end, self.tff, self.tfq = "censored", -1, -1, 0
        K = len(ESTIMATORS)
        self.v = [float(ahead_q)] * K            # V starts at the size ahead at entry
        self.ef = [0.0] * K                      # estimated filled quantity
        self.et = [-1] * K                       # estimated first-fill time
        self.vff = [math.nan] * K                # V after the clip holding X's first true fill
        self.need_vff = False
        self.q0ex = ahead_q                      # Q_ex at the start of the current clip
        self.tr_from = tr_from                   # this event's T prints before the start are not seen
        # v0.5: post = X was filled and the estimators run on to the horizon (X is no longer at the level);
        # stop = why the estimators stopped ("" while running, "owner", "horizon", "clear", "end")
        self.post, self.stop, self.stop_t = False, "", -1


class Reference:
    lot_round = False     # SPEC v0.6 fill_lot_rounding (set by run())
    def __init__(self, tick: int, label_ticks: int = 5, shadow_size: int = 1, horizon_ns: int = 60_000_000_000,
                 sample_levels: int = 10):
        self.tick, self.label_ticks, self.S, self.H = tick, label_ticks, shadow_size, horizon_ns
        self.N = sample_levels
        self.books = {"B": {}, "A": {}}          # side -> price -> OrderedDict(oid -> order)
        self.orders = {}                         # oid -> {"side","price","size","label"}
        self.stats = dict(records=0, events=0, adds=0, cancels=0, modifies=0, clears=0, trades=0, fills=0,
                          dup_add=0, unknown_cancel=0, modify_unknown=0, implicit_clear=0,
                          fill_unknown=0, maybe_bad_book=0, shadow_fifo_anomaly=0)
        self.fifo_ranks, self.fifo_ts = [], []
        self.labels = []                         # list of dicts
        self.shadows, self.done_shadows = [], []
        self._ev_fill = {}                       # oid -> cumulative F size in current event
        self._ev_ranks = []                      # (index into fifo_ranks, [(ahead oid, size)]) ranked at the boundary
        self._ev_trades = []                     # (price, aggressor side, size) in current event
        self._ev_touched = set()                 # (side, price) touched in current event
        self._prev_snapshot = False
        self._prev_action = None
        self._next_sid = 0
        self._q_before = {}                      # (side, price) -> Q at start of event (for touched levels)
        self.order_tracks = False                # SPEC 9, set by run()
        self.track_h = 0                         # SPEC 9 v0.5 track horizon (ns), set by run(); 0 = v0.4
        self._post = []                          # tracks whose X was filled, estimators still running (v0.5)
        self._last_boundary = -1                 # ts_event of the last event boundary (-1: none yet)
        self.tracks = []                         # every track, in start order
        self._tracks_at = {}                     # (side, price) -> {label index: live track}
        self._started = []                       # tracks started mid-event (tr_from > 0) in this event
        self._vff_pending = []                   # tracks whose first true fill is in this event

    # ------------------------------------------------------------------ book primitives
    def _level(self, side, price, create=False):
        lv = self.books[side].get(price)
        if lv is None and create:
            lv = OrderedDict()
            self.books[side][price] = lv
        return lv

    def _qty(self, side, price) -> int:
        lv = self.books[side].get(price)
        return sum(o["size"] for o in lv.values()) if lv else 0

    def best(self, side):
        b = self.books[side]
        if not b:
            return None
        return max(b) if side == "B" else min(b)

    def _touch(self, side, price):
        key = (side, price)
        if key not in self._ev_touched:
            self._ev_touched.add(key)
            self._q_before[key] = self._qty(side, price)

    def _remove(self, oid, reason):
        o = self.orders.pop(oid)
        lv = self.books[o["side"]][o["price"]]
        del lv[oid]
        if not lv:
            del self.books[o["side"]][o["price"]]
        for sh in self.shadows:
            if sh.side == o["side"] and sh.price == o["price"]:
                sh.ahead.pop(oid, None)
        self._end_label(o, reason)

    def _append(self, oid, side, price, size, label_ok):
        best = self.best(side)                   # best BEFORE the add (SPEC 6)
        lv = self._level(side, price, create=True)
        ahead_n, ahead_q = len(lv), sum(x["size"] for x in lv.values())
        o = {"side": side, "price": price, "size": size, "label": None, "fsu": 0}   # fsu: fill_since_update
        lv[oid] = o
        self.orders[oid] = o
        if label_ok:
            if best is None:
                dist = 0
            else:
                dist = (best - price) // self.tick if side == "B" else (price - best) // self.tick
                dist = max(dist, 0)
            if dist <= self.label_ticks:
                lab = dict(order_id=oid, t_add=self._ts, side=side, price=price, size=size, dist=int(dist),
                           rank_at_add=ahead_n, size_ahead_at_add=ahead_q, outcome="censored", t_end=-1, filled_qty=0)
                self.labels.append(lab)
                o["label"] = lab
                if self.order_tracks and dist == 0:      # SPEC 9: joins at or better than the best
                    li = len(self.labels) - 1
                    trk = _Track(li, oid, side, price, self._ts, size, ahead_q, len(self._ev_trades))
                    lab["track"] = trk
                    self.tracks.append(trk)
                    self._tracks_at.setdefault((side, price), {})[li] = trk
                    if trk.tr_from > 0:
                        self._started.append(trk)

    def _end_label(self, o, reason):
        lab = o.get("label")
        if lab is not None and lab["outcome"] == "censored" and lab["t_end"] == -1:
            lab["outcome"], lab["t_end"] = reason, self._ts
            o["label"] = None
            trk = lab.get("track")
            if trk is not None:
                self._end_track(trk, reason)

    # ------------------------------------------------------------------ order-anchored tracks (SPEC 9)
    def _track_trades(self, trk):
        """Tr (opposite-aggressor T volume at X's price) and trade-through, over this event's T prints
        after the track's start."""
        opp = "A" if trk.side == "B" else "B"
        tr, through = 0, False
        for px, ag, sz in self._ev_trades[trk.tr_from:]:
            if ag != opp:
                continue
            if px == trk.price:
                tr += sz
            if px < trk.price if trk.side == "B" else px > trk.price:
                through = True
        return tr, through

    def _update_track(self, trk, tr, through, q1ex):
        """SPEC 5 update with Q replaced by Q_ex (the level without X); q0 is Q_ex at the clip start."""
        qa = max(trk.q0ex - tr, 0)
        dec = max(qa - q1ex, 0)
        S = float(trk.size0)
        for k in range(len(ESTIMATORS)):
            v = trk.v[k]
            unfilled = trk.ef[k] < S
            if tr > 0:
                excess = tr - v
                v = max(v - tr, 0.0)
                if excess > 0 and unfilled:
                    trk.ef[k] = min(S, trk.ef[k] + excess)
            if through and unfilled:
                trk.ef[k] = S
            if trk.et[k] < 0 and (trk.ef[k] >= 0.5 if self.lot_round else trk.ef[k] > 0):   # v0.6 fill_lot_rounding
                trk.et[k] = self._ts
            if dec > 0:
                v = v - phi(k, v, max(qa - v, 0.0)) * dec
            v = min(max(v, 0.0), float(q1ex))
            trk.v[k] = v
        trk.q0ex = q1ex
        if trk.need_vff:
            trk.vff, trk.need_vff = list(trk.v), False

    def _end_track(self, trk, reason):
        """X's life ended at this record: apply the final clip (records after the last boundary, up to
        and including this one; X is no longer at the level), unless the book was cleared. With a track
        horizon (v0.5), a filled X's estimators keep running (from the next record on); otherwise they stop."""
        if reason != "cleared":
            tr, through = self._track_trades(trk)
            self._update_track(trk, tr, through, self._qty(trk.side, trk.price))
        trk.outcome, trk.t_end = reason, self._ts
        if reason == "filled" and self.track_h > 0:
            trk.post = True
            trk.tr_from = len(self._ev_trades)       # the rest of this event's prints are seen, earlier ones not
            self._started.append(trk)
            self._post.append(trk)
            return
        self._stop_track(trk, "clear" if reason == "cleared" else "owner")

    def _stop_track(self, trk, why):
        """The estimators stop here (v0.5 bookkeeping; with no horizon this is the end of X's life)."""
        trk.stop, trk.stop_t = why, self._ts
        group = self._tracks_at[(trk.side, trk.price)]
        del group[trk.li]
        if not group:
            del self._tracks_at[(trk.side, trk.price)]

    def _reached(self, side, price):
        """Whether any opposite-aggressor T print of this event is at `price` or worse (a necessary
        condition for Tr > 0 or a trade-through on any track there)."""
        opp = "A" if side == "B" else "B"
        return any(ag == opp and (px <= price if side == "B" else px >= price) for px, ag, _ in self._ev_trades)

    def _end_event_tracks(self):
        for (side, price), group in self._tracks_at.items():
            touched = (side, price) in self._ev_touched
            if not touched and not self._reached(side, price):
                continue                                   # no track here can move in this event
            q = self._qty(side, price)
            for trk in group.values():
                tr, through = self._track_trades(trk)
                if touched or tr > 0 or through:
                    # X, in this life, still rests at (side, price); after a fill (v0.5) X is gone: Q_ex = Q
                    x = 0 if trk.post else self.orders[trk.oid]["size"]
                    self._update_track(trk, tr, through, q - x)
        if self._post:                                     # v0.5: the first boundary at or past t_start + H
            keep = []
            for trk in self._post:
                if self._ts >= trk.t_start + self.track_h:
                    self._stop_track(trk, "horizon")
                else:
                    keep.append(trk)
            self._post = keep
        for trk in self._vff_pending:                      # if its update was skipped, V is unchanged
            if trk.need_vff and trk.outcome == "censored":     # still live (a cleared one keeps NaN)
                trk.vff, trk.need_vff = list(trk.v), False
        self._vff_pending.clear()
        for trk in self._started:
            trk.tr_from = 0
        self._started.clear()

    def _clear(self):
        for oid in list(self.orders):
            o = self.orders[oid]
            self._end_label(o, "cleared")
        self.books = {"B": {}, "A": {}}
        self.orders = {}
        for trk in self._post:                   # v0.5: a clear also stops estimators running past a fill
            self._stop_track(trk, "clear")
        self._post = []
        for sh in self.shadows:
            sh.end = "cleared"
        self.done_shadows.extend(self.shadows)
        self.shadows = []

    # ------------------------------------------------------------------ record handlers
    def _on_add(self, oid, side, price, size, snap):
        self.stats["adds"] += 1
        self._touch(side, price)
        if oid in self.orders:
            if not snap:
                self.stats["dup_add"] += 1
            o = self.orders[oid]
            self._touch(o["side"], o["price"])
            self._remove(oid, "cancelled")
        self._append(oid, side, price, size, label_ok=not snap)

    def _on_cancel(self, oid, size):
        self.stats["cancels"] += 1
        o = self.orders.get(oid)
        if o is None:
            self.stats["unknown_cancel"] += 1
            return
        self._touch(o["side"], o["price"])
        new = o["size"] - size
        if size == 0 or new <= 0:
            reason = "filled" if self._ev_fill.get(oid, 0) > 0 else "cancelled"
            self._remove(oid, reason)
        else:
            o["size"] = new
            o["fsu"] = 0
            for sh in self.shadows:
                if oid in sh.ahead:
                    sh.ahead[oid] = new

    def _requeues(self, o, size) -> bool:
        """SPEC 2: an M at the same price requeues iff the new size exceeds what is left after the
        fills reported since the order's last book change (an iceberg refill is such an increase)."""
        return size > o["size"] - o["fsu"]

    def _on_modify(self, oid, side, price, size):
        self.stats["modifies"] += 1
        o = self.orders.get(oid)
        if o is None:
            self.stats["modify_unknown"] += 1
            self._touch(side, price)
            self._append(oid, side, price, size, label_ok=True)
            return
        self._touch(o["side"], o["price"])
        if price != o["price"] or side != o["side"]:
            self._touch(side, price)
            self._remove(oid, "moved_price")
            self._append(oid, side, price, size, label_ok=True)
        elif size <= 0:
            reason = "filled" if self._ev_fill.get(oid, 0) > 0 else "cancelled"
            self._remove(oid, reason)
        elif self._requeues(o, size):
            self._remove(oid, "requeued")
            self._append(oid, side, price, size, label_ok=True)   # new order dict: fsu starts at 0
        else:
            o["size"] = size
            o["fsu"] = 0
            for sh in self.shadows:
                if oid in sh.ahead:
                    sh.ahead[oid] = size

    def _rank_fill(self, oid, lv):
        """SPEC 3: note which orders are ahead of `oid` now, and their sizes; the rank is computed
        at the event boundary (`_close_ranks`), with the event's total fills per order."""
        ahead = []
        for other_id, other in lv.items():
            if other_id == oid:
                break
            ahead.append((other_id, other["size"]))
        self._ev_ranks.append((len(self.fifo_ranks), ahead))
        self.fifo_ranks.append(None)

    def _close_ranks(self):
        for i, ahead in self._ev_ranks:
            self.fifo_ranks[i] = sum(1 for aid, sz in ahead if self._ev_fill.get(aid, 0) < sz)
        self._ev_ranks.clear()

    def _on_fill(self, oid, side, price, size):
        self.stats["fills"] += 1
        o = self.orders.get(oid)
        if o is None:
            self.stats["fill_unknown"] += 1
            self.fifo_ranks.append(-1)
        else:
            self._rank_fill(oid, self.books[o["side"]][o["price"]])
            o["fsu"] += size
            lab = o.get("label")
            if lab is not None:
                lab["filled_qty"] += size
                trk = lab.get("track")
                if trk is not None:                  # SPEC 9 truth: X's own F records
                    trk.tfq += size
                    if trk.tff < 0:
                        trk.tff, trk.need_vff = self._ts, True
                        self._vff_pending.append(trk)
        self.fifo_ts.append(self._ts)
        self._ev_fill[oid] = self._ev_fill.get(oid, 0) + size
        # truth shadows (SPEC 4): fills of orders behind the shadow, or trade-through.
        # Unknown-id fills are ignored here: their resting side is not observable.
        if o is None:
            return
        bside, bprice = o["side"], o["price"]
        for sh in self.shadows:
            if sh.filled >= sh.size or sh.side != bside:
                continue
            hit = (bprice == sh.price and oid not in sh.ahead) or sh.worse(bprice)
            if hit:
                if bprice == sh.price:
                    # ahead volume not yet consumed by earlier F records in this event
                    left = sum(max(sz - self._ev_fill.get(aid, 0), 0) for aid, sz in sh.ahead.items())
                    if left > 0:
                        self.stats["shadow_fifo_anomaly"] += 1
                take = min(size, sh.size - sh.filled)
                sh.filled += take
                if sh.filled >= sh.size and sh.t_fill < 0:
                    sh.t_fill = self._ts

    # ------------------------------------------------------------------ event boundary
    def _end_event(self):
        self.stats["events"] += 1
        self._last_boundary = self._ts
        self._close_ranks()
        # estimator updates (SPEC 5) for every live shadow
        for sh in self.shadows:
            opp = "A" if sh.side == "B" else "B"
            tr = sum(sz for px, ag, sz in self._ev_trades if ag == opp and px == sh.price)
            through = any(ag == opp and sh.worse(px) for px, ag, sz in self._ev_trades)
            key = (sh.side, sh.price)
            touched = key in self._ev_touched
            if not touched and tr == 0 and not through:
                continue
            q1 = self._qty(sh.side, sh.price)
            q0 = self._q_before.get(key, q1)
            qa = max(q0 - tr, 0)
            dec = max(qa - q1, 0)
            for k in range(len(ESTIMATORS)):
                v = sh.est_v[k]
                unfilled = sh.est_filled[k] < sh.size
                if tr > 0:
                    excess = tr - v
                    v = max(v - tr, 0.0)
                    if excess > 0 and unfilled:
                        sh.est_filled[k] = min(float(sh.size), sh.est_filled[k] + excess)
                        if self.lot_round and sh.est_filled[k] >= sh.size - 0.5:   # v0.6: hftbacktest half-lot rounding
                            sh.est_filled[k] = float(sh.size)
                if through and unfilled:
                    sh.est_filled[k] = float(sh.size)
                if unfilled and sh.est_filled[k] >= sh.size:
                    sh.est_t[k] = self._ts
                if dec > 0:
                    v = v - phi(k, v, max(qa - v, 0.0)) * dec
                v = min(max(v, 0.0), float(q1))
                sh.est_v[k] = v
            if touched and sh.filled < sh.size:
                vt = sh.v_true()
                sh.nobs += 1
                for k in range(len(ESTIMATORS)):
                    d = sh.est_v[k] - vt
                    sh.err[k] += d
                    sh.abserr[k] += abs(d)
        if self._tracks_at or self._vff_pending or self._started:
            self._end_event_tracks()
        self._ev_fill.clear()
        self._ev_trades.clear()
        self._ev_touched.clear()
        self._q_before.clear()

    def _retire_and_place(self, place_times, pi):
        # retire finished shadows
        keep = []
        for sh in self.shadows:
            all_done = sh.filled >= sh.size and all(f >= sh.size for f in sh.est_filled)
            if all_done or self._ts >= sh.t_end:
                if sh.filled >= sh.size:
                    sh.end = "filled"
                self.done_shadows.append(sh)
            else:
                keep.append(sh)
        self.shadows = keep
        # place new shadows whose time has come (one bid and one ask per placement time)
        while pi < len(place_times) and place_times[pi] <= self._ts:
            for side in ("B", "A"):
                px = self.best(side)
                if px is not None:
                    self.shadows.append(_Shadow(self._next_sid, side, px, self._ts, self.H, self.S, self.books[side][px]))
                    self._next_sid += 1
            pi += 1
        return pi

    def _sample(self, t):
        row = []
        for side in ("B", "A"):
            prices = sorted(self.books[side], reverse=(side == "B"))[: self.N]
            for i in range(self.N):
                if i < len(prices):
                    lv = self.books[side][prices[i]]
                    row += [prices[i], sum(o["size"] for o in lv.values()), len(lv)]
                else:
                    row += [0, 0, 0]
        return row

    def _queue_snapshot(self, q, levels):
        """SPEC 8: one row per resting order at the best `levels` prices of each side, FIFO order."""
        rows = []
        for side in ("B", "A"):
            for li, px in enumerate(sorted(self.books[side], reverse=(side == "B"))[:levels]):
                lv = self.books[side][px]                    # OrderedDict in priority order (front first)
                n, total = len(lv), sum(o["size"] for o in lv.values())
                ahead = 0
                for rank, (oid, o) in enumerate(lv.items()):
                    rows.append([q, oid - (1 << 64) if oid >= (1 << 63) else oid, 1 if side == "B" else -1, px,
                                 li, rank, o["size"], ahead, n, total])
                    ahead += o["size"]
        return rows

    # ------------------------------------------------------------------ driver
    def run(self, rec: dict, place_times=(), sample_times=(), queue_times=(), queue_levels: int = 5,
            order_tracks: bool = False, track_horizon_ns: int = 0, fill_lot_rounding: bool = False):
        """rec: dict of equal-length numpy arrays (ts_event, order_id, price, size, flags, action, side).
        action/side are single-byte codes (b'A' or ord). Returns a results dict (see SPEC);
        results["tracks"] (SPEC 9) only with order_tracks=True. track_horizon_ns = H > 0 (SPEC 9, v0.5) runs
        a filled order's estimators on to t_start + H and adds the track keys est_stop_t and e0."""
        self.order_tracks = bool(order_tracks)
        self.track_h = check_track_horizon(track_horizon_ns)
        self.lot_round = bool(fill_lot_rounding)      # SPEC v0.6: estimated fills rounded to whole lots (>= 0.5 lot)
        place_times = np.asarray(place_times, dtype=np.int64)
        sample_times = np.asarray(sample_times, dtype=np.int64)
        queue_times = check_queue_args(queue_times, queue_levels)
        queue_levels = int(queue_levels)
        pi = si = qi = 0
        samples, queue = [], []
        n = len(rec["ts_event"])
        act = _codes(rec["action"])
        sid = _codes(rec["side"])
        for i in range(n):
            self._ts = int(rec["ts_event"][i])
            # samples are taken at boundaries; emit any sample time strictly before this record
            while si < len(sample_times) and sample_times[si] < self._ts and self._at_boundary:
                samples.append(self._sample(sample_times[si])); si += 1
            # queue snapshots: same rule as samples (SPEC 8)
            while qi < len(queue_times) and queue_times[qi] < self._ts and self._at_boundary:
                queue += self._queue_snapshot(qi, queue_levels); qi += 1
            a, s = act[i], sid[i]
            oid, px, sz, fl = int(rec["order_id"][i]), int(rec["price"][i]), int(rec["size"][i]), int(rec["flags"][i])
            self.stats["records"] += 1
            if fl & F_MAYBE_BAD_BOOK:
                self.stats["maybe_bad_book"] += 1
            snap = bool(fl & F_SNAPSHOT)
            if snap and not self._prev_snapshot and self._prev_action != "R" and self.orders:
                self.stats["implicit_clear"] += 1
                self._clear()
            if a == "A":
                self._on_add(oid, s, px, sz, snap)
            elif a == "C":
                self._on_cancel(oid, sz)
            elif a == "M":
                self._on_modify(oid, s, px, sz)
            elif a == "R":
                self.stats["clears"] += 1
                self._clear()
            elif a == "T":
                self.stats["trades"] += 1
                self._ev_trades.append((px, s, sz))
            elif a == "F":
                self._on_fill(oid, s, px, sz)
            self._prev_snapshot, self._prev_action = snap, a
            self._at_boundary = bool(fl & F_LAST)
            if self._at_boundary:
                self._end_event()
                pi = self._retire_and_place(place_times, pi)
        while si < len(sample_times):
            samples.append(self._sample(sample_times[si])); si += 1
        while qi < len(queue_times):
            queue += self._queue_snapshot(qi, queue_levels); qi += 1
        self._close_ranks()                      # an event left open by a last record without F_LAST (SPEC 3)
        for sh in self.shadows:
            self.done_shadows.append(sh)
        self.shadows = []
        out = self._results(samples)
        out["queue"] = np.array(queue, dtype=np.int64).reshape(-1, len(QUEUE_COLUMNS))
        if self.order_tracks:
            out["tracks"] = self._track_results()
        return out

    def _e0(self, trk):
        """SPEC 9 (v0.5) E0 per estimator, in ns: |tau_est_k - tau_true|, both censored at H. -1 = not scored:
        the estimators were stopped by a clear, or by the end of the stream before t_start + H was reached."""
        H = self.track_h
        deadline = trk.t_start + H
        if not (trk.stop in ("owner", "horizon") or (trk.stop == "end" and self._last_boundary >= deadline)):
            return [-1] * len(ESTIMATORS)
        tau_true = trk.tff - trk.t_start if 0 <= trk.tff <= deadline else H
        lim = deadline if trk.stop_t < 0 else min(trk.stop_t, deadline)
        return [abs((t - trk.t_start if 0 <= t <= lim else H) - tau_true) for t in trk.et]

    def _track_results(self):
        T, K = self.tracks, len(ESTIMATORS)
        i64 = lambda xs: np.array(xs, dtype=np.int64)  # noqa: E731
        f2 = lambda xs: np.array(xs, dtype=np.float64).reshape(-1, K)  # noqa: E731
        for trk in T:                                  # estimators still running when the stream ends
            if trk.stop == "":
                trk.stop, trk.stop_t = "end", -1
        out = {
            "label_index": i64([t.li for t in T]),
            "order_id": i64([t.oid - (1 << 64) if t.oid >= (1 << 63) else t.oid for t in T]),
            "side": i64([1 if t.side == "B" else -1 for t in T]),
            "price": i64([t.price for t in T]),
            "t_start": i64([t.t_start for t in T]),
            "size0": i64([t.size0 for t in T]),
            "size_ahead0": i64([t.ahead0 for t in T]),
            "outcome": i64([OUTCOMES.index(t.outcome) for t in T]),
            "t_end": i64([t.t_end for t in T]),
            "true_first_fill_t": i64([t.tff for t in T]),
            "true_filled_qty": i64([t.tfq for t in T]),
            "est_first_fill_t": i64([t.et for t in T]).reshape(-1, K),
            "est_filled_qty": f2([t.ef for t in T]),
            "v_est_at_true_first_fill": f2([t.vff for t in T]),
            "v_est_at_end": f2([t.v for t in T]),
        }
        if self.track_h > 0:
            out["est_stop_t"] = i64([t.stop_t for t in T])
            out["e0"] = i64([self._e0(t) for t in T]).reshape(-1, K)
        return out

    _at_boundary = True
    _ts = 0

    def _results(self, samples):
        sh = sorted(self.done_shadows, key=lambda x: x.sid)
        K = len(ESTIMATORS)
        out = {
            "stats": dict(self.stats),
            "fifo_rank": np.array(self.fifo_ranks, dtype=np.int64),
            "fifo_ts": np.array(self.fifo_ts, dtype=np.int64),
            "shadows": {
                "sid": np.array([x.sid for x in sh], dtype=np.int64),
                "side": np.array([1 if x.side == "B" else -1 for x in sh], dtype=np.int64),
                "price": np.array([x.price for x in sh], dtype=np.int64),
                "t_place": np.array([x.t_place for x in sh], dtype=np.int64),
                "q0": np.array([x.q0 for x in sh], dtype=np.int64),
                "end": np.array([{"horizon": 0, "filled": 1, "cleared": 2}[x.end] for x in sh], dtype=np.int64),
                "true_filled": np.array([int(x.filled >= x.size) for x in sh], dtype=np.int64),
                "true_t_fill": np.array([x.t_fill for x in sh], dtype=np.int64),
                "est_filled": np.array([[int(x.est_filled[k] >= x.size) for k in range(K)] for x in sh], dtype=np.int64).reshape(-1, K),
                "est_t_fill": np.array([x.est_t for x in sh], dtype=np.int64).reshape(-1, K),
                "err_sum": np.array([x.err for x in sh], dtype=np.float64).reshape(-1, K),
                "abserr_sum": np.array([x.abserr for x in sh], dtype=np.float64).reshape(-1, K),
                "nobs": np.array([x.nobs for x in sh], dtype=np.int64),
            },
            "labels": {k: np.array([lab[k] if k != "outcome" else OUTCOMES.index(lab[k]) for lab in self.labels],
                                   dtype=np.int64)
                       for k in ["order_id", "t_add", "price", "size", "dist", "rank_at_add", "size_ahead_at_add",
                                 "outcome", "t_end", "filled_qty"]},
            "samples": np.array(samples, dtype=np.int64).reshape(-1, 6 * self.N),
        }
        out["labels"]["side"] = np.array([1 if lab["side"] == "B" else -1 for lab in self.labels], dtype=np.int64)
        return out


def _codes(arr):
    """Normalize action/side arrays (bytes 'S1', ints, or str) to a list of 1-char str."""
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        return [x.decode() for x in a]
    if a.dtype.kind in "iu":
        return [chr(int(x)) for x in a]
    return [str(x) for x in a]

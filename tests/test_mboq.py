"""mbo-queue tests. Run:  python3 tests/test_mboq.py   (or pytest)

Every test states the known answer it checks. Synthetic data only, except one test that decodes
Databento's own public 4-record sample file to check the real DBN path end to end; it is skipped
unless the file has been fetched with scripts/fetch_test_data.sh.

The installed `mboq` is tested if there is one (e.g. a wheel), otherwise the source tree's src/.
Set MBOQ_REQUIRE_CORE=1 to fail, instead of skipping the C++ tests, when the C++ core is missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import mboq  # noqa: F401  (installed package)
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))

from mboq.reference import (ESTIMATORS, OUTCOMES, QUEUE_COLUMNS, TRACK_COLUMNS, TRACK_HORIZON_COLUMNS,  # noqa: E402
                            Reference)
from mboq.synthetic import SyntheticExchange, generate  # noqa: E402

try:
    from mboq import fast
    HAVE_CORE = True
except ImportError:  # C++ core not built
    if os.environ.get("MBOQ_REQUIRE_CORE") == "1":
        raise
    HAVE_CORE = False

TICK = 10_000_000
L, SNAP = 128, 32


def recs(rows):
    """rows: (ts, action, side, price_ticks, size, order_id, flags)"""
    return {
        "ts_event": np.array([r[0] for r in rows], dtype=np.uint64),
        "action": np.array([r[1].encode() for r in rows], dtype="S1"),
        "side": np.array([r[2].encode() for r in rows], dtype="S1"),
        "price": np.array([r[3] * TICK for r in rows], dtype=np.int64),
        "size": np.array([r[4] for r in rows], dtype=np.uint32),
        "order_id": np.array([r[5] for r in rows], dtype=np.uint64),
        "flags": np.array([r[6] for r in rows], dtype=np.uint8),
    }


def both(rec, **kw):
    out = [Reference(tick=TICK, **{k: v for k, v in kw.items()
                                   if k in ("label_ticks", "shadow_size", "horizon_ns", "sample_levels")})
           .run(rec, **{k: v for k, v in kw.items() if k in ("place_times", "sample_times", "queue_times",
                                                              "queue_levels", "order_tracks", "track_horizon_ns",
                                                              "fill_lot_rounding")})]
    if HAVE_CORE:
        out.append(fast.run(rec, tick=TICK, **kw))
    return out


# a small book: bids at 100 (orders 1:3, 2:2), ask at 101 (order 9:5)
BASE = [(1, "A", "B", 100, 3, 1, L), (2, "A", "B", 100, 2, 2, L), (3, "A", "A", 101, 5, 9, L)]


def test_shadow_fills_after_ahead_queue_is_consumed():
    # shadow placed at t=4 at bid 100 behind 5 lots. Sell 6: F(1,3) F(2,2) F(3,1 behind) -> shadow filled.
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),            # shadow is placed at the t=4 boundary, i.e. behind order 3
                   (10, "T", "A", 100, 6, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 2, 2, 0),
                   (10, "F", "B", 100, 1, 3, 0), (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 2, 2, 0),
                   (10, "M", "B", 100, 3, 3, L)]
    for r in both(recs(rows), place_times=[4], horizon_ns=100):
        sh = r["shadows"]
        b = sh["side"] == 1
        # shadow at t=4 is placed behind orders 1,2,3 (9 lots) -> 6 lots only consume the ahead queue
        assert sh["q0"][b][0] == 9 and sh["true_filled"][b][0] == 0
        # risk_averse and every estimator agree: trade 6 < queue 9, no fill
        assert (sh["est_filled"][b][0] == 0).all()


def test_shadow_filled_by_behind_order_fill():
    # shadow placed at t=3 behind orders 1,2 (5 lots); order 3 joins behind at t=4; sell 6 fills 1,2 and 1 lot of 3
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),
                   (10, "T", "A", 100, 6, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 2, 2, 0),
                   (10, "F", "B", 100, 1, 3, 0), (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 2, 2, 0),
                   (10, "M", "B", 100, 3, 3, L)]
    for r in both(recs(rows), place_times=[3], horizon_ns=100):
        sh = r["shadows"]
        b = sh["side"] == 1
        assert sh["q0"][b][0] == 5 and sh["true_filled"][b][0] == 1 and sh["true_t_fill"][b][0] == 10
        assert (sh["est_filled"][b][0] == 1).all()          # trade 6 > every estimate of a 5-lot queue
        assert r["stats"]["shadow_fifo_anomaly"] == 0


def test_cancel_ahead_vs_behind_moves_only_the_truth():
    # shadow behind 1,2 (5 lots); order 3 joins behind; order 1 (ahead) cancels 3 lots; then sell 3.
    # Truth: queue ahead = 2 -> sell 3 fills order 2 (2) and 1 lot behind -> shadow filled.
    # risk_averse assumes the cancel was behind -> V stays 5 -> no fill. front_optimistic -> V=2 -> fill.
    rows = BASE + [(4, "A", "B", 100, 4, 3, L), (5, "C", "B", 100, 3, 1, L),
                   (10, "T", "A", 100, 3, 0, 0), (10, "F", "B", 100, 2, 2, 0), (10, "F", "B", 100, 1, 3, 0),
                   (10, "C", "B", 100, 2, 2, 0), (10, "M", "B", 100, 3, 3, L)]
    for r in both(recs(rows), place_times=[3], horizon_ns=100):
        sh = r["shadows"]
        b = sh["side"] == 1
        assert sh["true_filled"][b][0] == 1
        assert sh["est_filled"][b][0][ESTIMATORS.index("risk_averse")] == 0
        assert sh["est_filled"][b][0][ESTIMATORS.index("front_optimistic")] == 1
        # after the cancel (t=5): truth ahead 2; front_optimistic says 2 (err 0), risk_averse says 5 (err +3)
        err_at_cancel = sh["err_sum"][b][0]
        assert err_at_cancel[ESTIMATORS.index("front_optimistic")] == 0.0
        assert err_at_cancel[ESTIMATORS.index("risk_averse")] == 3.0


def test_size_increase_loses_priority_size_decrease_keeps_it():
    # shadow at t=3 behind orders 1 (3) and 2 (2); order 3 (4) joins behind it at t=4.
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),
                   (5, "M", "B", 100, 1, 1, L),            # order 1 shrinks 3 -> 1: stays ahead (truth V 5 -> 3)
                   (6, "M", "B", 100, 6, 2, L)]            # order 2 grows 2 -> 6: requeues behind (truth V 3 -> 1)
    for r in both(recs(rows), place_times=[3], horizon_ns=100):
        sh = r["shadows"]
        b = sh["side"] == 1
        # t=4: Q 5 -> 9, no decrease: every estimator 5, truth 5 (err 0).
        # t=5: Q 9 -> 7, dec 2. risk_averse 5 (err +2); front_optimistic 3 (err 0).
        # t=6: Q 7 -> 11, an increase: estimators unchanged; truth 1. risk_averse err +4; front_optimistic +2.
        # Estimates are capped at the visible level size Q, so the cap never binds here.
        assert sh["nobs"][b][0] == 3
        assert sh["err_sum"][b][0][ESTIMATORS.index("risk_averse")] == 6.0
        assert sh["err_sum"][b][0][ESTIMATORS.index("front_optimistic")] == 2.0
        lab = r["labels"]
        i2 = np.where(lab["order_id"] == 2)[0]
        assert OUTCOMES[lab["outcome"][i2[0]]] == "requeued" and lab["rank_at_add"][i2[1]] == 2


def test_trade_through_fills_every_estimator():
    # sell sweeps 100 (5 lots) and continues to 99: a bid shadow at 100 must be filled by truth and all estimators
    rows = BASE + [(4, "A", "B", 99, 2, 4, L),
                   (10, "T", "A", 100, 5, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 2, 2, 0),
                   (10, "T", "A", 99, 1, 0, 0), (10, "F", "B", 99, 1, 4, 0),
                   (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 2, 2, 0), (10, "M", "B", 99, 1, 4, L)]
    for r in both(recs(rows), place_times=[3], horizon_ns=100):
        sh = r["shadows"]
        b = sh["side"] == 1
        assert sh["true_filled"][b][0] == 1 and (sh["est_filled"][b][0] == 1).all()


def test_labels_outcomes():
    rows = BASE + [(4, "A", "B", 100, 1, 5, L),
                   (10, "T", "A", 100, 3, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "C", "B", 100, 3, 1, L),  # 1 filled
                   (11, "C", "B", 100, 2, 2, L),                                                            # 2 cancelled
                   (12, "M", "B", 99, 1, 5, L)]                                                             # 5 moved
    for r in both(recs(rows)):
        lab = r["labels"]
        out = {int(o): OUTCOMES[int(x)] for o, x in zip(lab["order_id"], lab["outcome"])}
        assert out[1] == "filled" and out[2] == "cancelled" and out[9] == "censored"
        i5 = np.where(lab["order_id"] == 5)[0]
        assert OUTCOMES[lab["outcome"][i5[0]]] == "moved_price" and lab["rank_at_add"][i5[0]] == 2
        assert lab["size_ahead_at_add"][i5[0]] == 5 and lab["dist"][i5[1]] == 0   # bid side empty before the re-add
        assert int(lab["filled_qty"][lab["order_id"] == 1][0]) == 3


def test_fifo_violation_is_detected():
    # a fill that hits order 2 while order 1 (ahead, not consumed) still rests: rank 1
    rows = BASE + [(10, "T", "A", 100, 2, 0, 0), (10, "F", "B", 100, 2, 2, 0), (10, "C", "B", 100, 2, 2, L)]
    for r in both(recs(rows)):
        assert list(r["fifo_rank"]) == [1]


def test_snapshot_clear_and_implicit_clear():
    # R + snapshot; then a normal event; then a snapshot block with no R -> implicit clear (SPEC 2)
    rows = BASE + [(20, "R", "N", 0, 0, 0, 0), (21, "A", "B", 100, 7, 1, SNAP), (21, "A", "A", 102, 1, 9, SNAP | L),
                   (25, "A", "B", 99, 3, 7, L),
                   (30, "A", "B", 100, 4, 1, SNAP), (30, "A", "A", 103, 2, 8, SNAP | L)]
    for r in both(recs(rows), sample_times=[25, 35]):
        s = r["samples"]
        assert s[0][0] == 100 * TICK and s[0][1] == 7 and s[0][30] == 102 * TICK
        assert s[1][1] == 4 and s[1][3] == 0 and s[1][30] == 103 * TICK and r["stats"]["implicit_clear"] == 1


def test_rebuild_matches_generator_and_fifo_is_exact():
    for mode in ("volume", "tail", "front", "order"):
        rec, st, truth, ex = generate(n_events=8000, sample_every=40, seed=7, cancel_mode=mode,
                                      p_clear=0.002, p_implicit_snap=0.002)
        for r in both(rec, sample_times=st):
            assert np.array_equal(r["samples"], truth), mode
            fr = r["fifo_rank"]
            assert (fr[fr >= 0] == 0).all() and r["stats"]["fill_unknown"] == 0


def _estimator_summary(mode, seed=3):
    rec, st, truth, ex = generate(n_events=30000, sample_every=1000, seed=seed, cancel_mode=mode, p_modify=0.0)
    pt = np.arange(rec["ts_event"][200], rec["ts_event"][-1] - 120_000_000, 100_000_000)
    r = (fast.run(rec, tick=ex.tick, place_times=pt, horizon_ns=60_000_000) if HAVE_CORE else
         Reference(tick=ex.tick, horizon_ns=60_000_000).run(rec, place_times=pt))
    sh = r["shadows"]
    ok = sh["end"] != 2
    n = sh["nobs"][ok].sum()
    return sh, ok, sh["err_sum"][ok].sum(0) / n


def test_known_answer_tail_cancels_make_risk_averse_exact():
    sh, ok, bias = _estimator_summary("tail")
    k = ESTIMATORS.index("risk_averse")
    assert bias[k] == 0.0 and sh["abserr_sum"][ok, k].sum() == 0.0
    assert np.array_equal(sh["est_filled"][ok, k], sh["true_filled"][ok])
    f = sh["true_filled"][ok] == 1
    assert np.array_equal(sh["est_t_fill"][ok, k][f], sh["true_t_fill"][ok][f])


def test_known_answer_front_cancels_make_front_optimistic_exact():
    sh, ok, bias = _estimator_summary("front")
    k = ESTIMATORS.index("front_optimistic")
    assert bias[k] == 0.0 and np.array_equal(sh["est_filled"][ok, k], sh["true_filled"][ok])


def test_known_answer_volume_cancels_make_linear_model_nearly_unbiased():
    sh, ok, bias = _estimator_summary("volume")
    assert abs(bias[ESTIMATORS.index("prob_power1")]) < 0.25
    assert bias[ESTIMATORS.index("risk_averse")] > 1.0       # conservative model overstates the queue


def test_no_lookahead_in_book_samples():
    rec, st, truth, ex = generate(n_events=4000, sample_every=100, seed=5)
    cut = int(st[len(st) // 2])
    mutated = {k: v.copy() for k, v in rec.items()}
    after = rec["ts_event"] > cut
    mutated["size"][after] = 1 + (mutated["size"][after] * 7) % 13      # scramble the future
    for f in ([lambda x: Reference(tick=ex.tick).run(x, sample_times=st)] +
              ([lambda x: fast.run(x, tick=ex.tick, sample_times=st)] if HAVE_CORE else [])):
        a, b = f(rec)["samples"], f(mutated)["samples"]
        early = st <= cut
        assert np.array_equal(a[early], b[early])


def test_cpp_equals_reference_bitwise():
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    for seed in range(4):
        for mode in ("volume", "tail", "front", "order"):
            rec, st, truth, ex = generate(n_events=4000, sample_every=37, seed=100 + seed, cancel_mode=mode,
                                          p_clear=0.003, p_implicit_snap=0.003)
            pt = np.arange(rec["ts_event"][50], rec["ts_event"][-1], 30_000_000)
            a = Reference(tick=ex.tick, horizon_ns=40_000_000).run(rec, place_times=pt, sample_times=st)
            b = fast.run(rec, tick=ex.tick, horizon_ns=40_000_000, place_times=pt, sample_times=st)
            assert {k: int(v) for k, v in a["stats"].items()} == {k: int(v) for k, v in b["stats"].items()}
            for grp in ("shadows", "labels"):
                for k in a[grp]:
                    assert np.array_equal(np.asarray(a[grp][k]), np.asarray(b[grp][k])), (seed, mode, grp, k)
            assert np.array_equal(a["fifo_rank"], b["fifo_rank"]) and np.array_equal(a["samples"], b["samples"])


class _ProbeReference(Reference):
    """The reference plus counters that show a comparison is not vacuous (reads state, changes nothing)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.max_live = self.max_levels = self.through_obs = self.price_moves = self.expiry_ties = self.retouch = 0

    def _touch(self, side, price):
        if (side, price) in self._ev_touched and any(s.side == side and s.price == price for s in self.shadows):
            self.retouch += 1                                               # Q0 must stay the first touch's Q
        super()._touch(side, price)

    def _retire_and_place(self, place_times, pi):
        self.expiry_ties += sum(self._ts == s.t_end for s in self.shadows)   # boundary exactly at a horizon end
        return super()._retire_and_place(place_times, pi)

    def _end_event(self):
        live = self.shadows
        self.max_live = max(self.max_live, len(live))
        self.max_levels = max(self.max_levels, len({(s.side, s.price) for s in live}))
        self.through_obs += sum(any(ag == ("A" if s.side == "B" else "B") and s.worse(px)
                                    for px, ag, _ in self._ev_trades) for s in live)
        super()._end_event()

    def _on_modify(self, oid, side, price, size):
        o = self.orders.get(oid)
        if o is not None and o["price"] != price:
            self.price_moves += 1
        super()._on_modify(oid, side, price, size)


def _assert_same_output(a, b, ctx):
    """Every output of the two engines equal: stats, and each array's dtype, shape and bytes."""
    assert {k: int(v) for k, v in a["stats"].items()} == {k: int(v) for k, v in b["stats"].items()}, ctx
    for grp in ("shadows", "labels"):
        assert set(a[grp]) == set(b[grp]), (ctx, grp)
        for k in a[grp]:
            x, y = np.asarray(a[grp][k]), np.asarray(b[grp][k])
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (ctx, grp, k)
    for k in ("fifo_rank", "fifo_ts", "samples", "queue"):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (ctx, k)


def test_cpp_equals_reference_bitwise_many_shadows():
    # Known answer: identical output from both engines (every stats counter, FIFO ranks, shadows, labels,
    # samples, queue snapshots; dtype, shape and bytes) when well over 100 shadows are live at once at many
    # prices of both sides. The streams have market orders that sweep several levels (trade-throughs),
    # price-change modifies, clears and snapshots without a clear (half of them in the middle of an event),
    # shadows larger than one lot (partial truth fills), and either ts_event on a 1 ms grid (event boundaries
    # land exactly on a shadow's horizon end) or snapshot records whose ts_event goes back in time, as in
    # Databento's snapshot. The counters below check that each of these actually happens.
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    rng = np.random.default_rng(11)
    tot = dict(live=0, levels=0, through=0, moves=0, cleared=0, clears=0, implicit=0, mid_event_clears=0,
               back_steps=0, ties=0, retouch=0, shadows=0)
    for j, mode in enumerate(("order", "volume", "tail", "front")):
        rec, st, truth, ex = generate(n_events=5000, sample_every=41, seed=300 + j, cancel_mode=mode, p_modify=0.15,
                                      p_market=0.18, max_mkt=30, p_clear=0.003, p_implicit_snap=0.003)
        if j % 2 == 0:                            # ts_event on a 1 ms grid: horizon ends tie with boundaries
            rec["ts_event"] = rec["ts_event"] // 1_000_000 * 1_000_000
        else:                                     # snapshot records keep older ts_event (non-monotone stream)
            snap = (rec["flags"] & SNAP) != 0
            rec["ts_event"][snap] -= rng.integers(0, 300_000_000, int(snap.sum())).astype(np.uint64)
        # merge events (as one CME packet can carry several actions): half of the clears (R, or a snapshot
        # without R) then happen mid-event, after the event touched levels, and 20% of the other boundaries go
        snap = (rec["flags"] & SNAP) != 0
        starts = np.where((rec["action"] == b"R") | (snap & ~np.concatenate([[False], snap[:-1]])))[0]
        starts = starts[(starts > 0) & (rec["action"][starts - 1] != b"R")]
        merge = starts[rng.uniform(size=len(starts)) < 0.5] - 1
        tot["mid_event_clears"] += len(merge)
        last = np.where((rec["flags"][:-1] & L) != 0)[0]
        merge = np.concatenate([merge, last[rng.uniform(size=len(last)) < 0.2]])
        rec["flags"][merge] &= np.uint8(255 - L)
        ts = rec["ts_event"].astype(np.int64)
        pt = np.arange(ts[30], ts[-1], 1_500_000)  # one bid + one ask shadow about every event
        qt = np.sort(np.concatenate([st[::2], rng.integers(ts[0], ts[-1], 40)]))
        kw = dict(label_ticks=(2, 5)[j % 2], shadow_size=(3, 1)[j % 2], horizon_ns=150_000_000, sample_levels=6)
        ref = _ProbeReference(tick=ex.tick, **kw)
        a = ref.run(rec, place_times=pt, sample_times=st, queue_times=qt, queue_levels=4)
        b = fast.run(rec, tick=ex.tick, place_times=pt, sample_times=st, queue_times=qt, queue_levels=4, **kw)
        _assert_same_output(a, b, (j, mode))
        tot["live"] = max(tot["live"], ref.max_live); tot["levels"] = max(tot["levels"], ref.max_levels)
        tot["through"] += ref.through_obs; tot["moves"] += ref.price_moves
        tot["cleared"] += int((a["shadows"]["end"] == 2).sum()); tot["shadows"] += len(a["shadows"]["sid"])
        tot["clears"] += a["stats"]["clears"]; tot["implicit"] += a["stats"]["implicit_clear"]
        tot["back_steps"] += int((np.diff(ts) < 0).sum())
        tot["ties"] += ref.expiry_ties; tot["retouch"] += ref.retouch
    assert tot["live"] >= 100 and tot["levels"] >= 6 and tot["shadows"] > 5000, tot
    assert tot["through"] > 100 and tot["moves"] > 100 and tot["cleared"] > 0, tot
    assert tot["clears"] > 0 and tot["implicit"] > 0 and tot["mid_event_clears"] > 0, tot
    assert tot["back_steps"] > 0 and tot["ties"] > 0 and tot["retouch"] > 100, tot


def test_start_at_snapshot_drops_incomplete_prefix():
    # Live records before midnight reference orders never seen (no snapshot at the reopen), then
    # Databento's snapshot: R + F_SNAPSHOT adds whose ts_event is the orders' OLD entry time.
    # Known answer: 3 records dropped; first live ts = 1000; t range from live records only; book
    # = snapshot + live; A1 pairs: level bid 100 has 2 adds in ts order -> 1 pair, in order.
    from mboq.io import start_at_snapshot
    S = SNAP | 8
    rows = [(900, "A", "B", 99, 1, 50, L), (901, "C", "B", 100, 7, 77, L), (902, "F", "B", 100, 1, 78, L),
            (10, "R", "N", 0, 0, 0, S),
            (5, "A", "B", 100, 3, 1, S), (7, "A", "B", 100, 2, 2, S), (6, "A", "A", 101, 5, 9, S | L),
            (1000, "A", "B", 100, 4, 3, L)]
    rec, info = start_at_snapshot(recs(rows))
    assert info["snapshot_found"] and info["dropped_before_snapshot"] == 3 and info["snapshot_records"] == 3
    assert info["first_live_ts"] == 1000 and len(rec["ts_event"]) == 5
    assert info["a1_pairs_same_level"] == 1 and info["a1_pairs_ts_nondecreasing"] == 1
    for r in both(rec, sample_times=[1000]):
        assert list(r["samples"][0][:6]) == [100 * TICK, 9, 3, 0, 0, 0] and r["stats"]["clears"] == 1
    none, info2 = start_at_snapshot(recs(BASE))
    assert not info2["snapshot_found"] and len(none["ts_event"]) == len(BASE)


def test_databento_sample_file_decodes_and_runs():
    path = ROOT / "tests" / "data" / "databento_sample_GLBX_mbo.dbn.zst"
    if not path.exists():
        print("   (skipped: sample file absent; run scripts/fetch_test_data.sh)")
        return
    try:
        from mboq.io import load_mbo
    except ImportError:
        print("   (skipped: databento not installed)")
        return
    rec, meta = load_mbo(path)
    assert meta["schema"] == "mbo" and len(rec["ts_event"]) == 4
    for r in both(rec, sample_times=[int(rec["ts_event"][-1])]):
        assert r["stats"]["adds"] == 4 and r["samples"][0][1] == 2      # best bid 3675.75 x 2 from the snapshot


# ---------------------------------------------------------------------------- queue snapshots (SPEC 8)
QC = {c: j for j, c in enumerate(QUEUE_COLUMNS)}
QROWS = [  # book build-up, one record per event: bids 100: 1(3) 2(2) 3(4); 99: 4(5) 5(1); 98: 6(8)
    (1, "A", "B", 100, 3, 1, L), (2, "A", "B", 100, 2, 2, L), (3, "A", "B", 100, 4, 3, L),
    (4, "A", "B", 99, 5, 4, L), (5, "A", "B", 99, 1, 5, L), (6, "A", "B", 98, 8, 6, L),
    (7, "A", "A", 101, 5, 9, L), (8, "A", "A", 102, 2, 11, L), (9, "A", "A", 101, 1, 10, L),  # asks 101: 9(5) 10(1); 102: 11(2)
    (10, "M", "B", 100, 6, 1, L),                       # size increase 3 -> 6: order 1 goes to the back of 100
    (11, "M", "B", 100, 3, 3, L),                       # size decrease 4 -> 3: order 3 keeps its place
    (12, "M", "A", 102, 5, 9, L),                       # price change 101 -> 102: order 9 joins the back of 102
    (13, "T", "A", 100, 4, 0, 0), (13, "F", "B", 100, 2, 2, 0), (13, "F", "B", 100, 2, 3, 0),  # sell 4 at 100:
    (13, "C", "B", 100, 2, 2, 0), (13, "M", "B", 100, 1, 3, L),                              # 2 filled, 3 partial 3 -> 1
    (30, "A", "B", 100, 9, 8, L)]                       # after the t=20 snapshot: only the t=100 one sees it


def _q(rows):
    """Expected rows given as (q, id, side, price_ticks, level, rank, size, ahead, count, total)."""
    out = np.array(rows, dtype=np.int64).reshape(-1, len(QUEUE_COLUMNS))
    out[:, QC["price"]] *= TICK
    return out


def test_queue_snapshot_hand_computed():
    # Known answer (hand-computed, queue_levels=2 so bid 98 is cut). Snapshot times 0, 9, 11, 20, 100:
    # t=0 precedes every record (empty book: index 0 has no rows); t=100 is after the last record.
    qt = [0, 9, 11, 20, 100]
    asks_before = [(-1, 101, 0, 0, 5, 0, 2, 6), (-1, 101, 0, 1, 1, 5, 2, 6), (-1, 102, 1, 0, 2, 0, 1, 2)]
    exp = _q(
        # t=9: build-up complete, FIFO = arrival order
        [(1, 1, 1, 100, 0, 0, 3, 0, 3, 9), (1, 2, 1, 100, 0, 1, 2, 3, 3, 9), (1, 3, 1, 100, 0, 2, 4, 5, 3, 9),
         (1, 4, 1, 99, 1, 0, 5, 0, 2, 6), (1, 5, 1, 99, 1, 1, 1, 5, 2, 6)]
        + [(1, oid) + a for oid, a in zip((9, 10, 11), asks_before)]
        # t=11: order 1 (grew) is now last at 100; order 3 (shrank to 3) kept rank 1
        + [(2, 2, 1, 100, 0, 0, 2, 0, 3, 11), (2, 3, 1, 100, 0, 1, 3, 2, 3, 11), (2, 1, 1, 100, 0, 2, 6, 5, 3, 11),
           (2, 4, 1, 99, 1, 0, 5, 0, 2, 6), (2, 5, 1, 99, 1, 1, 1, 5, 2, 6)]
        + [(2, oid) + a for oid, a in zip((9, 10, 11), asks_before)]
        # t=20: order 2 filled and gone, order 3 (partial fill, 1 left) at the front; order 9 behind 11 at 102
        + [(3, 3, 1, 100, 0, 0, 1, 0, 2, 7), (3, 1, 1, 100, 0, 1, 6, 1, 2, 7),
           (3, 4, 1, 99, 1, 0, 5, 0, 2, 6), (3, 5, 1, 99, 1, 1, 1, 5, 2, 6),
           (3, 10, -1, 101, 0, 0, 1, 0, 1, 1), (3, 11, -1, 102, 1, 0, 2, 0, 2, 7), (3, 9, -1, 102, 1, 1, 5, 2, 2, 7)]
        # t=100: order 8 (t=30) joined the back of 100
        + [(4, 3, 1, 100, 0, 0, 1, 0, 3, 16), (4, 1, 1, 100, 0, 1, 6, 1, 3, 16), (4, 8, 1, 100, 0, 2, 9, 7, 3, 16),
           (4, 4, 1, 99, 1, 0, 5, 0, 2, 6), (4, 5, 1, 99, 1, 1, 1, 5, 2, 6),
           (4, 10, -1, 101, 0, 0, 1, 0, 1, 1), (4, 11, -1, 102, 1, 0, 2, 0, 2, 7), (4, 9, -1, 102, 1, 1, 5, 2, 2, 7)])
    for r in both(recs(QROWS), queue_times=qt, queue_levels=2):
        assert r["queue"].dtype == np.int64 and r["queue"].shape == (31, 10)
        assert np.array_equal(r["queue"], exp)
    # default queue_levels=5 also shows bid 98 (level 2: order 6, 8 lots alone); no queue_times -> (0, 10)
    t20 = exp[exp[:, 0] == 3].copy()
    t20[:, 0] = 0
    exp5 = np.concatenate([t20[:4], _q([(0, 6, 1, 98, 2, 0, 8, 0, 1, 8)]), t20[4:]])
    for r in both(recs(QROWS), queue_times=[20]):
        assert r["queue"].shape == (8, 10) and np.array_equal(r["queue"], exp5)
    for r in both(recs(QROWS)):
        assert r["queue"].shape == (0, 10) and r["queue"].dtype == np.int64
    for r in both(recs(QROWS), queue_times=[20], queue_levels=0):
        assert r["queue"].shape == (0, 10)
    # invalid inputs are refused by both engines
    for run in [lambda **k: Reference(tick=TICK).run(recs(QROWS), **k)] + \
               ([lambda **k: fast.run(recs(QROWS), tick=TICK, **k)] if HAVE_CORE else []):
        for bad in ({"queue_times": [20, 9]}, {"queue_times": [9], "queue_levels": -1}):
            try:
                run(**bad)
                raise AssertionError(f"accepted {bad}")
            except ValueError:
                pass


def _queue_engines(tick):
    """Both engines with one call signature (sample_levels is a constructor argument of Reference)."""
    def ref(x, sample_levels=10, **k):
        return Reference(tick=tick, sample_levels=sample_levels).run(x, **k)
    return [ref] + ([lambda x, **k: fast.run(x, tick=tick, **k)] if HAVE_CORE else [])


def test_queue_snapshot_no_lookahead():
    # Known answer: rows for every queue time t <= cut are identical when each record with ts_event > cut is
    # (a) scrambled (size, price, side, order id) or (b) deleted. Control: later snapshots do change under (a).
    rec, st, truth, ex = generate(n_events=4000, sample_every=100, seed=5, p_clear=0.002, p_implicit_snap=0.002)
    qt = np.sort(np.concatenate([st, st + 1, st - 1]))
    cut = int(st[len(st) // 2])
    after = rec["ts_event"] > cut
    scr = {k: v.copy() for k, v in rec.items()}
    scr["size"][after] = 1 + (scr["size"][after] * 7) % 13
    scr["price"][after] += 3 * ex.tick
    scr["side"][after] = np.where(scr["side"][after] == b"B", b"A", b"B")
    scr["order_id"][after] = scr["order_id"][after] + 17
    cutoff = {k: v[~after].copy() for k, v in rec.items()}
    for f in _queue_engines(ex.tick):
        a, b, c = (f(x, queue_times=qt, queue_levels=5)["queue"] for x in (rec, scr, cutoff))
        early = lambda q: q[qt[q[:, QC["queue_time_index"]]] <= cut]  # noqa: E731
        assert len(early(a)) > 1000
        assert np.array_equal(early(a), early(b)) and np.array_equal(early(a), early(c))
        assert not np.array_equal(a[len(early(a)):], b[len(early(b)):])   # the scramble does reach later rows


def test_queue_snapshot_cpp_equals_reference_bitwise():
    # Known answer: identical `queue` arrays (shape, dtype, every value) from both engines on randomized streams
    # with clears and implicit snapshots; every pre-existing output is still equal across engines and is
    # unchanged by asking for queue snapshots.
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    rng = np.random.default_rng(8)
    n_rows = clears = implicit = 0
    for seed in range(4):
        for j, mode in enumerate(("volume", "tail", "front", "order")):
            rec, st, truth, ex = generate(n_events=4000, sample_every=37, seed=200 + seed, cancel_mode=mode,
                                          p_clear=0.003, p_implicit_snap=0.003)
            ts = rec["ts_event"].astype(np.int64)
            pt = np.arange(ts[50], ts[-1], 30_000_000)
            qt = np.sort(np.concatenate([st[::3], rng.integers(ts[0] - 10, ts[-1] + 10, 60), st[5:8], [ts[0] - 1]]))
            ql = (1, 3, 5, 10)[(seed + j) % 4]
            kw = dict(horizon_ns=40_000_000, place_times=pt, sample_times=st)
            a = Reference(tick=ex.tick, horizon_ns=40_000_000).run(rec, place_times=pt, sample_times=st,
                                                                   queue_times=qt, queue_levels=ql)
            b = fast.run(rec, tick=ex.tick, queue_times=qt, queue_levels=ql, **kw)
            a0 = Reference(tick=ex.tick, horizon_ns=40_000_000).run(rec, place_times=pt, sample_times=st)
            b0 = fast.run(rec, tick=ex.tick, **kw)
            assert a["queue"].dtype == b["queue"].dtype == np.int64 and a["queue"].shape[1] == 10
            assert np.array_equal(a["queue"], b["queue"]), (seed, mode)
            for x, y in ((a, b), (a, a0), (b, b0)):
                assert {k: int(v) for k, v in x["stats"].items()} == {k: int(v) for k, v in y["stats"].items()}
                for grp in ("shadows", "labels"):
                    for k in x[grp]:
                        assert np.array_equal(np.asarray(x[grp][k]), np.asarray(y[grp][k])), (seed, mode, grp, k)
                for k in ("fifo_rank", "fifo_ts", "samples"):
                    assert np.array_equal(x[k], y[k]), (seed, mode, k)
            n_rows += len(a["queue"]); clears += a["stats"]["clears"]; implicit += a["stats"]["implicit_clear"]
    assert n_rows > 10_000 and clears > 0 and implicit > 0          # the comparison is not vacuous


def test_queue_snapshot_consistent_with_book_samples():
    # Known answer: for every snapshot, side and level, the rows reproduce the section 7 sample at the same
    # time (price, size, count; same levels present), ranks are 0..count-1, size_ahead is the exclusive
    # prefix sum of size, and size_ahead + size of the last order equals level_total_size.
    N = 6
    for mode in ("volume", "order"):
        rec, st, truth, ex = generate(n_events=6000, sample_every=29, seed=31, cancel_mode=mode,
                                      p_clear=0.003, p_implicit_snap=0.003)
        qt = np.sort(np.concatenate([st, st[::2] + 1]))         # event times and times between events
        for f in _queue_engines(ex.tick):
            r = f(rec, sample_times=qt, sample_levels=N, queue_times=qt, queue_levels=N)
            q, s = r["queue"], r["samples"]
            assert len(q) > 5000
            for qi in range(len(qt)):
                rows_q = q[q[:, QC["queue_time_index"]] == qi]
                for side, off in ((1, 0), (-1, 3 * N)):
                    rs = rows_q[rows_q[:, QC["side"]] == side]
                    present = [lvl for lvl in range(N) if s[qi, off + 3 * lvl + 2] > 0]
                    assert sorted(set(rs[:, QC["level_index"]].tolist())) == present, (mode, qi, side)
                    for lvl in present:
                        g = rs[rs[:, QC["level_index"]] == lvl]
                        px, size, count = s[qi, off + 3 * lvl: off + 3 * lvl + 3]
                        assert (g[:, QC["price"]] == px).all() and len(g) == count
                        assert (g[:, QC["level_total_size"]] == size).all() and (g[:, QC["level_order_count"]] == count).all()
                        assert np.array_equal(g[:, QC["rank"]], np.arange(count))
                        sz = g[:, QC["size"]]
                        assert np.array_equal(g[:, QC["size_ahead"]], np.concatenate([[0], np.cumsum(sz)[:-1]]))
                        assert g[-1, QC["size_ahead"]] + sz[-1] == size and sz.sum() == size


def test_queue_snapshot_equals_generator_fifo_queues():
    # Known answer: the synthetic exchange's own price-time queues (oid, size in priority order) after the
    # sampled events, in all four cancel regimes, with clears and snapshot blocks.
    for mode in ("volume", "tail", "front", "order"):
        ex = SyntheticExchange(seed=41, cancel_mode=mode, p_clear=0.003, p_implicit_snap=0.003)
        times, exp = [], []
        for j in range(5000):
            ex.step()
            if j % 31 == 30:
                qi = len(times)
                times.append(ex.ts)
                for side, sc in (("B", 1), ("A", -1)):
                    for lvl, px in enumerate(sorted(ex.books[side], reverse=(side == "B"))[:4]):
                        lv = ex.books[side][px]
                        ahead = 0
                        for rank, (oid, size) in enumerate(lv.items()):
                            exp.append((qi, oid, sc, px * ex.tick, lvl, rank, size, ahead, len(lv), sum(lv.values())))
                            ahead += size
        exp = np.array(exp, dtype=np.int64)
        rec = ex.records()
        for f in _queue_engines(ex.tick):
            assert np.array_equal(f(rec, queue_times=np.array(times), queue_levels=4)["queue"], exp), mode


# ---------------------------------------------------------------------------- SPEC v0.3
class _V02Rules(Reference):
    """Control only: the SPEC v0.2 rules. A same-price M is compared with the rebuilt size (ignoring
    fill_since_update), and the FIFO rank nets out only F records listed earlier in the event."""

    def _requeues(self, o, size):
        return size > o["size"]

    def _rank_fill(self, oid, lv):
        rank = 0
        for other_id, other in lv.items():
            if other_id == oid:
                break
            if self._ev_fill.get(other_id, 0) < other["size"]:
                rank += 1
        self.fifo_ranks.append(rank)


class _V02RequeueOnly(Reference):
    """Control: v0.2 M rule, v0.3 FIFO rank."""
    _requeues = _V02Rules._requeues


class _V02RankOnly(Reference):
    """Control: v0.3 M rule, v0.2 FIFO rank."""
    _rank_fill = _V02Rules._rank_fill


def _bid_queue(q, qi, price_ticks):
    """(order_id, size) in FIFO order at one bid price in snapshot qi."""
    g = q[(q[:, QC["queue_time_index"]] == qi) & (q[:, QC["side"]] == 1) & (q[:, QC["price"]] == price_ticks * TICK)]
    return [(int(a), int(b)) for a, b in zip(g[:, QC["order_id"]], g[:, QC["size"]])]


def test_v03_iceberg_refill_requeues():
    # Known answer (hand-computed). Bids at 100: 1(3) 2(2); the shadow is placed at t=3 behind them (V_true 5);
    # order 3 (4) joins behind it at t=4. Order 1 is a native iceberg: at t=10 a 3-lot sell empties its display
    # (F 3) and the refill shows 3 again (M 3). v0.3: rem = 3 - 3 = 0 < 3, so order 1 requeues behind 2 and 3 and
    # leaves the shadow's ahead set (V_true 2). At t=20 a 3-lot sell fills 2 (2 lots) and 1 lot of 3 (behind the
    # shadow): every F has rank 0, the shadow fills at t=20 with no FIFO anomaly. Estimators: the t=10 trade (3)
    # takes every estimate 5 -> 2 (no unexplained decrease: Q 9 -> 9), equal to the truth, so err_sum = 0
    # (samples at t=4: 5 vs 5, t=10: 2 vs 2; at t=20 the truth fills, no sample).
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),
                   (10, "T", "A", 100, 3, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "M", "B", 100, 3, 1, L),
                   (20, "T", "A", 100, 3, 0, 0), (20, "F", "B", 100, 2, 2, 0), (20, "F", "B", 100, 1, 3, 0),
                   (20, "C", "B", 100, 2, 2, 0), (20, "M", "B", 100, 3, 3, L)]
    kw = dict(place_times=[3], horizon_ns=100, queue_times=[15, 25])
    for r in both(recs(rows), **kw):
        assert list(r["fifo_rank"]) == [0, 0, 0] and r["stats"]["shadow_fifo_anomaly"] == 0
        assert _bid_queue(r["queue"], 0, 100) == [(2, 2), (3, 4), (1, 3)]
        assert _bid_queue(r["queue"], 1, 100) == [(3, 3), (1, 3)]
        sh = r["shadows"]
        b = sh["side"] == 1
        assert sh["true_filled"][b][0] == 1 and sh["true_t_fill"][b][0] == 20 and sh["nobs"][b][0] == 2
        assert (sh["err_sum"][b][0] == 0.0).all() and (sh["est_filled"][b][0] == 1).all()
        lab = r["labels"]
        i1 = np.where(lab["order_id"] == 1)[0]          # the displayed tranche ends 'requeued' at the refill
        assert [OUTCOMES[x] for x in lab["outcome"][i1]] == ["requeued", "censored"]
        assert list(lab["filled_qty"][i1]) == [3, 0] and list(lab["t_end"][i1]) == [10, -1]
        assert lab["t_add"][i1[1]] == 10 and lab["rank_at_add"][i1[1]] == 2 and lab["size_ahead_at_add"][i1[1]] == 6
    # control, v0.2 rules: the refill keeps priority (3 is not > 3), so the t=20 fills rank 1 (order 1 still
    # "ahead"), the shadow's ahead set keeps order 1 (err -3 at t=10) and the fill of order 3 is an anomaly
    c = _V02Rules(tick=TICK, horizon_ns=100).run(recs(rows), place_times=[3], queue_times=[15, 25])
    assert list(c["fifo_rank"]) == [0, 1, 1] and c["stats"]["shadow_fifo_anomaly"] == 1
    assert _bid_queue(c["queue"], 0, 100) == [(1, 3), (2, 2), (3, 4)]
    assert (c["shadows"]["err_sum"][c["shadows"]["side"] == 1][0] == -3.0).all()


def test_v03_fill_since_update_keeps_priority_after_partial_fill():
    # Known answer (hand-computed) for the same-price M rule with fill_since_update (fsu), bid 100 queue after
    # each event. Start 1(3) 2(2).
    #  t=10 F 1 on 1, M 1 -> 2: rem = 3 - 1 = 2, not larger: keeps priority      [1(2) 2(2)]
    #  t=11 M 1 -> 2 (fsu was reset by the last M): rem 2, keeps priority         [1(2) 2(2)]
    #  t=12 F 1 on 1, M 1 -> 2: rem = 1, larger: requeues (v0.2 would keep it)   [2(2) 1(2)]
    #  t=13 F 1 on 2 alone (no C/M): fsu(2) = 1, book unchanged                   [2(2) 1(2)]
    #  t=14 M 2 -> 2: rem = 2 - 1 = 1, larger: requeues (fsu spans events)         [1(2) 2(2)]
    #  t=15 F 1 on 1, partial C 1: size 1, fsu reset                              [1(1) 2(2)]
    #  t=16 M 1 -> 1: rem 1, keeps priority (no reset would requeue it)           [1(1) 2(2)]
    #  t=17 F 1 on 1 alone: fsu(1) = 1
    #  t=18 R + snapshot 1(1) 2(2): snapshot adds start at fsu 0                  [1(1) 2(2)]
    #  t=19 M 1 -> 1: rem 1, keeps priority (no reset by R would requeue it)      [1(1) 2(2)]
    rows = BASE + [(10, "T", "A", 100, 1, 0, 0), (10, "F", "B", 100, 1, 1, 0), (10, "M", "B", 100, 2, 1, L),
                   (11, "M", "B", 100, 2, 1, L),
                   (12, "T", "A", 100, 1, 0, 0), (12, "F", "B", 100, 1, 1, 0), (12, "M", "B", 100, 2, 1, L),
                   (13, "T", "A", 100, 1, 0, 0), (13, "F", "B", 100, 1, 2, L),
                   (14, "M", "B", 100, 2, 2, L),
                   (15, "T", "A", 100, 1, 0, 0), (15, "F", "B", 100, 1, 1, 0), (15, "C", "B", 100, 1, 1, L),
                   (16, "M", "B", 100, 1, 1, L),
                   (17, "T", "A", 100, 1, 0, 0), (17, "F", "B", 100, 1, 1, L),
                   (18, "R", "N", 0, 0, 0, 0), (18, "A", "B", 100, 1, 1, SNAP), (18, "A", "B", 100, 2, 2, SNAP),
                   (18, "A", "A", 101, 5, 9, SNAP | L),
                   (19, "M", "B", 100, 1, 1, L)]
    qt = list(range(10, 20))
    exp = [[(1, 2), (2, 2)], [(1, 2), (2, 2)], [(2, 2), (1, 2)], [(2, 2), (1, 2)]] + [[(1, 2), (2, 2)]] + \
          [[(1, 1), (2, 2)]] * 5
    for r in both(recs(rows), queue_times=qt):
        assert [_bid_queue(r["queue"], i, 100) for i in range(10)] == exp
        assert list(r["fifo_rank"]) == [0, 0, 0, 0, 0]
        lab = r["labels"]
        out = [(int(o), int(t), OUTCOMES[int(x)], int(e), int(f)) for o, t, x, e, f in
               zip(lab["order_id"], lab["t_add"], lab["outcome"], lab["t_end"], lab["filled_qty"])]
        # lives: 1 requeued at 12 (fills at 10, 12), 2 requeued at 14 (fill at 13); new lives cleared by the R
        assert out == [(1, 1, "requeued", 12, 2), (2, 2, "requeued", 14, 1), (9, 3, "cleared", 18, 0),
                       (1, 12, "cleared", 18, 2), (2, 14, "cleared", 18, 0)], out
    # control, v0.2 rule: t=12 and t=14 keep priority, so order 2's lone fill at t=13 ranks 1
    c = _V02Rules(tick=TICK).run(recs(rows), queue_times=qt)
    assert [_bid_queue(c["queue"], i, 100) for i in range(5)] == [[(1, 2), (2, 2)]] * 5
    assert list(c["fifo_rank"]) == [0, 0, 1, 0, 0]
    # an M to size 0 removes the order even when the reported fills exceed its size (rem = 3 - 4 < 0): it is
    # checked before the requeue test, so order 1 leaves as 'filled' instead of requeueing with size 0
    over = BASE + [(10, "T", "A", 100, 4, 0, 0), (10, "F", "B", 100, 4, 1, 0), (10, "M", "B", 100, 0, 1, L)]
    for r in both(recs(over), queue_times=[20]):
        assert _bid_queue(r["queue"], 0, 100) == [(2, 2)]
        lab = r["labels"]
        assert [OUTCOMES[x] for x in lab["outcome"][lab["order_id"] == 1]] == ["filled"]


def test_v03_out_of_order_fills_rank_with_event_totals():
    # Known answer (hand-computed) for the SPEC 3 rank: ahead-ness and sizes at the F record, fills = the event's
    # totals. Bids at 100: 1(3) 2(2) and, from t=4, 3(4); a bid shadow is placed at t=3 (behind 1 and 2).
    three = BASE + [(4, "A", "B", 100, 4, 3, L)]
    cases = [
        # (a) the feed lists the fills in reverse priority order; 1 and 2 are fully consumed in the event: all
        #     rank 0 (v0.2: 2, 1, 0). The shadow fills (order 3 is behind it); shadow_fifo_anomaly keeps the v0.2
        #     definition (ahead volume not consumed by EARLIER F records = 5 at F(3)), so it counts 1.
        (three + [(10, "T", "A", 100, 6, 0, 0), (10, "F", "B", 100, 1, 3, 0), (10, "F", "B", 100, 2, 2, 0),
                  (10, "F", "B", 100, 3, 1, 0), (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 2, 2, 0),
                  (10, "M", "B", 100, 3, 3, L)], [0, 0, 0], [2, 1, 0], 1),
        # (b) order 2 is only partly filled in the event (1 of 2): F(3) ranks 1 (v0.2: 2)
        (three + [(10, "T", "A", 100, 5, 0, 0), (10, "F", "B", 100, 1, 3, 0), (10, "F", "B", 100, 3, 1, 0),
                  (10, "F", "B", 100, 1, 2, 0), (10, "C", "B", 100, 3, 1, 0), (10, "M", "B", 100, 1, 2, 0),
                  (10, "M", "B", 100, 3, 3, L)], [1, 0, 0], [2, 0, 0], 1),
        # (c) ahead-ness is read at the F record: order 1 is cancelled (not filled) later in the event -> rank 1
        (BASE + [(10, "T", "A", 100, 1, 0, 0), (10, "F", "B", 100, 1, 2, 0), (10, "C", "B", 100, 3, 1, 0),
                 (10, "M", "B", 100, 1, 2, L)], [1], [1], 0),
        # (d) iceberg 1 ahead of 2: the feed lists 2's fill first, then 1's display fill and the refill (requeue):
        #     the event's 3 lots on 1 consume its 3-lot display, so F(2) ranks 0 (v0.2: 1)
        (BASE + [(10, "T", "A", 100, 4, 0, 0), (10, "F", "B", 100, 1, 2, 0), (10, "F", "B", 100, 3, 1, 0),
                 (10, "M", "B", 100, 3, 1, 0), (10, "M", "B", 100, 1, 2, L)], [0, 0], [1, 0], 0),
        # (e) the stream ends inside an event (no F_LAST): its ranks close at the end of the stream
        (BASE + [(10, "T", "A", 100, 5, 0, 0), (10, "F", "B", 100, 2, 2, 0), (10, "F", "B", 100, 3, 1, 0)],
         [0, 0], [1, 0], 0),
    ]
    for j, (rows, v03, v02, anomaly) in enumerate(cases):
        for r in both(recs(rows), place_times=[3], horizon_ns=100, queue_times=[100]):
            assert list(r["fifo_rank"]) == v03, (j, list(r["fifo_rank"]))
            assert r["stats"]["shadow_fifo_anomaly"] == anomaly, j
            assert list(r["fifo_ts"]) == [10] * len(v03)
        c = _V02Rules(tick=TICK, horizon_ns=100).run(recs(rows), place_times=[3])
        assert list(c["fifo_rank"]) == v02, (j, list(c["fifo_rank"]))
    r = both(recs(cases[3][0]), queue_times=[100])
    assert all(_bid_queue(x["queue"], 0, 100) == [(2, 1), (1, 3)] for x in r)       # the refill went to the back


def _iceberg_exchange(mode, seed, p_iceberg, p_fill_shuffle, n=5000, every=31, levels=4):
    """Run the synthetic exchange; return its records, the snapshot times and its own FIFO queues there."""
    ex = SyntheticExchange(seed=seed, cancel_mode=mode, p_clear=0.003, p_implicit_snap=0.003, max_mkt=20,
                           p_iceberg=p_iceberg, p_fill_shuffle=p_fill_shuffle)
    times, exp, truth = [], [], []
    for j in range(n):
        ex.step()
        if j % every == every - 1:
            qi = len(times)
            times.append(ex.ts)
            truth.append(ex.truth_sample())
            for side, sc in (("B", 1), ("A", -1)):
                for lvl, px in enumerate(sorted(ex.books[side], reverse=(side == "B"))[:levels]):
                    lv = ex.books[side][px]
                    ahead = 0
                    for rank, (oid, size) in enumerate(lv.items()):
                        exp.append((qi, oid, sc, px * ex.tick, lvl, rank, size, ahead, len(lv), sum(lv.values())))
                        ahead += size
    return (ex.records(), np.array(times, dtype=np.int64), np.array(exp, dtype=np.int64).reshape(-1, 10),
            np.array(truth, dtype=np.int64), ex)


def _fifo_rate(r):
    fr = r["fifo_rank"]
    return float((fr[fr >= 0] == 0).mean())


def test_v03_rebuild_and_fifo_exact_with_icebergs_and_shuffled_fills():
    # Known answer: with native icebergs (refills requeue at the exchange) and F records listed out of priority
    # order within events, the v0.3 rebuild equals the synthetic exchange's book (section 7 samples) and its FIFO
    # queues (section 8 snapshots), and FIFO-consistency is exactly 100%, in both engines and all cancel regimes.
    # Controls: under the v0.2 rules the queues differ and FIFO-consistency is below 100%; each v0.2 rule alone
    # breaks the behaviour it does not handle (M rule: icebergs; rank rule: shuffled fills).
    tot = dict(refill_not_larger=0, out_of_order=0)
    for k, mode in enumerate(("volume", "tail", "front", "order")):
        rec, qt, exp, truth, ex = _iceberg_exchange(mode, 60 + k, 0.3, 0.5)
        tot["refill_not_larger"] += ex.counts["refill_not_larger"]; tot["out_of_order"] += ex.counts["out_of_order"]
        for f in _queue_engines(ex.tick):
            r = f(rec, queue_times=qt, queue_levels=4, sample_times=qt)
            assert np.array_equal(r["queue"], exp) and np.array_equal(r["samples"], truth), mode
            assert _fifo_rate(r) == 1.0 and r["stats"]["fill_unknown"] == 0, mode
        c = _V02Rules(tick=ex.tick).run(rec, queue_times=qt, queue_levels=4, sample_times=qt)
        assert not np.array_equal(c["queue"], exp) and np.array_equal(c["samples"], truth)  # same sizes, wrong order
        assert _fifo_rate(c) < 0.9, (mode, _fifo_rate(c))
    assert tot["refill_not_larger"] > 500 and tot["out_of_order"] > 300, tot
    # each v0.2 rule alone, on the stream with only the behaviour the other rule handles: still exact
    rec, qt, exp, truth, ex = _iceberg_exchange("volume", 70, 0.3, 0.0)
    assert ex.counts["refill_not_larger"] > 100 and ex.counts["shuffled"] == 0
    assert _fifo_rate(_V02RankOnly(tick=ex.tick).run(rec)) == 1.0
    r = _V02RequeueOnly(tick=ex.tick).run(rec, queue_times=qt, queue_levels=4)
    assert _fifo_rate(r) < 0.9 and not np.array_equal(r["queue"], exp)
    rec, qt, exp, truth, ex = _iceberg_exchange("volume", 71, 0.0, 0.5)
    assert ex.counts["out_of_order"] > 50 and ex.counts["icebergs"] == 0
    assert _fifo_rate(_V02RequeueOnly(tick=ex.tick).run(rec)) == 1.0
    r = _V02RankOnly(tick=ex.tick).run(rec, queue_times=qt, queue_levels=4)
    assert _fifo_rate(r) < 0.95 and np.array_equal(r["queue"], exp)     # the book is right, only the rank is not


def test_v03_equals_v02_without_icebergs_or_shuffled_fills():
    # Known answer: on the synthetic exchange with both switches off (every F applied by a C/M in the same event,
    # fills listed in priority order), every output of the v0.3 reference equals the v0.2 rules' output.
    for seed, mode in enumerate(("volume", "tail", "front", "order")):
        rec, st, truth, ex = generate(n_events=4000, sample_every=37, seed=80 + seed, cancel_mode=mode,
                                      p_clear=0.003, p_implicit_snap=0.003, max_mkt=(12, 30)[seed % 2])
        pt = np.arange(int(rec["ts_event"][50]), int(rec["ts_event"][-1]), 30_000_000)
        kw = dict(place_times=pt, sample_times=st, queue_times=st[::3])
        a = Reference(tick=ex.tick, horizon_ns=40_000_000).run(rec, **kw)
        b = _V02Rules(tick=ex.tick, horizon_ns=40_000_000).run(rec, **kw)
        _assert_same_output(a, b, mode)


class _ProbeV03(_ProbeReference):
    """Counts the v0.3 rule changes at work (reads state, changes nothing): same-price M records that requeue
    only because of fill_since_update, and F records whose rank falls when later F records of the event count."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.fsu_requeues = self.netted_later = 0
        self._imm = {}

    def _requeues(self, o, size):
        q = super()._requeues(o, size)
        self.fsu_requeues += q and size <= o["size"]
        return q

    def _rank_fill(self, oid, lv):
        imm = 0
        for other_id, other in lv.items():
            if other_id == oid:
                break
            imm += self._ev_fill.get(other_id, 0) < other["size"]
        self._imm[len(self.fifo_ranks)] = imm
        super()._rank_fill(oid, lv)

    def _close_ranks(self):
        idx = [i for i, _ in self._ev_ranks]
        super()._close_ranks()
        self.netted_later += sum(self.fifo_ranks[i] < self._imm[i] for i in idx)
        self._imm.clear()


def test_v03_cpp_equals_reference_bitwise_icebergs_and_shuffled_fills():
    # Known answer: identical output from both engines (every counter, FIFO rank, shadow, label, sample and queue
    # row; dtype, shape, bytes) on randomized streams with native iceberg refills and out-of-order fills, plus
    # many live shadows, sweeps, clears, snapshots without R, 20% of event boundaries removed (so fills and
    # their C/M can fall in different events, and ranks are netted across merged events) and 3% of F records
    # deleted (so some ranks stay > 0 after netting). Counters show both v0.3 rules change the outcome many
    # times, and that the streams produce requeued labels, nonzero ranks and FIFO anomalies.
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    rng = np.random.default_rng(13)
    tot = dict(fsu_requeues=0, netted_later=0, requeued=0, anomaly=0, nonzero_rank=0, live=0, shadows=0)
    for j, mode in enumerate(("order", "volume", "tail", "front")):
        rec, st, truth, ex = generate(n_events=5000, sample_every=41, seed=400 + j, cancel_mode=mode, p_modify=0.12,
                                      p_market=0.18, max_mkt=30, p_clear=0.003, p_implicit_snap=0.003,
                                      p_iceberg=0.35, p_fill_shuffle=0.5)
        last = np.where((rec["flags"][:-1] & L) != 0)[0]
        rec["flags"][last[rng.uniform(size=len(last)) < 0.2]] &= np.uint8(255 - L)
        keep = ~((rec["action"] == b"F") & (rng.uniform(size=len(rec["action"])) < 0.03))
        rec = {k: v[keep] for k, v in rec.items()}               # 3% of fills lost: real nonzero ranks
        if j % 2:
            rec["flags"][-1] &= np.uint8(255 - L)                 # the stream ends inside an event
        ts = rec["ts_event"].astype(np.int64)
        pt = np.arange(ts[30], ts[-1], 1_500_000)
        qt = np.sort(np.concatenate([st[::2], rng.integers(ts[0], ts[-1], 40)]))
        kw = dict(label_ticks=(2, 5)[j % 2], shadow_size=(3, 1)[j % 2], horizon_ns=150_000_000, sample_levels=6)
        ref = _ProbeV03(tick=ex.tick, **kw)
        a = ref.run(rec, place_times=pt, sample_times=st, queue_times=qt, queue_levels=4)
        b = fast.run(rec, tick=ex.tick, place_times=pt, sample_times=st, queue_times=qt, queue_levels=4, **kw)
        _assert_same_output(a, b, (j, mode))
        tot["fsu_requeues"] += ref.fsu_requeues; tot["netted_later"] += ref.netted_later
        tot["requeued"] += int((a["labels"]["outcome"] == OUTCOMES.index("requeued")).sum())
        tot["anomaly"] += a["stats"]["shadow_fifo_anomaly"]; tot["nonzero_rank"] += int((a["fifo_rank"] > 0).sum())
        tot["live"] = max(tot["live"], ref.max_live); tot["shadows"] += len(a["shadows"]["sid"])
    assert tot["fsu_requeues"] > 500 and tot["netted_later"] > 300 and tot["requeued"] > 500, tot
    assert tot["anomaly"] > 0 and tot["nonzero_rank"] > 0 and tot["live"] >= 100 and tot["shadows"] > 5000, tot


# ---------------------------------------------------------------------------- order-anchored tracks (SPEC 9)
LOG = ESTIMATORS.index("prob_log")
RA, FO = ESTIMATORS.index("risk_averse"), ESTIMATORS.index("front_optimistic")


def _phi_hand(front, back):
    """The SPEC 5 front shares written out by hand, in ESTIMATORS order."""
    import math
    return [0.0, front / (front + back), front ** 2 / (front ** 2 + back ** 2), front ** 3 / (front ** 3 + back ** 3),
            math.log(1 + front) / (math.log(1 + front) + math.log(1 + back)), 1.0]


def _trk(r, oid, nth=0):
    """One track row (as a dict) by order id; nth-th life of that id."""
    t = r["tracks"]
    i = np.where(t["order_id"] == oid)[0][nth]
    return {k: t[k][i] for k in t}


def _vsame(got, exp):
    """Exact for every estimator but prob_log (math.log vs log1p may differ in the last bit)."""
    got, exp = np.asarray(got, dtype=float), np.asarray(exp, dtype=float)
    mask = np.arange(len(ESTIMATORS)) != LOG
    return (np.array_equal(got[mask], exp[mask], equal_nan=True)
            and np.allclose(got[LOG], exp[LOG], rtol=1e-12, atol=1e-12, equal_nan=True))


def test_track_fill_reached_by_trades_on_the_front():
    # Known answer (hand-computed). Bids at 100: 1(3) 2(2); X = order 3 (4 lots) joins at t=4, V_true = 5.
    # t=10: sell 6 fills 1 (3), 2 (2) and 1 lot of X. Every estimator starts at V = 5, so the 6-lot print
    # leaves excess 1: X's estimated fill = its true fill = 1 lot at t=10; V after the event = 0 (Q_ex = 0:
    # only X is left at 100). Orders 1, 2 and 9 also start tracks (dist 0: first at their side or at the best).
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),
                   (10, "T", "A", 100, 6, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 2, 2, 0),
                   (10, "F", "B", 100, 1, 3, 0), (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 2, 2, 0),
                   (10, "M", "B", 100, 3, 3, L)]
    for r in both(recs(rows), order_tracks=True):
        t = r["tracks"]
        assert sorted(t) == sorted(TRACK_COLUMNS) and list(t["order_id"]) == [1, 2, 9, 3]
        assert list(t["label_index"]) == [0, 1, 2, 3] and list(t["size_ahead0"]) == [0, 3, 0, 5]
        x = _trk(r, 3)
        assert (x["side"], x["price"], x["t_start"], x["size0"]) == (1, 100 * TICK, 4, 4)
        assert OUTCOMES[x["outcome"]] == "censored" and x["t_end"] == -1
        assert x["true_first_fill_t"] == 10 and x["true_filled_qty"] == 1
        assert (x["est_first_fill_t"] == 10).all() and (x["est_filled_qty"] == 1.0).all()
        assert (x["v_est_at_true_first_fill"] == 0.0).all() and (x["v_est_at_end"] == 0.0).all()
        # orders 1 and 2: filled completely at t=10 by truth and by every estimator (V 0 and 3 < 6)
        for oid, size in ((1, 3), (2, 2)):
            y = _trk(r, oid)
            assert OUTCOMES[y["outcome"]] == "filled" and y["t_end"] == 10 and y["true_filled_qty"] == size
            assert (y["est_filled_qty"] == size).all() and (y["est_first_fill_t"] == 10).all()
        z = _trk(r, 9)                                  # the ask never trades: no fill, V stays 0, V at fill NaN
        assert z["true_first_fill_t"] == -1 and (z["est_first_fill_t"] == -1).all()
        assert np.isnan(z["v_est_at_true_first_fill"]).all() and (z["v_est_at_end"] == 0).all()
        assert t["est_filled_qty"].dtype == np.float64 and t["est_first_fill_t"].shape == (4, len(ESTIMATORS))


def test_track_cancel_ahead_with_each_phi():
    # Known answer (hand-computed). X = order 3 (4) behind 1(3) 2(2): V0 = 5. Order 4 (4) joins behind X at t=5.
    # t=6: order 1 (ahead) cancels 3. Q_ex 9 -> 6, D = 3, front F = V = 5, back B = Qa - V = 4, so
    # V_k = 5 - 3 * phi_k(5, 4) (risk_averse 5, power1 5 - 15/9, ..., front_optimistic 2 = the truth).
    # t=10: sell 3 fills 2 (2) and 1 lot of X. Tr = 3: only front_optimistic (V 2) has excess (1 lot, = the true
    # fill); every other V > 3. No unexplained decrease (Qa = 6 - 3 = 3 < Q1_ex = 4). V after: max(V_k - 3, 0).
    rows = BASE + [(4, "A", "B", 100, 4, 3, L), (5, "A", "B", 100, 4, 4, L), (6, "C", "B", 100, 3, 1, L),
                   (10, "T", "A", 100, 3, 0, 0), (10, "F", "B", 100, 2, 2, 0), (10, "F", "B", 100, 1, 3, 0),
                   (10, "C", "B", 100, 2, 2, 0), (10, "M", "B", 100, 3, 3, L)]
    v6 = [min(max(5.0 - p * 3.0, 0.0), 6.0) for p in _phi_hand(5.0, 4.0)]
    assert v6[RA] == 5.0 and v6[FO] == 2.0 and all(3.0 < v for v in v6[:FO])
    v10 = [min(max(v - 3.0, 0.0), 4.0) for v in v6]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 3)
        assert x["true_first_fill_t"] == 10 and x["true_filled_qty"] == 1
        assert list(x["est_first_fill_t"]) == [-1] * FO + [10]
        assert list(x["est_filled_qty"]) == [0.0] * FO + [1.0]
        assert _vsame(x["v_est_at_true_first_fill"], v10) and _vsame(x["v_est_at_end"], v10)
        assert x["v_est_at_end"][RA] == 2.0 and x["v_est_at_end"][FO] == 0.0


class _TrackNoExclusion(Reference):
    """Control only: tracks see the full Q(s, p), X included (SPEC 9 without its one change)."""

    def _append(self, oid, side, price, size, label_ok):
        n = len(self.tracks)
        super()._append(oid, side, price, size, label_ok)
        if len(self.tracks) > n:
            self.tracks[-1].q0ex += size                 # Q just after X's record, X included

    def _update_track(self, trk, tr, through, q1ex):
        o = self.orders.get(trk.oid)
        live = o is not None and o.get("label") is not None and o["label"].get("track") is trk
        super()._update_track(trk, tr, through, q1ex + (o["size"] if live else 0))


def test_track_excludes_x_own_size_from_q():
    # Known answer (hand-computed). Bids at 100: 1(3), then X = order 3 (5), then 4 (4) behind; V0 = 3.
    #  t=10 sell 1 hits order 1 (3 -> 2): every V 3 -> 2 (= truth).
    #  t=11 X itself cancels 2 lots (5 -> 3): Q_ex unchanged (6 -> 6), so no estimator moves (V 2 = truth).
    #  t=12 sell 3: 2 lots of order 1, then 1 lot of X (partial fill, 3 -> 2). Tr = 3 > V = 2: every estimator
    #       credits exactly the 1 true lot at t=12 and V = 0.
    # Control (X's size inside Q): at t=11 the cancel looks like D = 2 lots gone, front_optimistic drops to V = 0
    # and at t=12 credits 3 lots (a 2-lot overfill); power1 moves by 2 * 2/11.
    rows = [(1, "A", "B", 100, 3, 1, L), (2, "A", "A", 101, 5, 9, L), (3, "A", "B", 100, 5, 3, L),
            (4, "A", "B", 100, 4, 4, L),
            (10, "T", "A", 100, 1, 0, 0), (10, "F", "B", 100, 1, 1, 0), (10, "M", "B", 100, 2, 1, L),
            (11, "C", "B", 100, 2, 3, L),
            (12, "T", "A", 100, 3, 0, 0), (12, "F", "B", 100, 2, 1, 0), (12, "F", "B", 100, 1, 3, 0),
            (12, "C", "B", 100, 2, 1, 0), (12, "M", "B", 100, 2, 3, L)]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 3)
        assert x["size_ahead0"] == 3 and x["true_first_fill_t"] == 12 and x["true_filled_qty"] == 1
        assert (x["est_first_fill_t"] == 12).all() and (x["est_filled_qty"] == 1.0).all()
        assert (x["v_est_at_true_first_fill"] == 0.0).all() and (x["v_est_at_end"] == 0.0).all()
    c = _trk(_TrackNoExclusion(tick=TICK).run(recs(rows), order_tracks=True), 3)
    assert c["est_filled_qty"][FO] == 3.0 and c["est_filled_qty"][RA] == 1.0
    p1 = 2.0 - (2.0 / 11.0) * 2.0                     # V after the t=11 "decrease" under the control
    assert c["est_filled_qty"][ESTIMATORS.index("prob_power1")] == 3.0 - p1
    # The cap V <= Q_ex: when only X is left at the level, every estimator knows nothing is ahead. Order 1 (3)
    # ahead of X cancels: Q_ex 3 -> 0, so even risk_averse (phi = 0) has V = min(3, 0) = 0 (the truth);
    # the control caps at Q = 5 and keeps risk_averse at 3.
    rows = [(1, "A", "B", 100, 3, 1, L), (2, "A", "A", 101, 5, 9, L), (3, "A", "B", 100, 5, 3, L),
            (4, "C", "B", 100, 3, 1, L)]
    for r in both(recs(rows), order_tracks=True):
        assert (_trk(r, 3)["v_est_at_end"] == 0.0).all()
    c = _trk(_TrackNoExclusion(tick=TICK).run(recs(rows), order_tracks=True), 3)
    assert c["v_est_at_end"][RA] == 3.0 and c["v_est_at_end"][FO] == 0.0


def test_track_cancelled_before_any_estimated_fill():
    # Known answer (hand-computed). X = order 3 (4) behind 1(3) 2(2): V0 = 5. t=10 sell 2 (order 1: 3 -> 1):
    # V = 3 for every estimator (no excess). t=11 X cancels: outcome cancelled at 11, no true or estimated
    # fill, V at true first fill NaN, V at end 3 (the last clip: Q_ex 3 -> 3, nothing moves).
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),
                   (10, "T", "A", 100, 2, 0, 0), (10, "F", "B", 100, 2, 1, 0), (10, "M", "B", 100, 1, 1, L),
                   (11, "C", "B", 100, 4, 3, L)]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "cancelled" and x["t_end"] == 11
        assert x["true_first_fill_t"] == -1 and x["true_filled_qty"] == 0
        assert (x["est_first_fill_t"] == -1).all() and (x["est_filled_qty"] == 0.0).all()
        assert np.isnan(x["v_est_at_true_first_fill"]).all() and (x["v_est_at_end"] == 3.0).all()


def test_track_false_fill_before_cancel():
    # Known answer (hand-computed). X = order 3 (4) behind 1(3) 2(2): V0 = 5; order 4 (4) joins behind X (t=5)
    # and cancels at t=6 (a cancel BEHIND X; the truth stays 5). Q_ex 9 -> 5, D = 4, F = 5, B = 4:
    # V_k = 5 - 4 * phi_k (front_optimistic 1). t=10 sell 2 hits order 1 only: front_optimistic has excess
    # 1 and books a fill of 1 lot that never happened; the others (V > 2) do not. t=11 X cancels unfilled:
    # front_optimistic's track is a false fill (est_filled_qty 1, true 0).
    rows = BASE + [(4, "A", "B", 100, 4, 3, L), (5, "A", "B", 100, 4, 4, L), (6, "C", "B", 100, 4, 4, L),
                   (10, "T", "A", 100, 2, 0, 0), (10, "F", "B", 100, 2, 1, 0), (10, "M", "B", 100, 1, 1, L),
                   (11, "C", "B", 100, 4, 3, L)]
    v6 = [min(max(5.0 - p * 4.0, 0.0), 5.0) for p in _phi_hand(5.0, 4.0)]
    assert v6[FO] == 1.0 and all(v > 2.0 for v in v6[:FO])
    vend = [min(max(v - 2.0, 0.0), 3.0) for v in v6]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "cancelled" and x["t_end"] == 11 and x["true_filled_qty"] == 0
        assert list(x["est_filled_qty"]) == [0.0] * FO + [1.0] and list(x["est_first_fill_t"]) == [-1] * FO + [10]
        assert np.isnan(x["v_est_at_true_first_fill"]).all() and _vsame(x["v_est_at_end"], vend)


def test_track_clips_at_start_end_requeue_clear_and_open_event():
    # Known answers (hand-computed) for the clip rules.
    # (a) X = order 3 joins at t=10 in the same event as, and after, a 3-lot sell that took order 1: the print is
    #     not seen by X's estimators (V0 = 2, order 2 ahead; no estimated fill; had it been seen, V = 0 and a
    #     1-lot fill). (b) t=20: X cancels, then a 2-lot sell fills order 2 in the same event: X's last clip ends
    #     at its cancel and does not see the print (no fill, V 2). Order 2 (track from t=2, V0 = 3) sees every
    #     print of both events: the t=10 print takes V to 0 with no excess; at t=20 it is filled (truth and every
    #     estimator, 2 lots at t=20).
    rows = BASE + [(10, "T", "A", 100, 3, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "C", "B", 100, 3, 1, 0),
                   (10, "A", "B", 100, 4, 3, L),
                   (20, "C", "B", 100, 4, 3, 0), (20, "T", "A", 100, 2, 0, 0), (20, "F", "B", 100, 2, 2, 0),
                   (20, "C", "B", 100, 2, 2, L)]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 3)
        assert x["t_start"] == 10 and x["size_ahead0"] == 2 and OUTCOMES[x["outcome"]] == "cancelled"
        assert (x["est_first_fill_t"] == -1).all() and (x["v_est_at_end"] == 2.0).all()
        y = _trk(r, 2)
        assert OUTCOMES[y["outcome"]] == "filled" and y["true_first_fill_t"] == 20 and y["true_filled_qty"] == 2
        assert (y["est_first_fill_t"] == 20).all() and (y["est_filled_qty"] == 2.0).all()
    # (c) iceberg refill of order 1 at t=10 (F 3, M 3): its first life ends 'requeued' with its 3-lot fill seen
    #     by every estimator (V0 = 0); the refill is a new dist-0 life (best 100 before it) with its own track:
    #     V0 = 2 (order 2 ahead) and the event's print, listed before the refill, is not seen.
    # (d) t=30: a 1-lot sell fills order 2, then R + snapshot in the same event: live tracks end 'cleared' at 30,
    #     no last clip (order 2: true first fill 30, no estimated fill, V at true first fill NaN); snapshot adds
    #     start no labels and no tracks.
    rows = BASE + [(10, "T", "A", 100, 3, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "M", "B", 100, 3, 1, L),
                   (30, "T", "A", 100, 1, 0, 0), (30, "F", "B", 100, 1, 2, 0), (30, "R", "N", 0, 0, 0, 0),
                   (30, "A", "B", 100, 1, 2, SNAP), (30, "A", "B", 100, 3, 1, SNAP), (30, "A", "A", 101, 5, 9, SNAP | L)]
    for r in both(recs(rows), order_tracks=True):
        t = r["tracks"]
        assert list(t["order_id"]) == [1, 2, 9, 1] and [OUTCOMES[o] for o in t["outcome"]] == \
            ["requeued", "cleared", "cleared", "cleared"]
        a, b = _trk(r, 1, 0), _trk(r, 1, 1)
        assert a["t_end"] == 10 and a["true_filled_qty"] == 3 and (a["est_filled_qty"] == 3.0).all()
        assert (a["est_first_fill_t"] == 10).all() and (a["v_est_at_true_first_fill"] == 0.0).all()
        assert b["t_start"] == 10 and b["size_ahead0"] == 2 and b["label_index"] == 3
        assert (b["est_first_fill_t"] == -1).all() and (b["v_est_at_end"] == 2.0).all()
        y = _trk(r, 2)
        assert y["t_end"] == 30 and y["true_first_fill_t"] == 30 and y["true_filled_qty"] == 1
        assert (y["est_first_fill_t"] == -1).all() and np.isnan(y["v_est_at_true_first_fill"]).all()
        assert len(r["labels"]["order_id"]) == 4
    # (e) the stream ends inside an event (no F_LAST) holding X's first fills: that event is not applied, so the
    #     tracks are censored with no estimated fill and V at true first fill NaN.
    rows = BASE + [(4, "A", "B", 100, 4, 3, L),
                   (10, "T", "A", 100, 6, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 2, 2, 0),
                   (10, "F", "B", 100, 1, 3, 0)]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "censored" and x["true_first_fill_t"] == 10 and x["true_filled_qty"] == 1
        assert (x["est_filled_qty"] == 0.0).all() and np.isnan(x["v_est_at_true_first_fill"]).all()
        assert (x["v_est_at_end"] == 5.0).all()
    # (f) X = order 2 (V0 = 3, behind order 1) gets an F at t=10 in an event with no print and no A/C/M at its
    #     level (e.g. the T was lost): that clip is skipped and V at the true first fill is the V it leaves, 3, not
    #     the V after the next applied clip (t=11: order 1 cancels, Q_ex 3 -> 0, every V 0).
    rows = BASE + [(10, "F", "B", 100, 1, 2, L), (11, "C", "B", 100, 3, 1, L)]
    for r in both(recs(rows), order_tracks=True):
        x = _trk(r, 2)
        assert x["true_first_fill_t"] == 10 and x["true_filled_qty"] == 1 and (x["est_first_fill_t"] == -1).all()
        assert (x["v_est_at_true_first_fill"] == 3.0).all() and (x["v_est_at_end"] == 0.0).all()
    # (g) rule 2 reaches a level that the event neither touched nor printed at: a sell print at 99 (bid order 4,
    #     dist 1: a label, no track) while bids rest at 100, and a buy print at 102 while ask 9 rests at 101.
    #     Every estimator fills tracks 1, 2 (bid 100) at t=10 and track 9 (ask 101) at t=11 completely: false fills
    #     (no F on them). Not FIFO-consistent data; it pins the rule for both engines.
    rows = BASE + [(4, "A", "B", 99, 2, 4, L), (5, "A", "A", 102, 1, 10, L),
                   (10, "T", "A", 99, 1, 0, 0), (10, "F", "B", 99, 1, 4, 0), (10, "M", "B", 99, 1, 4, L),
                   (11, "T", "B", 102, 1, 0, 0), (11, "F", "A", 102, 1, 10, 0), (11, "C", "A", 102, 1, 10, L)]
    for r in both(recs(rows), order_tracks=True):
        assert list(r["tracks"]["order_id"]) == [1, 2, 9]
        for oid, size, t in ((1, 3, 10), (2, 2, 10), (9, 5, 11)):
            x = _trk(r, oid)
            assert x["true_filled_qty"] == 0 and (x["est_filled_qty"] == size).all() and (x["est_first_fill_t"] == t).all()


class _TrackProbe(Reference):
    """The reference plus a check (reads state, changes nothing): after every applied clip while X still rests,
    estimator `k`'s V against X's true queue ahead (the sizes in front of X in its level's FIFO)."""

    def __init__(self, k, *a, **kw):
        super().__init__(*a, **kw)
        self.k, self.checks, self.wrong, self.nonzero = k, 0, 0, 0

    def _update_track(self, trk, tr, through, q1ex):
        super()._update_track(trk, tr, through, q1ex)
        o = self.orders.get(trk.oid)
        if o is None or o.get("label") is None or o["label"].get("track") is not trk:
            return                                         # the last clip: X has left the level
        vt = 0
        for oid, other in self.books[trk.side][trk.price].items():
            if oid == trk.oid:
                break
            vt += other["size"]
        self.checks += 1
        self.wrong += trk.v[self.k] != vt
        self.nonzero += vt > 0


def test_track_known_answers_tail_and_front_cancels():
    # Known answer (SPEC 9 "Known answers"): with tail cancels risk_averse, with front cancels front_optimistic,
    # is exact for real orders too: V = X's true queue ahead after every applied clip while X rests, estimated
    # first-fill time = true first-fill time on every non-cleared track (both -1 when X never fills), and V at the
    # true first fill = 0 on every filled track. Controls: in the same streams the other model is not exact; with
    # tail cancels front_optimistic fills early (estimated first fill before the true one, or with no true fill),
    # with front cancels risk_averse still sees queue ahead when X is filled. (Estimated filled QUANTITY is not
    # exact: after X's own partial cancel, rule 1 can credit more than X has left.)
    for mode, k, other in (("tail", RA, FO), ("front", FO, RA)):
        rec, st, truth, ex = generate(n_events=12000, sample_every=1000, seed=21, cancel_mode=mode, p_modify=0.0,
                                      p_clear=0.002)
        probe = _TrackProbe(k, tick=ex.tick)
        runs = [probe.run(rec, order_tracks=True)] + ([fast.run(rec, tick=ex.tick, order_tracks=True)] if HAVE_CORE else [])
        assert probe.checks > 10000 and probe.nonzero > 5000 and probe.wrong == 0, (mode, probe.checks, probe.wrong)
        for r in runs:
            t = r["tracks"]
            ok = t["outcome"] != OUTCOMES.index("cleared")
            f = ok & (t["true_filled_qty"] > 0)
            assert ok.sum() > 1000 and f.sum() > 300 and (~ok).sum() > 0, mode
            assert np.array_equal(t["est_first_fill_t"][ok, k], t["true_first_fill_t"][ok]), mode
            assert (t["v_est_at_true_first_fill"][f, k] == 0.0).all(), mode
            eo, to = t["est_first_fill_t"][ok, other], t["true_first_fill_t"][ok]
            assert (eo != to).sum() > 50, mode
            if mode == "tail":
                assert ((eo >= 0) & ((to < 0) | (eo < to))).sum() > 50
            else:
                assert (t["v_est_at_true_first_fill"][f, other] > 0).sum() > 50


def test_track_rows_are_the_dist0_labels():
    # Known answer: tracks are exactly the dist-0 labels, in label order, and repeat their fields (id, side, price,
    # t_add, size, size ahead, outcome, t_end, filled quantity), on streams with icebergs, shuffled fills, clears
    # and snapshots; a true first fill exists iff the filled quantity is > 0 and lies within [t_start, t_end];
    # estimated filled quantities are within [0, size0]; and no track exists when label_ticks < 0.
    for j, mode in enumerate(("volume", "order")):
        rec, st, truth, ex = generate(n_events=5000, sample_every=100, seed=600 + j, cancel_mode=mode,
                                      p_clear=0.003, p_implicit_snap=0.003, p_iceberg=0.3, p_fill_shuffle=0.5)
        for f in _queue_engines(ex.tick):
            r = f(rec, order_tracks=True)
            t, lab = r["tracks"], r["labels"]
            li = np.where(lab["dist"] == 0)[0]
            assert len(li) > 1000 and len(li) < len(lab["dist"]) and np.array_equal(t["label_index"], li)
            for a, b in (("order_id", "order_id"), ("side", "side"), ("price", "price"), ("t_start", "t_add"),
                         ("size0", "size"), ("size_ahead0", "size_ahead_at_add"), ("outcome", "outcome"),
                         ("t_end", "t_end"), ("true_filled_qty", "filled_qty")):
                assert np.array_equal(t[a], lab[b][li]), (mode, a)
            filled = t["true_filled_qty"] > 0
            assert np.array_equal(t["true_first_fill_t"] >= 0, filled)
            assert (t["true_first_fill_t"][filled] >= t["t_start"][filled]).all()
            ended = filled & (t["t_end"] >= 0)
            assert (t["true_first_fill_t"][ended] <= t["t_end"][ended]).all()
            ef = t["est_filled_qty"]
            assert (ef >= 0).all() and (ef <= t["size0"][:, None]).all()
            assert np.array_equal(t["est_first_fill_t"] >= 0, ef > 0)
            assert np.isnan(t["v_est_at_true_first_fill"][~filled]).all()
    r0 =Reference(tick=ex.tick, label_ticks=-1).run(rec, order_tracks=True)
    assert len(r0["tracks"]["order_id"]) == 0 and r0["tracks"]["v_est_at_end"].shape == (0, len(ESTIMATORS))
    if HAVE_CORE:
        r1 = fast.run(rec, tick=ex.tick, label_ticks=-1, order_tracks=True)
        assert all(np.asarray(r1["tracks"][k]).shape == np.asarray(r0["tracks"][k]).shape for k in TRACK_COLUMNS)


def _assert_same_tracks(a, b, ctx):
    assert set(a["tracks"]) == set(b["tracks"]) == set(TRACK_COLUMNS), ctx
    for k in TRACK_COLUMNS:
        x, y = np.asarray(a["tracks"][k]), np.asarray(b["tracks"][k])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (ctx, k)


def test_track_cpp_equals_reference_bitwise():
    # Known answer: identical tracks from both engines (every column; dtype, shape, bytes, NaN included) and
    # identical other outputs, on randomized streams with native iceberg refills, out-of-order fills, sweeps,
    # price-change modifies, clears, snapshots without R, many shadows, 20% of event boundaries removed (so tracks
    # start and end mid-event, and fills and their C/M fall in different events), 3% of F records deleted, and
    # streams ending inside an event. Counters show each track situation occurs.
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    rng = np.random.default_rng(17)
    tot = dict(tracks=0, outcomes=set(), false_fill=0, missed=0, nan_filled=0, mid_start=0)
    for j, mode in enumerate(("order", "volume", "tail", "front", "order", "volume")):
        rec, st, truth, ex = generate(n_events=5000, sample_every=41, seed=800 + j, cancel_mode=mode, p_modify=0.12,
                                      p_market=0.18, max_mkt=30, p_clear=0.003, p_implicit_snap=0.003,
                                      p_iceberg=(0.35, 0.0)[j // 4], p_fill_shuffle=(0.5, 0.0)[j // 4])
        last = np.where((rec["flags"][:-1] & L) != 0)[0]
        rec["flags"][last[rng.uniform(size=len(last)) < 0.2]] &= np.uint8(255 - L)
        keep = ~((rec["action"] == b"F") & (rng.uniform(size=len(rec["action"])) < 0.03))
        rec = {k: v[keep] for k, v in rec.items()}
        if j % 2:
            rec["flags"][-1] &= np.uint8(255 - L)
        ts = rec["ts_event"].astype(np.int64)
        pt = np.arange(ts[30], ts[-1], 1_500_000)
        qt = np.sort(np.concatenate([st[::2], rng.integers(ts[0], ts[-1], 40)]))
        kw = dict(label_ticks=(2, 5)[j % 2], shadow_size=(3, 1)[j % 2], horizon_ns=150_000_000, sample_levels=6)
        rk = dict(place_times=pt, sample_times=st, queue_times=qt, queue_levels=4, order_tracks=True)
        a = Reference(tick=ex.tick, **kw).run(rec, **rk)
        b = fast.run(rec, tick=ex.tick, **rk, **kw)
        _assert_same_output(a, b, (j, mode))
        _assert_same_tracks(a, b, (j, mode))
        t = a["tracks"]
        tot["tracks"] += len(t["order_id"]); tot["outcomes"] |= set(t["outcome"].tolist())
        tot["false_fill"] += int(((t["est_filled_qty"] > 0) & (t["true_filled_qty"] == 0)[:, None]).sum())
        tot["missed"] += int(((t["est_first_fill_t"] < 0) & (t["true_filled_qty"] > 0)[:, None]).sum())
        tot["nan_filled"] += int(np.isnan(t["v_est_at_true_first_fill"][t["true_filled_qty"] > 0]).any(1).sum())
        # tracks that started mid-event: a record of the same event (same ts here) precedes the start record
        tot["mid_start"] += int(np.isin(t["t_start"], ts[1:][(rec["flags"][:-1] & L) == 0]).sum())
    assert tot["tracks"] > 5000 and tot["outcomes"] == set(range(len(OUTCOMES))), tot
    assert tot["false_fill"] > 100 and tot["missed"] > 100 and tot["nan_filled"] > 0 and tot["mid_start"] > 100, tot


# Digests of every output (stats, shadows, labels, FIFO ranks, samples, queue rows: dtype, shape, bytes) of the
# committed v0.3 engines (both identical; commit 57febcf, before SPEC 9 existed) on _digest_streams(). They depend
# on numpy's Generator streams (produced with numpy 2.2.6).
V03_DIGESTS = ["c6c230003034bfe5a2a69bbd98557d607950d2be0aa24d8585f5f47ed9557dc1",
               "55e30f963d6ddab18c61e712879cc11d560171b0d9d38eb810297e842e20d3d3",
               "38d326610672e4c698b395fbbe18c51cb809f01fbae7c3412594703006ece233",
               "772ccc24483a07ea2a34c27f12c2591d121c53405342a636364a6cf970501555"]
# The full digests include the estimators' floating-point state, which goes through libm (pow, log1p), so they
# were recorded on, and are checked on, macOS arm64 only (Apple libm); C++ == Python is checked on every platform
# by the bitwise tests. Everywhere: the input streams and the book-only outputs (labels, FIFO ranks, samples,
# queue rows), which use no floating point, must equal these stored digests (recorded with numpy 2.5.4).
REFERENCE_PLATFORM = sys.platform == "darwin" and __import__("platform").machine() == "arm64"
STREAM_DIGESTS = ["5f6794940de3f041945a55dc3e028068e8385d6ed0b6efdb0c7d2f982f5d2660",
                  "5dc9a6218603465d17bd880faa0b98795ca194e5b00024187b855015c22c81a9",
                  "92c33fe8bbf8bd8e91ec39b77436c8b456d8c2dc1ccaa7604d3c5387daa727e1",
                  "000c58d792d3d1908a7d08e9c07799d7829514982f198845f782ac4f26276713"]
V03_BOOK_DIGESTS = ["055e46878e8c13be9b089261e8876a6fd5581d2e64034a69f2dd479c3535a438",
                    "24140b11f058472e5b5b76b0db34a9fb763917b670623a55ae0718d5f08534b5",
                    "9817f2d83e9e09119ed51a0fbccfd74a762a2a70e0ccf554c485074fa8f9141a",
                    "e03b45399ec240b31891061c844a97628d0498c419dc9d4daabd85246c61bcab"]


def _digest_streams():
    out = []
    for j, mode in enumerate(("order", "volume", "tail", "front")):
        rec, st, truth, ex = generate(n_events=3000, sample_every=37, seed=700 + j, cancel_mode=mode, p_modify=0.12,
                                      p_market=0.18, max_mkt=30, p_clear=0.003, p_implicit_snap=0.003,
                                      p_iceberg=(0.0, 0.35)[j % 2], p_fill_shuffle=(0.0, 0.5)[j % 2])
        ts = rec["ts_event"].astype(np.int64)
        kw = dict(label_ticks=(2, 5)[j % 2], shadow_size=(3, 1)[j % 2], horizon_ns=150_000_000, sample_levels=6)
        rk = dict(place_times=np.arange(ts[30], ts[-1], 3_000_000), sample_times=st, queue_times=st[::2], queue_levels=4)
        out.append((rec, ex.tick, kw, rk))
    return out


def _digest(r):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(r["stats"]):
        h.update(f"{k}={int(r['stats'][k])};".encode())
    for grp in ("shadows", "labels"):
        for k in sorted(r[grp]):
            x = np.asarray(r[grp][k])
            h.update(f"{grp}.{k}:{x.dtype.str}:{x.shape};".encode())
            h.update(x.tobytes())
    for k in ("fifo_rank", "fifo_ts", "samples", "queue"):
        x = np.asarray(r[k])
        h.update(f"{k}:{x.dtype.str}:{x.shape};".encode())
        h.update(x.tobytes())
    return h.hexdigest()


def _book_digest(r):
    """Digest of the outputs that involve no floating point: labels, FIFO ranks, book samples, queue rows."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(r["labels"]):
        x = np.asarray(r["labels"][k])
        h.update(f"labels.{k}:{x.dtype.str}:{x.shape};".encode())
        h.update(x.tobytes())
    for k in ("fifo_rank", "fifo_ts", "samples", "queue"):
        x = np.asarray(r[k])
        h.update(f"{k}:{x.dtype.str}:{x.shape};".encode())
        h.update(x.tobytes())
    return h.hexdigest()


def _stream_digest(rec):
    import hashlib
    return hashlib.sha256(b"".join(np.ascontiguousarray(rec[c]).tobytes() for c in sorted(rec))).hexdigest()


def test_digest_streams_and_book_outputs_match_on_every_platform():
    # Known answer: the synthetic input streams and the book-only outputs of both engines equal the stored digests
    # on every platform (no floating point is involved); the full v0.3 digests also on the reference platform.
    for j, ((rec, tick, kw, rk), s_want, b_want, want) in enumerate(
            zip(_digest_streams(), STREAM_DIGESTS, V03_BOOK_DIGESTS, V03_DIGESTS)):
        assert _stream_digest(rec) == s_want, f"synthetic input stream {j} differs on this platform"
        runs = [Reference(tick=tick, **kw).run(rec, **rk)] + ([fast.run(rec, tick=tick, **kw, **rk)] if HAVE_CORE else [])
        for r in runs:
            assert _book_digest(r) == b_want, f"book-only outputs of stream {j} differ"
            if REFERENCE_PLATFORM:
                assert _digest(r) == want, j


def test_track_flag_off_and_on_leave_every_other_output_unchanged():
    # Known answer: with order_tracks off (the default) both engines return exactly the committed v0.3 engines'
    # outputs and no "tracks" key; with it on, the same outputs (every other output unchanged) plus "tracks". On the
    # reference platform "the same" is the stored v0.3 digest; elsewhere it is the flag-off run's digest.
    for (rec, tick, kw, rk), stored in zip(_digest_streams(), V03_DIGESTS):
        runs = [lambda **f: Reference(tick=tick, **kw).run(rec, **rk, **f)]
        if HAVE_CORE:
            runs.append(lambda **f: fast.run(rec, tick=tick, **kw, **rk, **f))
        for run in runs:
            want = stored if REFERENCE_PLATFORM else _digest(run())
            for flag in ({}, {"order_tracks": False}, {"order_tracks": True}):
                r = run(**flag)
                assert ("tracks" in r) == bool(flag.get("order_tracks")), flag
                assert set(r) - {"tracks"} == {"stats", "fifo_rank", "fifo_ts", "shadows", "labels", "samples", "queue"}
                assert _digest(r) == want, flag


# ---------------------------------------------------------------------------- SPEC 9 v0.5: track horizon and E0
SEC = 1_000_000_000
H60 = 60 * SEC
# Bids at 100: order 1 (3 lots); X = order 3 (2 lots) joins at t_start = 1 s behind it (V0 = 3); order 4 (5 lots) joins
# behind X at 2 s. Ask: order 9 (5) at 101. Every hand-computed case below follows X (order 3) with H = 60 s, so the
# horizon is t_start + H = 61 s.
BASE5 = [(0, "A", "B", 100, 3, 1, L), (0, "A", "A", 101, 5, 9, L), (1 * SEC, "A", "B", 100, 2, 3, L),
         (2 * SEC, "A", "B", 100, 5, 4, L)]
LATE = [(70 * SEC, "A", "A", 102, 1, 10, L)]                 # the first boundary past 61 s, away from X's level
FRONT_CANCEL = [(5 * SEC, "C", "B", 100, 3, 1, L)]           # order 1 (ahead of X) cancels: true V 3 -> 0
TAIL_CANCEL = [(5 * SEC, "C", "B", 100, 3, 4, L)]            # order 4 (behind X) cancels 3 of 5: true V stays 3
X_FILL_10 = [(10 * SEC, "T", "A", 100, 2, 0, 0), (10 * SEC, "F", "B", 100, 2, 3, 0), (10 * SEC, "C", "B", 100, 2, 3, L)]
SELL_8 = [(8 * SEC, "T", "A", 100, 1, 0, 0), (8 * SEC, "F", "B", 100, 1, 1, 0), (8 * SEC, "M", "B", 100, 2, 1, L)]
X_FILL_12 = [(12 * SEC, "T", "A", 100, 4, 0, 0), (12 * SEC, "F", "B", 100, 2, 1, 0), (12 * SEC, "F", "B", 100, 2, 3, 0),
             (12 * SEC, "C", "B", 100, 2, 1, 0), (12 * SEC, "C", "B", 100, 2, 3, L)]


def _sell_on_4(t_sec, n, left):
    """A sell print of n lots at 100 that fills order 4 (n lots; `left` remain, M in place or C to 0)."""
    end = (t_sec * SEC, "M", "B", 100, left, 4, L) if left > 0 else (t_sec * SEC, "C", "B", 100, n, 4, L)
    return [(t_sec * SEC, "T", "A", 100, n, 0, 0), (t_sec * SEC, "F", "B", 100, n, 4, 0), end]


def _v5_cancel3():
    """V_k after either 5 s cancel: MBP sees the same thing for both (Q_ex 8 -> 5, D = 3, front F = V = 3,
    back B = Qa - V = 5): V_k = 3 - 3 phi_k(3, 5); risk_averse 3, front_optimistic 0."""
    return [min(max(3.0 - p * 3.0, 0.0), 5.0) for p in _phi_hand(3.0, 5.0)]


def _runs_h(rows, H=H60):
    """(v0.4 run, v0.5 run) pairs, one per engine."""
    return list(zip(both(recs(rows), order_tracks=True), both(recs(rows), order_tracks=True, track_horizon_ns=H)))


def test_v05_late_fill_is_measured_not_censored():
    # Known answer (hand-computed; estimator risk_averse, and every estimator from the phi formulas). BASE5, then a
    # front cancel at 5 s (risk_averse keeps V = 3, front_optimistic 0; true V 0). t=10 s: a 2-lot sell fills X
    # (outcome filled, true first fill 10 s): Tr = 2, risk_averse V 3 -> 1 and no excess. t=15 s: a 2-lot sell hits
    # order 4: risk_averse has excess 1 and fills. v0.4 (H = 0): the estimators stop at X's fill, so risk_averse has
    # no estimated fill. v0.5 (H = 60 s): they run on (Q_ex = Q, X gone) to the first boundary >= 61 s (70 s):
    # risk_averse est first fill 15 s, E0 = |(15 - 1) - (10 - 1)| s = 5 s. Every estimator k fills at 10 s if
    # V_k < 2 (E0 0, e.g. front_optimistic), else at 15 s (E0 5 s).
    rows = BASE5 + FRONT_CANCEL + X_FILL_10 + _sell_on_4(15, 2, 3) + LATE
    v5 = _v5_cancel3()
    exp_et = [10 * SEC if v < 2.0 else 15 * SEC for v in v5]
    assert v5[RA] == 3.0 and v5[FO] == 0.0 and exp_et[RA] == 15 * SEC and exp_et[FO] == 10 * SEC
    for r0, r in _runs_h(rows):
        x0, x = _trk(r0, 3), _trk(r, 3)
        assert set(r0["tracks"]) == set(TRACK_COLUMNS) and set(r["tracks"]) == set(TRACK_COLUMNS + TRACK_HORIZON_COLUMNS)
        assert OUTCOMES[x["outcome"]] == "filled" and x["t_end"] == 10 * SEC and x["true_first_fill_t"] == 10 * SEC
        assert list(x0["est_first_fill_t"]) == [t if t == 10 * SEC else -1 for t in exp_et]   # v0.4: no late fill
        assert x0["est_first_fill_t"][RA] == -1 and x0["est_filled_qty"][RA] == 0.0
        assert list(x["est_first_fill_t"]) == exp_et and x["est_stop_t"] == 70 * SEC
        assert list(x["e0"]) == [t - 10 * SEC for t in exp_et] and x["e0"][RA] == 5 * SEC and x["e0"][FO] == 0
        assert x["est_filled_qty"][RA] == 1.0 and x["v_est_at_end"][RA] == 0.0
        assert _vsame(x["v_est_at_true_first_fill"], x0["v_est_at_true_first_fill"])     # unchanged from v0.4


def test_v05_early_fill():
    # Known answer (hand-computed; estimator front_optimistic early, risk_averse exact). BASE5, a tail cancel at 5 s
    # (true V stays 3; MBP sees the same decrease as a front cancel, so V_k = 3 - 3 phi_k(3, 5): front_optimistic 0).
    # t=8 s: a 1-lot sell on order 1: front_optimistic has excess 1 and fills at 8 s; risk_averse V 3 -> 2.
    # t=12 s: a 4-lot sell fills order 1 (2) and X (2): true first fill 12 s; risk_averse excess 2, fills at 12 s.
    # E0: front_optimistic tau_true - tau_est = 11 s - 7 s = 4 s; risk_averse 0; estimator k: 4 s iff V_k < 1.
    rows = BASE5 + TAIL_CANCEL + SELL_8 + X_FILL_12 + LATE
    v5 = _v5_cancel3()
    exp_et = [8 * SEC if v < 1.0 else 12 * SEC for v in v5]
    for r0, r in _runs_h(rows):
        x0, x = _trk(r0, 3), _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "filled" and x["true_first_fill_t"] == 12 * SEC
        assert list(x["est_first_fill_t"]) == exp_et == list(x0["est_first_fill_t"])  # early fills: v0.4 saw them too
        assert list(x["e0"]) == [12 * SEC - t for t in exp_et] and x["e0"][FO] == 4 * SEC and x["e0"][RA] == 0
        assert x["est_stop_t"] == 70 * SEC


def test_v05_false_fill_on_a_cancelled_order():
    # Known answer (hand-computed; estimator front_optimistic false fill, risk_averse none). As the early-fill case
    # up to 8 s (front_optimistic fills at 8 s), then X cancels at 10 s (no true fill: tau_true = H). E0 for
    # front_optimistic = H - tau_est = 60 s - 7 s = 53 s; risk_averse never fills: H - H = 0; estimator k:
    # 53 s iff V_k < 1.
    rows = BASE5 + TAIL_CANCEL + SELL_8 + [(10 * SEC, "C", "B", 100, 2, 3, L)] + LATE
    v5 = _v5_cancel3()
    for r0, r in _runs_h(rows):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "cancelled" and x["true_filled_qty"] == 0 and x["est_stop_t"] == 10 * SEC
        assert list(x["e0"]) == [H60 - 7 * SEC if v < 1.0 else 0 for v in v5] and x["e0"][FO] == 53 * SEC


def test_v05_missed_fill():
    # Known answer (hand-computed; estimator risk_averse missed, front_optimistic exact). As the late-fill case, but
    # the 2-lot sell on order 4 comes at 70 s (past the 61 s horizon): risk_averse fills at 70 s, at the boundary
    # where its estimators stop (est_stop_t 70 s), which is too late to count: tau_est = H, E0 = H - tau_true =
    # 60 s - 9 s = 51 s (estimator k: 51 s iff V_k >= 2, else 0). A further sell at 80 s that would credit
    # risk_averse its second lot is not seen (est_filled_qty stays 1): the estimators stopped at the horizon.
    rows = BASE5 + FRONT_CANCEL + X_FILL_10 + _sell_on_4(70, 2, 3) + _sell_on_4(80, 3, 0)
    v5 = _v5_cancel3()
    for r0, r in _runs_h(rows):
        x0, x = _trk(r0, 3), _trk(r, 3)
        assert x0["est_first_fill_t"][RA] == -1 and x["est_first_fill_t"][RA] == 70 * SEC
        assert x["est_stop_t"] == 70 * SEC and x["est_filled_qty"][RA] == 1.0
        assert list(x["e0"]) == [51 * SEC if v >= 2.0 else 0 for v in v5] and x["e0"][RA] == 51 * SEC
        assert x["e0"][FO] == 0
    # A boundary exactly at t_start + H (61 s, an ask add) is the stop (">="): the 62 s sell is not seen.
    rows = BASE5 + FRONT_CANCEL + X_FILL_10 + [(61 * SEC, "A", "A", 102, 1, 10, L)] + _sell_on_4(62, 2, 3)
    for r0, r in _runs_h(rows):
        x = _trk(r, 3)
        assert x["est_stop_t"] == 61 * SEC and x["est_first_fill_t"][RA] == -1 and x["e0"][RA] == 51 * SEC


def test_v05_owner_action_stops_the_estimators():
    # Known answer (hand-computed; estimators risk_averse and front_optimistic). As the false-fill case, with X's life
    # ending at 10 s by each owner action: a cancel, a price change (to 99), a size increase (requeue, 2 -> 3).
    # At 12 s a 4-lot sell takes orders 1 and 4: risk_averse (V = 2) would have excess 2 if it still ran, but the
    # estimators stopped at X's end (est_stop_t 10 s): no risk_averse fill, every v0.4 column of X's track equals the
    # H = 0 run, and E0 is risk_averse 0 (H - H), front_optimistic 53 s (false fill at 8 s).
    sell12 = [(12 * SEC, "T", "A", 100, 4, 0, 0), (12 * SEC, "F", "B", 100, 2, 1, 0), (12 * SEC, "F", "B", 100, 2, 4, 0),
              (12 * SEC, "C", "B", 100, 2, 1, 0), (12 * SEC, "C", "B", 100, 2, 4, L)]
    for outcome, end in (("cancelled", (10 * SEC, "C", "B", 100, 2, 3, L)), ("moved_price", (10 * SEC, "M", "B", 99, 2, 3, L)),
                         ("requeued", (10 * SEC, "M", "B", 100, 3, 3, L))):
        rows = BASE5 + TAIL_CANCEL + SELL_8 + [end] + sell12 + LATE
        for r0, r in _runs_h(rows):
            x0, x = _trk(r0, 3), _trk(r, 3)
            assert OUTCOMES[x["outcome"]] == outcome and x["t_end"] == 10 * SEC and x["est_stop_t"] == 10 * SEC
            assert x["est_first_fill_t"][RA] == -1 and x["v_est_at_end"][RA] == 2.0
            assert all(np.array_equal(x[k], x0[k], equal_nan=True) for k in TRACK_COLUMNS), outcome
            assert x["e0"][RA] == 0 and x["e0"][FO] == 53 * SEC


def test_v05_clear_and_stream_end_leave_e0_unscored():
    # Known answers (hand-computed). e0 = -1 marks a track that is not scored.
    # (a) R at 5 s while X rests: outcome cleared, est_stop_t 5 s, e0 -1 (estimators risk_averse .. front_optimistic).
    for r0, r in _runs_h(BASE5 + [(5 * SEC, "R", "N", 0, 0, 0, L)] + LATE):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "cleared" and x["est_stop_t"] == 5 * SEC and (x["e0"] == -1).all()
    # (b) X filled at 10 s (late-fill case), then R at 12 s while its estimators run on: the label outcome stays filled
    #     (truth unchanged), est_stop_t 12 s, e0 -1: a clear ends the track.
    for r0, r in _runs_h(BASE5 + FRONT_CANCEL + X_FILL_10 + [(12 * SEC, "R", "N", 0, 0, 0, L)] + LATE):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "filled" and x["true_first_fill_t"] == 10 * SEC
        assert x["est_stop_t"] == 12 * SEC and (x["e0"] == -1).all()
    # (c) the late-fill case without the 70 s event: the stream ends (last boundary 15 s) before the horizon (61 s), so
    #     est_stop_t = -1 and e0 = -1, although risk_averse's late fill at 15 s is recorded.
    for r0, r in _runs_h(BASE5 + FRONT_CANCEL + X_FILL_10 + _sell_on_4(15, 2, 3)):
        x = _trk(r, 3)
        assert x["est_first_fill_t"][RA] == 15 * SEC and x["est_stop_t"] == -1 and (x["e0"] == -1).all()
    # (d) X still resting at the end of the stream (censored) is scored iff the last boundary is >= t_start + H: with
    #     the tail cancel and the 8 s sell (front_optimistic false fill at 8 s) and LATE, e0 = 53 s for
    #     front_optimistic and 0 for risk_averse; without LATE (last boundary 8 s), -1.
    for r0, r in _runs_h(BASE5 + TAIL_CANCEL + SELL_8 + LATE):
        x = _trk(r, 3)
        assert OUTCOMES[x["outcome"]] == "censored" and x["est_stop_t"] == -1
        assert x["e0"][FO] == 53 * SEC and x["e0"][RA] == 0
    for r0, r in _runs_h(BASE5 + TAIL_CANCEL + SELL_8):
        assert (_trk(r, 3)["e0"] == -1).all()
    # (e) a negative horizon is an error in both engines; H = 0 is the default (no new keys)
    for f in [lambda: Reference(tick=TICK).run(recs(BASE5), order_tracks=True, track_horizon_ns=-1)] + \
            ([lambda: fast.run(recs(BASE5), tick=TICK, order_tracks=True, track_horizon_ns=-1)] if HAVE_CORE else []):
        try:
            f()
            raise AssertionError("negative track_horizon_ns accepted")
        except ValueError:
            pass


def _last_boundary_ts(rec):
    b = np.where((rec["flags"] & L) != 0)[0]
    return int(rec["ts_event"][b[-1]]) if len(b) else -1


def test_v05_known_answers_e0_is_zero_for_the_exact_model():
    # Known answer (SPEC 9 v0.5 "Known answers"): on the synthetic exchange without icebergs or shuffled fills,
    # E0 = 0 on every scored track for the model that is exact for real orders: risk_averse with tail cancels (with
    # p_modify = 0 and with the generator's default modifies) and front_optimistic with front cancels and
    # p_modify = 0; generate(n_events=20000, seed=21, p_clear=0.0005), H = 200 ms (~100 events). Cleared tracks are
    # never scored, and the only unscored non-cleared tracks are filled ones whose run-on a clear cut before
    # t_start + H, or ones the stream end cuts before t_start + H. Controls: the other model has E0 > 0 on many tracks;
    # with front cancels, risk_averse fills late, after X's true fill, on many tracks (v0.4 recorded these as no
    # fill); some filled tracks have their true fill after H (tau_true = H).
    H = 200_000_000
    for mode, k, other, pm in (("tail", RA, FO, 0.0), ("tail", RA, FO, None), ("front", FO, RA, 0.0)):
        gk = {} if pm is None else {"p_modify": pm}
        rec, st, truth, ex = generate(n_events=20000, sample_every=1000, seed=21, cancel_mode=mode, p_clear=0.0005, **gk)
        lb = _last_boundary_ts(rec)
        ts, r_at, bnd = rec["ts_event"].astype(np.int64), np.where(rec["action"] == b"R")[0], np.where(rec["flags"] & L)[0]
        assert (np.diff(ts) >= 0).all()
        runs = [Reference(tick=ex.tick).run(rec, order_tracks=True, track_horizon_ns=H)]
        if HAVE_CORE:
            runs.append(fast.run(rec, tick=ex.tick, order_tracks=True, track_horizon_ns=H))
        for r in runs:
            t = r["tracks"]
            ok = t["outcome"] != OUTCOMES.index("cleared")
            scored = t["e0"][:, k] >= 0
            assert not (scored & ~ok).any() and ((t["e0"] >= 0) == scored[:, None]).all()
            cut = ok & ~scored
            at_end = cut & (t["est_stop_t"] == -1)
            assert (t["t_start"][at_end] + H > lb).all(), mode
            by_clear = cut & ~at_end                     # a filled X whose estimators a clear stopped early
            assert (t["outcome"][by_clear] == OUTCOMES.index("filled")).all(), mode
            assert (t["est_stop_t"] >= t["t_end"])[by_clear].all(), mode
            # ... and the R came before the first boundary at or past t_start + H (the R itself may be later than it)
            for stop, dl in zip(t["est_stop_t"][by_clear], (t["t_start"] + H)[by_clear]):
                i_r = r_at[np.searchsorted(ts[r_at], stop)]
                assert ts[i_r] == stop and ts[bnd[bnd < i_r][-1]] < dl, mode
            assert scored.sum() > 3000 and cut.sum() < 50, (mode, scored.sum(), cut.sum())
            assert (t["e0"][scored, k] == 0).all(), (mode, pm, int((t["e0"][scored, k] != 0).sum()))
            assert (t["e0"][scored, other] > 0).sum() > 50, mode
            filled = scored & (t["outcome"] == OUTCOMES.index("filled"))
            assert (t["true_first_fill_t"][filled] - t["t_start"][filled] > H).sum() > 10, mode
            if mode == "front":
                late = filled & (t["est_first_fill_t"][:, other] > t["t_end"])
                assert late.sum() > 50, mode


def _assert_same_tracks_h(a, b, ctx):
    cols = TRACK_COLUMNS + TRACK_HORIZON_COLUMNS
    assert set(a["tracks"]) == set(b["tracks"]) == set(cols), ctx
    for k in cols:
        x, y = np.asarray(a["tracks"][k]), np.asarray(b["tracks"][k])
        assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), (ctx, k)


def test_v05_cpp_equals_reference_bitwise_with_track_horizon():
    # Known answer: with H > 0 both engines give identical tracks (every column incl. est_stop_t and e0; dtype, shape,
    # bytes) and identical other outputs, on randomized streams with native iceberg refills, out-of-order fills,
    # sweeps, price-change modifies, clears, snapshots without R, shadows, 20% of event boundaries removed, 3% of F
    # records deleted and streams ending inside an event; H from 10 ms to 300 ms. Relative to H = 0, only filled
    # tracks may change, and only in est_first_fill_t (only where v0.4 had none), est_filled_qty and v_est_at_end.
    # Counters show each v0.5 situation occurs (late fills after X's fill, horizon stops, clears and stream ends
    # during the run-on, scored censored tracks).
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    rng = np.random.default_rng(23)
    tot = dict(late=0, horizon_stop=0, clear_post=0, end_post=0, censored_scored=0, true_after_h=0)
    for j, mode in enumerate(("order", "volume", "tail", "front", "order", "volume")):
        rec, st, truth, ex = generate(n_events=4000, sample_every=41, seed=900 + j, cancel_mode=mode, p_modify=0.12,
                                      p_market=0.18, max_mkt=30, p_clear=0.003, p_implicit_snap=0.003,
                                      p_iceberg=(0.35, 0.0)[j // 4], p_fill_shuffle=(0.5, 0.0)[j // 4])
        last = np.where((rec["flags"][:-1] & L) != 0)[0]
        rec["flags"][last[rng.uniform(size=len(last)) < 0.2]] &= np.uint8(255 - L)
        keep = ~((rec["action"] == b"F") & (rng.uniform(size=len(rec["action"])) < 0.03))
        rec = {k: v[keep] for k, v in rec.items()}
        if j % 2:
            rec["flags"][-1] &= np.uint8(255 - L)
        ts = rec["ts_event"].astype(np.int64)
        H = (10_000_000, 300_000_000, 60_000_000)[j % 3]
        kw = dict(label_ticks=(2, 5)[j % 2], shadow_size=(3, 1)[j % 2], horizon_ns=150_000_000, sample_levels=6)
        rk = dict(place_times=np.arange(ts[30], ts[-1], 1_500_000), sample_times=st, queue_times=st[::2], queue_levels=4,
                  order_tracks=True)
        a = Reference(tick=ex.tick, **kw).run(rec, **rk, track_horizon_ns=H)
        b = fast.run(rec, tick=ex.tick, **rk, **kw, track_horizon_ns=H)
        a0 = Reference(tick=ex.tick, **kw).run(rec, **rk)
        _assert_same_output(a, b, (j, mode))
        _assert_same_output(a, a0, (j, mode, "H=0"))
        _assert_same_tracks_h(a, b, (j, mode))
        t, t0 = a["tracks"], a0["tracks"]
        filled = t["outcome"] == OUTCOMES.index("filled")
        for k in TRACK_COLUMNS:
            same = np.array_equal(t[k], t0[k], equal_nan=True)
            if k in ("est_first_fill_t", "est_filled_qty", "v_est_at_end"):
                same = np.array_equal(t[k][~filled], t0[k][~filled], equal_nan=True)
            assert same, (j, k)
        had = t0["est_first_fill_t"] >= 0
        assert np.array_equal(t["est_first_fill_t"][had], t0["est_first_fill_t"][had])
        assert (t["est_filled_qty"] >= t0["est_filled_qty"]).all()
        tend = t["t_end"][:, None]
        tot["late"] += int((filled[:, None] & ~had & (t["est_first_fill_t"] >= 0) & (t["est_first_fill_t"] >= tend)).sum())
        tot["horizon_stop"] += int((filled & (t["est_stop_t"] >= t["t_start"] + H)).sum())
        tot["clear_post"] += int((filled & (t["est_stop_t"] >= 0) & (t["e0"][:, 0] < 0)).sum())
        tot["end_post"] += int((filled & (t["est_stop_t"] < 0)).sum())
        tot["censored_scored"] += int(((t["outcome"] == 0) & (t["e0"][:, 0] >= 0)).sum())
        tot["true_after_h"] += int((filled & (t["true_first_fill_t"] - t["t_start"] > H)).sum())
        assert ((t["e0"] >= -1) & (t["e0"] <= H)).all()
    assert all(v > 0 for v in tot.values()) and tot["late"] > 100 and tot["horizon_stop"] > 500, tot


# Digests of results["tracks"] (TRACK_COLUMNS: dtype, shape, bytes) of the committed v0.4 engines (both identical;
# commit 5dc6644, before the track horizon existed) on _digest_streams() with order_tracks=True.
V04_TRACK_DIGESTS = ["bfb6b951cb8d039e4e2525dc7e14bdb85de02ad53b10a7a9264b185b7f519be1",
                     "7ebadd18bd8e8715a5962500241846b68ce5e3d8ffc0fc6c82f6e47799b15909",
                     "dcacb2717bca4524e2a4c839ff500e26cc78a2924433b8e3cb600556938ee96a",
                     "83966ec3c6bed1c7eb579ff307e7e5a8e514c82144894455daa5e5d193923a84"]


def _track_digest(r):
    import hashlib
    h = hashlib.sha256()
    for k in TRACK_COLUMNS:
        x = np.asarray(r["tracks"][k])
        h.update(f"{k}:{x.dtype.str}:{x.shape};".encode())
        h.update(x.tobytes())
    return h.hexdigest()


def test_v05_zero_horizon_is_v04_exactly():
    # Known answer: with track_horizon_ns = 0 (the default, or explicit) both engines return exactly the committed
    # v0.4 tracks (no est_stop_t / e0 keys) and the v0.3 output of everything else; with H > 0 every non-track output
    # keeps its v0.3 value. On the reference platform these are the stored digests; elsewhere (libm differs) the
    # digests of the engine's own default run, so the equalities between flags are still checked.
    for (rec, tick, kw, rk), stored, stored_t in zip(_digest_streams(), V03_DIGESTS, V04_TRACK_DIGESTS):
        runs = [lambda **f: Reference(tick=tick, **kw).run(rec, **rk, order_tracks=True, **f)]
        if HAVE_CORE:
            runs.append(lambda **f: fast.run(rec, tick=tick, **kw, **rk, order_tracks=True, **f))
        for run in runs:
            if REFERENCE_PLATFORM:
                want, want_t = stored, stored_t
            else:
                r0 = run()
                want, want_t = _digest(r0), _track_digest(r0)
            for flag in ({}, {"track_horizon_ns": 0}):
                r = run(**flag)
                assert set(r["tracks"]) == set(TRACK_COLUMNS) and _track_digest(r) == want_t and _digest(r) == want
            r = run(track_horizon_ns=50_000_000)
            assert set(r["tracks"]) == set(TRACK_COLUMNS + TRACK_HORIZON_COLUMNS) and _digest(r) == want


def test_v06_lot_rounding_shadow_known_answer():
    # Known answer (hand-computed). Shadow on the bid at 100 placed at the t=3 boundary: V0 = Q = 5 (orders 1:3, 2:2). Order 3 (5)
    # joins behind at t=5. t=6: order 2 shrinks to 1 (keeps priority): Q 10 -> 9, D = 1, F = 5, B = Qa - V = 5, so
    # phi = 0.5 for power1/2/3 and log -> V = 4.5; front_optimistic V = 4; risk_averse V = 5. t=10: sell 5 at 100
    # fills 1 (3), 2 (1) and 1 lot of order 3 (behind the shadow: the truth shadow fills). Tr = 5: excess 0.5 for the
    # probabilistic models, 1 for front_optimistic, 0 for risk_averse. v0.5 rule (fill at >= 1 lot): only
    # front_optimistic fills; v0.6 fill_lot_rounding (hftbacktest: round(excess / lot) >= 1): the four probabilistic
    # models fill too.
    rows = BASE + [(5, "A", "B", 100, 5, 3, L), (6, "M", "B", 100, 1, 2, L),
                   (10, "T", "A", 100, 5, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 1, 2, 0),
                   (10, "F", "B", 100, 1, 3, 0), (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 1, 2, 0),
                   (10, "M", "B", 100, 4, 3, L)]
    for lr, want in ((False, [0, 0, 0, 0, 0, 1]), (True, [0, 1, 1, 1, 1, 1])):
        for r in both(recs(rows), place_times=[3], fill_lot_rounding=lr):
            sh = r["shadows"]
            b = int(np.where(np.asarray(sh["side"]) == 1)[0][0])
            assert int(sh["true_filled"][b]) == 1
            assert list(np.asarray(sh["est_filled"])[b]) == want, (lr, list(np.asarray(sh["est_filled"])[b]))
            assert [int(x) for x in np.asarray(sh["est_t_fill"])[b]] == [10 if w else -1 for w in want]


def test_v06_lot_rounding_track_known_answer():
    # Known answer (hand-computed). X = order 3 (1 lot) joins the bid at 100 at t=4 behind 1 (3) and 2 (2): V0 = 5.
    # Order 4 (8) joins behind X at t=5 (Q_ex = 13). t=6: order 2 shrinks to 1: D = 1, F = 5, B = 8: V = 5 - phi_k:
    # power1 5/13 -> excess at t=10 (Tr = 5) 0.385; power2 25/89 -> 0.281; power3 125/637 -> 0.196; log
    # log(6)/(log(6)+log(9)) -> 0.449; front_optimistic V = 4 -> excess 1; risk_averse V = 5 -> 0. X is filled at t=10.
    # v0.5 rule (first fill when the estimated filled quantity > 0): all but risk_averse at t=10; v0.6
    # fill_lot_rounding (>= 0.5 lot): only front_optimistic.
    rows = BASE + [(4, "A", "B", 100, 1, 3, L), (5, "A", "B", 100, 8, 4, L), (6, "M", "B", 100, 1, 2, L),
                   (10, "T", "A", 100, 5, 0, 0), (10, "F", "B", 100, 3, 1, 0), (10, "F", "B", 100, 1, 2, 0),
                   (10, "F", "B", 100, 1, 3, 0), (10, "C", "B", 100, 3, 1, 0), (10, "C", "B", 100, 1, 2, 0),
                   (10, "C", "B", 100, 1, 3, L)]
    for lr, want in ((False, [-1, 10, 10, 10, 10, 10]), (True, [-1, -1, -1, -1, -1, 10])):
        for r in both(recs(rows), order_tracks=True, fill_lot_rounding=lr):
            x = _trk(r, 3)
            assert x["true_first_fill_t"] == 10
            assert [int(v) for v in x["est_first_fill_t"]] == want, (lr, list(x["est_first_fill_t"]))


def test_v06_cpp_equals_reference_bitwise_with_lot_rounding():
    # Known answer: with fill_lot_rounding=True both engines give identical outputs (shadows, tracks with a track
    # horizon, labels, FIFO ranks, samples, queue snapshots) on randomized streams; the option changes shadow and track
    # estimator fills somewhere (it is exercised), and with it off the output equals the v0.5 run.
    if not HAVE_CORE:
        print("   (skipped: C++ core not built)")
        return
    changed = 0
    for j, mode in enumerate(("order", "volume", "tail", "front")):
        rec, st, truth, ex = generate(n_events=4000, sample_every=41, seed=1300 + j, cancel_mode=mode, p_modify=0.12,
                                      p_market=0.18, max_mkt=30, p_clear=0.002, p_iceberg=0.2, p_fill_shuffle=0.3)
        ts = rec["ts_event"].astype(np.int64)
        kw = dict(label_ticks=5, shadow_size=(1, 3)[j % 2], horizon_ns=150_000_000, sample_levels=6)
        rk = dict(place_times=np.arange(ts[30], ts[-1], 1_500_000), sample_times=st, queue_times=st[::2], queue_levels=4,
                  order_tracks=True, track_horizon_ns=60_000_000)
        a = Reference(tick=ex.tick, **kw).run(rec, **rk, fill_lot_rounding=True)
        b = fast.run(rec, tick=ex.tick, **rk, **kw, fill_lot_rounding=True)
        _assert_same_output(a, b, (j, mode, "lot rounding"))
        _assert_same_tracks_h(a, b, (j, mode, "lot rounding"))
        a0 = Reference(tick=ex.tick, **kw).run(rec, **rk)
        b0 = fast.run(rec, tick=ex.tick, **rk, **kw, fill_lot_rounding=False)
        _assert_same_output(a0, b0, (j, mode, "off"))
        changed += int((np.asarray(a["shadows"]["est_filled"]) != np.asarray(a0["shadows"]["est_filled"])).sum())
        changed += int((np.asarray(a["tracks"]["est_first_fill_t"]) != np.asarray(a0["tracks"]["est_first_fill_t"])).sum())
    assert changed > 0


if __name__ == "__main__":
    tests =[(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as ex:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(ex).__name__}: {ex}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed" + ("" if HAVE_CORE else " (C++ core not built)"))
    sys.exit(1 if failed else 0)

#!/usr/bin/env python3
"""Throughput of the C++ core vs the Python reference on identical synthetic MBO input.

  python3 scripts/bench.py            # 150k events, both engines, writes results/bench.json
  python3 scripts/bench.py --quick    # CI smoke version
Reports records/second for (a) book rebuild + FIFO check + labels, (b) the same plus shadows
placed every 1 s of stream time with a 60 s horizon (the run_day.py protocol), (c) (b) plus
order-anchored tracks (SPEC 9, order_tracks=True) and (d) (c) with the v0.5 track horizon of 60 s
(filled orders' estimators run on to t_start + 60 s), and checks that both engines return identical
output.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import mboq  # noqa: F401  (installed package)
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
from mboq import fast  # noqa: E402
from mboq.reference import Reference  # noqa: E402
from mboq.synthetic import generate  # noqa: E402


def timeit(f, reps=3):
    best, out = 1e9, None
    for _ in range(reps):
        t = time.perf_counter(); out = f(); best = min(best, time.perf_counter() - t)
    return best, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    n = 15_000 if a.quick else 150_000
    rec, _, _, ex = generate(n_events=n, sample_every=n, seed=2026, cancel_mode="order", mean_gap_ns=20_000_000)
    N = len(rec["ts_event"]); ts = rec["ts_event"]
    place = np.arange(int(ts[0]) + 5_000_000_000, int(ts[-1]) - 60_000_000_000, 1_000_000_000, dtype=np.int64)
    res = {"records": N, "stream_seconds": float((int(ts[-1]) - int(ts[0])) / 1e9), "shadow_placements": int(2 * len(place)),
           "machine": f"{platform.processor() or platform.machine()} | {platform.system()} | python {platform.python_version()}"}
    configs = (("book", {}), ("book+shadows", {"place_times": place}),
               ("book+shadows+tracks", {"place_times": place, "order_tracks": True}),   # SPEC 9 on
               ("book+shadows+tracks+H60s", {"place_times": place, "order_tracks": True,  # SPEC 9 v0.5 run-on
                                             "track_horizon_ns": 60_000_000_000}))
    for name, kw in configs:
        tc, oc = timeit(lambda: fast.run(rec, tick=ex.tick, **kw))
        tp, op = timeit(lambda: Reference(tick=ex.tick).run(rec, **kw), reps=1)
        same = all(np.array_equal(np.asarray(op["shadows"][k]), np.asarray(oc["shadows"][k])) for k in op["shadows"]) \
            and all(np.array_equal(op["labels"][k], oc["labels"][k]) for k in op["labels"]) \
            and ("tracks" in op) == ("tracks" in oc) \
            and all(np.asarray(op["tracks"][k]).tobytes() == np.asarray(oc["tracks"][k]).tobytes() for k in op.get("tracks", {}))
        if "tracks" in oc:
            res["tracks"] = int(len(oc["tracks"]["order_id"]))
        if "e0" in oc.get("tracks", {}):              # v0.5: filled tracks whose estimators ran on after the fill
            res["tracks_filled_run_on"] = int((oc["tracks"]["outcome"] == 1).sum())
        res[name] = {"cpp_records_per_s": N / tc, "python_records_per_s": N / tp, "speedup": tp / tc, "identical_output": bool(same)}
        print(f"{name:<26} C++ {N / tc:12,.0f} rec/s   Python {N / tp:10,.0f} rec/s   x{tp / tc:6.0f}   identical={same}")
        assert same
    out = ROOT / "results" / "bench.json"; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Known-answer validation of the six MBP queue estimators on the synthetic CME exchange.

For each cancel-location regime and 5 seeds: 30,000 events, no modifies (so the regime is the only
source of estimator error), shadows every 100 ms of stream time, 60 ms horizon. Writes
results/synthetic_validation.json and prints a markdown table (mean and s.d. across seeds).
SYNTHETIC: these numbers validate the method; they say nothing about real markets.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import mboq  # noqa: F401  (installed package)
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
from mboq import fast  # noqa: E402
from mboq.reference import ESTIMATORS  # noqa: E402
from mboq.synthetic import generate  # noqa: E402

SEEDS, N = [1, 2, 3, 4, 5], 30_000
out = {"design": __doc__.strip(), "seeds": SEEDS, "n_events": N, "regimes": {}}
for mode in ("tail", "front", "volume", "order"):
    rows = []
    for seed in SEEDS:
        rec, _, _, ex = generate(n_events=N, sample_every=N, seed=seed, cancel_mode=mode, p_modify=0.0)
        pt = np.arange(int(rec["ts_event"][200]), int(rec["ts_event"][-1]) - 120_000_000, 100_000_000)
        sh = fast.run(rec, tick=ex.tick, place_times=pt, horizon_ns=60_000_000)["shadows"]
        ok = sh["end"] != 2
        n = sh["nobs"][ok].sum()
        tf = sh["true_filled"][ok]
        rows.append({"true_fill": float(tf.mean()), "n_shadows": int(ok.sum()),
                     "bias": (sh["err_sum"][ok].sum(0) / n).tolist(), "mae": (sh["abserr_sum"][ok].sum(0) / n).tolist(),
                     "fill_bias_pp": (100 * (sh["est_filled"][ok].mean(0) - tf.mean())).tolist()})
    agg = {k: {"mean": np.mean([r[k] for r in rows], 0).tolist(), "sd": np.std([r[k] for r in rows], 0, ddof=1).tolist()}
           for k in ("bias", "mae", "fill_bias_pp")}
    agg["true_fill"] = float(np.mean([r["true_fill"] for r in rows]))
    agg["n_shadows_per_seed"] = int(np.mean([r["n_shadows"] for r in rows]))
    out["regimes"][mode] = agg
print("| regime | estimator | queue error, mean (lots) | MAE (lots) | fill-rate bias (pp) |")
print("|---|---|---|---|---|")
for mode, a in out["regimes"].items():
    for k, name in enumerate(ESTIMATORS):
        print(f"| {mode} | {name} | {a['bias']['mean'][k]:+.3f} ± {a['bias']['sd'][k]:.3f} | "
              f"{a['mae']['mean'][k]:.3f} | {a['fill_bias_pp']['mean'][k]:+.2f} ± {a['fill_bias_pp']['sd'][k]:.2f} |")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "synthetic_validation.json").write_text(json.dumps(out, indent=2))

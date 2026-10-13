#!/usr/bin/env python3
"""Run mbo-queue on one (day, contract) of Databento MBO data and score the queue estimators.

  # a full-session MBO file (and optionally MBP-10 for reconciliation):
  python3 scripts/run_day.py --mbo F.dbn.zst --mbp10 G.dbn.zst --tick 0.01 --day YYYY-MM-DD --label CLZ6
  # pipeline check on synthetic data (no Databento needed):
  python3 scripts/run_day.py --synthetic

The MBO file must contain Databento's 00:00 UTC snapshot (request the whole session) so the book
can be rebuilt. Writes <out>/<day>_<label>/: summary.json (input SHA-256s, git hash, config, every
number), report.md, and shadows/labels parquet. Output computed from licensed market data falls
under that data's license.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    import mboq  # noqa: F401  (installed package)
except ImportError:
    sys.path.insert(0, str(ROOT / "src"))
from mboq.reference import ESTIMATORS, OUTCOMES  # noqa: E402

NY = ZoneInfo("America/New_York")
NS = 1_000_000_000
# Run protocol (written into summary.json; bump the version if you change anything here)
CONFIG = {
    "version": "run_day-v0.6",
    "engine_spec": "mbo-queue SPEC v0.6 (defaults: no order tracks, no fill lot rounding)",
    "shadow_size": 1,
    "horizon_s": 60,
    "placement": "one bid + one ask shadow at the best price every 1.000 s, grid offset 0.5 s",
    "stream_start": "Databento's 00:00 UTC synthetic snapshot (R + F_SNAPSHOT); earlier records dropped",
    "placement_start": "5 min after the first live record that follows the snapshot (20:05 ET prior day in EDT)",
    "placement_end": "end of MBO coverage minus the horizon",
    "event_window_s": [-120, 120],
    "label_ticks": 5,
    "markout_horizons_s": [1, 10],
    "mid_grid_ms": 100,
    "reconcile_grid_s": 1,
}


def git_hash() -> str:
    try:
        h = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
        d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "."], capture_output=True,
                           text=True, timeout=5).stdout.strip()   # mbo-queue only, not the parent repo
        return (h or "no-git") + ("-dirty" if d else "")
    except Exception:  # noqa: BLE001
        return "no-git"


def et_ns(day: str, clock: time, offset_days: int = 0) -> int:
    d = date.fromisoformat(day) + timedelta(days=offset_days)
    return int(datetime.combine(d, clock, tzinfo=NY).timestamp() * NS)


def engine():
    try:
        from mboq import fast
        return "c++", fast.run
    except ImportError:
        from mboq.reference import Reference

        def run(rec, tick, **kw):
            r = Reference(tick=tick, label_ticks=kw.get("label_ticks", 5), shadow_size=kw.get("shadow_size", 1),
                          horizon_ns=kw.get("horizon_ns", 60 * NS), sample_levels=kw.get("sample_levels", 10))
            return r.run(rec, place_times=kw.get("place_times", ()), sample_times=kw.get("sample_times", ()))
        return "python-reference", run


# ------------------------------------------------------------------------------ analysis helpers
def mid_path(run, rec, tick, t0, t1, step_ns):
    grid = np.arange(t0, t1, step_ns, dtype=np.int64)
    s = run(rec, tick=tick, sample_times=grid, sample_levels=1)["samples"]
    bid, ask = s[:, 0].astype(float), s[:, 3].astype(float)
    mid = np.where((bid > 0) & (ask > 0), (bid + ask) / 2, np.nan)
    return grid, mid


def asof(grid, values, t):
    i = np.searchsorted(grid, t, side="right") - 1
    out = np.full(len(t), np.nan)
    ok = i >= 0
    out[ok] = values[i[ok]]
    return out


def markouts(grid, mid, t_fill, price, side, tick, h_ns):
    m = asof(grid, mid, t_fill + h_ns)
    return side * (m - price) / tick            # >0 good for the passive order


def shadow_benchmark(sh, grid, mid, tick, horizons_ns, mask):
    """Per-estimator fill-rate bias, confusion, queue error, markouts and P&L-bias decomposition."""
    ok = mask & (sh["end"] != 2)
    n = int(ok.sum())
    tf = sh["true_filled"][ok].astype(bool)
    side, price = sh["side"][ok], sh["price"][ok].astype(float)
    out = {"n_shadows": n, "true_fill_rate": float(tf.mean()) if n else None, "estimators": {}}
    if n == 0:
        return out
    nobs = sh["nobs"][ok].sum()
    mt = {h: markouts(grid, mid, sh["true_t_fill"][ok][tf], price[tf], side[tf], tick, h * NS) for h in horizons_ns}
    out["true_markout_ticks"] = {f"+{h}s": float(np.nanmean(v)) if len(v) else None for h, v in mt.items()}
    for k, name in enumerate(ESTIMATORS):
        ef = sh["est_filled"][ok, k].astype(bool)
        e = {
            "fill_rate": float(ef.mean()),
            "fill_rate_bias_pp": float(100 * (ef.mean() - tf.mean())),
            "false_fills": int((ef & ~tf).sum()), "missed_fills": int((~ef & tf).sum()),
            "queue_error_mean_lots": float(sh["err_sum"][ok, k].sum() / nobs) if nobs else None,
            "queue_error_mae_lots": float(sh["abserr_sum"][ok, k].sum() / nobs) if nobs else None,
        }
        both = ef & tf
        if both.any():
            dt = (sh["est_t_fill"][ok, k][both] - sh["true_t_fill"][ok][both]) / 1e6
            e["fill_time_error_ms_median"] = float(np.median(dt))
        for h in horizons_ns:
            me = markouts(grid, mid, sh["est_t_fill"][ok, k][ef], price[ef], side[ef], tick, h * NS)
            m_t, m_e = float(np.nanmean(mt[h])) if len(mt[h]) else np.nan, float(np.nanmean(me)) if len(me) else np.nan
            n_t, n_e = int(tf.sum()), int(ef.sum())
            e[f"markout_+{h}s_ticks"] = m_e
            # P&L(est) - P&L(true) = (n_e - n_t) * m_t  +  n_e * (m_e - m_t)
            e[f"pnl_bias_+{h}s"] = {"total_ticks": n_e * m_e - n_t * m_t,
                                    "fill_count_part": (n_e - n_t) * m_t,
                                    "adverse_selection_part": n_e * (m_e - m_t)}
        out["estimators"][name] = e
    return out


def reconcile(samples, mbp, grid):
    """Compare the rebuilt book with MBP-10 at each grid time (as-of)."""
    i = np.searchsorted(mbp["ts_event"].astype(np.int64), grid, side="right") - 1
    ok = i >= 0
    s, j = samples[ok], i[ok]
    if len(j) == 0:
        return {"n": 0}
    res = {"n": int(len(j))}
    for side, off, px, sz in (("bid", 0, "bid_px", "bid_sz"), ("ask", 30, "ask_px", "ask_sz")):
        p_mbo, q_mbo = s[:, off], s[:, off + 1]
        p_mbp, q_mbp = mbp[px][j, 0], mbp[sz][j, 0].astype(np.int64)
        same = p_mbo == p_mbp
        res[f"{side}_L1_price_match"] = float(same.mean())
        res[f"{side}_L1_size_match_given_price"] = float((q_mbo[same] == q_mbp[same]).mean()) if same.any() else None
        gap = (q_mbp - q_mbo)[same]
        res[f"{side}_L1_size_gap_mbp_minus_mbo_mean"] = float(gap.mean()) if same.any() else None
        res[f"{side}_L1_size_gap_positive_share"] = float((gap > 0).mean()) if same.any() else None
        lv = []
        for L in range(10):
            pm, qm = s[:, off + 3 * L], s[:, off + 3 * L + 1]
            pp, qp = mbp[px][j, L], mbp[sz][j, L].astype(np.int64)
            m = (pm == pp) & (pp != 0)
            lv.append(float((qm[m] == qp[m]).mean()) if m.any() else None)
        res[f"{side}_size_match_by_level"] = lv
    return res


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day")
    ap.add_argument("--mbo"); ap.add_argument("--mbp10"); ap.add_argument("--tick", type=float)
    ap.add_argument("--label"); ap.add_argument("--instrument-id", type=int)
    ap.add_argument("--release-et", default="10:30", help="scheduled release time, HH:MM US/Eastern")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "mboq"))
    a = ap.parse_args()
    which, run = engine()
    tick_ns = None
    inputs = {}
    if a.synthetic:
        from mboq.synthetic import generate
        rec, _, _, ex = generate(n_events=120_000, sample_every=10_000, seed=42, cancel_mode="order")
        tick_ns, label, day = ex.tick, "SYNTHETIC", "synthetic"
        t_lo, t_hi = int(rec["ts_event"][0]), int(rec["ts_event"][-1])
        release = (t_lo + t_hi) // 2
        place0 = t_lo + 5 * NS
        mbp = None
        inputs = {"synthetic": "mboq.synthetic.generate(n_events=120000, seed=42, cancel_mode='order')"}
    else:
        from mboq.io import load_mbo, load_mbp10, start_at_snapshot
        if not (a.mbo and a.tick and a.day and a.label):
            ap.error("--mbo, --tick, --day and --label are required (or use --synthetic)")
        mbo_path, mbp_path, tick, label, iid = a.mbo, a.mbp10, a.tick, a.label, a.instrument_id
        day = a.day
        tick_ns = int(round(tick * 1e9))
        rec, meta = load_mbo(mbo_path, iid)
        rec, snapinfo = start_at_snapshot(rec)
        if not snapinfo["snapshot_found"] or snapinfo["first_live_ts"] is None:
            sys.exit("No 00:00 UTC snapshot in this MBO file: the book cannot be rebuilt. "
                     "Request the whole session from Databento.")
        meta["start_at_snapshot"] = snapinfo
        inputs["mbo"] = meta
        mbp = None
        if mbp_path:
            mbp, mmeta = load_mbp10(mbp_path, iid)
            inputs["mbp-10"] = mmeta
        hh, mm = (int(x) for x in a.release_et.split(":"))
        release = et_ns(day, time(hh, mm))
        # snapshot records carry old ts_event values: the time range comes from live records
        t_lo, t_hi = snapinfo["first_live_ts"], int(rec["ts_event"].max())
        place0 = t_lo + 5 * 60 * NS
    H = CONFIG["horizon_s"] * NS
    place = np.arange(place0 + NS // 2, t_hi - H, NS, dtype=np.int64)
    print(f"engine={which}  records={len(rec['ts_event']):,}  shadows planned={2 * len(place):,}")
    r = run(rec, tick=tick_ns, place_times=place, label_ticks=CONFIG["label_ticks"],
            shadow_size=CONFIG["shadow_size"], horizon_ns=H)
    sh = r["shadows"]
    grid, mid = mid_path(run, rec, tick_ns, t_lo, t_hi + H, CONFIG["mid_grid_ms"] * 1_000_000)
    ev = (sh["t_place"] >= release + CONFIG["event_window_s"][0] * NS) & (sh["t_place"] < release + CONFIG["event_window_s"][1] * NS)
    hz = CONFIG["markout_horizons_s"]
    fr = r["fifo_rank"]
    known = fr >= 0
    win = (r["fifo_ts"] >= release - 600 * NS) & (r["fifo_ts"] < release + 600 * NS)
    summary = {
        "status": "DESCRIPTIVE: one day, one contract",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_hash(),
        "engine": which, "day": day, "label": label, "tick_ns": tick_ns, "config": CONFIG, "inputs": inputs,
        "stats": {k: int(v) for k, v in r["stats"].items()},
        "fifo_consistency": {
            "session_rate": float((fr[known] == 0).mean()) if known.any() else None,
            "session_n_fills_known_id": int(known.sum()), "fill_unknown": int((~known).sum()),
            "release_pm10min_rate": float((fr[known & win] == 0).mean()) if (known & win).any() else None,
            "rank_distribution": {str(k): int(v) for k, v in zip(*np.unique(fr, return_counts=True))},
            "validated_for_queue_claims": bool(known.any() and (fr[known] == 0).mean() >= 0.95),
        },
        "shadow_benchmark": {
            "all": shadow_benchmark(sh, grid, mid, tick_ns, hz, np.ones(len(sh["sid"]), bool)),
            "release_window": shadow_benchmark(sh, grid, mid, tick_ns, hz, ev),
            "outside_release_window": shadow_benchmark(sh, grid, mid, tick_ns, hz, ~ev),
        },
        "labels": {OUTCOMES[int(k)]: int(v) for k, v in zip(*np.unique(r["labels"]["outcome"], return_counts=True))},
    }
    if mbp is not None and len(mbp["ts_event"]):
        g = np.arange(int(mbp["ts_event"].min()) + NS, int(mbp["ts_event"].max()), CONFIG["reconcile_grid_s"] * NS, dtype=np.int64)
        smp = run(rec, tick=tick_ns, sample_times=g, sample_levels=10)["samples"]
        summary["reconciliation_vs_mbp10"] = reconcile(smp, mbp, g)
    outdir = Path(a.out) / f"{day}_{label}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    try:
        import pandas as pd
        pd.DataFrame({k: (v if np.ndim(v) == 1 else list(v)) for k, v in sh.items()}).to_parquet(outdir / "shadows.parquet")
        pd.DataFrame(r["labels"]).to_parquet(outdir / "labels.parquet")
    except Exception as ex:  # noqa: BLE001
        print("parquet skipped:", ex)
    (outdir / "report.md").write_text(report_md(summary))
    print((outdir / "report.md").read_text())
    print(f"wrote {outdir}")


def report_md(s):
    f = s["fifo_consistency"]
    L = [f"# mbo-queue run: {s['day']} {s['label']}", "", f"**{s['status']}**", "",
         f"engine {s['engine']}, git {s['git']}, records {s['stats']['records']:,}, events {s['stats']['events']:,}", "",
         "## Rebuild quality",
         f"- FIFO-consistency (session): {f['session_rate']} on {f['session_n_fills_known_id']:,} fills with known ids; "
         f"unknown-id fills {f['fill_unknown']:,}; release +/-10 min: {f['release_pm10min_rate']}",
         f"- validated for queue-position claims (>= 95%): {f['validated_for_queue_claims']}",
         f"- unknown cancels {s['stats']['unknown_cancel']:,}, modify-unknown {s['stats']['modify_unknown']:,}, "
         f"implicit clears {s['stats']['implicit_clear']}, F_MAYBE_BAD_BOOK records {s['stats']['maybe_bad_book']:,}"]
    if "reconciliation_vs_mbp10" in s:
        rc = s["reconciliation_vs_mbp10"]
        L += [f"- vs MBP-10 ({rc.get('n', 0)} one-second checks): bid L1 price match {rc.get('bid_L1_price_match')}, "
              f"size match {rc.get('bid_L1_size_match_given_price')}; ask L1 price match {rc.get('ask_L1_price_match')}, "
              f"size match {rc.get('ask_L1_size_match_given_price')}. Mean MBP-minus-MBO L1 size gap: bid "
              f"{rc.get('bid_L1_size_gap_mbp_minus_mbo_mean')}, ask {rc.get('ask_L1_size_gap_mbp_minus_mbo_mean')} "
              "(a positive gap is consistent with implied liquidity in MBP but not MBO; verify before saying so)"]
    for key in ("all", "release_window", "outside_release_window"):
        b = s["shadow_benchmark"][key]
        L += ["", f"## Queue estimators vs the order-ID book: {key.replace('_', ' ')}",
              f"{b['n_shadows']:,} shadows; true fill rate within {s['config']['horizon_s']} s: {b['true_fill_rate']}", "",
              "| estimator | fill rate | bias (pp) | false / missed fills | queue error mean / MAE (lots) | P&L bias +10s (ticks): count part / adverse-selection part |",
              "|---|---|---|---|---|---|"]
        for name, e in b.get("estimators", {}).items():
            pb = e.get("pnl_bias_+10s", {})
            L.append(f"| {name} | {e['fill_rate']:.4f} | {e['fill_rate_bias_pp']:+.2f} | {e['false_fills']} / {e['missed_fills']} | "
                     f"{_f(e['queue_error_mean_lots'])} / {_f(e['queue_error_mae_lots'])} | "
                     f"{_f(pb.get('fill_count_part'))} / {_f(pb.get('adverse_selection_part'))} |")
    L += ["", "One day, one contract: descriptive only. Inference needs many events and days."]
    return "\n".join(L) + "\n"


def _f(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.3f}"


if __name__ == "__main__":
    main()

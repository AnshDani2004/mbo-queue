# mbo-queue

Exact FIFO queue reconstruction from CME market-by-order (MBO) data, and a benchmark of the
queue-position estimators that market-making backtests use when they only see aggregate depth.

## Why

A backtest of a passive (resting) order has to guess where that order sits in the queue, because
most market data (market-by-price, MBP) only shows the total size at each price. When the size at
your price drops and no trade printed, someone cancelled, but were they ahead of you or behind
you? Open-source backtesters answer with a family of rules: hftbacktest's `RiskAdverseQueueModel`
(cancels are always behind you) and its probabilistic power and log models, which follow
Rigtorp's estimator.

CME MBO data, as distributed by Databento, carries an order ID on every add, cancel, modify and
fill, so the true queue can be rebuilt. mbo-queue rebuilds it and measures how wrong each
estimator is: in queue position, in fill rate, in fill time, and in the P&L of a passive order.

## What it computes

Each item is one section of [SPEC.md](SPEC.md), which is the rule set both engines implement.

1. **Input.** A single-instrument stream of Databento MBO records, read in file order and only at
   event boundaries (records ending with the `F_LAST` flag).
2. **Book rebuild.** Adds join the back of their price level, size increases and price changes
   lose priority, size decreases keep it, fills leave the book through the cancel or modify that
   follows them, and a native iceberg refill goes to the back as it does at CME.
3. **FIFO-consistency rate.** For every fill, whether the filled order is the one the rebuild says
   is at the front; this is the quality gate (at least 95%) before trusting queue positions.
4. **Shadow orders.** A hypothetical order placed at the back of the best level, whose true queue
   position is the remaining size of the specific orders that were ahead of it (no market impact).
5. **MBP estimators.** Six queue-position rules (`risk_averse`, `prob_power1/2/3`, `prob_log`,
   `front_optimistic`) run on the same shadows while seeing only aggregate depth and trade prints.
6. **Resting-order labels.** For every real order near the touch: queue ahead at entry, distance
   from the best price, and its outcome (filled, cancelled, moved, requeued, cleared, censored)
   with the time to that outcome, for competing-risks fill models.
7. **Book samples.** The top N levels at requested times, for reconciliation with MBP-10 and for
   mid-price paths.
8. **Queue-position snapshots.** At requested times, every real resting order at the best levels
   with its rank, size and size ahead, read without look-ahead.
9. **Order-anchored tracks.** The six estimators run for real orders instead of hypothetical ones
   and scored against each order's own fill records, optionally keeping them running after the
   true fill (a horizon H) so that late estimated fills are measured.

`scripts/run_day.py` turns these outputs into per-estimator metrics: fill-rate bias, false and
missed fills, queue-position error, fill-time error, markouts at +1 s and +10 s, and a split of the
P&L bias into a fill-count part and an adverse-selection-given-a-fill part:
`P&L_est - P&L_true = (N_est - N_true) * m_true + N_est * (m_est - m_true)`.

## Install

```bash
pip install mbo-queue            # the engines; needs only numpy
pip install "mbo-queue[data]"    # plus databento, pandas and pyarrow, to read DBN files
```

Wheels are built for CPython 3.10 to 3.14 on Linux (x86_64, aarch64) and macOS (arm64, x86_64).
Elsewhere pip builds from source, which needs a C++20 compiler.

## Quickstart

Run the six estimators against the true queue on the built-in synthetic exchange, where cancels
hit resting orders in proportion to their size:

```python
import numpy as np
from mboq import fast
from mboq.reference import ESTIMATORS
from mboq.synthetic import generate

rec, _, _, ex = generate(n_events=30_000, seed=1, cancel_mode="volume", p_modify=0.0)
ts = rec["ts_event"].astype(np.int64)
place = np.arange(ts[200], ts[-1] - 120_000_000, 100_000_000)       # a shadow every 100 ms
sh = fast.run(rec, tick=ex.tick, place_times=place, horizon_ns=60_000_000)["shadows"]
ok = sh["end"] != 2                                                  # drop shadows cut by a clear
nobs, true_fill = sh["nobs"][ok].sum(), sh["true_filled"][ok].mean()
print(f"{'estimator':<18}{'queue error (lots)':>20}{'fill-rate bias (pp)':>22}")
for k, name in enumerate(ESTIMATORS):
    err = sh["err_sum"][ok, k].sum() / nobs
    bias = 100 * (sh["est_filled"][ok, k].mean() - true_fill)
    print(f"{name:<18}{err:>+20.3f}{bias:>+22.2f}")
```

Output (synthetic data):

```
estimator           queue error (lots)   fill-rate bias (pp)
risk_averse                     +1.738                 -1.00
prob_power1                     -0.028                 -0.42
prob_power2                     -0.109                 -0.50
prob_power3                     -0.119                 -0.42
prob_log                        +0.548                 -0.25
front_optimistic                -0.246                 +0.50
```

`risk_averse` assumes every cancel is behind you, so it thinks the queue ahead is longer than it
is; the linear model (`prob_power1`) is close to unbiased here because that is how this synthetic
exchange chooses which order cancels.

`mboq.fast.run` (C++) and `mboq.reference.Reference(...).run` (pure Python) take the same records
and return the same dict of numpy arrays: `stats`, `fifo_rank`, `shadows`, `labels`, `samples`,
`queue`, and `tracks` with `order_tracks=True`.

Options of `run()` that change the rules are all off by default, and each one switched off gives
the previous version's output byte for byte:

| option | default | what it does (SPEC) |
|---|---|---|
| `order_tracks` | `False` | order-anchored estimator tracks for real orders (section 9, v0.4) |
| `track_horizon_ns` | `0` | keep a filled order's estimators running until `t_start + H` (section 9, v0.5) |
| `fill_lot_rounding` | `False` | round estimated fills to whole lots, as hftbacktest does (v0.6) |
| `truth_record_price` | `False` | read a shadow's true fills at the `F` record's own price and side (section 4, v0.7) |

`truth_record_price` is a sensitivity, not a correction. By default a fill is located where the
filled order rests in the rebuilt book. On CME data a known-id `F` record can carry a price other
than its order's resting price, and it is not settled which of the two locates the fill. With
`truth_record_price=True` the record's own price and side are used instead (side `N` takes the
order's side): a fill at the shadow's price fills it unless the filled order rests at that level
ahead of the shadow, and a fill at a worse price is a trade-through. Running both settings shows
how much a result depends on which reading of those records is right. Only the shadows' truth
side changes (true fills and fill times, the queue-error sums taken while the truth is unfilled,
the shadow's end) and the `shadow_fifo_anomaly` counter; labels, FIFO ranks, book samples, queue
snapshots and order tracks are unaffected.

## Running on Databento data

```bash
python scripts/run_day.py --mbo session.mbo.dbn.zst --mbp10 session.mbp-10.dbn.zst \
    --tick 0.01 --day YYYY-MM-DD --label CLZ6 --release-et 10:30
python scripts/run_day.py --synthetic          # the same pipeline on synthetic data
```

The MBO file must cover the whole session, because the book can only be rebuilt from Databento's
00:00 UTC snapshot. `run_day.py` places a bid and an ask shadow every second, scores the six
estimators overall and in a window around a scheduled release time, reconciles the rebuilt book
with MBP-10 if given, and writes `summary.json` (with input SHA-256s and the run configuration) and
`report.md`. Results computed from licensed market data are covered by that data's license.

## How it is validated

Everything below runs in the test suite or the scripts in this repository, on synthetic data.

* **Known answers per estimator and regime.** The synthetic CME-style FIFO exchange
  (`mboq/synthetic.py`) emits Databento-format records and has a switch for where in the queue
  cancels happen, so the right answer is known. `scripts/synthetic_validation.py` (5 seeds,
  30,000 events each):

  | cancel regime | expected | result |
  |---|---|---|
  | at the tail | `risk_averse` exact | zero queue error, zero fill-rate bias |
  | at the front | `front_optimistic` exact | zero queue error, zero fill-rate bias |
  | proportional to size | linear model close to unbiased | mean queue error -0.022 lots (s.d. 0.040 across seeds) vs +1.769 for `risk_averse` |
  | proportional to order count | linear model close to unbiased | mean queue error +0.003 lots (s.d. 0.030) vs +1.710 for `risk_averse` |

* **Rebuild equals the exchange.** The rebuilt book and queues equal the generator's own at every
  sampled event in all four regimes, including book clears, snapshots without a clear, native
  iceberg refills and fills listed out of priority order; FIFO-consistency is then exactly 100%.
* **Two independent implementations, bitwise equal.** The Python reference (explicit ahead sets)
  and the C++20 core (priority stamps and intrusive lists) return byte-identical output (every
  counter and array: dtype, shape and bytes) on randomized streams with every feature switched on.
  The C++ core is compiled with `-ffp-contract=off` so no fused multiply-add breaks this.
* **Hand-computed scenarios** for each rule: cancel ahead vs behind, priority loss, trade-through,
  FIFO violations, label outcomes, iceberg refills, out-of-order fills, every track and horizon case.
* **No look-ahead.** Book samples and queue snapshots before time T do not change when every record
  after T is scrambled or deleted.
* **Backward compatibility.** Each new feature is off by default, and turning it off reproduces the
  earlier version. Stored SHA-256 digests of earlier versions' outputs are checked in full on macOS
  arm64, where they were recorded. The estimators' floating-point state goes through the system
  math library (`pow`, `log1p`), whose last bits can differ between platforms. So elsewhere only
  the input streams and the book outputs (labels, FIFO ranks, samples, queue rows) are compared
  with stored digests, and the flag-off/flag-on equalities are checked within the run.
* **Real decode path.** One test decodes Databento's public 4-record sample file (fetched by
  `scripts/fetch_test_data.sh`, pinned by commit and SHA-256; the test skips without it).

```bash
bash scripts/fetch_test_data.sh      # optional: the DBN decode test
python tests/test_mboq.py            # 53 tests (or: pytest tests)
python scripts/synthetic_validation.py
```

## Performance

Synthetic stream from `scripts/bench.py` (229,313 records, 5,838 shadow placements, 28,588 order
tracks), one thread, Apple M3, macOS, Python 3.10, mbo-queue 0.6.0 built from the sdist. Every run
checks that the C++ and Python outputs are identical.

| workload | C++ core | Python reference |
|---|---:|---:|
| rebuild + FIFO check + labels | 18.9M records/s | 26.5k records/s |
| plus shadows, 6 estimators each | 3.79M records/s | 2.5k records/s |
| plus order tracks | 0.57M records/s | 1.8k records/s |
| plus a 60 s track horizon | 0.52M records/s | 1.6k records/s |

Order tracks are off by default and cost nothing when off. When they are on, most of the time goes
to the per-track estimator updates the spec requires at every touched level. `pow` and `log1p`
stay library calls so that the C++ output matches Python bit for bit.

## Limitations

* **Displayed outright book only.** MBO does not include implied (spread-generated) liquidity or
  the hidden part of iceberg orders, so "truth" here is the order-ID-resolved displayed outright
  book. Its quality is measured (FIFO-consistency, MBP-10 reconciliation), not assumed.
* **No-impact shadows.** A shadow order never trades and does not change anyone's behaviour, the
  same assumption every replay backtester makes; results apply to small orders. Order-anchored
  tracks (section 9) use real orders, at the price of selection (orders that joined at the touch
  are not a random sample of placements).
* **Snapshot ordering.** Snapshot records are assumed to arrive in queue order within each price
  level (SPEC assumption A1). The FIFO-consistency gate tests this on every file.
* **CME semantics.** The rebuild rules are those of Databento's CME Globex MBO feed. Other venues
  may need different rules.

## Research use

mbo-queue underpins an order-level study of liquidity in CME energy futures (paper forthcoming).
This repository contains only the engine, its synthetic tests and generic scripts; no market data
and no results computed from market data.

## How to cite

See [CITATION.cff](CITATION.cff), or:

> Dani, A. (2026). mbo-queue: exact FIFO queue reconstruction from CME market-by-order data and a
> benchmark of MBP queue-position estimators (version 0.7.0). https://github.com/AnshDani2004/mbo-queue

## Layout

`SPEC.md` (the rules) · `src/mboq/reference.py` (readable Python) · `cpp/mboq_core.cpp` (C++20 +
pybind11) · `src/mboq/fast.py` (the C++ wrapper) · `src/mboq/synthetic.py` (known-answer exchange)
· `src/mboq/io.py` (DBN loading) · `scripts/` · `tests/` · [CHANGELOG.md](CHANGELOG.md).

## License

MIT for the code (see [LICENSE](LICENSE)). No market data is included in this repository or its
packages.

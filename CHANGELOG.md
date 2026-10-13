# Changelog

Versions follow SPEC.md: package version 0.N.x implements SPEC v0.N. Every version keeps the
Python reference (`mboq.reference`) and the C++ core (`mboq._core`) bitwise equal, and every
additive feature is off by default so earlier outputs are unchanged (checked by stored digests).

## Unreleased

- Tests: the DBN decode test now skips, instead of failing, when the sample file has been fetched
  but `databento` is not installed (a plain `pip install mbo-queue` without `[data]`). Test-only; the
  0.6.0 packages are unaffected.

## 0.6.0 (2026-10-13)

First public release.

- New `run()` option `fill_lot_rounding` (default off, so v0.5 outputs are byte-identical): estimated
  fills are rounded to whole lots as hftbacktest does. A shadow of size S counts as filled once its
  estimated fill reaches S - 0.5 lots; a track's estimated first fill is the first time its
  estimated filled quantity reaches 0.5 lot. Only the fractional estimators (prob_power1/2/3,
  prob_log) are affected.
- SPEC section 4, wording only: a truth fill is read at the filled order's resting position in the
  rebuilt book (its side and price), not the `F` record's own price field. The engines always did this.
- Packaging: pre-built wheels for CPython 3.10 to 3.14 on manylinux (x86_64, aarch64) and macOS
  (arm64, x86_64); optional extra `[data]` for reading Databento files.

## 0.5 (2026-10-12)

- Order-track horizon `track_horizon_ns` (default 0 = v0.4 exactly). With H > 0, a filled order's
  estimators keep running after the true fill until `t_start + H`, as a backtester that does not
  see the fill would, so late estimated fills are measured instead of cut off.
- New per-track outcome E0: the absolute error of the estimated time to first fill, each time
  capped at H. Early, late, false and missed fills all count. New track keys `est_stop_t`, `e0`.

## 0.4 (2026-10-12)

- Order-anchored estimator tracks (SPEC section 9, `order_tracks=True`, off by default): the six
  estimators run for every real order that joins at the touch, starting from its true size ahead
  and seeing the level size without the order itself, scored against the order's own fill records.

## 0.3 (2026-10-11)

- Book rebuild: a same-price modify is compared with the size left after the fills reported on the
  order since its last book change, so a native iceberg refill loses priority, as at CME.
- FIFO-consistency: an order ahead of the filled one is netted out if fills anywhere in the same
  matching event consume it (the feed can list an event's fills out of priority order).
- With icebergs and shuffled fill listing switched off, every output equals v0.2.

## 0.2 (2026-10-10)

- Queue-position snapshots (SPEC section 8): at requested times, every real resting order at the
  best levels with its rank, size, size ahead and the level totals, with no look-ahead.

## 0.1 (2026-10-10)

- Book rebuild from Databento MBO records with CME semantics (SPEC section 2).
- FIFO-consistency rate as the quality gate for queue-position claims (section 3).
- Shadow orders with the true queue position under a no-impact assumption (section 4).
- Six MBP queue-position estimators tracked on the same shadows (section 5): risk_averse,
  prob_power1/2/3, prob_log, front_optimistic.
- Resting-order labels for competing-risks fill models (section 6) and book samples (section 7).
- Synthetic CME-style FIFO exchange with known answers; Python reference and C++20/pybind11 core.

# Changelog

Versions follow SPEC.md: package version 0.N.x implements SPEC v0.N. Every version keeps the
Python reference (`mboq.reference`) and the C++ core (`mboq._core`) bitwise equal, and every
additive feature is off by default so earlier outputs are unchanged (checked by stored digests).

## 0.7.1 (unreleased)

Input validation and SPEC clarifications only. No rule changes: on every valid input the output of
both engines is byte-identical to 0.7.0 (checked on 48 synthetic runs per engine, with every option
on and action/side given as bytes, str and integer codes).

- Malformed input now raises `ValueError`, with the same message, in the Python reference and the
  C++ core. In 0.7.0 the two engines disagreed on these inputs, or the C++ core had undefined behaviour:
  - an `A` or `M` record with a side other than `B` or `A` (Python raised `KeyError`, C++ booked the
    order on the ask side). `C`, `R`, `T`, `F` and `N` records may still have side `N`, as in CME data.
  - record columns of unequal length, or not 1-D (C++ read past the end of the shorter column);
  - `sample_levels < 1` (C++ divided by zero) and `tick < 1` (C++ integer division by zero);
  - unsorted `place_times` or `sample_times` (silently accepted; samples were read at the wrong
    state). `queue_times` was already checked.
  The C++ binding (`mboq._core.run`) runs the same checks, so it is safe to call directly.
- Action and side codes are read by one function for both engines (`mboq.reference.code_array`):
  bytes, str, integer codes 0-255 and object arrays of these. In 0.7.0 an object array of bytes was
  silently ignored by the Python reference and rejected by the C++ core. Floats, bools,
  multi-character codes and codes above 255 raise `ValueError`.
- Order ids >= 2^63 in `labels["order_id"]` are stored as the int64 with the same bits, as queue
  rows and tracks already were (the Python reference raised `OverflowError`; C++ was unchanged).
- SPEC: section 1 lists the input checks; section 2 states that an implicit clear happens only if
  the book is non-empty; section 8 states that times still pending at the end of a stream whose last
  record lacks `F_LAST` read the final, possibly mid-event, state. Both describe what the engines
  already did.
- Tests: one hand test per check above (both engines raise the same exception and message), and a
  randomized C++-vs-Python differential test on 400 seeded random streams with side-`N` `T` and `F`
  records. CI: a new job runs the test suite and the benchmark smoke test against a C++ core built
  with AddressSanitizer and UndefinedBehaviorSanitizer.
- The module docstring of the C++ core now says SPEC v0.7 (it said v0.5).

## 0.7.0 (2026-10-14)

- New `run()` option `truth_record_price` (SPEC v0.7, section 4; default off, so v0.6 outputs are
  byte-identical): a sensitivity in which a known-id `F` record moves shadows by the record's own
  price and side (side `N` takes the filled order's side) instead of the filled order's resting
  position in the rebuilt book. A fill at the shadow's price fills it unless the filled order rests
  at that level ahead of the shadow; a fill at a worse price is a trade-through. Only shadow truth
  outputs and the `shadow_fifo_anomaly` counter can change. Two new tests: a hand-computed case,
  and C++ = Python bitwise on randomized streams with repriced and side-`N` `F` records.
- Tests: the DBN decode test now skips, instead of failing, when the sample file has been fetched
  but `databento` is not installed (a plain `pip install mbo-queue` without `[data]`). Test-only.

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

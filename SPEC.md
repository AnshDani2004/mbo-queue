# mbo-queue specification (v0.6, 2026-10-13)

> **Note on this public copy.** The rules in this document are identical to the frozen SPEC v0.6
> used in the study of CME energy futures that this engine underpins (paper forthcoming). This
> copy only removes references to private study files and one example count computed from market
> data; no rule, definition or default differs.

**Change log**

* **v0.6** (2026-10-13) (a) section 4
  wording clarified, no behaviour change: a truth fill is read at the filled order's resting position in the
  rebuilt book (its side and price), not the `F` record's own price field; the engines always did this. (b) New
  `run()` parameter `fill_lot_rounding` (default False = v0.5 exactly, byte-identical; tested): with True, estimated
  fills are rounded to whole lots as hftbacktest does (`round(overdraw / lot) >= 1`): a shadow of size S counts as
  filled once its cumulative estimated fill reaches S - 0.5 lots (one-lot shadows: 0.5), and a track's estimated
  first fill is the first time its estimated filled quantity reaches 0.5 lot (instead of > 0). It affects only
  estimators whose V can be fractional (prob_power1/2/3, prob_log); risk_averse and front_optimistic are unchanged.

* **v0.5** (2026-10-12) section 9 only: the `run()` parameter `track_horizon_ns` = H (default 0 =
  v0.4 exactly, tested against stored v0.4 digests). With H > 0, a filled order's estimators keep
  running after its fill until the first event boundary at or past `t_start + H` (lateness is
  measured, not censored at the fill), and two track keys are added: `est_stop_t` and `e0`, the
  absolute first-fill-time error censored at H (outcome E0). Sections 1-8 are unchanged
  and with any H every non-track output is byte-identical.

* **v0.4** (2026-10-12) adds section 9 (order-anchored estimator tracks), an additive output behind
  the `run()` flag `order_tracks` (default off). Sections 1-8 are unchanged; with the flag off every
  output is byte-identical to v0.3, and with it on every output other than `results["tracks"]` is
  too (tested against the committed v0.3 engines).

* **v0.3** (2026-10-11). Two rule changes; every other rule is unchanged.
  1. Section 2, `M` at an unchanged price: the new size is compared with the quantity left after
     the `F` records reported on the order since its last book change (`fill_since_update`), not
     with its rebuilt size. A native iceberg refill (an `F` that empties the display, then an `M`
     that restores it) therefore loses priority, as it does at the exchange. Sections 4, 6 and 8,
     which follow "loses priority", now say "requeue" and point to this rule. The table now also
     states what both engines already did: a same-price `M` to size 0 removes the order; it is
     checked before the requeue test (which matters only if reported fills exceed the size).
     Assumption A3 is added.
  2. Section 3, FIFO-consistency rank: an order ahead of the filled order is netted out if `F`
     records anywhere in the same event (also those listed after the ranked one) consume it, i.e.
     the event's total fills per order at the event boundary are used, not only earlier records.
     The section 4 counter `shadow_fifo_anomaly` keeps its v0.2 definition.
  On the synthetic exchange with native icebergs and out-of-order fill listing switched off, every
  v0.3 output equals the v0.2 output (tested).
* **v0.2** (2026-10-10) adds section 8 (queue-position snapshots), an additive output. Sections 1-7
  are unchanged from v0.1 and every v0.1 output is identical with or without the new inputs
  (tested).

Both implementations (`mboq.reference`, pure Python, and `mboq._core`, C++20) implement exactly
this document. A test asserts they produce identical output on randomized synthetic streams.
Anything not specified here is a bug. Assumptions that real data must confirm are marked **A#**.

## 1. Input

A single-instrument stream of Databento MBO records in file order, with fields
`ts_event` (ns, uint64), `order_id` (uint64), `price` (int64, 1e-9 units), `size` (uint32),
`flags` (uint8), `action` (`A C M R T F N`), `side` (`A` ask, `B` bid, `N` none).
Flags: `F_LAST=128`, `F_SNAPSHOT=32`, `F_MAYBE_BAD_BOOK=4`.

An **event** is a run of records ending with a record that has `F_LAST` set. Book state is only
read at event boundaries (after an `F_LAST` record), because intermediate states are not
consistent (Databento convention).

## 2. Book rebuild (Databento semantics for CME)

| Action | Effect |
|---|---|
| `A` | Append the order to the back of its (side, price) level. If the id already exists: if `F_SNAPSHOT` is set, replace it (remove, then append); otherwise count `dup_add` and replace. |
| `C` | Reduce the order's size by `size`; remove it if the result is <= 0 or if `size == 0`. Unknown id: count `unknown_cancel`, no effect. |
| `M` | Unknown id: treat as `A` and count `modify_unknown`. Price change: remove and append to the back of the new level. Same price, new size 0: remove the order (outcome as for a `C`). Same price, otherwise, with `rem = size - fill_since_update` (the quantity left after the reported fills): new size > `rem`: **requeue**, i.e. move to the back of the same level (loses priority), as for any size increase; new size <= `rem`: update in place (keeps priority). |
| `R` | Clear the whole book. |
| `T` `F` `N` | No book change (fills leave the book through the `C`/`M` records that follow). An `F` on a known id adds its size to that order's `fill_since_update`. |

**`fill_since_update`.** Every resting order carries `fill_since_update`, the total size of the
`F` records on its id since its last book change. It is 0 when the order is appended (an `A`,
including snapshot adds; an `M` for an unknown id; a price-change `M`; a requeue), grows with every
`F` on the order's id (in any event, until the next book change), and is reset to 0 after any `C`
or `M` on the order is applied. An `R` (or an implicit clear) removes every order and with it every
`fill_since_update`. Example, a 5-lot order: `F 2` then `M 3` is not an increase (rem = 3) and
keeps priority; `F 2` then `M 4` is an increase (rem = 3) and requeues; `F 5` then `M 5` requeues.

Rationale: CME native icebergs refill their displayed quantity under the same order id, and the
refill gets new time priority (it joins the back of the level). Databento shows the refill as an
`F` that empties the display followed by an `M` that restores the size (`F 5`, `M 5` on a 5-lot
display). Compared with the rebuilt size (5) that `M` is not an increase, so v0.2 kept the order's
priority; compared with what is left after the fills (0) it is an increase, which is what the
exchange did. A partial fill followed by an `M` to the post-fill size is unaffected.

**Implicit clear.** If a record with `F_SNAPSHOT` arrives while the previous record did not have
`F_SNAPSHOT` and the previous record was not `R`, the book is cleared first (count
`implicit_clear`). This guards against a mid-stream snapshot without a preceding `R`.

**A1.** Snapshot `A` records arrive in queue (priority) order within each price level.
**A2.** On CME, `F` records carry the resting order's id and the filled quantity, and `T` records
carry the aggressor side. Both are tested on real data by the FIFO-consistency rate (section 3).
**A3.** A native iceberg refill is reported as the `F` record(s) that empty the display followed by
an `M` at the same price on the same id that shows the new display. Also tested by the
FIFO-consistency rate: if refills kept priority, the next fills at that level would rank > 0.

## 3. FIFO-consistency (validates A1, A2, A3 and the rebuild)

For each `F` record with a known order id X at level L: the *rank* of X is the number of orders
ahead of X in L that are not fully consumed by the `F` records of the same event, counting every
`F` record of that event, also those listed after X's `F` in the file. Which orders are ahead of X,
and each one's size, are read from the queue at the moment X's `F` record is processed (section 2
state, so a requeued order is no longer ahead and an order cancelled later in the event still is).
Whether an ahead order Y is consumed is decided at the event boundary, with the event's total fills
per order: Y is consumed iff the total size of the event's `F` records on Y's id >= Y's size at the
moment X's `F` was processed. Consistent iff rank = 0. Rationale: within one matching event the feed
can list fills out of priority order; under FIFO the exchange filled X only after
exhausting every order ahead, so the question is whether the event as a whole consumed them. Ranks
are output in file order of the `F` records and are final at the event boundary (an event still
open at the end of the stream, whose last record lacks `F_LAST`, closes there).
`F` with an unknown id counts as `fill_unknown` (rank = -1). The **FIFO-consistency rate** is the
share of known-id `F` records with rank 0. A rebuild is "validated" for queue-position claims only
if this rate is >= 95%.

## 4. Shadow orders (true queue position under the no-impact assumption)

A shadow is a hypothetical order of size S on side s at price p, placed at the first event
boundary at or after its placement time, at the **back** of the level (behind every order resting
there at that moment). It never enters the book. Placement price: the best price of side s at that
boundary (no shadow if that side is empty).

* **Ahead set.** The orders resting at (s, p) at placement. An ahead order leaves the set when it
  is removed, when its price changes, or when it is requeued (section 2: an `M` at the same price
  whose new size exceeds its size minus `fill_since_update`, e.g. an iceberg refill; it loses
  priority and requeues behind the shadow). In-place updates keep it in the set.
  `V_true` = sum of the current sizes of the ahead set.
* **Fills (truth).** While unfilled, an `F` record on side s
  (a) at price p for an order **not** in the ahead set, or
  (b) at a price worse than p for the shadow (bid: < p; ask: > p)
  where side and price are those of the **filled order's resting position in the rebuilt book** (the order
  the record's id names), not the record's own price field (v0.6 clarification; on CME data some known-id `F`
  records carry a different price, so the record's own price field is not used),
  fills the shadow by min(record size, remaining), at that record's `ts_event`. Rationale: under
  FIFO the exchange could only have reached that order after exhausting the ahead set and the
  shadow. A case-(a) fill while ahead volume not yet consumed by earlier `F` records of the same event is
  positive is counted as `shadow_fifo_anomaly` (expected 0 on FIFO data). This counter keeps its
  v0.2 definition in v0.3: ahead volume = sum over the ahead set of max(current size - `F` size on
  that id in the event so far, 0), evaluated when the record is processed; it does not use the
  section 3 end-of-event netting, so fills listed out of priority order within an event can make
  it positive on FIFO data.
* **Life.** Until `ts_place + horizon`, until the truth and every estimator (section 5) have
  filled, or until an `R` (end reason `cleared`; such shadows are excluded from metrics). The
  truth stops accruing fills once filled; estimators keep running for the whole horizon because a
  backtester would not know the true outcome. `F` records with an unknown id never move shadows.

## 5. MBP queue-position estimators (what a backtester without order IDs does)

Each shadow is also tracked by K estimators that see only what an MBP feed plus trade prints show:
the aggregate size Q at (s, p) after each event, and `T` prints. Per event, with `Tr` = volume of
`T` prints at price p whose aggressor is on the opposite side, and `Q0` = Q before the event:

1. Trades consume from the front: `excess = Tr - V`; `V = max(V - Tr, 0)`; if `excess > 0` the
   shadow fills by min(excess, remaining).
2. Trade-through: any opposite-aggressor `T` print at a price worse than p (bid: < p; ask: > p)
   fills the shadow completely.
3. Unexplained decrease (cancels): `Qa = max(Q0 - Tr, 0)`, `D = max(Qa - Q1, 0)` where Q1 is Q
   after the event. Front share `phi`: `V = V - phi * D`, then `V = min(max(V, 0), Q1)`.
4. Increases do not move V (new orders join behind).

Initial `V = Q` at placement. Estimators (`phi` with front F = V, back B = max(Qa - V, 0)):

| id | name | phi |
|---|---|---|
| 0 | `risk_averse` (hftbacktest RiskAdverseQueueModel) | 0 |
| 1 | `prob_power1` (linear; uniform cancel location by volume) | F / (F + B) |
| 2 | `prob_power2` | F^2 / (F^2 + B^2) |
| 3 | `prob_power3` | F^3 / (F^3 + B^3) |
| 4 | `prob_log` | log(1+F) / (log(1+F) + log(1+B)) |
| 5 | `front_optimistic` (all cancels ahead) | 1 |

If F + B = 0, phi = 0. Estimator fill times are the `ts_event` of the event's last record.
With `fill_lot_rounding` (v0.6) the estimated fill is rounded to whole lots as in hftbacktest: a shadow counts as
filled when its cumulative estimated fill reaches size - 0.5 lots (it is then set to the full size); otherwise
(default) when it reaches the full size.
V is updated every event for the whole life even after the estimator has filled (fill accounting
stops, tracking does not). Queue-position error is sampled after every event that touches level
(s, p) while the truth shadow is unfilled: sum and absolute sum of `V_est - V_true`, and count.

## 6. Resting-order labels (real orders; competing risks)

For every non-snapshot `A`, every `M` that moves an order to the back of a level (price change or
requeue, section 2; this starts a new life), and every `M` for an unknown id, whose
price is within `label_ticks` of the same-side best price **before** the add (for an `M`, the
book after X has left its old position and before it is appended: an order alone at the best that
moves to a worse price becomes the new best and has distance 0; clarification 2026-10-12, the
engines always worked this way): record id, side,
price, size, distance in ticks (0 = at the best or better), orders and size ahead at entry
(= the level's count and size before the append). Outcome is the first of:
`filled` (size reaches 0 through C/M records that follow fills in the same event),
`cancelled` (size reaches 0 otherwise), `moved_price`, `requeued` (a requeue, section 2),
`cleared` (R), or `censored` (end of stream). Filled quantity is the sum of `F` sizes on that id
during the life. Distance uses the instrument tick (`tick` parameter, in price units). A native
iceberg's displayed tranche is one life: the refill ends it as `requeued` (its filled quantity
includes the fills that emptied it) and starts a new life at the back of the level.

## 7. Book samples

At each requested time t (sorted), the state at the last event boundary with `ts_event <= t`:
best N levels per side (price, size, order count). Used to reconcile against MBP-10 and to build
mid-price paths for markouts.

## 8. Queue-position snapshots

The full FIFO queue of the best levels at requested times: which real resting orders were where
in line. Used for markouts of resting orders by queue position at a fixed time (e.g. release - 1 s).

**Inputs.** `queue_times`: int64 ns, sorted non-decreasing (duplicates allowed; each entry is its
own snapshot), empty by default. `queue_levels`: int >= 0, default 5. Unsorted `queue_times` or a
negative `queue_levels` is an error (`ValueError`), not silently repaired.

**Timing (identical to section 7).** Snapshot q (the q-th entry, 0-based, of `queue_times`, time
t) reads the book at the last event boundary with `ts_event <= t`. Operationally, in both engines:
before record i is processed, every pending queue time t with `t < ts_event[i]` is emitted, in
order, provided the engine is at an event boundary (record i-1 had `F_LAST`, or i = 0); a pending
time is otherwise held until the next boundary. Times still pending after the last record are
emitted from the final state. Hence:
* a record with `ts_event == t` is included, a record with `ts_event > t` is not, and records after
  t cannot change the snapshot (no look-ahead);
* t before the first record gives the empty book (no rows for that q);
* an intermediate (mid-event) state is never read. If one event's records straddle t (a record
  with `ts_event <= t` followed in the same event by one with `ts_event > t`), the snapshot is
  taken after that whole event, exactly like a section 7 sample. This requires `ts_event` to differ
  inside one event; the synthetic exchange never does that.

**Rows.** For each snapshot q, then side (bid first, then ask, as in section 7), then level index
l = 0 .. min(`queue_levels`, number of non-empty levels on that side) - 1 (l = 0 is the best price;
bids by descending price, asks by ascending), then each order resting at that level in FIFO
priority order (front first), one row. Priority is the section 2 queue: an order joins the back on
`A`, on an `M` that changes price, on a requeue (an `M` at the same price whose new size exceeds
its size minus `fill_since_update`, which includes an iceberg refill), and on an `M` for an unknown
id; an in-place `M` and a partial `C` keep its place; `T`/`F` records do not move it. Missing levels
produce no rows (no padding); an empty side produces no rows; a snapshot with an empty book
produces no rows, so a value of q can be absent.

**Output.** `results["queue"]`: a 2-D int64 array of shape (rows, 10), (0, 10) when `queue_times` is
empty or every snapshot is empty. Column order (`mboq.reference.QUEUE_COLUMNS`):

| col | name | meaning |
|---|---|---|
| 0 | `queue_time_index` | q, the index into `queue_times` |
| 1 | `order_id` | the order's id; uint64 bits stored as int64 (ids >= 2^63 read negative; `.view(np.uint64)` recovers them) |
| 2 | `side` | +1 bid, -1 ask (the shadows/labels convention) |
| 3 | `price` | int64, 1e-9 units |
| 4 | `level_index` | l, 0 = best price on that side |
| 5 | `rank` | number of orders ahead of it at the level (0 = front) |
| 6 | `size` | its current displayed size |
| 7 | `size_ahead` | sum of `size` of the orders ahead of it at the level |
| 8 | `level_order_count` | number of orders at the level |
| 9 | `level_total_size` | total displayed size at the level |

**Invariants (tested).** Within each (q, side, l): `rank` runs 0 .. count - 1, `size_ahead` is the
exclusive running sum of `size`, `size_ahead + size` of the last row equals `level_total_size`, and
`level_total_size` / `level_order_count` equal the section 7 size / count at the same time and
level (with `sample_levels >= queue_levels`). Snapshots read the book only; no other output changes.

Rows describe the displayed outright book as rebuilt here (implied liquidity and hidden iceberg
quantity are not in MBO; see section 2 for how refreshes requeue). Turning `rank` or `size_ahead`
into queue-position quintiles, and the markouts themselves, belong to the analysis protocol,
not to this engine.

## 9. Order-anchored estimator tracks

Section 4 asks the estimators about a hypothetical order. A track asks them about a real one: a
resting order X that joined at the touch, followed for its whole life, with the section 5
estimators run as a backtester would run them for its own order, and scored against X's own
fills. Off by default; `run(..., order_tracks=True)` turns it on. With it off nothing is computed
and `results` has no `"tracks"` key; with it on, no other output changes.

**Start.** Every section 6 label with `dist == 0` starts one track, at the label's record (the `A`,
the price-change or requeue `M`, or the `M` for an unknown id). `dist == 0` means X joins at or
better than the same-side best price before the record, or that side was empty (section 6 clamps
`dist` at 0). Tracks are therefore a subset of the labels (none if `label_ticks < 0`), each row
carries its `label_index`, and a requeue (e.g. an iceberg refill) that ends one life starts a new
track only if the new life is itself a `dist == 0` label. At the start: side s, price p, size
`S0` = the label's size, `size_ahead0` = the label's `size_ahead_at_add` (the true size ahead, V_true
at entry).

**Q_ex.** The estimators see the level aggregate without X: `Q_ex = Q(s, p) - size of X`, where
"size of X" is the current size of the order with X's id resting at (s, p) in this life (0 once
the life has ended). X's owner knows its own partial fills, size decreases and refills, so they
are not taken for queue changes. This is the only change to the section 5 rules.

**Clips.** The estimators are updated once per *clip*: the records of one event that fall inside
X's life. A clip is the whole event, except that the first clip starts after X's start record and
the last one ends with (and includes) the record that ends X's life. Per clip, with `Tr` and the
trade-through test computed from the clip's `T` prints only (prints earlier in the start event, or
later in the end event, are not seen):
* `Q0` = Q_ex at the start of the clip: `size_ahead0` for the first clip (Q_ex just after X's
  record), otherwise Q_ex at the previous event boundary (a level only changes through records that
  touch it, so this is the section 5 Q before the event).
* `Q1` = Q_ex at the end of the clip: at the event boundary, or, for the last clip, just after the
  record that ends the life (X is no longer at the level, so Q1 = Q(s, p) at that moment; for a
  requeue this excludes X's new tranche too).
* Then exactly section 5, for each estimator k with initial `V = size_ahead0`: rule 1 (`excess = Tr
  - V`, `V = max(V - Tr, 0)`, `excess > 0` credits X's estimated filled quantity by `excess`), rule 2
  (an opposite-aggressor print at a price worse than p fills X completely), rule 3 (`Qa = max(Q0 -
  Tr, 0)`, `D = max(Qa - Q1, 0)`, `V = V - phi_k * D` with the section 5 phi_k, then `V = min(max(V,
  0), Q1)`), rule 4. The estimated filled quantity is capped at `S0`; fill accounting stops at `S0`,
  V tracking does not. `Tr` includes the fills on X itself (an MBP backtester cannot split prints).
* A clip that ends at an event boundary is skipped (V unchanged) when level (s, p) was not touched
  in the event (section 5 sense: any `A`/`C`/`M` there, including X's own start record), `Tr = 0`
  and there is no trade-through, exactly as section 5 skips shadows. The last clip is never skipped
  (X's own ending record touches the level).
* Estimated first-fill time `est_first_fill_t[k]` = `ts_event` of the record at which the update is
  applied (the event's last record; the ending record for the last clip) the first time the
  estimated filled quantity becomes > 0 (with `fill_lot_rounding`, v0.6: reaches 0.5 lot).

**Truth.** X's own `F` records during its life (the same records as the label's `filled_qty`):
`true_first_fill_t` = `ts_event` of the first one (-1 if none), `true_filled_qty` = the sum of
their sizes. `v_est_at_true_first_fill[k]` = estimator k's V after the clip that contains X's first
`F` (a skipped clip leaves V as it was): the queue the estimator still believed ahead of X when X
was in fact filled (0 = no lag). NaN if X has no `F`, or if that clip is never closed (a clear, or
an event still open at the end of the stream).

**End.** The track's truth ends with X's label life, with the same outcome and `t_end` (section 6:
`filled`, `cancelled`, `moved_price`, `requeued`, `cleared`, `censored`). With `track_horizon_ns = 0`
(the default) the estimators run until then and not after. An `R` (or implicit clear) ends the track
as `cleared` with no final clip; cleared tracks are excluded from metrics (the outcome code is the
flag). At the end of the stream a live track is `censored` (`t_end = -1`); an event left open there
(last record without `F_LAST`) is not applied, as for shadows.

**Horizon (v0.5).** `run(..., track_horizon_ns=H)`, an int >= 0 (negative: `ValueError`), used only
with `order_tracks`. H = 0 is the rule above exactly (no new keys). With H > 0 the estimators stop:
* when X's life ends by its owner's action (`cancelled`, `moved_price`, `requeued`), at the ending
  record, after the final clip, as with H = 0 (the strategy would have stopped too);
* when X's life ends `filled`: not then. After the final clip they keep being updated, as a
  backtester that does not know the order was filled would keep updating them: the next clip is
  the rest of the end event (its records after the ending record; prints before it are not seen
  again), then one clip per event, exactly the clip rules above with `Q_ex = Q(s, p)` (X is gone,
  even if its id later rests at (s, p) again). They stop at the first event boundary with
  `ts_event >= t_start + H`, after that event's update (as shadows retire after the horizon event);
  if X's fill is itself at or past `t_start + H`, that is the end event's boundary. Estimated fills
  and V keep accruing during this run-on (`est_first_fill_t`, `est_filled_qty` capped at S0,
  `v_est_at_end` are read at the stop); `v_est_at_true_first_fill` and every truth column are as
  with H = 0;
* at a clear (`R` or implicit), also during the run-on of a filled X (its outcome stays `filled`);
* at the end of the stream (censored X, or a run-on not yet at its horizon).
While X still rests, the estimators run whatever its age (only the run-on after a fill is bounded by
H), so for every track whose outcome is not `filled` all H = 0 columns are unchanged, and for a filled
one only `est_first_fill_t` (where H = 0 had -1), `est_filled_qty` and `v_est_at_end` can change.

**E0 (v0.5).** Per track and estimator k, in ns, with `tau_true` = `true_first_fill_t - t_start` if X
has a true fill with `true_first_fill_t - t_start <= H`, else H; `tau_est_k` = `est_first_fill_t[k] -
t_start` if estimator k filled at or before `min(est_stop_t, t_start + H)` (an `est_stop_t` of -1 is
read as no limit), else H: **`e0[k] = |tau_est_k - tau_true|`**, in [0, H]. Early, late, false and
missed fills all score: an early fill `tau_true - tau_est`, a late one `tau_est - tau_true`, a false
fill (X never filled within H) `H - tau_est`, a missed one `H - tau_true`. `e0 = -1` (every k) marks a
track that is **not scored** because what happened within H is not fully observed: its estimators
were stopped by a clear (every `cleared` track, and a filled X whose run-on a clear cut), or by the
end of the stream while the last applied event boundary was before `t_start + H` (a censored X, or a
run-on, near the end of the file). A censored X whose last applied boundary is at or past `t_start + H`
is scored (an order resting through the horizon without a fill has `tau_true = H`).

**Output.** `results["tracks"]`, a dict of arrays in track start order (= label order), keys
`mboq.reference.TRACK_COLUMNS`:

| key | dtype, shape | meaning |
|---|---|---|
| `label_index` | int64 (n,) | row of X's life in `results["labels"]` |
| `order_id` | int64 (n,) | X's id (uint64 bits as int64, as in section 8) |
| `side` | int64 (n,) | +1 bid, -1 ask |
| `price` | int64 (n,) | p, 1e-9 units |
| `t_start` | int64 (n,) | `ts_event` of the start record (= label `t_add`) |
| `size0` | int64 (n,) | S0 |
| `size_ahead0` | int64 (n,) | true size ahead at entry (= label `size_ahead_at_add`) |
| `outcome` | int64 (n,) | index into `OUTCOMES` (5 = cleared: exclude) |
| `t_end` | int64 (n,) | `ts_event` of the ending record, -1 if censored |
| `true_first_fill_t` | int64 (n,) | -1 if none |
| `true_filled_qty` | int64 (n,) | sum of X's `F` sizes in the life |
| `est_first_fill_t` | int64 (n, K) | -1 if none; columns in `ESTIMATORS` order |
| `est_filled_qty` | float64 (n, K) | at track end, <= S0 |
| `v_est_at_true_first_fill` | float64 (n, K) | NaN if X never filled (see Truth) |
| `v_est_at_end` | float64 (n, K) | V after the last applied clip (v0.5: including the run-on) |
| `est_stop_t` | int64 (n,) | only with H > 0 (`TRACK_HORIZON_COLUMNS`): `ts_event` of the record (owner action, `R`) or event boundary (horizon) at which the estimators stopped; -1 if they ran to the end of the stream |
| `e0` | int64 (n, K) | only with H > 0: E0 in ns, -1 = not scored |

An estimated fill of a track that has no true fill (`est_filled_qty > 0`, `true_filled_qty = 0`)
is a *false fill*; a true fill with `est_first_fill_t = -1` is a *missed fill*. Rates, P&L and
adverse-selection splits built on these columns belong to the analysis protocol, not to the engine.

**Known answers (tested).** On the synthetic exchange (no modifies, icebergs or shuffled fills,
where section 5's known answers hold): with tail cancels `risk_averse`, and with front cancels
`front_optimistic`, has V equal to X's true queue ahead after every applied clip while X rests,
`est_first_fill_t == true_first_fill_t` on every non-cleared track, and
`v_est_at_true_first_fill == 0` on every filled one. With modifies (generator default), front cancels leave no
estimator exact: a size increase moves an order ahead of X to the back, shrinking X's true queue ahead
while the level size grows, which MBP data cannot show (front_optimistic: 82.5% time match, mean |V|
11.5 lots at the true fill; measured 2026-10-12, 4 seeds x 30,000 events). Tail cancels keep
`risk_averse` exact with modifies. Note: `v_est_at_true_first_fill` cannot reveal early fills (an
estimator that has already filled sits at V = 0); early and false fills must be read from
`est_first_fill_t` (with tail cancels the other five estimators fill early on 4-5% of tracks).
v0.5: on the same exchange (no icebergs or shuffled fills), `e0 = 0` on every scored track for
`risk_averse` with tail cancels (with and without modifies) and for `front_optimistic` with front
cancels and no modifies (the exact model's first fill is the true one, and the run-on cannot change a
first fill already made); the other model has `e0 > 0` on many tracks, and with front cancels
`risk_averse` fills after X's true fill on many tracks, which H = 0 records as no fill. Tested with
H = 200 ms on `generate(n_events=20000, seed=21, p_clear=0.0005)`.

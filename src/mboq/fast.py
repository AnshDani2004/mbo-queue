"""Thin wrapper: run the C++ core (`mboq._core`) with the same interface as `Reference.run`."""
from __future__ import annotations

import numpy as np

from . import _core
from .reference import (ESTIMATORS, OUTCOMES, QUEUE_COLUMNS, TRACK_COLUMNS, TRACK_HORIZON_COLUMNS,  # noqa: F401
                        check_engine_args, check_queue_args, check_records, check_sorted_times,
                        check_track_horizon, code_array)  # (re-exported names)


def run(rec: dict, tick: int, place_times=(), sample_times=(), label_ticks: int = 5, shadow_size: int = 1,
        horizon_ns: int = 60_000_000_000, sample_levels: int = 10, queue_times=(), queue_levels: int = 5,
        order_tracks: bool = False, track_horizon_ns: int = 0, fill_lot_rounding: bool = False,
        truth_record_price: bool = False) -> dict:
    """results["tracks"] (SPEC 9) only with order_tracks=True; with it off every output is unchanged.
    track_horizon_ns > 0 (SPEC 9, v0.5) adds the track keys est_stop_t and e0; 0 is v0.4 exactly.
    fill_lot_rounding=True (SPEC v0.6): estimated fills rounded to whole lots as in hftbacktest (>= 0.5 lot);
    False is v0.5 exactly. truth_record_price=True (SPEC v0.7, sensitivity only): shadow truth fills read the
    F record's own side and price instead of the filled order's resting position; False is v0.6 exactly."""
    # 0.7.1: the same validation, in the same order, as Reference.__init__ and Reference.run
    check_engine_args(tick, sample_levels)
    check_records(rec)
    track_horizon_ns = check_track_horizon(track_horizon_ns)
    place_times = check_sorted_times(place_times, "place_times")
    sample_times = check_sorted_times(sample_times, "sample_times")
    queue_times = check_queue_args(queue_times, queue_levels)
    return _core.run(np.ascontiguousarray(rec["ts_event"], dtype=np.uint64),
                     np.ascontiguousarray(rec["order_id"], dtype=np.uint64),
                     np.ascontiguousarray(rec["price"], dtype=np.int64),
                     np.ascontiguousarray(rec["size"], dtype=np.uint64),
                     np.ascontiguousarray(rec["flags"], dtype=np.uint8),
                     code_array(rec["action"], "action"), code_array(rec["side"], "side"),
                     np.ascontiguousarray(place_times, dtype=np.int64),
                     np.ascontiguousarray(sample_times, dtype=np.int64),
                     int(tick), int(label_ticks), int(shadow_size), int(horizon_ns), int(sample_levels),
                     np.ascontiguousarray(queue_times, dtype=np.int64), int(queue_levels), bool(order_tracks),
                     track_horizon_ns, bool(fill_lot_rounding), bool(truth_record_price))

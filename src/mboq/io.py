"""Load Databento DBN files (MBO, MBP-10) into the column dicts the engines expect.

Reads in chunks so a full CME session (tens of millions of MBO records) fits in memory as
compact columns. Requires the `databento` package.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MBO_COLS = ("ts_event", "order_id", "price", "size", "flags", "action", "side")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunks(store, count=2_000_000):
    try:
        yield from store.to_ndarray(count=count)
    except TypeError:                        # older databento: no chunking
        yield store.to_ndarray()


def load_mbo(path, instrument_id: int | None = None):
    """Return (columns, meta). If the file holds several instruments, pass instrument_id."""
    import databento as db
    store = db.DBNStore.from_file(str(path))
    parts = {c: [] for c in MBO_COLS}
    ids = {}
    for arr in _chunks(store):
        if len(arr) == 0:
            continue
        iid = arr["instrument_id"]
        u, c = np.unique(iid, return_counts=True)
        for a, b in zip(u, c):
            ids[int(a)] = ids.get(int(a), 0) + int(b)
        if instrument_id is not None:
            arr = arr[iid == instrument_id]
        for c in MBO_COLS:
            parts[c].append(np.asarray(arr[c]))
    rec = {c: (np.concatenate(v) if v else np.array([])) for c, v in parts.items()}
    if instrument_id is None and len(ids) > 1:
        raise ValueError(f"{path}: several instruments {sorted(ids)}; pass instrument_id")
    meta = {"path": str(path), "sha256": sha256(path), "schema": str(store.schema), "dataset": store.dataset,
            "records": int(len(rec["ts_event"])), "instrument_counts": ids,
            "databento_version": getattr(db, "__version__", "?")}
    return rec, meta


def start_at_snapshot(rec: dict):
    """Drop every record before Databento's synthetic snapshot (an R, then F_SNAPSHOT records).

    CME sends no snapshot at the session reopen: a stream that starts
    at 17:00 ET is missing every order that rested through the halt, so the book is complete only
    from the 00:00 UTC snapshot on. Snapshot records keep each order's original ts_event, so
    ts_event is not monotone inside the snapshot: always read in file order.

    Returns (columns from the R on, info). info["snapshot_found"] is False if there is no snapshot.
    Also reports a check of SPEC assumption A1: within each (side, price) level, the snapshot's A
    records should be in priority order, i.e. ts_event non-decreasing in file order.
    """
    act = np.asarray(rec["action"])
    r_code = b"R" if act.dtype.kind == "S" else (ord("R") if act.dtype.kind in "iu" else "R")
    snap = (np.asarray(rec["flags"]).astype(np.int64) & 32) > 0
    is_r = act == r_code
    cand = np.flatnonzero(is_r[:-1] & snap[1:]) if len(act) > 1 else np.array([], dtype=np.int64)
    if not len(cand):
        return rec, {"snapshot_found": False, "dropped_before_snapshot": 0}
    i0 = int(cand[0])
    out = {k: np.asarray(v)[i0:] for k, v in rec.items()}
    s = snap[i0:]
    after = np.flatnonzero(~s[1:])
    j = int(after[0]) + 1 if len(after) else len(s)          # first live record after the snapshot
    blk = slice(1, j)
    a_code = b"A" if act.dtype.kind == "S" else (ord("A") if act.dtype.kind in "iu" else "A")
    is_add = np.asarray(out["action"])[blk] == a_code
    side = np.asarray(out["side"])[blk][is_add]
    price = np.asarray(out["price"])[blk][is_add].astype(np.int64)
    ts = np.asarray(out["ts_event"])[blk][is_add].astype(np.int64)
    order = np.lexsort((np.arange(len(ts)), price, side))     # group by level, keep file order
    same = (side[order][1:] == side[order][:-1]) & (price[order][1:] == price[order][:-1])
    in_order = (ts[order][1:] >= ts[order][:-1]) & same
    info = {"snapshot_found": True, "dropped_before_snapshot": i0, "snapshot_records": j - 1,
            "first_live_index": j, "first_live_ts": int(out["ts_event"][j]) if j < len(s) else None,
            "a1_pairs_same_level": int(same.sum()), "a1_pairs_ts_nondecreasing": int(in_order.sum())}
    return out, info


def load_mbp10(path, instrument_id: int | None = None):
    """Return (dict with ts_event and (n,10) arrays bid_px, bid_sz, ask_px, ask_sz), meta)."""
    import databento as db
    store = db.DBNStore.from_file(str(path))
    ts, bp, bs, ap, az = [], [], [], [], []
    for arr in _chunks(store):
        if instrument_id is not None:
            arr = arr[arr["instrument_id"] == instrument_id]
        if len(arr) == 0:
            continue
        ts.append(np.asarray(arr["ts_event"]))
        bp.append(np.stack([arr[f"bid_px_{i:02d}"] for i in range(10)], 1))
        bs.append(np.stack([arr[f"bid_sz_{i:02d}"] for i in range(10)], 1))
        ap.append(np.stack([arr[f"ask_px_{i:02d}"] for i in range(10)], 1))
        az.append(np.stack([arr[f"ask_sz_{i:02d}"] for i in range(10)], 1))
    cat = lambda v: np.concatenate(v) if v else np.zeros((0, 10), dtype=np.int64)  # noqa: E731
    out = {"ts_event": np.concatenate(ts) if ts else np.array([], dtype=np.uint64),
           "bid_px": cat(bp), "bid_sz": cat(bs), "ask_px": cat(ap), "ask_sz": cat(az)}
    meta = {"path": str(path), "sha256": sha256(path), "schema": str(store.schema), "records": int(len(out["ts_event"]))}
    return out, meta

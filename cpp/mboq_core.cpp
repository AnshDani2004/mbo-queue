// mbo-queue C++20 core. Implements SPEC.md; see src/mboq/reference.py for the readable version.
//
// Data structures differ from the reference on purpose (independent check of the spec):
//  * orders live in a pooled array with an intrusive doubly linked FIFO list per price level;
//  * every (re)queue gets a monotonically increasing priority stamp, so "order X is ahead of
//    shadow S" is simply `same level && X.stamp < S.stamp` (no explicit ahead sets);
//  * each shadow keeps V_true incrementally instead of re-summing its ahead set;
//  * a shadow index: live shadows are grouped by the (side, price) they rest at, so a record or an
//    event visits only the shadows it can affect (same level, or a trade-through from a worse
//    price), and finished shadows are retired from a horizon heap plus a list of shadows that just
//    filled, instead of re-scanning every live shadow at every event boundary;
//  * queue snapshots (SPEC 8) walk the intrusive lists and report the levels' running totals;
//  * FIFO ranks (SPEC 3, final at the event boundary) keep only the ahead orders not yet consumed
//    when the F is processed (the reference keeps every ahead order and filters at the boundary);
//  * order-anchored tracks (SPEC 9, only with order_tracks) have their own (side, price) index, so an
//    event boundary visits only the tracks at levels it touched or traded at / through (the
//    reference visits every live track), and X's size is read through its pool slot; with a track
//    horizon (v0.5) the tracks running past a fill are retired from a min-heap on t_start + H (the
//    reference scans its list of them at every boundary).
// None of this changes the arithmetic: each shadow's update is the same sequence of operations as
// in the reference, and shadows never interact, so the order in which they are visited is free.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace mboq {

constexpr uint8_t F_LAST = 128, F_SNAPSHOT = 32, F_MAYBE_BAD_BOOK = 4;
constexpr int K = 6;  // estimators, same order as reference.ESTIMATORS
constexpr int QCOLS = 10;  // columns of results["queue"], same order as reference.QUEUE_COLUMNS
constexpr uint32_t NIL = 0xFFFFFFFFu;

enum Outcome : int64_t { CENSORED = 0, FILLED = 1, CANCELLED = 2, MOVED_PRICE = 3, REQUEUED = 4, CLEARED = 5 };
// why a track's estimators stopped (SPEC 9, v0.5; reference _Track.stop "", owner, horizon, clear, end)
enum TrackStop : int { RUNNING = 0, STOP_OWNER = 1, STOP_HORIZON = 2, STOP_CLEAR = 3, STOP_END = 4 };

// results["stats"] counters. Counted in a plain array (a std::map<std::string> lookup per record
// was ~5% of the run) and copied into the `stats` map, under these names, when the run ends.
enum Stat {
    ST_RECORDS, ST_EVENTS, ST_ADDS, ST_CANCELS, ST_MODIFIES, ST_CLEARS, ST_TRADES, ST_FILLS, ST_DUP_ADD,
    ST_UNKNOWN_CANCEL, ST_MODIFY_UNKNOWN, ST_IMPLICIT_CLEAR, ST_FILL_UNKNOWN, ST_MAYBE_BAD_BOOK,
    ST_SHADOW_FIFO_ANOMALY, NSTAT
};
static const char* const STAT_NAMES[NSTAT] = {
    "records", "events", "adds", "cancels", "modifies", "clears", "trades", "fills", "dup_add",
    "unknown_cancel", "modify_unknown", "implicit_clear", "fill_unknown", "maybe_bad_book", "shadow_fifo_anomaly"};

struct Order {
    uint64_t id = 0;
    char side = 'N';
    int64_t price = 0, size = 0;
    int64_t fsu = 0;  // fill_since_update (SPEC 2): F size on this id since its last book change
    uint64_t stamp = 0;
    int64_t label = -1;
    uint32_t prev = NIL, next = NIL;
};

struct Level {
    uint32_t head = NIL, tail = NIL;
    int64_t total = 0, count = 0;
};

struct Label {
    int64_t order_id, t_add, side, price, size, dist, rank_at_add, size_ahead_at_add;
    int64_t outcome = CENSORED, t_end = -1, filled_qty = 0;
    int64_t track = -1;  // index into Engine::tracks (SPEC 9), -1 if none
};

// SPEC 9: one real resting order X from its entry at (or better than) the best to the end of its life
struct Track {
    int64_t label_index, order_id, side, price, t_start, size0, ahead0;
    int64_t outcome = CENSORED, t_end = -1, tff = -1, tfq = 0;
    int64_t q0ex;          // Q_ex (the level without X) at the start of the current clip
    size_t tr_from;        // index into the event's T prints: earlier ones precede the start
    uint32_t oix;          // X's pool slot (stable during its life)
    uint32_t pos;          // index in its TrackLevel's members
    char cside;
    bool live = true, need_vff = false;
    bool post = false;     // v0.5: X was filled, the estimators run on to t_start + H (X has left the level)
    int stop = RUNNING;    // TrackStop
    int64_t stop_t = -1;   // est_stop_t
    double v[K], ef[K], vff[K];
    int64_t et[K];
};

struct TrackLevel {
    int64_t price;
    std::vector<uint32_t> members;  // live tracks at this (side, price); their order is irrelevant
    uint64_t touched_ev = 0, listed_ev = 0;
};

struct Shadow {
    int64_t sid;
    char side;
    int64_t price, t_place, t_end, size, q0;
    uint64_t stamp;
    int64_t vtrue, filled = 0, t_fill = -1, end = 0;  // end: 0 horizon, 1 filled, 2 cleared
    double est_v[K], est_filled[K] = {0, 0, 0, 0, 0, 0}, err[K] = {0, 0, 0, 0, 0, 0}, abserr[K] = {0, 0, 0, 0, 0, 0};
    int64_t est_t[K] = {-1, -1, -1, -1, -1, -1};
    int64_t nobs = 0;
    bool live = false;    // slot in use (Engine::shadows_ is a pool with stable slot numbers)
    bool queued = false;  // already in Engine::check_ for the next retire pass
    bool worse(int64_t px) const { return side == 'B' ? px < price : px > price; }
    bool all_done() const {  // truth and every estimator filled (SPEC 4 "Life")
        bool d = filled >= size;
        for (int k = 0; k < K && d; ++k) d = est_filled[k] >= (double)size;
        return d;
    }
};

// The shadow index entry: every live shadow resting at one price of one side.
struct ShadowLevel {
    int64_t price;
    std::vector<uint32_t> members;  // slots in Engine::shadows_, in placement order = increasing stamp
    uint64_t touched_ev = 0;        // number of the event in which the level was last touched
    int64_t q_before = 0;           // Q at the level's first touch in that event (SPEC 5 Q0)
    uint64_t listed_ev = 0;         // number of the event in which it was put on Engine::affected_
};

// Not inlined, so std::pow always reaches libm with a run-time exponent, as Python's `front ** k`
// does. Inlined into an unrolled k loop, clang may rewrite pow(x, 2.0) as x * x, which equals
// libm's pow only if pow is correctly rounded (the engines must be bitwise equal).
__attribute__((noinline)) static double phi(int k, double front, double back) {
    if (k == 0) return 0.0;
    if (k == 5) return 1.0;
    double a, b;
    if (k >= 1 && k <= 3) {
        a = std::pow(front, (double)k);
        b = std::pow(back, (double)k);
    } else {
        a = std::log1p(front);
        b = std::log1p(back);
    }
    return a + b > 0 ? a / (a + b) : 0.0;
}

class Engine {
   public:
    Engine(int64_t tick, int64_t label_ticks, int64_t shadow_size, int64_t horizon, int64_t nlev, int64_t qlev,
           bool order_tracks, int64_t track_horizon, bool fill_lot_rounding = false, bool truth_record_price = false)
        : tick_(tick), label_ticks_(label_ticks), S_(shadow_size), H_(horizon), N_(nlev), QL_(qlev),
          TH_(track_horizon), tracks_on_(order_tracks), LR_(fill_lot_rounding), TRP_(truth_record_price) {
        ids_.reserve(1 << 20);
        pool_.reserve(1 << 20);
    }

    // ---------------------------------------------------------------- outputs
    std::map<std::string, int64_t> stats;  // filled from st_ at the end of run()
    std::vector<int64_t> fifo_rank, fifo_ts;
    std::vector<Label> labels;
    std::vector<Shadow> done;
    std::vector<int64_t> samples;
    std::vector<int64_t> queue;  // row-major (rows, QCOLS)
    std::vector<Track> tracks;   // SPEC 9, in start order

    void run(const uint64_t* ts, const uint64_t* oid, const int64_t* px, const uint64_t* sz, const uint8_t* fl,
             const uint8_t* act, const uint8_t* side, size_t n, const int64_t* place, size_t np, const int64_t* samp,
             size_t ns, const int64_t* qt, size_t nq) {
        size_t pi = 0, si = 0, qi = 0;
        for (size_t i = 0; i < n; ++i) {
            ts_ = (int64_t)ts[i];
            while (si < ns && samp[si] < ts_ && at_boundary_) {
                sample();
                ++si;
            }
            while (qi < nq && qt[qi] < ts_ && at_boundary_) {  // same rule as samples (SPEC 8)
                queue_snapshot((int64_t)qi);
                ++qi;
            }
            const char a = (char)act[i], s = (char)side[i];
            const uint64_t id = oid[i];
            const int64_t p = px[i], q = (int64_t)sz[i];
            const uint8_t f = fl[i];
            st_[ST_RECORDS]++;
            if (f & F_MAYBE_BAD_BOOK) st_[ST_MAYBE_BAD_BOOK]++;
            const bool snap = (f & F_SNAPSHOT) != 0;
            if (snap && !prev_snapshot_ && prev_action_ != 'R' && !ids_.empty()) {
                st_[ST_IMPLICIT_CLEAR]++;
                clear_all();
            }
            switch (a) {
                case 'A': on_add(id, s, p, q, snap); break;
                case 'C': on_cancel(id, q); break;
                case 'M': on_modify(id, s, p, q); break;
                case 'R':
                    st_[ST_CLEARS]++;
                    clear_all();
                    break;
                case 'T':
                    st_[ST_TRADES]++;
                    ev_trades_.push_back({p, s, q});
                    break;
                case 'F': on_fill(id, s, p, q); break;
                default: break;
            }
            prev_snapshot_ = snap;
            prev_action_ = a;
            at_boundary_ = (f & F_LAST) != 0;
            if (at_boundary_) {
                end_event();
                pi = retire_and_place(place, np, pi);
            }
        }
        while (si < ns) {
            sample();
            ++si;
        }
        while (qi < nq) {
            queue_snapshot((int64_t)qi);
            ++qi;
        }
        close_ranks();  // an event left open by a last record without F_LAST (SPEC 3)
        for (auto& t : tracks)  // estimators still running when the stream ends (v0.5)
            if (t.stop == RUNNING) { t.stop = STOP_END; t.stop_t = -1; }
        for (auto& sh : shadows_)
            if (sh.live) done.push_back(sh);
        std::sort(done.begin(), done.end(), [](const Shadow& x, const Shadow& y) { return x.sid < y.sid; });
        for (int j = 0; j < NSTAT; ++j) stats[STAT_NAMES[j]] = st_[j];
    }

    // SPEC 9 (v0.5) E0 of track t, estimator k, in ns: |tau_est_k - tau_true|, both censored at H. -1 = not
    // scored (the estimators were stopped by a clear, or by the end of the stream before t_start + H)
    int64_t e0(const Track& t, int k) const {
        const int64_t deadline = t.t_start + TH_;
        if (!(t.stop == STOP_OWNER || t.stop == STOP_HORIZON || (t.stop == STOP_END && last_boundary_ >= deadline)))
            return -1;
        const int64_t tau_true = (t.tff >= 0 && t.tff <= deadline) ? t.tff - t.t_start : TH_;
        const int64_t lim = t.stop_t < 0 ? deadline : std::min(t.stop_t, deadline);
        const int64_t tau_est = (t.et[k] >= 0 && t.et[k] <= lim) ? t.et[k] - t.t_start : TH_;
        const int64_t d = tau_est - tau_true;
        return d < 0 ? -d : d;
    }

   private:
    struct Trade { int64_t px; char ag; int64_t sz; };
    int64_t tick_, label_ticks_, S_, H_, N_, QL_;
    int64_t TH_;                 // SPEC 9 v0.5 track horizon (ns); 0 = v0.4
    int64_t last_boundary_ = -1;  // ts_event of the last event boundary (-1: none yet)
    bool tracks_on_;
    bool LR_ = false;             // SPEC v0.6 fill_lot_rounding (hftbacktest half-lot rule)
    bool TRP_ = false;            // SPEC v0.7 truth_record_price (sensitivity): shadow truth at the F record's price
    int64_t st_[NSTAT] = {};
    int64_t ts_ = 0;
    bool at_boundary_ = true, prev_snapshot_ = false;
    char prev_action_ = 0;
    uint64_t stamp_ = 0;
    int64_t next_sid_ = 0;
    std::vector<Order> pool_;
    std::vector<uint32_t> free_;
    std::unordered_map<uint64_t, uint32_t> ids_;
    std::map<int64_t, Level, std::greater<int64_t>> bids_;
    std::map<int64_t, Level> asks_;
    std::unordered_map<uint64_t, int64_t> ev_fill_;
    std::vector<Trade> ev_trades_;
    // SPEC 3 ranks are final at the event boundary. An F whose ahead orders are all consumed when it
    // is processed has rank 0 already (fills only grow within an event); otherwise the ahead orders
    // not yet consumed, with their sizes at that moment, wait in pend_ahead_ until close_ranks().
    struct Ahead { uint64_t id; int64_t size; };
    struct PendRank { size_t idx, start, n; };  // fifo_rank index; its entries in pend_ahead_
    std::vector<Ahead> pend_ahead_;
    std::vector<PendRank> pend_rank_;
    uint64_t ev_no_ = 1;  // number of the current event (ShadowLevel marks compare against it)

    // ---- live shadows and the shadow index
    struct Expiry {  // horizon heap entry; stale once the slot is retired or reused (sid differs)
        int64_t t_end, sid;
        uint32_t slot;
        bool operator>(const Expiry& o) const { return t_end != o.t_end ? t_end > o.t_end : sid > o.sid; }
    };
    struct Listed { int side; size_t level; };  // a ShadowLevel to update at this event boundary
    std::vector<Shadow> shadows_;               // slot pool; shadows_[slot].live marks a live shadow
    std::vector<uint32_t> free_slots_;
    int64_t nlive_ = 0;
    std::vector<ShadowLevel> sl_[2];            // [0] bid, [1] ask levels holding live shadows, by price
    std::vector<Listed> affected_;              // levels touched or traded at in the current event
    std::vector<uint32_t> check_;               // slots that may be all_done(): just filled or placed
    std::priority_queue<Expiry, std::vector<Expiry>, std::greater<Expiry>> expiry_;  // earliest t_end on top

    // ---- live order-anchored tracks (SPEC 9) and their index
    std::map<int64_t, TrackLevel> tl_[2];        // [0] bid, [1] ask: levels holding live tracks
    std::vector<TrackLevel*> tl_listed_[2];      // levels touched or traded at in the current event
    std::vector<uint32_t> trk_started_;          // tracks started mid-event (tr_from > 0, reset at the boundary)
    std::vector<uint32_t> vff_pending_;          // tracks whose first true fill is in the current event
    int64_t ntrk_live_ = 0;                      // tracks whose estimators run (X live, or v0.5 past a fill)
    struct PostDue {                             // v0.5: a filled X's track, due to stop at t_start + H
        int64_t deadline;
        uint32_t ti;
        bool operator>(const PostDue& o) const { return deadline != o.deadline ? deadline > o.deadline : ti > o.ti; }
    };
    std::priority_queue<PostDue, std::vector<PostDue>, std::greater<PostDue>> post_due_;

    static int side_ix(char side) { return side == 'B' ? 0 : side == 'A' ? 1 : -1; }
    // the index entry for (side, price), or nullptr; the side is matched exactly ('B' or 'A'),
    // like the `sh.side == o.side` comparisons of a scan over all shadows
    ShadowLevel* find_sl(char side, int64_t price, size_t* pos = nullptr) {
        const int s = side_ix(side);
        if (s < 0) return nullptr;
        auto& v = sl_[s];
        auto it = std::lower_bound(v.begin(), v.end(), price, [](const ShadowLevel& L, int64_t p) { return L.price < p; });
        if (it == v.end() || it->price != price) return nullptr;
        if (pos) *pos = (size_t)(it - v.begin());
        return &*it;
    }
    void queue_check(uint32_t slot) {
        Shadow& sh = shadows_[slot];
        if (!sh.queued) { sh.queued = true; check_.push_back(slot); }
    }

    // ---------------------------------------------------------------- book primitives
    Level* level(char side, int64_t price) {
        if (side == 'B') {
            auto it = bids_.find(price);
            return it == bids_.end() ? nullptr : &it->second;
        }
        auto it = asks_.find(price);
        return it == asks_.end() ? nullptr : &it->second;
    }
    Level& level_create(char side, int64_t price) { return side == 'B' ? bids_[price] : asks_[price]; }
    void level_erase(char side, int64_t price) {
        if (side == 'B') bids_.erase(price); else asks_.erase(price);
    }
    int64_t qty(char side, int64_t price) {
        Level* lv = level(side, price);
        return lv ? lv->total : 0;
    }
    bool best(char side, int64_t& out) {
        if (side == 'B') {
            if (bids_.empty()) return false;
            out = bids_.begin()->first;
            return true;
        }
        if (asks_.empty()) return false;
        out = asks_.begin()->first;
        return true;
    }
    // SPEC 5 Q0: Q of a level at its first touch in the current event. Only levels that hold live
    // shadows use it, and that set cannot change inside an event (shadows are placed and retired
    // at event boundaries; a clear empties it), so Q0 is kept on the index entry, which is also
    // listed for the end-of-event estimator update.
    void touch(char side, int64_t price) {
        if (ntrk_live_ > 0) track_touch(side, price);
        if (nlive_ == 0) return;
        size_t pos;
        ShadowLevel* L = find_sl(side, price, &pos);
        if (!L || L->touched_ev == ev_no_) return;
        L->touched_ev = ev_no_;
        L->q_before = qty(side, price);
        list_level(side_ix(side), pos);
    }
    void list_level(int s, size_t pos) {
        ShadowLevel& L = sl_[s][pos];
        if (L.listed_ev != ev_no_) { L.listed_ev = ev_no_; affected_.push_back({s, pos}); }
    }
    int64_t ev_fill(uint64_t id) {
        auto it = ev_fill_.find(id);
        return it == ev_fill_.end() ? 0 : it->second;
    }

    // shadows at the order's level for which the order is ahead (o.stamp < shadow stamp); the
    // level's members are in increasing stamp order, so those shadows are a suffix of the list
    void adjust_vtrue(const Order& o, int64_t delta) {
        if (nlive_ == 0) return;
        ShadowLevel* L = find_sl(o.side, o.price);
        if (!L) return;
        for (auto it = L->members.rbegin(); it != L->members.rend(); ++it) {
            Shadow& sh = shadows_[*it];
            if (o.stamp >= sh.stamp) break;
            sh.vtrue += delta;
        }
    }

    void end_label(Order& o, int64_t reason) {
        if (o.label >= 0) {
            Label& lab = labels[(size_t)o.label];
            if (lab.outcome == CENSORED && lab.t_end == -1) {
                lab.outcome = reason;
                lab.t_end = ts_;
                if (lab.track >= 0) end_track((uint32_t)lab.track, reason);
            }
            o.label = -1;
        }
    }

    // ---------------------------------------------------------------- order-anchored tracks (SPEC 9)
    void track_list(int s, TrackLevel& L) {
        if (L.listed_ev != ev_no_) { L.listed_ev = ev_no_; tl_listed_[s].push_back(&L); }
    }
    void track_touch(char side, int64_t price) {
        const int s = side_ix(side);
        if (s < 0) return;
        auto it = tl_[s].find(price);
        if (it == tl_[s].end() || it->second.touched_ev == ev_no_) return;
        it->second.touched_ev = ev_no_;
        track_list(s, it->second);
    }
    void start_track(int64_t li, uint32_t oix, uint64_t id, char side, int64_t price, int64_t size, int64_t ahead_q) {
        const int s = side_ix(side);
        if (s < 0) return;
        const uint32_t ti = (uint32_t)tracks.size();
        tracks.emplace_back();
        Track& t = tracks.back();
        t.label_index = li; t.order_id = (int64_t)id; t.side = s == 0 ? 1 : -1; t.price = price; t.t_start = ts_;
        t.size0 = size; t.ahead0 = ahead_q; t.q0ex = ahead_q; t.tr_from = ev_trades_.size(); t.oix = oix; t.cside = side;
        for (int k = 0; k < K; ++k) {
            t.v[k] = (double)ahead_q; t.ef[k] = 0.0; t.vff[k] = std::numeric_limits<double>::quiet_NaN(); t.et[k] = -1;
        }
        labels[(size_t)li].track = ti;
        auto it = tl_[s].find(price);
        if (it == tl_[s].end()) it = tl_[s].emplace(price, TrackLevel{price, {}, 0, 0}).first;
        TrackLevel& L = it->second;
        t.pos = (uint32_t)L.members.size();
        L.members.push_back(ti);
        L.touched_ev = ev_no_;  // X's own record touched the level
        track_list(s, L);
        if (t.tr_from > 0) trk_started_.push_back(ti);
        ++ntrk_live_;
    }
    // Tr and trade-through over the current event's T prints from index `from` (SPEC 5 rules 1, 2)
    void track_trades(char side, int64_t price, size_t from, int64_t& tr, bool& through) const {
        const char opp = side == 'B' ? 'A' : 'B';
        tr = 0;
        through = false;
        for (size_t j = from; j < ev_trades_.size(); ++j) {
            const Trade& x = ev_trades_[j];
            if (x.ag != opp) continue;
            if (x.px == price) tr += x.sz;
            if (side == 'B' ? x.px < price : x.px > price) through = true;
        }
    }
    // SPEC 5 update with Q replaced by Q_ex (the level without X); Q0_ex is t.q0ex
    void update_track(Track& t, int64_t tr, bool through, int64_t q1ex) {
        const int64_t qa = std::max<int64_t>(t.q0ex - tr, 0);
        const int64_t dec = std::max<int64_t>(qa - q1ex, 0);
        const double S = (double)t.size0;
        for (int k = 0; k < K; ++k) {
            double v = t.v[k];
            const bool unfilled = t.ef[k] < S;
            if (tr > 0) {
                const double excess = (double)tr - v;
                v = std::max(v - (double)tr, 0.0);
                if (excess > 0 && unfilled) t.ef[k] = std::min(S, t.ef[k] + excess);
            }
            if (through && unfilled) t.ef[k] = S;
            if (t.et[k] < 0 && (LR_ ? t.ef[k] >= 0.5 : t.ef[k] > 0)) t.et[k] = ts_;   // v0.6 fill_lot_rounding
            if (dec > 0) v = v - phi(k, v, std::max((double)qa - v, 0.0)) * (double)dec;
            v = std::min(std::max(v, 0.0), (double)q1ex);
            t.v[k] = v;
        }
        t.q0ex = q1ex;
        if (t.need_vff) { std::copy(t.v, t.v + K, t.vff); t.need_vff = false; }
    }
    // X's life ended at this record (called after X left the level): the final clip, then retire, or, for a
    // filled X with a track horizon (v0.5), keep the estimators running from the next record on
    void end_track(uint32_t ti, int64_t reason) {
        Track& t = tracks[ti];
        t.outcome = reason;
        t.t_end = ts_;
        t.live = false;
        if (reason == CLEARED) {  // clear_all() resets the whole index
            t.stop = STOP_CLEAR;
            t.stop_t = ts_;
            return;
        }
        int64_t tr;
        bool through;
        track_trades(t.cside, t.price, t.tr_from, tr, through);
        update_track(t, tr, through, qty(t.cside, t.price));
        if (reason == FILLED && TH_ > 0) {
            t.post = true;
            t.tr_from = ev_trades_.size();  // the rest of this event's prints are seen, earlier ones not
            trk_started_.push_back(ti);
            post_due_.push({t.t_start + TH_, ti});
            return;
        }
        stop_track(ti, STOP_OWNER);
    }
    // the estimators stop at this record / boundary: retire the track from the index
    void stop_track(uint32_t ti, int why) {
        Track& t = tracks[ti];
        t.stop = why;
        t.stop_t = ts_;
        const int s = side_ix(t.cside);
        TrackLevel& L = tl_[s].find(t.price)->second;
        const uint32_t last = L.members.back();
        L.members[t.pos] = last;
        tracks[last].pos = t.pos;
        L.members.pop_back();
        track_list(s, L);  // an emptied level is erased at the boundary
        --ntrk_live_;
    }
    void end_event_tracks() {
        if (!ev_trades_.empty()) {  // add the levels the trades reach (touched ones are listed already)
            for (int s = 0; s < 2; ++s) {
                if (tl_[s].empty()) continue;
                const char opp = s == 0 ? 'A' : 'B';
                bool any = false;
                int64_t lim = 0;  // bids: the lowest print (levels above it are traded through); asks: the highest
                for (const Trade& x : ev_trades_) {
                    if (x.ag != opp) continue;
                    auto it = tl_[s].find(x.px);
                    if (it != tl_[s].end()) track_list(s, it->second);
                    if (!any || (s == 0 ? x.px < lim : x.px > lim)) lim = x.px;
                    any = true;
                }
                if (!any) continue;
                if (s == 0) { for (auto it = tl_[0].upper_bound(lim); it != tl_[0].end(); ++it) track_list(0, it->second); }
                else { for (auto it = tl_[1].begin(); it != tl_[1].end() && it->first < lim; ++it) track_list(1, it->second); }
            }
        }
        for (int s = 0; s < 2; ++s) {
            const char side = s == 0 ? 'B' : 'A';
            for (TrackLevel* L : tl_listed_[s]) {
                if (L->members.empty()) continue;
                const bool touched = L->touched_ev == ev_no_;
                const int64_t q = qty(side, L->price);
                int64_t tr0;
                bool through0;
                track_trades(side, L->price, 0, tr0, through0);
                for (uint32_t ti : L->members) {
                    Track& t = tracks[ti];
                    int64_t tr = tr0;
                    bool through = through0;
                    if (t.tr_from > 0) track_trades(side, L->price, t.tr_from, tr, through);
                    // after a fill (v0.5) X is gone and its pool slot may be reused: Q_ex = Q
                    if (touched || tr > 0 || through) update_track(t, tr, through, q - (t.post ? 0 : pool_[t.oix].size));
                }
            }
        }
        while (!post_due_.empty() && post_due_.top().deadline <= ts_) {  // v0.5: first boundary at/past t_start + H
            const uint32_t ti = post_due_.top().ti;
            post_due_.pop();
            if (tracks[ti].stop == RUNNING) stop_track(ti, STOP_HORIZON);
        }
        for (int s = 0; s < 2; ++s) {
            for (TrackLevel* L : tl_listed_[s])
                if (L->members.empty()) tl_[s].erase(L->price);  // each level is listed at most once
            tl_listed_[s].clear();
        }
        for (uint32_t ti : vff_pending_) {  // a skipped update leaves V unchanged by the event
            Track& t = tracks[ti];
            if (t.live && t.need_vff) { std::copy(t.v, t.v + K, t.vff); t.need_vff = false; }
        }
        vff_pending_.clear();
        for (uint32_t ti : trk_started_) tracks[ti].tr_from = 0;
        trk_started_.clear();
    }

    void remove(uint64_t id, int64_t reason) {
        uint32_t ix = ids_.at(id);
        Order& o = pool_[ix];
        adjust_vtrue(o, -o.size);
        Level* lv = level(o.side, o.price);
        if (o.prev != NIL) pool_[o.prev].next = o.next; else lv->head = o.next;
        if (o.next != NIL) pool_[o.next].prev = o.prev; else lv->tail = o.prev;
        lv->total -= o.size;
        lv->count -= 1;
        if (lv->count == 0) level_erase(o.side, o.price);
        end_label(o, reason);
        ids_.erase(id);
        free_.push_back(ix);
    }

    void append(uint64_t id, char side, int64_t price, int64_t size, bool label_ok) {
        int64_t b = 0;
        const bool has_best = best(side, b);  // best BEFORE the add
        Level& lv = level_create(side, price);
        const int64_t ahead_n = lv.count, ahead_q = lv.total;
        uint32_t ix;
        if (!free_.empty()) { ix = free_.back(); free_.pop_back(); } else { ix = (uint32_t)pool_.size(); pool_.emplace_back(); }
        Order& o = pool_[ix];
        o = Order{};
        o.id = id; o.side = side; o.price = price; o.size = size; o.stamp = ++stamp_;
        o.prev = lv.tail; o.next = NIL;
        if (lv.tail != NIL) pool_[lv.tail].next = ix; else lv.head = ix;
        lv.tail = ix;
        lv.total += size;
        lv.count += 1;
        ids_[id] = ix;
        if (label_ok) {
            int64_t dist = 0;
            if (has_best) {
                dist = side == 'B' ? (b - price) / tick_ : (price - b) / tick_;
                if (dist < 0) dist = 0;
            }
            if (dist <= label_ticks_) {
                labels.push_back(Label{(int64_t)id, ts_, side == 'B' ? 1 : -1, price, size, dist, ahead_n, ahead_q});
                o.label = (int64_t)labels.size() - 1;
                if (tracks_on_ && dist == 0) start_track(o.label, ix, id, side, price, size, ahead_q);  // SPEC 9
            }
        }
    }

    void set_size(Order& o, int64_t nsz) {
        int64_t d = nsz - o.size;
        adjust_vtrue(o, d);
        Level* lv = level(o.side, o.price);
        lv->total += d;
        o.size = nsz;
    }

    void clear_all() {
        for (auto& kv : ids_) end_label(pool_[kv.second], CLEARED);
        ids_.clear();
        pool_.clear();
        free_.clear();
        bids_.clear();
        asks_.clear();
        for (auto& sh : shadows_)
            if (sh.live) { sh.end = 2; done.push_back(sh); }
        shadows_.clear();
        free_slots_.clear();
        nlive_ = 0;
        sl_[0].clear();
        sl_[1].clear();
        affected_.clear();
        check_.clear();
        expiry_ = {};
        for (int s = 0; s < 2; ++s)  // v0.5: estimators running past a fill stop too
            for (auto& kv : tl_[s])
                for (uint32_t ti : kv.second.members)
                    if (tracks[ti].stop == RUNNING) { tracks[ti].stop = STOP_CLEAR; tracks[ti].stop_t = ts_; }
        post_due_ = {};
        for (int s = 0; s < 2; ++s) { tl_[s].clear(); tl_listed_[s].clear(); }  // tracks ended 'cleared' above
        trk_started_.clear();
        vff_pending_.clear();
        ntrk_live_ = 0;
    }

    // ---------------------------------------------------------------- handlers
    void on_add(uint64_t id, char side, int64_t price, int64_t size, bool snap) {
        st_[ST_ADDS]++;
        touch(side, price);
        auto it = ids_.find(id);
        if (it != ids_.end()) {
            if (!snap) st_[ST_DUP_ADD]++;
            const Order& o = pool_[it->second];
            touch(o.side, o.price);
            remove(id, CANCELLED);
        }
        append(id, side, price, size, !snap);
    }

    void on_cancel(uint64_t id, int64_t size) {
        st_[ST_CANCELS]++;
        auto it = ids_.find(id);
        if (it == ids_.end()) { st_[ST_UNKNOWN_CANCEL]++; return; }
        Order& o = pool_[it->second];
        touch(o.side, o.price);
        const int64_t nsz = o.size - size;
        if (size == 0 || nsz <= 0) remove(id, ev_fill(id) > 0 ? FILLED : CANCELLED);
        else { set_size(o, nsz); o.fsu = 0; }
    }

    void on_modify(uint64_t id, char side, int64_t price, int64_t size) {
        st_[ST_MODIFIES]++;
        auto it = ids_.find(id);
        if (it == ids_.end()) {
            st_[ST_MODIFY_UNKNOWN]++;
            touch(side, price);
            append(id, side, price, size, true);
            return;
        }
        Order& o = pool_[it->second];
        touch(o.side, o.price);
        if (price != o.price || side != o.side) {
            touch(side, price);
            remove(id, MOVED_PRICE);
            append(id, side, price, size, true);
        } else if (size <= 0) {
            remove(id, ev_fill(id) > 0 ? FILLED : CANCELLED);
        } else if (size > o.size - o.fsu) {  // SPEC 2 requeue: more than is left after the reported fills
            remove(id, REQUEUED);
            append(id, side, price, size, true);  // a fresh Order: fsu = 0
        } else {
            set_size(o, size);
            o.fsu = 0;
        }
    }

    void on_fill(uint64_t id, char rec_side, int64_t rec_price, int64_t size) {
        st_[ST_FILLS]++;
        auto it = ids_.find(id);
        if (it == ids_.end()) {
            st_[ST_FILL_UNKNOWN]++;
            fifo_rank.push_back(-1);
            fifo_ts.push_back(ts_);
            ev_fill_[id] += size;
            return;
        }
        const uint32_t ix = it->second;
        pool_[ix].fsu += size;
        const Order o = pool_[ix];  // copy: fields read below
        const Level* lv = level(o.side, o.price);
        const size_t start = pend_ahead_.size();
        for (uint32_t j = lv->head; j != NIL && j != ix; j = pool_[j].next)
            if (ev_fill(pool_[j].id) < pool_[j].size) pend_ahead_.push_back({pool_[j].id, pool_[j].size});
        if (pend_ahead_.size() > start) pend_rank_.push_back({fifo_rank.size(), start, pend_ahead_.size() - start});
        fifo_rank.push_back(0);  // final unless pending (close_ranks)
        fifo_ts.push_back(ts_);
        if (o.label >= 0) {
            Label& lab = labels[(size_t)o.label];
            lab.filled_qty += size;
            if (lab.track >= 0) {  // SPEC 9 truth: X's own F records
                Track& t = tracks[(size_t)lab.track];
                t.tfq += size;
                if (t.tff < 0) { t.tff = ts_; t.need_vff = true; vff_pending_.push_back((uint32_t)lab.track); }
            }
        }
        ev_fill_[id] += size;
        // truth shadows (SPEC 4) on the order's side that this fill reaches: (a) at the order's
        // price, placed before the order joined (stamp below the order's), or (b) trade-through,
        // the fill is at a price worse than the shadow's. SPEC v0.7 truth_record_price (sensitivity,
        // default off): read side and price from the F record instead (side N -> the order's side);
        // an order not resting at that level is in no shadow's ahead set there.
        const char fside = (TRP_ && (rec_side == 'B' || rec_side == 'A')) ? rec_side : o.side;
        const int64_t fprice = TRP_ ? rec_price : o.price;
        const bool here = fside == o.side && fprice == o.price;
        const Level* flv = here ? lv : level(fside, fprice);
        const int s = nlive_ > 0 ? side_ix(fside) : -1;
        if (s < 0) return;
        for (ShadowLevel& L : sl_[s]) {
            const bool same = L.price == fprice;
            const bool through = s == 0 ? fprice < L.price : fprice > L.price;  // Shadow::worse
            if (!same && !through) continue;
            for (uint32_t slot : L.members) {
                Shadow& sh = shadows_[slot];
                if (same && here && sh.stamp > o.stamp) break;  // members by stamp: the rest are behind the order
                if (sh.filled >= sh.size) continue;
                if (same) {
                    int64_t left = 0;
                    if (flv != nullptr)
                        for (uint32_t j = flv->head; j != NIL && pool_[j].stamp < sh.stamp; j = pool_[j].next)
                            left += std::max<int64_t>(pool_[j].size - ev_fill(pool_[j].id), 0);
                    if (left > 0) st_[ST_SHADOW_FIFO_ANOMALY]++;
                }
                const int64_t take = std::min(size, sh.size - sh.filled);
                sh.filled += take;
                if (sh.filled >= sh.size && sh.t_fill < 0) sh.t_fill = ts_;
                if (sh.filled >= sh.size) queue_check(slot);
            }
        }
    }

    // ---------------------------------------------------------------- event boundary
    void close_ranks() {
        for (const PendRank& p : pend_rank_) {
            int64_t r = 0;
            for (size_t k = p.start; k < p.start + p.n; ++k)
                if (ev_fill(pend_ahead_[k].id) < pend_ahead_[k].size) r++;
            fifo_rank[p.idx] = r;
        }
        pend_rank_.clear();
        pend_ahead_.clear();
    }

    void end_event() {
        st_[ST_EVENTS]++;
        last_boundary_ = ts_;
        close_ranks();
        if (nlive_ > 0) {
            // A shadow's estimators move only if its level was touched or traded at / through in
            // this event (SPEC 5). Touched levels are already listed; add the ones the trades reach.
            if (!ev_trades_.empty())
                for (int s = 0; s < 2; ++s)
                    for (size_t j = 0; j < sl_[s].size(); ++j) {
                        int64_t tr;
                        bool through;
                        trade_effect(s, sl_[s][j].price, tr, through);
                        if (tr != 0 || through) list_level(s, j);
                    }
            for (const Listed& a : affected_) update_level(a.side, sl_[a.side][a.level]);
        }
        if (ntrk_live_ > 0 || !tl_listed_[0].empty() || !tl_listed_[1].empty()) end_event_tracks();
        affected_.clear();
        ev_fill_.clear();
        ev_trades_.clear();
        ++ev_no_;
    }

    // this event's trade inputs for a shadow level on side s (SPEC 5): Tr, the opposite-aggressor
    // volume at its price, and whether an opposite-aggressor print was at a worse price
    void trade_effect(int s, int64_t price, int64_t& tr, bool& through) const {
        const char opp = s == 0 ? 'A' : 'B';
        tr = 0;
        through = false;
        for (const auto& t : ev_trades_) {
            if (t.ag != opp) continue;
            if (t.px == price) tr += t.sz;
            if (s == 0 ? t.px < price : t.px > price) through = true;  // Shadow::worse
        }
    }

    // SPEC 5 estimator update and the queue-position error sample for every shadow at one level
    void update_level(int s, ShadowLevel& L) {
        const char side = s == 0 ? 'B' : 'A';
        int64_t tr;
        bool through;
        trade_effect(s, L.price, tr, through);
        const bool touched = L.touched_ev == ev_no_;
        const int64_t q1 = qty(side, L.price);
        const int64_t q0 = touched ? L.q_before : q1;
        const int64_t qa = std::max<int64_t>(q0 - tr, 0);
        const int64_t dec = std::max<int64_t>(qa - q1, 0);
        for (uint32_t slot : L.members) {
            Shadow& sh = shadows_[slot];
            bool est_fill = false;
            for (int k = 0; k < K; ++k) {
                double v = sh.est_v[k];
                const bool unfilled = sh.est_filled[k] < (double)sh.size;
                if (tr > 0) {
                    const double excess = (double)tr - v;
                    v = std::max(v - (double)tr, 0.0);
                    if (excess > 0 && unfilled) {
                        sh.est_filled[k] = std::min((double)sh.size, sh.est_filled[k] + excess);
                        if (LR_ && sh.est_filled[k] >= (double)sh.size - 0.5) sh.est_filled[k] = (double)sh.size;  // v0.6
                    }
                }
                if (through && unfilled) sh.est_filled[k] = (double)sh.size;
                if (unfilled && sh.est_filled[k] >= (double)sh.size) { sh.est_t[k] = ts_; est_fill = true; }
                if (dec > 0) v = v - phi(k, v, std::max((double)qa - v, 0.0)) * (double)dec;
                v = std::min(std::max(v, 0.0), (double)q1);
                sh.est_v[k] = v;
            }
            if (touched && sh.filled < sh.size) {
                sh.nobs++;
                for (int k = 0; k < K; ++k) {
                    const double d = sh.est_v[k] - (double)sh.vtrue;
                    sh.err[k] += d;
                    sh.abserr[k] += std::fabs(d);
                }
            }
            if (est_fill) queue_check(slot);
        }
    }

    // SPEC 4 "Life": a shadow ends at the first boundary where it is all_done() or ts >= t_end.
    // all_done() can only become true when the truth or an estimator fills, and those moments
    // queue the shadow in check_. The horizon is a min-heap on t_end, compared with the current
    // boundary's ts (ts_event is not monotone around snapshots). Placements only happen at a
    // new running maximum of ts, so t_end is also in placement order; the heap does not rely on it.
    void retire(uint32_t slot) {
        Shadow& sh = shadows_[slot];
        if (sh.filled >= sh.size) sh.end = 1;
        sh.live = false;
        done.push_back(sh);
        const int s = side_ix(sh.side);
        size_t pos;
        auto& m = find_sl(sh.side, sh.price, &pos)->members;
        m.erase(std::find(m.begin(), m.end(), slot));  // erase keeps the stamp order
        if (m.empty()) sl_[s].erase(sl_[s].begin() + (std::ptrdiff_t)pos);
        free_slots_.push_back(slot);
        --nlive_;
    }

    size_t retire_and_place(const int64_t* place, size_t np, size_t pi) {
        for (uint32_t slot : check_) {
            Shadow& sh = shadows_[slot];
            sh.queued = false;
            if (sh.live && (sh.all_done() || ts_ >= sh.t_end)) retire(slot);
        }
        check_.clear();
        while (!expiry_.empty() && expiry_.top().t_end <= ts_) {
            const Expiry e = expiry_.top();
            expiry_.pop();
            if (shadows_[e.slot].live && shadows_[e.slot].sid == e.sid) retire(e.slot);
        }
        while (pi < np && place[pi] <= ts_) {
            for (char side : {'B', 'A'}) {
                int64_t b;
                if (!best(side, b)) continue;
                uint32_t slot;
                if (!free_slots_.empty()) { slot = free_slots_.back(); free_slots_.pop_back(); }
                else { slot = (uint32_t)shadows_.size(); shadows_.emplace_back(); }
                Shadow& sh = shadows_[slot];
                sh = Shadow{};
                sh.sid = next_sid_++;
                sh.side = side;
                sh.price = b;
                sh.t_place = ts_;
                sh.t_end = ts_ + H_;
                sh.size = S_;
                sh.q0 = level(side, b)->total;
                sh.vtrue = sh.q0;
                sh.stamp = ++stamp_;
                for (int k = 0; k < K; ++k) { sh.est_v[k] = (double)sh.q0; sh.est_t[k] = -1; sh.est_filled[k] = 0; sh.err[k] = 0; sh.abserr[k] = 0; }
                sh.live = true;
                ++nlive_;
                // index it: the newest stamp goes to the back, which keeps members in stamp order
                auto& v = sl_[side_ix(side)];
                auto it = std::lower_bound(v.begin(), v.end(), b, [](const ShadowLevel& L, int64_t p) { return L.price < p; });
                if (it == v.end() || it->price != b) it = v.insert(it, ShadowLevel{b, {}, 0, 0, 0});
                it->members.push_back(slot);
                expiry_.push({sh.t_end, sh.sid, slot});
                if (sh.all_done()) queue_check(slot);  // only with shadow_size <= 0
            }
            ++pi;
        }
        return pi;
    }

    void sample() {
        for (char side : {'B', 'A'}) {
            int64_t i = 0;
            auto emit = [&](int64_t p, const Level& lv) {
                samples.push_back(p); samples.push_back(lv.total); samples.push_back(lv.count);
            };
            if (side == 'B') { for (auto& kv : bids_) { if (i++ >= N_) break; emit(kv.first, kv.second); } }
            else { for (auto& kv : asks_) { if (i++ >= N_) break; emit(kv.first, kv.second); } }
            for (; i < N_; ++i) { samples.push_back(0); samples.push_back(0); samples.push_back(0); }
        }
    }

    // SPEC 8: one row per resting order at the best QL_ prices of each side, in FIFO order
    void queue_snapshot(int64_t q) {
        for (char side : {'B', 'A'}) {
            const int64_t sc = side == 'B' ? 1 : -1;
            int64_t li = 0;
            auto emit = [&](int64_t p, const Level& lv) {
                int64_t rank = 0, ahead = 0;
                for (uint32_t j = lv.head; j != NIL; j = pool_[j].next) {
                    const Order& o = pool_[j];
                    const int64_t row[QCOLS] = {q, (int64_t)o.id, sc, p, li, rank, o.size, ahead, lv.count, lv.total};
                    queue.insert(queue.end(), row, row + QCOLS);
                    ahead += o.size;
                    ++rank;
                }
                ++li;
            };
            if (side == 'B') { for (auto& kv : bids_) { if (li >= QL_) break; emit(kv.first, kv.second); } }
            else { for (auto& kv : asks_) { if (li >= QL_) break; emit(kv.first, kv.second); } }
        }
    }
};

template <typename T>
static py::array_t<T> to_np(const std::vector<T>& v) {
    py::array_t<T> a((py::ssize_t)v.size());
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

py::dict run(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ts,
             py::array_t<uint64_t, py::array::c_style | py::array::forcecast> oid,
             py::array_t<int64_t, py::array::c_style | py::array::forcecast> price,
             py::array_t<uint64_t, py::array::c_style | py::array::forcecast> size,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> flags,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> action,
             py::array_t<uint8_t, py::array::c_style | py::array::forcecast> side,
             py::array_t<int64_t, py::array::c_style | py::array::forcecast> place_times,
             py::array_t<int64_t, py::array::c_style | py::array::forcecast> sample_times, int64_t tick,
             int64_t label_ticks, int64_t shadow_size, int64_t horizon_ns, int64_t sample_levels,
             py::array_t<int64_t, py::array::c_style | py::array::forcecast> queue_times, int64_t queue_levels,
             bool order_tracks, int64_t track_horizon_ns, bool fill_lot_rounding, bool truth_record_price) {
    const size_t n = (size_t)ts.size();
    if (track_horizon_ns < 0) throw py::value_error("track_horizon_ns must be >= 0");
    Engine e(tick, label_ticks, shadow_size, horizon_ns, sample_levels, queue_levels, order_tracks, track_horizon_ns,
             fill_lot_rounding, truth_record_price);
    {
        py::gil_scoped_release nogil;
        e.run(ts.data(), oid.data(), price.data(), size.data(), flags.data(), action.data(), side.data(), n,
              place_times.data(), (size_t)place_times.size(), sample_times.data(), (size_t)sample_times.size(),
              queue_times.data(), (size_t)queue_times.size());
    }
    py::dict out, st, shd, lab;
    for (auto& kv : e.stats) st[py::str(kv.first)] = kv.second;
    out["stats"] = st;
    out["fifo_rank"] = to_np(e.fifo_rank);
    out["fifo_ts"] = to_np(e.fifo_ts);
    const size_t m = e.done.size();
    std::vector<int64_t> sid(m), sd(m), pr(m), tp(m), q0(m), en(m), tf(m), ttf(m), nobs(m);
    py::array_t<int64_t> ef({(py::ssize_t)m, (py::ssize_t)K}), et({(py::ssize_t)m, (py::ssize_t)K});
    py::array_t<double> er({(py::ssize_t)m, (py::ssize_t)K}), ae({(py::ssize_t)m, (py::ssize_t)K});
    auto efm = ef.mutable_unchecked<2>(); auto etm = et.mutable_unchecked<2>();
    auto erm = er.mutable_unchecked<2>(); auto aem = ae.mutable_unchecked<2>();
    for (size_t i = 0; i < m; ++i) {
        const Shadow& s = e.done[i];
        sid[i] = s.sid; sd[i] = s.side == 'B' ? 1 : -1; pr[i] = s.price; tp[i] = s.t_place; q0[i] = s.q0;
        en[i] = s.end; tf[i] = s.filled >= s.size ? 1 : 0; ttf[i] = s.t_fill; nobs[i] = s.nobs;
        for (int k = 0; k < K; ++k) {
            efm(i, k) = s.est_filled[k] >= (double)s.size ? 1 : 0; etm(i, k) = s.est_t[k];
            erm(i, k) = s.err[k]; aem(i, k) = s.abserr[k];
        }
    }
    shd["sid"] = to_np(sid); shd["side"] = to_np(sd); shd["price"] = to_np(pr); shd["t_place"] = to_np(tp);
    shd["q0"] = to_np(q0); shd["end"] = to_np(en); shd["true_filled"] = to_np(tf); shd["true_t_fill"] = to_np(ttf);
    shd["est_filled"] = ef; shd["est_t_fill"] = et; shd["err_sum"] = er; shd["abserr_sum"] = ae; shd["nobs"] = to_np(nobs);
    out["shadows"] = shd;
    const size_t L = e.labels.size();
    std::vector<int64_t> c[11];
    for (auto& v : c) v.resize(L);
    for (size_t i = 0; i < L; ++i) {
        const Label& x = e.labels[i];
        c[0][i] = x.order_id; c[1][i] = x.t_add; c[2][i] = x.price; c[3][i] = x.size; c[4][i] = x.dist;
        c[5][i] = x.rank_at_add; c[6][i] = x.size_ahead_at_add; c[7][i] = x.outcome; c[8][i] = x.t_end;
        c[9][i] = x.filled_qty; c[10][i] = x.side;
    }
    const char* names[11] = {"order_id", "t_add", "price", "size", "dist", "rank_at_add", "size_ahead_at_add",
                             "outcome", "t_end", "filled_qty", "side"};
    for (int j = 0; j < 11; ++j) lab[names[j]] = to_np(c[j]);
    out["labels"] = lab;
    auto smp = to_np(e.samples);
    out["samples"] = smp.reshape({(py::ssize_t)(e.samples.size() / (size_t)(6 * sample_levels)), (py::ssize_t)(6 * sample_levels)});
    auto qa = to_np(e.queue);
    out["queue"] = qa.reshape({(py::ssize_t)(e.queue.size() / (size_t)QCOLS), (py::ssize_t)QCOLS});
    if (order_tracks) {  // SPEC 9; keys and order as reference.TRACK_COLUMNS
        const size_t T = e.tracks.size();
        std::vector<int64_t> c1[11];
        for (auto& v : c1) v.resize(T);
        py::array_t<int64_t> et({(py::ssize_t)T, (py::ssize_t)K});
        py::array_t<double> ef({(py::ssize_t)T, (py::ssize_t)K}), vf({(py::ssize_t)T, (py::ssize_t)K}),
            ve({(py::ssize_t)T, (py::ssize_t)K});
        auto etm = et.mutable_unchecked<2>(); auto efm = ef.mutable_unchecked<2>();
        auto vfm = vf.mutable_unchecked<2>(); auto vem = ve.mutable_unchecked<2>();
        for (size_t i = 0; i < T; ++i) {
            const Track& t = e.tracks[i];
            c1[0][i] = t.label_index; c1[1][i] = t.order_id; c1[2][i] = t.side; c1[3][i] = t.price; c1[4][i] = t.t_start;
            c1[5][i] = t.size0; c1[6][i] = t.ahead0; c1[7][i] = t.outcome; c1[8][i] = t.t_end; c1[9][i] = t.tff;
            c1[10][i] = t.tfq;
            for (int k = 0; k < K; ++k) { etm(i, k) = t.et[k]; efm(i, k) = t.ef[k]; vfm(i, k) = t.vff[k]; vem(i, k) = t.v[k]; }
        }
        const char* tn[11] = {"label_index", "order_id", "side", "price", "t_start", "size0", "size_ahead0", "outcome",
                              "t_end", "true_first_fill_t", "true_filled_qty"};
        py::dict trk;
        for (int j = 0; j < 11; ++j) trk[tn[j]] = to_np(c1[j]);
        trk["est_first_fill_t"] = et; trk["est_filled_qty"] = ef; trk["v_est_at_true_first_fill"] = vf; trk["v_est_at_end"] = ve;
        if (track_horizon_ns > 0) {  // v0.5: reference.TRACK_HORIZON_COLUMNS
            std::vector<int64_t> stop(T);
            py::array_t<int64_t> e0({(py::ssize_t)T, (py::ssize_t)K});
            auto e0m = e0.mutable_unchecked<2>();
            for (size_t i = 0; i < T; ++i) {
                stop[i] = e.tracks[i].stop_t;
                for (int k = 0; k < K; ++k) e0m(i, k) = e.e0(e.tracks[i], k);
            }
            trk["est_stop_t"] = to_np(stop);
            trk["e0"] = e0;
        }
        out["tracks"] = trk;
    }
    return out;
}

}  // namespace mboq

PYBIND11_MODULE(_core, m) {
    m.doc() = "mbo-queue C++20 core (SPEC.md v0.5)";
    m.def("run", &mboq::run,
          "Run book rebuild, FIFO check, shadows, estimators, labels, samples, queue snapshots and order tracks");
}

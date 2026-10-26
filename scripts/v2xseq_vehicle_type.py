"""S. Vehicle type of the V2X-Seq leaders (docs/FINDINGS.md, section 'S. 선두 차량 차종').

The event tables carry no type column: dzp.v2xseq.load_tracks keeps VEHICLE rows with sub_type CAR / VAN / TRUCK /
BUS and maps them to agent_type, but events.extract_events does not write it. This script re-attaches the type by
track id from the raw trajectory files and reports, by vehicle class, the decision boundary (t_50), the after-onset
MLP performance in the dilemma zone and the staged decision, and the leader type seen by the follower events.

  types : per vehicle track (the split track_id of dzp.v2xseq.load_tracks) in every segment that has a yellow-onset
          event (data/raw/v2xseq/results/check_v4/events_all.csv): modal sub_type over the track's frames, number of
          distinct sub_types, median / min / max length -> data/raw/v2xseq/results/vehicle_type/track_types.csv
          (gitignored). Then the 1,876-row leader table with vehicle_type / vehicle_length / vehicle_class columns
          -> data/raw/v2xseq/results/leaders_v4_leads/events_through_vtype.csv (new file, the original is untouched).
  s1    : class distribution of the 1,867 leaders (t_r known) and of the 509 dilemma-zone leaders (d/v 2.5-5.5 s)
  s2    : t_50 (d/v <= 12 s, univariate logistic, cycle bootstrap) for large vehicles and for passenger cars, and
          the likelihood-ratio tests of a logistic model d/v + class + d/v x class against d/v alone
          (only when >= MIN_T50 large leaders; otherwise counts only)
  s3    : dilemma zone, MLP out-of-fold P(go) of results/v2xseq/leadtime_dz (23-fold LOIO): macro-F1 at 0 / 1.0 /
          2.0 s and the staged decision (theta = 0.05: mean commit time, risky / conservative errors) by class, with
          the a(t_c) of the large vehicles that actually stopped as in Q4 (only when >= MIN_DZ large leaders)
  s4    : dilemma-zone follower events (same all-events table, through, t_r known, d/v 2.5-5.5 s) by the class of
          their leader (ldr_id of dzp.features.build)
  s5    : leader outcome x follower outcome cross-tables (through followers, band followers) and, for the
          "leader stopped & follower went" cases, the lateral offsets on the approach from the raw tracks
          (lane change = |lateral shift between onset and crossing| > LANE_W / 2)

Classes: large = TRUCK or BUS, or median length >= LARGE_LEN_M; passenger = CAR or VAN below that; other = anything
else (counted only). Outputs under results/v2xseq/vehicle_type/.

  .venv/bin/python scripts/v2xseq_vehicle_type.py --tasks types,s1,s2,s3,s4,s5
"""
import argparse
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from dzp import bench_eval, v2xseq

LEADS = [0.0, 0.5, 1.0, 1.5, 2.0]
SHOW_LEADS = [0.0, 1.0, 2.0]
DZ_LO, DZ_HI, TT_MAX = 2.5, 5.5, 12.0
THETA = 0.05
LARGE_LEN_M = 6.0
MIN_T50, MIN_DZ = 30, 20
LARGE_TYPES, PASSENGER_TYPES = ("TRUCK", "BUS"), ("CAR", "VAN")
ROOT = "data/raw/v2xseq/full/extracted/V2X-Seq-TFD"
CACHE = "data/raw/v2xseq/results/cache"
LEADS_DIR = "data/raw/v2xseq/results/leaders_v4_leads"
ALL_EVENTS = "data/raw/v2xseq/results/check_v4/events_all.csv"
PREDS = "results/v2xseq/leadtime_dz"
RAW_OUT = "data/raw/v2xseq/results/vehicle_type"
VTYPE_TABLE = f"{LEADS_DIR}/events_through_vtype.csv"
_G = {}


# ------------------------------------------------------------------------------------------------ types
def vehicle_class(sub_type, length_m):
    """'large' (TRUCK / BUS, or any type with median length >= LARGE_LEN_M), 'passenger' (CAR / VAN), else 'other'."""
    st = np.asarray(sub_type, dtype=object)
    ln = np.asarray(length_m, dtype=float)
    out = np.full(len(st), "other", dtype=object)
    out[np.isin(st, PASSENGER_TYPES)] = "passenger"
    out[np.isin(st, LARGE_TYPES) | (np.isin(st, PASSENGER_TYPES) & (ln >= LARGE_LEN_M))] = "large"
    return out


def _init_types():
    _G["segs"] = {s.name: s for s in v2xseq.segments(ROOT, CACHE)}


def _segment_types(name):
    seg = _G["segs"][name]
    t = v2xseq.load_tracks(ROOT, seg)[["track_id", "orig_id", "timestamp", "length"]]
    raw = pd.concat([v2xseq._read(ROOT, "trajectories", s, ["timestamp", "id", "type", "sub_type"]) for s in seg.scenes], ignore_index=True)
    raw = raw[raw["type"] == "VEHICLE"].drop_duplicates(["id", "timestamp"])
    m = t.merge(raw[["id", "timestamp", "sub_type"]], left_on=["orig_id", "timestamp"], right_on=["id", "timestamp"], how="left")
    rows = []
    for tid, g in m.groupby("track_id", sort=False):
        vc = g["sub_type"].value_counts()
        rows.append(dict(segment=name, track_id=int(tid), sub_type=vc.index[0], n_sub_types=int(len(vc)), sub_type_share=float(vc.iloc[0] / len(g)),
                         n_frames=int(len(g)), length_median=float(g["length"].median()), length_min=float(g["length"].min()),
                         length_max=float(g["length"].max())))
    return rows


def types(args):
    os.makedirs(RAW_OUT, exist_ok=True)
    A = pd.read_csv(ALL_EVENTS)
    names = sorted(A["recording"].unique())
    print(f"[types] {len(names)} segments with events", flush=True)
    rows = []
    with get_context("spawn").Pool(args.workers, initializer=_init_types) as pool:
        for i, r in enumerate(pool.imap_unordered(_segment_types, names, chunksize=4), 1):
            rows += r
            if i % 100 == 0:
                print(f"[types] {i}/{len(names)}", flush=True)
    T = pd.DataFrame(rows)
    T["vehicle_class"] = vehicle_class(T["sub_type"], T["length_median"])
    T.to_csv(f"{RAW_OUT}/track_types.csv", index=False)
    print(f"[types] tracks {len(T)}; sub_type {T.sub_type.value_counts().to_dict()}; class {T.vehicle_class.value_counts().to_dict()}", flush=True)
    attach_leaders(T)


def attach_leaders(T=None):
    """Leader table (1,876 rows) + vehicle_type / vehicle_length_m / vehicle_class -> events_through_vtype.csv."""
    T = pd.read_csv(f"{RAW_OUT}/track_types.csv") if T is None else T
    ev = pd.read_csv(f"{LEADS_DIR}/events_through.csv")
    key = T.set_index(["segment", "track_id"])
    idx = list(zip(ev["segment"], ev["track_id"].astype(int)))
    found = key.index.isin(idx)
    sub = key.reindex(idx)
    ev["vehicle_type"] = sub["sub_type"].values
    ev["vehicle_type_n"] = sub["n_sub_types"].values
    ev["vehicle_type_share"] = sub["sub_type_share"].values
    ev["vehicle_length_m"] = sub["length_median"].values
    ev["vehicle_class"] = np.where(sub["sub_type"].notna().values, vehicle_class(sub["sub_type"].fillna("").values, sub["length_median"].values), "other")
    assert ev["vehicle_type"].notna().all(), f"{int(ev.vehicle_type.isna().sum())} leaders without a track type"
    ev.to_csv(VTYPE_TABLE, index=False)
    print(f"[types] leaders {len(ev)}: type {ev.vehicle_type.value_counts().to_dict()}, class {ev.vehicle_class.value_counts().to_dict()}, "
          f"tracks matched {int(found.sum())} of {len(key)}", flush=True)


# ------------------------------------------------------------------------------------------------ common
def _load():
    """1,876-row tables aligned with the prediction files; ok (t_r known, 1,867), band (509), class per row."""
    from staged_dz import load
    ev, FS, band0, crossed, usable = load(LEADS_DIR)
    V = pd.read_csv(VTYPE_TABLE)
    assert (V["segment"].values == ev["segment"].values).all() and (V["track_id"].values == ev["track_id"].values).all()
    ok = FS[0.0][0]["t_r"].notna().values
    return ev, FS, band0, crossed, usable, ok, V


def _dist(cls, y, label):
    n = len(cls)
    rows = []
    for c in ("large", "passenger", "other"):
        m = cls == c
        rows.append(dict(subset=label, vehicle_class=c, n=int(m.sum()), share=float(m.mean()) if n else np.nan,
                         n_go=int(y[m].sum()), n_stop=int((y[m] == 0).sum()), go_share=float(y[m].mean()) if m.any() else np.nan))
    rows.append(dict(subset=label, vehicle_class="all", n=n, share=1.0, n_go=int(y.sum()), n_stop=int((y == 0).sum()), go_share=float(y.mean())))
    return rows


def s1(args, out):
    ev, FS, band0, crossed, usable, ok, V = _load()
    y = ev["label"].values.astype(int)
    cls, typ = V["vehicle_class"].values, V["vehicle_type"].values
    rows = _dist(cls[ok], y[ok], "leaders_1867") + _dist(cls[band0], y[band0], "band_509")
    # raw sub_type and the length rule, for the record
    for lab, m in (("leaders_1867", ok), ("band_509", band0)):
        for t in ("CAR", "VAN", "TRUCK", "BUS"):
            mm = m & (typ == t)
            rows.append(dict(subset=lab, vehicle_class=f"type:{t}", n=int(mm.sum()), share=float(mm.sum() / m.sum()), n_go=int(y[mm].sum()),
                             n_stop=int((y[mm] == 0).sum()), go_share=float(y[mm].mean()) if mm.any() else np.nan,
                             n_len_ge_6=int((mm & (V["vehicle_length_m"].values >= LARGE_LEN_M)).sum()),
                             length_median=float(np.median(V["vehicle_length_m"].values[mm])) if mm.any() else np.nan))
        rows.append(dict(subset=lab, vehicle_class="type:mixed_sub_type_in_track", n=int((m & (V["vehicle_type_n"].values > 1)).sum()),
                         share=float((m & (V["vehicle_type_n"].values > 1)).sum() / m.sum())))
    D = pd.DataFrame(rows)
    D.to_csv(f"{out}/s1_class_distribution.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(D.round(4).to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ s2
def _t50(x, y):
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(max_iter=1000).fit(x[:, None], y)
    return -lr.intercept_[0] / lr.coef_[0, 0] if lr.coef_[0, 0] < 0 else np.nan


def _cyc_draws(cyc, mask, n_boot, seed=0):   # as scripts/pooled_eval.py cyc_draws
    idx = np.where(mask)[0]
    c = cyc[idx]
    uc, inv = np.unique(c, return_inverse=True)
    mem = [idx[inv == k] for k in range(len(uc))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([mem[k] for k in rng.integers(0, len(uc), len(uc))]) for _ in range(n_boot)]


def _loglik(X, y):
    lr = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-8).fit(X, y)
    p = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), lr


def lr_tests(tt, y, large):
    """Nested logistic models on d/v: M0 d/v; M1 + class; M2 + class + d/v x class. LR tests (chi-square)."""
    X0 = tt[:, None]
    X1 = np.column_stack([tt, large])
    X2 = np.column_stack([tt, large, tt * large])
    (l0, m0), (l1, m1), (l2, m2) = _loglik(X0, y), _loglik(X1, y), _loglik(X2, y)
    def test(name, la, lb, df):
        stat = 2 * (lb - la)
        return dict(test=name, df=df, lr_stat=stat, p_value=float(chi2.sf(stat, df)))
    rows = [test("intercept (M1 vs M0)", l0, l1, 1), test("slope given intercept (M2 vs M1)", l1, l2, 1), test("intercept + slope (M2 vs M0)", l0, l2, 2)]
    b2 = dict(b0=m2.intercept_[0], b_tt=m2.coef_[0, 0], b_large=m2.coef_[0, 1], b_tt_large=m2.coef_[0, 2])
    t50_pass = -b2["b0"] / b2["b_tt"]
    t50_large = -(b2["b0"] + b2["b_large"]) / (b2["b_tt"] + b2["b_tt_large"])
    return rows, dict(loglik_M0=l0, loglik_M1=l1, loglik_M2=l2, **b2, t50_passenger_M2=t50_pass, t50_large_M2=t50_large)


def s2(args, out):
    ev, FS, band0, crossed, usable, ok, V = _load()
    y = ev["label"].values.astype(int)
    tt = FS[0.0][0]["tt"].values
    cls = V["vehicle_class"].values
    cyc = bench_eval.cycles(ev)
    keep = ok & (tt <= TT_MAX)
    n_large = int((ok & (cls == "large")).sum())
    rows = []
    for c in ("all", "large", "passenger"):
        m = keep & ((cls == c) if c != "all" else True)
        r = dict(vehicle_class=c, n=int((ok & ((cls == c) if c != "all" else True)).sum()), n_tt_le_12=int(m.sum()), n_go=int(y[m].sum()), n_stop=int((y[m] == 0).sum()),
                 n_cycles=int(len(np.unique(cyc[m]))))
        if c == "large" and n_large < MIN_T50:
            r.update(t50=np.nan, note=f"not estimated: {n_large} large leaders < {MIN_T50}")
        else:
            bs = np.array([_t50(tt[ix], y[ix]) for ix in _cyc_draws(cyc, m, args.n_boot)])
            r.update(t50=_t50(tt[m], y[m]), t50_lo=float(np.nanpercentile(bs, 2.5)), t50_hi=float(np.nanpercentile(bs, 97.5)), t50_boot_undefined=int(np.isnan(bs).sum()))
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(f"{out}/s2_t50_by_class.csv", index=False)
    print(D.round(4).to_string(index=False), flush=True)
    if n_large >= MIN_T50:
        m = keep & np.isin(cls, ["large", "passenger"])
        tests, fit = lr_tests(tt[m], y[m], (cls[m] == "large").astype(float))
        Tt = pd.DataFrame(tests).assign(n=int(m.sum()), n_large=int((cls[m] == "large").sum()))
        Tt.to_csv(f"{out}/s2_lr_tests.csv", index=False)
        pd.DataFrame([fit]).to_csv(f"{out}/s2_lr_fit.csv", index=False)
        print(Tt.to_string(index=False), flush=True)
        print({k: round(v, 6) for k, v in fit.items()}, flush=True)
        # difference of the two separate t_50 with its bootstrap CI (same cycle draws for both classes)
        draws = _cyc_draws(cyc, m, args.n_boot)
        dif = np.array([_t50(tt[ix][cls[ix] == "large"], y[ix][cls[ix] == "large"]) - _t50(tt[ix][cls[ix] == "passenger"], y[ix][cls[ix] == "passenger"]) for ix in draws])
        pt = D.set_index("vehicle_class")
        dd = dict(d_t50_large_minus_passenger=pt.loc["large", "t50"] - pt.loc["passenger", "t50"], lo=float(np.nanpercentile(dif, 2.5)), hi=float(np.nanpercentile(dif, 97.5)),
                  boot_undefined=int(np.isnan(dif).sum()))
        pd.DataFrame([dd]).to_csv(f"{out}/s2_t50_difference.csv", index=False)
        print(dd, flush=True)


# ------------------------------------------------------------------------------------------------ s3
def _q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=int(len(x)), median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                min=float(x.min()), max=float(x.max())) if len(x) else dict(n=0)


def s3(args, out):
    from staged_dz import decide, stats
    ev, FS, band0, crossed, usable, ok, V = _load()
    y = ev["label"].values.astype(int)
    cls = V["vehicle_class"].values
    ib = np.where(band0)[0]
    n_large = int((cls[ib] == "large").sum())
    if n_large < MIN_DZ:
        pd.DataFrame([dict(n_band=len(ib), n_large=n_large, note=f"skipped: {n_large} large leaders in the band < {MIN_DZ}")]).to_csv(f"{out}/s3_skipped.csv", index=False)
        print(f"[s3] skipped: {n_large} large leaders in the band < {MIN_DZ}", flush=True)
        return
    P = np.column_stack([pd.read_csv(f"{PREDS}/preds_lead{L:+.1f}.csv")["mlp_dva"].values for L in LEADS])
    X = np.column_stack([crossed[L] for L in LEADS])
    cyc = bench_eval.cycles(ev)
    draws = _cyc_draws(cyc, band0, args.n_boot)          # whole-band cycle draws, both classes from the same draw
    groups = {"all": np.ones(len(y), bool), "large": cls == "large", "passenger": cls == "passenger"}
    rows = []
    # macro-F1 at fixed leads (usable: not crossed, P finite)
    for j, L in enumerate(LEADS):
        if L not in SHOW_LEADS:
            continue
        mask = band0 & usable[L] & np.isfinite(P[:, j])
        yh = (P[:, j] >= 0.5).astype(int)
        f1 = {}
        for g, gm in groups.items():
            m = mask & gm
            pt = bench_eval._macro_f1(y[m], yh[m]) if m.any() else np.nan
            bs = np.array([bench_eval._macro_f1(y[ix][mask[ix] & gm[ix]], yh[ix][mask[ix] & gm[ix]]) if (mask[ix] & gm[ix]).any() else np.nan for ix in draws])
            f1[g] = bs
            rows.append(dict(quantity="macro_f1", lead_s=L, vehicle_class=g, n=int(m.sum()), n_go=int(y[m].sum()), n_stop=int((y[m] == 0).sum()), value=pt,
                             lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)), n_boot_undefined=int(np.isnan(bs).sum())))
        d = f1["large"] - f1["passenger"]
        ml, mp = mask & groups["large"], mask & groups["passenger"]
        rows.append(dict(quantity="macro_f1_large_minus_passenger", lead_s=L, vehicle_class="large-passenger", n=int(mask.sum()),
                         value=bench_eval._macro_f1(y[ml], yh[ml]) - bench_eval._macro_f1(y[mp], yh[mp]),
                         lo=float(np.nanpercentile(d, 2.5)), hi=float(np.nanpercentile(d, 97.5)), n_boot_undefined=int(np.isnan(d).sum())))
    # staged decision theta = 0.05 on the band
    yh, tc = decide(P[ib], X[ib], THETA)
    yb, clb = y[ib], cls[ib]
    pos = {e: i for i, e in enumerate(ib)}
    for g in groups:
        gm = clb == g if g != "all" else np.ones(len(ib), bool)
        pt = stats(yb[gm], yh[gm], tc[gm])
        bs = []
        for ix in draws:
            ii = np.array([pos[e] for e in ix])
            gi = ii[(clb[ii] == g) if g != "all" else np.ones(len(ii), bool)]
            bs.append(stats(yb[gi], yh[gi], tc[gi]) if len(gi) else dict(macro_f1=np.nan, risky=np.nan, conservative=np.nan, t_mean=np.nan, t_median=np.nan))
        B = pd.DataFrame(bs)
        for k in ("t_mean", "macro_f1", "risky", "conservative"):
            rows.append(dict(quantity=f"staged_{k}", lead_s=np.nan, vehicle_class=g, n=int(gm.sum()), n_go=int(yb[gm].sum()), n_stop=int((yb[gm] == 0).sum()), value=pt[k],
                             lo=float(np.nanpercentile(B[k], 2.5)), hi=float(np.nanpercentile(B[k], 97.5)), n_boot_undefined=int(B[k].isna().sum()),
                             n_risky=int(np.sum((yb[gm] == 0) & (yh[gm] == 1))), n_conservative=int(np.sum((yb[gm] == 1) & (yh[gm] == 0)))))
    # a(t_c) of the leaders that actually stopped (Q4 method: feature value at the commit lead, track-ended rows excluded)
    per = []
    for i, e in enumerate(ib):
        if yb[i] != 0:
            continue
        f = FS[float(tc[i])][0].iloc[e]
        per.append(dict(ev=e, segment=ev["segment"].iloc[e], track_id=ev["track_id"].iloc[e], vehicle_class=clb[i], vehicle_type=V["vehicle_type"].iloc[e],
                        vehicle_length_m=V["vehicle_length_m"].iloc[e], t_c=float(tc[i]), yhat=int(yh[i]), clamped=int(f["clamped"]), d_tc=float(f["d"]), v_tc=float(f["v"]), a_tc=float(f["a"])))
    E = pd.DataFrame(per)
    os.makedirs(RAW_OUT, exist_ok=True)
    E.to_csv(f"{RAW_OUT}/s3_stop_commit.csv", index=False)
    for g in groups:
        S = E[(E.clamped == 0) & ((E.vehicle_class == g) if g != "all" else True)]
        for col in ("a_tc", "d_tc", "v_tc", "t_c"):
            rows.append(dict(quantity=f"stop_{col}", lead_s=np.nan, vehicle_class=g, n=len(S), **{k: v for k, v in _q(S[col]).items() if k != "n"}))
    D = pd.DataFrame(rows)
    D.to_csv(f"{out}/s3_band_by_class.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(D.round(4).to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ s4
def s4(args, out):
    T = pd.read_csv(f"{RAW_OUT}/track_types.csv")
    A = pd.read_csv(ALL_EVENTS)
    F = A[(A["ldr_id"] >= 0) & (A["movement"] == "through")].copy()
    key = T.set_index(["segment", "track_id"])
    idx = list(zip(F["recording"], F["ldr_id"].astype(int)))
    sub = key.reindex(idx)
    F["leader_type"] = sub["sub_type"].values
    F["leader_length_m"] = sub["length_median"].values
    F["leader_class"] = np.where(sub["sub_type"].notna().values, vehicle_class(sub["sub_type"].fillna("").values, sub["length_median"].values), "unknown")
    own = key.reindex(list(zip(F["recording"], F["track_id"].astype(int))))
    F["vehicle_class"] = np.where(own["sub_type"].notna().values, vehicle_class(own["sub_type"].fillna("").values, own["length_median"].values), "unknown")
    ok = F["t_r"].notna().values
    band = ok & (F["tt"].values >= DZ_LO) & (F["tt"].values <= DZ_HI)
    y = F["label"].values.astype(int)
    rows = []
    for lab, m in (("followers_through", ok), ("followers_band", band)):
        for c in ("large", "passenger", "other", "unknown"):
            mm = m & (F["leader_class"].values == c)
            rows.append(dict(subset=lab, leader_class=c, n=int(mm.sum()), share=float(mm.sum() / m.sum()), n_go=int(y[mm].sum()), n_stop=int((y[mm] == 0).sum()),
                             n_leader_stop_label=int((mm & (F["ldr_label"].values == 0)).sum()), n_leader_label_known=int((mm & F["ldr_label"].notna().values).sum()),
                             gap_median_m=float(np.median(F["ldr_gap"].values[mm])) if mm.any() else np.nan))
        rows.append(dict(subset=lab, leader_class="all", n=int(m.sum()), share=1.0, n_go=int(y[m].sum()), n_stop=int((y[m] == 0).sum()),
                         n_leader_stop_label=int((m & (F["ldr_label"].values == 0)).sum()), n_leader_label_known=int((m & F["ldr_label"].notna().values).sum()),
                         gap_median_m=float(np.median(F["ldr_gap"].values[m]))))
        for t in ("CAR", "VAN", "TRUCK", "BUS"):
            mm = m & (F["leader_type"].values == t)
            rows.append(dict(subset=lab, leader_class=f"type:{t}", n=int(mm.sum()), share=float(mm.sum() / m.sum()), n_go=int(y[mm].sum()), n_stop=int((y[mm] == 0).sum())))
    D = pd.DataFrame(rows)
    D.to_csv(f"{out}/s4_follower_leader_class.csv", index=False)
    F.to_csv(f"{RAW_OUT}/s4_followers.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(D.round(4).to_string(index=False), flush=True)
        print("follower own class (band):", pd.crosstab(F["vehicle_class"][band], F["leader_class"][band]).to_string(), flush=True)


# ------------------------------------------------------------------------------------------------ s5
def _s5_case(e):
    """Lateral offset of follower and leader on the approach (from the raw tracks): lane change check."""
    from dzp.geometry import build_approaches, project
    from dzp import events
    rec = v2xseq.load_recording(ROOT, _G["segs"][e["recording"]], _G["lanes"], shift_table=_G["shift"])
    ap = build_approaches(rec.map.stop_lines())[int(e["sid"])]
    out = dict(recording=e["recording"], track_id=int(e["track_id"]), ldr_id=int(e["ldr_id"]))
    for who, tid in (("fol", int(e["track_id"])), ("ldr", int(e["ldr_id"]))):
        k = rec.tracks[rec.tracks["track_id"] == tid].sort_values("frame_id")
        fr = k["frame_id"].values
        fx = k["x"].values + 0.5 * k["length"].values * np.cos(k["heading_rad"].values)
        fy = k["y"].values + 0.5 * k["length"].values * np.sin(k["heading_rad"].values)
        d, lat = project(ap, fx, fy)
        v = np.hypot(k["vx"].values, k["vy"].values)
        ts = (fr - e["f_yellow"]) * v2xseq.DT
        i0 = int(np.argmin(np.abs(fr - e["k0"])))
        after = fr >= e["k0"]
        cross = np.where((d[:-1] > 0) & (d[1:] <= 0) & after[1:])[0] + 1
        halt = np.where(after & (v < events.V_STOP) & (d > 0))[0]
        out.update({f"{who}_lat_onset": float(lat[i0]), f"{who}_d_onset": float(d[i0]), f"{who}_v_onset": float(v[i0]),
                    f"{who}_t_cross": float(ts[cross[0]]) if len(cross) else np.nan, f"{who}_lat_cross": float(lat[cross[0]]) if len(cross) else np.nan,
                    f"{who}_t_halt": float(ts[halt[0]]) if len(halt) else np.nan, f"{who}_d_halt": float(d[halt[0]]) if len(halt) else np.nan,
                    f"{who}_lat_halt": float(lat[halt[0]]) if len(halt) else np.nan, f"{who}_t_end": float(ts[-1]), f"{who}_d_end": float(d[-1]),
                    f"{who}_lat_min": float(lat[after].min()), f"{who}_lat_max": float(lat[after].max())})
        if who == "fol":
            fol = (ts, d, lat)
    # leader's position when the follower crosses, and the follower's lateral shift
    ts_f, d_f, lat_f = fol
    out["fol_lat_shift"] = out["fol_lat_cross"] - out["fol_lat_onset"] if np.isfinite(out["fol_t_cross"]) else np.nan
    out["lane_change"] = bool(abs(out["fol_lat_shift"]) > LANE_HALF) if np.isfinite(out["fol_lat_shift"]) else None
    return out


LANE_HALF = v2xseq.LANE_W / 2


def s5(args, out):
    from v2xseq_v11 import _init, _G as G11
    A = pd.read_csv(ALL_EVENTS)
    F = A[(A["ldr_id"] >= 0) & (A["movement"] == "through") & A["t_r"].notna()].copy()
    band = (F["tt"].values >= DZ_LO) & (F["tt"].values <= DZ_HI)
    F["leader_outcome"] = F["ldr_label"].map({0: "leader_stop", 1: "leader_go"}).fillna("leader_unknown")
    F["follower_outcome"] = F["label"].map({0: "follower_stop", 1: "follower_go"})
    rows = []
    for lab, m in (("followers_through", np.ones(len(F), bool)), ("followers_band", band)):
        ct = pd.crosstab(F["leader_outcome"][m], F["follower_outcome"][m])
        for lo in ("leader_go", "leader_stop", "leader_unknown"):
            for fo in ("follower_go", "follower_stop"):
                rows.append(dict(subset=lab, leader_outcome=lo, follower_outcome=fo, n=int(ct.loc[lo, fo]) if lo in ct.index and fo in ct.columns else 0))
    C = pd.DataFrame(rows)
    C.to_csv(f"{out}/s5_leader_follower_crosstab.csv", index=False)
    print(C.pivot_table(index=["subset", "leader_outcome"], columns="follower_outcome", values="n").to_string(), flush=True)
    X = F[(F["ldr_label"] == 0) & (F["label"] == 1)]
    print(f"[s5] leader stop & follower go: {len(X)} (band {int(((X.tt >= DZ_LO) & (X.tt <= DZ_HI)).sum())})", flush=True)
    _init("results/v2xseq/stopline_shift.csv")
    _G.update(G11)
    cases = pd.DataFrame([_s5_case(e) for _, e in X.iterrows()])
    cases.insert(3, "band", ((X["tt"].values >= DZ_LO) & (X["tt"].values <= DZ_HI)))
    cases.insert(4, "lane", X["lane"].values)
    cases.insert(5, "tt_onset", X["tt"].values)
    cases.insert(6, "ldr_gap_onset", X["ldr_gap"].values)
    cases.to_csv(f"{out}/s5_leader_stop_follower_go.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(cases.round(3).T.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="types,s1,s2,s3,s4")
    ap.add_argument("--out", default="results/v2xseq/vehicle_type")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for t in args.tasks.split(","):
        print(f"==== {t}", flush=True)
        {"types": lambda: types(args), "attach": attach_leaders, "s1": lambda: s1(args, args.out), "s2": lambda: s2(args, args.out),
         "s3": lambda: s3(args, args.out), "s4": lambda: s4(args, args.out), "s5": lambda: s5(args, args.out)}[t]()


if __name__ == "__main__":
    main()

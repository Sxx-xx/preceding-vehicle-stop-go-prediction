"""Q. Paper v1.1 supplementary numbers on the V2X-Seq leaders (docs/FINDINGS.md, section 'Q. v1.1 보완 분석').

Same event table (data/raw/v2xseq/results/leaders_v4_leads, 1,867 leaders after dropping t_r NaN), same models'
out-of-fold P(go) (results/v2xseq/leadtime_dz, 23-fold LOIO) and the same bootstrap as the N / D / P sections.

  traj : per-leader post-onset trajectory summary from the raw tracks (first halt v < 1 m/s upstream, first stop-line
         crossing, speeds in between) -> data/raw/v2xseq/results/v11/post_onset.csv (gitignored)
  q1   : label boundary cases: halted upstream (v < 1 m/s, d > 0) and then crossed the line; by crossing phase
         (yellow / within GO_GRACE_S = 3 s of red onset / later); what events.label_event gives them
  q2   : stop-line calibration sensitivity: t_50 (d/v <= 12 s, cycle bootstrap) and the 23-fold calibrated rule A
         macro-F1 (all / band) for the tables made with no shift, the 5 %, 10 % (= results/v2xseq/eval/v4) and
         20 % halt-position quantile (scripts/v2xseq_calibrate.py --lead-q, scripts/v2xseq_extract.py)
  q3   : theta sensitivity of the staged decision (MLP, band 509): theta = 0.02, 0.05, 0.10, 0.20 and all at 2.0 s
  q4   : margin at the commit time t_c (MLP, theta = 0.05 and all at 2.0 s) for the band leaders that actually
         stopped: d(t_c), time to the actual halt, a(t_c), and the constant deceleration a follower at the same
         speed tau = 2.0 / 1.5 s behind needs to stop 2 m behind the leader: v^2 / (2 (tau v + d(t_c) - 2))
         (the leader taken to stop at the line, as specified) and the variant with the leader's actual halt
         position, v^2 / (2 (tau v + d(t_c) - d_halt - 2)).

Outputs under results/v2xseq/v11/: q1_label_boundary.csv, q2_stopline_sensitivity.csv (+ q2_fold_t50.csv),
q3_theta_sensitivity.csv, q4_commit_margin.csv; per-event tables under data/raw/v2xseq/results/v11/.

  .venv/bin/python scripts/v2xseq_v11.py --tasks traj,q1,q3,q4
  .venv/bin/python scripts/v2xseq_v11.py --tasks q2      # after the three scripts/v2xseq_extract.py runs
"""
import argparse
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dzp import bench_eval, events, v2xseq
from dzp.geometry import build_approaches, project

LEADS = [0.0, 0.5, 1.0, 1.5, 2.0]
DZ_LO, DZ_HI, TT_MAX = 2.5, 5.5, 12.0
THETAS = [0.02, 0.05, 0.10, 0.20]
TAUS = [2.0, 1.5]
FOLLOW_MARGIN, A_HARD = 2.0, 3.0
ROOT = "data/raw/v2xseq/full/extracted/V2X-Seq-TFD"
CACHE = "data/raw/v2xseq/results/cache"
LEADS_DIR = "data/raw/v2xseq/results/leaders_v4_leads"
PREDS = "results/v2xseq/leadtime_dz"
RAW_OUT = "data/raw/v2xseq/results/v11"
_G = {}


# ------------------------------------------------------------------------------------------------ traj
def _init(shift):
    _G["lanes"] = v2xseq.lane_index(ROOT, CACHE)
    _G["segs"] = {s.name: s for s in v2xseq.segments(ROOT, CACHE)}
    _G["shift"] = v2xseq.load_shift_table(shift) if shift else None


def _segment(item):
    name, rows = item
    rec = v2xseq.load_recording(ROOT, _G["segs"][name], _G["lanes"], shift_table=_G["shift"])
    aps = build_approaches(rec.map.stop_lines())
    t = rec.tracks
    out = []
    for e in rows:
        k = t[t["track_id"] == e["track_id"]].sort_values("frame_id")
        fr = k["frame_id"].values
        fx = k["x"].values + 0.5 * k["length"].values * np.cos(k["heading_rad"].values)
        fy = k["y"].values + 0.5 * k["length"].values * np.sin(k["heading_rad"].values)
        d, _ = project(aps[int(e["sid"])], fx, fy)
        v, a = np.hypot(k["vx"].values, k["vy"].values), k["a_lon"].values
        after = fr >= e["k0"]
        ts = (fr - e["f_yellow"]) * v2xseq.DT
        halt = np.where(after & (v < events.V_STOP) & (d > 0))[0]
        cross = np.where((d[:-1] > 0) & (d[1:] <= 0) & after[1:])[0] + 1   # first frame at/past the line
        ih = halt[0] if len(halt) else None
        ic = cross[0] if len(cross) else None
        r = dict(segment=name, track_id=e["track_id"], f_yellow=e["f_yellow"], red_at_s=(e["f_red"] - e["f_yellow"]) * v2xseq.DT,
                 t_end_s=float(ts[-1]), n_after=int(after.sum()),
                 t_halt_s=float(ts[ih]) if ih is not None else np.nan, d_halt_m=float(d[ih]) if ih is not None else np.nan,
                 t_cross_s=float(ts[ic]) if ic is not None else np.nan, v_cross=float(v[ic]) if ic is not None else np.nan,
                 min_v_upstream_after=float(v[after & (d > 0)].min()) if (after & (d > 0)).any() else np.nan,
                 min_d_after=float(d[after].min()))
        r["halt_before_cross"] = bool(ih is not None and (ic is None or ih < ic))
        r["halt_then_cross"] = bool(ih is not None and ic is not None and ih < ic)
        if r["halt_then_cross"]:
            seg = slice(ih, ic + 1)
            r["v_max_halt_to_cross"], r["dwell_s"] = float(v[seg].max()), float(ts[ic] - ts[ih])
            r["v_min_halt_to_cross"] = float(v[seg].min())
        else:
            r["v_max_halt_to_cross"] = r["dwell_s"] = r["v_min_halt_to_cross"] = np.nan
        # halt position at the lowest speed of the first halt run (where the car actually stood)
        if ih is not None:
            run = ih
            while run + 1 < len(v) and v[run + 1] < events.V_STOP and d[run + 1] > 0:
                run += 1
            r["t_halt_end_s"], r["d_halt_min_m"] = float(ts[run]), float(d[ih:run + 1].min())
        else:
            r["t_halt_end_s"] = r["d_halt_min_m"] = np.nan
        out.append(r)
    return out


def traj(args):
    from v2xseq_eval import load_tables
    ev, _, _ = load_tables(LEADS_DIR)
    os.makedirs(RAW_OUT, exist_ok=True)
    items = [(s, g[["track_id", "sid", "k0", "f_yellow", "f_red"]].to_dict("records")) for s, g in ev.groupby("segment")]
    print(f"traj: {len(ev)} leaders in {len(items)} segments", flush=True)
    rows = []
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.shift_table,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_segment, items, chunksize=8)):
            rows.extend(r)
            if i % 200 == 0:
                print(f"  {i}/{len(items)}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(f"{RAW_OUT}/post_onset.csv", index=False)
    print(f"traj: wrote {len(T)} rows -> {RAW_OUT}/post_onset.csv", flush=True)


def _load_post(ev):
    T = pd.read_csv(f"{RAW_OUT}/post_onset.csv")
    m = ev[["segment", "track_id", "label", "cross_phase", "t_cross_s", "yellow_s"]].merge(T, on=["segment", "track_id"], how="left")
    assert len(m) == len(ev)   # rows without a post-onset summary (t_r NaN leaders, outside the 1,867) stay NaN
    return m


# ------------------------------------------------------------------------------------------------ q1
def q1(args, out):
    from run_benchmark import at
    from v2xseq_eval import load_tables
    ev, F, _ = load_tables(LEADS_DIR)
    T = _load_post(ev)
    band = ((F["tt"] >= DZ_LO) & (F["tt"] <= DZ_HI)).values
    grace = events.GO_GRACE_S
    tc = T["t_cross_s"].values if "t_cross_s_y" not in T else T["t_cross_s_y"].values
    tcx = T["t_cross_s_y"].values if "t_cross_s_y" in T else T["t_cross_s"].values
    red = T["red_at_s"].values
    htc = T["halt_then_cross"].values.astype(bool)
    phase = np.where(~htc, "none", np.where(tcx < red, "yellow", np.where(tcx <= red + grace, "red_within_3s", "red_after_3s")))
    T["cross_phase_traj"] = phase
    n = len(ev)
    rows = []
    for name, m in [("all_1867", np.ones(n, bool)), ("band_509", band)]:
        for ph in ["yellow", "red_within_3s", "red_after_3s"]:
            k = m & htc & (phase == ph)
            rows.append(dict(set=name, n_set=int(m.sum()), crossing=ph, n=int(k.sum()), share=float(k.sum() / m.sum()),
                             code_label_stop=int((ev["label"].values[k] == 0).sum()), code_label_go=int((ev["label"].values[k] == 1).sum()),
                             code_phase_halt=int((ev["cross_phase"].values[k] == "halt").sum()),
                             dwell_median_s=float(np.median(T["dwell_s"].values[k])) if k.any() else np.nan,
                             v_max_between_median=float(np.median(T["v_max_halt_to_cross"].values[k])) if k.any() else np.nan,
                             d_halt_median_m=float(np.median(T["d_halt_m"].values[k])) if k.any() else np.nan))
        k = m & htc & (phase != "red_after_3s")
        rows.append(dict(set=name, n_set=int(m.sum()), crossing="yellow_or_red_within_3s", n=int(k.sum()), share=float(k.sum() / m.sum()),
                         code_label_stop=int((ev["label"].values[k] == 0).sum()), code_label_go=int((ev["label"].values[k] == 1).sum()),
                         code_phase_halt=int((ev["cross_phase"].values[k] == "halt").sum()),
                         dwell_median_s=float(np.median(T["dwell_s"].values[k])) if k.any() else np.nan,
                         v_max_between_median=float(np.median(T["v_max_halt_to_cross"].values[k])) if k.any() else np.nan,
                         d_halt_median_m=float(np.median(T["d_halt_m"].values[k])) if k.any() else np.nan))
        k = m & htc
        rows.append(dict(set=name, n_set=int(m.sum()), crossing="any_after_halt", n=int(k.sum()), share=float(k.sum() / m.sum()),
                         code_label_stop=int((ev["label"].values[k] == 0).sum()), code_label_go=int((ev["label"].values[k] == 1).sum()),
                         code_phase_halt=int((ev["cross_phase"].values[k] == "halt").sum()),
                         dwell_median_s=float(np.median(T["dwell_s"].values[k])) if k.any() else np.nan,
                         v_max_between_median=float(np.median(T["v_max_halt_to_cross"].values[k])) if k.any() else np.nan,
                         d_halt_median_m=float(np.median(T["d_halt_m"].values[k])) if k.any() else np.nan))
    Q = pd.DataFrame(rows)
    Q.to_csv(f"{out}/q1_label_boundary.csv", index=False)
    # consistency with the labels: every halt-before-cross leader must be label 0 / phase 'halt'
    hb = T["halt_before_cross"].values.astype(bool)
    chk = dict(n=n, n_stop=int((ev.label == 0).sum()), n_go=int((ev.label == 1).sum()),
               halt_before_cross=int(hb.sum()), halt_before_cross_label0=int(((ev.label.values == 0) & hb).sum()),
               label0_without_traj_halt=int(((ev.label.values == 0) & ~hb).sum()),
               halt_then_cross=int(htc.sum()), halt_no_cross=int((hb & ~htc).sum()),
               cross_no_halt_before=int((~hb & T["t_cross_s_y" if "t_cross_s_y" in T else "t_cross_s"].notna().values).sum()))
    pd.Series(chk).to_csv(f"{out}/q1_consistency.csv", header=["value"])
    T.assign(band=band).to_csv(f"{RAW_OUT}/q1_events.csv", index=False)
    with pd.option_context("display.width", 250):
        print(Q.round(4).to_string(index=False), flush=True)
        print(pd.Series(chk).to_string(), flush=True)


# ------------------------------------------------------------------------------------------------ q2
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


def q2(args, out):
    from v2xseq_eval import load_tables, loio_folds
    base = pd.read_csv(f"{LEADS_DIR}/events_through.csv")
    base_keys = set(zip(base["segment"], base["track_id"].astype(int)))
    tabs = [("q10 (baseline, leaders_v4)", "data/raw/v2xseq/results/leaders_v4", "results/v2xseq/stopline_shift.csv"),
            ("no shift", "data/raw/v2xseq/results/leaders_v11_noshift", ""),
            ("q05", "data/raw/v2xseq/results/leaders_v11_q05", "results/v2xseq/stopline_shift_q05.csv"),
            ("q20", "data/raw/v2xseq/results/leaders_v11_q20", "results/v2xseq/stopline_shift_q20.csv")]
    rows, folds_rows = [], []
    for name, path, shift in tabs:
        if not os.path.exists(f"{path}/events_through.csv"):
            print(f"[q2] missing {path}, skipped", flush=True)
            continue
        ev, F, S = load_tables(path)
        y = ev["label"].values.astype(int)
        tt = F["tt"].values
        cyc = bench_eval.cycles(ev)
        band = (tt >= DZ_LO) & (tt <= DZ_HI)
        keep = tt <= TT_MAX
        t = _t50(tt[keep], y[keep])
        bs = np.array([_t50(tt[ix], y[ix]) for ix in _cyc_draws(cyc, keep, args.n_boot)])
        folds = loio_folds(ev, 1)
        P = np.full(len(y), np.nan)
        ft = []
        for site, trn, te in folds:
            k = trn[tt[trn] <= TT_MAX]
            ts = _t50(tt[k], y[k])
            P[te] = (tt[te] < ts).astype(float)
            ft.append(ts)
            folds_rows.append(dict(table=name, test_site=site, n_test=len(te), t50_train=ts))
        r = dict(table=name, shift_table=shift, n=len(y), n_go=int(y.sum()), n_tt_le_12=int(keep.sum()), n_band=int(band.sum()), n_band_go=int(y[band].sum()),
                 same_events_as_baseline=int(sum((s, int(i)) in base_keys for s, i in zip(ev["segment"], ev["track_id"]))),
                 d0_median=float(np.median(F["d"].values)), t50=t, t50_lo=float(np.nanpercentile(bs, 2.5)), t50_hi=float(np.nanpercentile(bs, 97.5)),
                 t50_boot_undefined=int(np.isnan(bs).sum()), folds=len(folds), fold_t50_min=float(np.nanmin(ft)), fold_t50_max=float(np.nanmax(ft)))
        for sub, m in [("all", np.ones(len(y), bool)), ("band", band)]:
            b = bench_eval.bootstrap(ev, {"ruleA_calibrated": (P, P)}, m, args.n_boot)
            b = b[b.metric == "macro_f1"].iloc[0]
            r.update({f"f1_{sub}": b.value, f"f1_{sub}_lo": b.ci_lo, f"f1_{sub}_hi": b.ci_hi})
        rows.append(r)
        print(f"[q2] {name}: n {len(y)} t50 {t:.4f} [{r['t50_lo']:.4f}, {r['t50_hi']:.4f}] f1 all {r['f1_all']:.4f} band {r['f1_band']:.4f}", flush=True)
    Q = pd.DataFrame(rows)
    Q.to_csv(f"{out}/q2_stopline_sensitivity.csv", index=False)
    pd.DataFrame(folds_rows).to_csv(f"{out}/q2_fold_t50.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(Q.round(4).to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ q3 / q4
def _staged():
    from staged_dz import load
    ev, FS, band0, crossed, usable = load(LEADS_DIR)
    y = ev["label"].values.astype(int)
    P = np.column_stack([pd.read_csv(f"{PREDS}/preds_lead{L:+.1f}.csv")["mlp_dva"].values for L in LEADS])
    X = np.column_stack([crossed[L] for L in LEADS])
    ib = np.where(band0)[0]
    assert np.isfinite(P[ib, 0]).all()
    return ev, FS, y, P, X, ib


def q3(args, out):
    from staged_dz import at_lead, decide, stats
    ev, FS, y, P, X, ib = _staged()
    Pm, Xm, yb = P[ib], X[ib], y[ib]
    pol = {f"theta={th}": decide(Pm, Xm, th) for th in THETAS}
    pol["all@2.0s"] = at_lead(Pm, Xm, len(LEADS) - 1)
    cyc = bench_eval.cycles(ev)[ib]
    uc = np.unique(cyc)
    mem = {c: np.where(cyc == c)[0] for c in uc}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([mem[c] for c in rng.choice(uc, len(uc))]) for _ in range(args.n_boot)]
    boot = {p: pd.DataFrame([stats(yb[d], yh[d], tc[d]) for d in draws]) for p, (yh, tc) in pol.items()}
    rows = []
    for p, (yh, tc) in pol.items():
        pt = stats(yb, yh, tc)
        r = dict(model="mlp_dva", policy=p, n=len(ib), n_go=int(yb.sum()), n_stop=int((yb == 0).sum()))
        for k in ("t_mean", "macro_f1", "risky", "conservative"):
            lo, hi = np.nanpercentile(boot[p][k], [2.5, 97.5])
            r.update({k: pt[k], f"{k}_lo": lo, f"{k}_hi": hi})
        r["t_median"] = pt["t_median"]
        r["n_risky"] = int(np.sum((yb == 0) & (yh == 1)))
        r["n_conservative"] = int(np.sum((yb == 1) & (yh == 0)))
        r["open_until_2s"] = float(np.mean(tc >= 2.0))
        r["n_open_until_2s"] = int(np.sum(tc >= 2.0))
        r.update({f"committed_at_{L:.1f}s": float(np.mean(tc == L)) for L in LEADS})
        r["d_macro_f1_vs_2s"] = pt["macro_f1"] - stats(yb, *pol["all@2.0s"])["macro_f1"]
        r["d_macro_f1_vs_2s_lo"], r["d_macro_f1_vs_2s_hi"] = np.nanpercentile(boot[p]["macro_f1"] - boot["all@2.0s"]["macro_f1"], [2.5, 97.5])
        rows.append(r)
    Q = pd.DataFrame(rows)
    Q.to_csv(f"{out}/q3_theta_sensitivity.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(Q.round(4).to_string(index=False), flush=True)


def _q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=int(len(x)), median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                min=float(x.min()), max=float(x.max())) if len(x) else dict(n=0)


def q4(args, out):
    from staged_dz import at_lead, decide
    ev, FS, y, P, X, ib = _staged()
    T = _load_post(ev)
    stop = ib[y[ib] == 0]
    assert T["t_end_s"].iloc[stop].notna().all() and T["t_halt_s"].iloc[stop].notna().all()
    pols = {"theta=0.05": decide(P[ib], X[ib], 0.05), "all@2.0s": at_lead(P[ib], X[ib], len(LEADS) - 1)}
    rows, per = [], []
    for pname, (yh, tc) in pols.items():
        pos = {e: i for i, e in enumerate(ib)}
        for e in stop:
            i = pos[e]
            L = float(tc[i])
            f = FS[L][0].iloc[e]
            d, v, a = float(f["d"]), float(f["v"]), float(f["a"])
            rec = dict(policy=pname, ev=e, segment=ev["segment"].iloc[e], track_id=ev["track_id"].iloc[e], t_c=L, yhat=int(yh[i]),
                       clamped=int(f["clamped"]), d_tc=d, v_tc=v, a_tc=a, t_halt_s=float(T["t_halt_s"].iloc[e]), d_halt_m=float(T["d_halt_m"].iloc[e]),
                       d_halt_min_m=float(T["d_halt_min_m"].iloc[e]), t_to_halt=float(T["t_halt_s"].iloc[e]) - L)
            for tau in TAUS:
                den = 2 * (tau * v + d - FOLLOW_MARGIN)
                den2 = 2 * (tau * v + d - float(T["d_halt_m"].iloc[e]) - FOLLOW_MARGIN)
                rec[f"a_req_tau{tau}"] = v ** 2 / den if den > 0 else np.inf
                rec[f"a_req_actual_tau{tau}"] = v ** 2 / den2 if den2 > 0 else np.inf
            per.append(rec)
    E = pd.DataFrame(per)
    E.to_csv(f"{RAW_OUT}/q4_events.csv", index=False)
    for pname in pols:
        for sub, m in [("all_actual_stops", E.policy == pname), ("committed_stop_only", (E.policy == pname) & (E.yhat == 0))]:
            S = E[m & (E.clamped == 0)]
            base = dict(policy=pname, subset=sub, n=int(m.sum()), n_track_ended_before_tc=int((E[m].clamped == 1).sum()), n_used=len(S),
                        t_c_mean=float(E[m].t_c.mean()), n_halt_before_tc=int((S.t_to_halt < 0).sum()))
            for col, lab in [("d_tc", "d_tc_m"), ("t_to_halt", "t_to_halt_s"), ("a_tc", "a_tc_mps2"), ("v_tc", "v_tc_mps")]:
                base.update({f"{lab}_{k}": v for k, v in _q(S[col]).items() if k != "n"})
            rows.append(base)
            for tau in TAUS:
                for kind in ("a_req", "a_req_actual"):
                    x = S[f"{kind}_tau{tau}"].values
                    fin = np.isfinite(x)
                    r = dict(policy=pname, subset=sub, n=int(m.sum()), n_used=len(S), quantity=f"{kind}_tau{tau}", n_infinite=int((~fin).sum()),
                             **{k: v for k, v in _q(x[fin]).items() if k != "n"}, share_gt_3=float(np.mean(x > A_HARD)), n_gt_3=int(np.sum(x > A_HARD)))
                    rows.append(r)
    Q = pd.DataFrame(rows)
    Q.to_csv(f"{out}/q4_commit_margin.csv", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 60):
        print(Q.round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="traj,q1,q2,q3,q4")
    ap.add_argument("--out", default="results/v2xseq/v11")
    ap.add_argument("--shift-table", default="results/v2xseq/stopline_shift.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(RAW_OUT, exist_ok=True)
    for t in args.tasks.split(","):
        print(f"=== {t}", flush=True)
        {"traj": lambda a: traj(a), "q1": lambda a: q1(a, a.out), "q2": lambda a: q2(a, a.out), "q3": lambda a: q3(a, a.out), "q4": lambda a: q4(a, a.out)}[t](args)


if __name__ == "__main__":
    main()

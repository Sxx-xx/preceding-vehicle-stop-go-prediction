"""R. Follower-policy replay on the V2X-Seq dilemma-zone leaders (509): a virtual AV behind each leader, four policies.

Simulator and policy definitions: src/dzp/policy_replay.py. Leader trajectories: raw 10 Hz tracks from yellow
onset (segment loader of scripts/v2xseq_v11.py), extended at constant speed beyond the track end (count and mean
extrapolated time reported). Commit time t_c and decision for P3: scripts/staged_dz.py decide() on the MLP
out-of-fold P(go) of results/v2xseq/leadtime_dz (the per-event values behind results/v2xseq/staged/staged.csv).

Metrics per event (horizon: leader-stop events until the leader halt + 3 s, leader-go events until the AV crosses):
  go events   delay of the AV's stop-line crossing vs P1 (s), minimum speed, maximum deceleration,
              'unnecessary braking' (a < -1 m/s^2 for >= 0.5 s)
  stop events maximum deceleration, minimum gap, minimum TTC, collision (g <= 0), hard braking (a < -3 m/s^2)
Summary: mean / median / [25, 75] per policy, cycle bootstrap (1,000) 95 % CI of the means and of P3 - P2, P3 - P4.
Sensitivity: tau in {1.0, 1.5, 2.0}, a_av in {2.0, 3.0}, a_l in {1.5, 2.0, 3.0}, theta in {0.05, 0.10}.

Outputs: results/v2xseq/policy/ (events_default.csv, summary_default.csv, sensitivity.csv, extrapolation.csv),
results/figures/fig_r_policy_replay.png; leader arrays under data/raw/v2xseq/results/policy/ (gitignored).

  .venv/bin/python scripts/v2xseq_policy.py [--tasks traj,run]
  .venv/bin/python scripts/v2xseq_policy.py --tasks run --clean-d     # robustness: leader d non-increasing (*_cleand files)
"""
import argparse
import itertools
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval, v2xseq
from dzp import policy_replay as pr
from dzp.geometry import build_approaches, project
from v2xseq_v11 import LEADS_DIR, PREDS, ROOT, _G, _init

RAW = "data/raw/v2xseq/results/policy"
OUT = "results/v2xseq/policy"
FIG = "results/figures/fig_r_policy_replay.png"
LEADS = [0.0, 0.5, 1.0, 1.5, 2.0]
DEFAULT = dict(tau=1.5, a_av=3.0, a_l=2.0, theta=0.05)
GRID = dict(tau=[1.0, 1.5, 2.0], a_av=[2.0, 3.0], a_l=[1.5, 2.0, 3.0], theta=[0.05, 0.10])
T = int(round(pr.T_CAP / pr.DT)) + 1
P_SHORT = {"P1_naive": "P1", "P2_conservative": "P2", "P3_staged": "P3", "P4_oracle": "P4"}


# ------------------------------------------------------------------------------------------------ leader arrays
def _traj_segment(item):
    name, rows = item
    rec = v2xseq.load_recording(ROOT, _G["segs"][name], _G["lanes"], shift_table=_G["shift"])
    aps = build_approaches(rec.map.stop_lines())
    t = rec.tracks
    out = []
    for e in rows:
        k = t[t["track_id"] == e["track_id"]].sort_values("frame_id")
        fr = k["frame_id"].values.astype(int)
        fx = k["x"].values + 0.5 * k["length"].values * np.cos(k["heading_rad"].values)
        fy = k["y"].values + 0.5 * k["length"].values * np.sin(k["heading_rad"].values)
        d, _ = project(aps[int(e["sid"])], fx, fy)
        v = np.hypot(k["vx"].values, k["vy"].values)
        off = fr - int(e["k0"])
        m = off >= 0
        off, d, v = off[m], d[m], v[m]
        full = np.arange(0, off[-1] + 1)
        d = np.interp(full, off, d)          # fills frame gaps (none expected) by linear interpolation
        v = np.interp(full, off, v)
        out.append(dict(ev=e["ev"], d=d, v=v, length=float(k["length"].median()), n_obs=len(full), n_gap=int(len(full) - len(off))))
    return out


def traj(args):
    from staged_dz import load
    ev, FS, band0, crossed, usable = load(LEADS_DIR)
    ib = np.where(band0)[0]
    os.makedirs(RAW, exist_ok=True)
    sub = ev.iloc[ib].assign(ev=ib)
    items = [(s, g[["ev", "track_id", "sid", "k0"]].to_dict("records")) for s, g in sub.groupby("segment")]
    print(f"traj: {len(ib)} band leaders in {len(items)} segments", flush=True)
    res = []
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.shift_table,)) as pool:
        for r in pool.imap_unordered(_traj_segment, items, chunksize=4):
            res.extend(r)
    res.sort(key=lambda r: r["ev"])
    D, V = np.full((len(res), T), np.nan), np.full((len(res), T), np.nan)
    n_obs = np.array([r["n_obs"] for r in res])
    for i, r in enumerate(res):
        D[i], V[i], _ = pr.extend_leader(r["d"], r["v"], T)
    np.savez_compressed(f"{RAW}/leader_arrays.npz", ev=np.array([r["ev"] for r in res]), d=D, v=V, n_obs=n_obs,
                        length=np.array([r["length"] for r in res]), n_gap=np.array([r["n_gap"] for r in res]))
    print(f"traj: saved {len(res)} leaders, frame gaps filled {int(sum(r['n_gap'] for r in res))}", flush=True)


# ------------------------------------------------------------------------------------------------ run
def _decisions(theta):
    from staged_dz import decide, load
    ev, FS, band0, crossed, usable = load(LEADS_DIR)
    y = ev["label"].values.astype(int)
    P = np.column_stack([pd.read_csv(f"{PREDS}/preds_lead{L:+.1f}.csv")["mlp_dva"].values for L in LEADS])
    X = np.column_stack([crossed[L] for L in LEADS])
    ib = np.where(band0)[0]
    yh, tc = decide(P[ib], X[ib], theta)
    return ev, ib, y[ib], yh, tc


def _draws(cyc, n_boot, seed=0):
    uc, inv = np.unique(cyc, return_inverse=True)
    mem = [np.where(inv == k)[0] for k in range(len(uc))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([mem[k] for k in rng.integers(0, len(uc), len(uc))]) for _ in range(n_boot)]


def _q(x):
    x = np.asarray(x, float)
    f = x[~np.isnan(x)]          # NaN = undefined for the event (delay when the AV never crosses); inf (TTC) is kept
    return dict(n=int(len(x)), n_finite=int(np.isfinite(x).sum()), mean=float(f[np.isfinite(f)].mean()) if np.isfinite(f).any() else np.nan,
                median=float(np.median(f)) if len(f) else np.nan, q25=float(np.percentile(f, 25)) if len(f) else np.nan,
                q75=float(np.percentile(f, 75)) if len(f) else np.nan, min=float(f.min()) if len(f) else np.nan, max=float(f.max()) if len(f) else np.nan)


def run_combo(Z, y, yh, tc, t_halt, tau, a_av, a_l):
    res = {}
    for p in pr.POLICIES:
        res[p] = pr.simulate(Z["d"], Z["v"], Z["length"], y, tc, yh, t_halt, p, tau=tau, a_av=a_av, a_l=a_l)
    for p in pr.POLICIES:
        res[p]["delay"] = res[p]["t_cross"] - res["P1_naive"]["t_cross"]
    return res


GO_METRICS = [("delay", "mean"), ("v_min", "mean"), ("max_decel", "mean"), ("unnecessary_brake", "share")]
STOP_METRICS = [("max_decel", "mean"), ("g_min", "mean"), ("ttc_min", "median"), ("collision", "share"), ("hard_brake", "share"), ("emergency", "share")]


def summarise(res, y, draws, with_ci=True):
    go, st = y == 1, y == 0
    rows = []
    for subset, m, metrics in (("go", go, GO_METRICS), ("stop", st, STOP_METRICS)):
        idx = np.where(m)[0]
        for name, kind in metrics:
            vals = {p: np.asarray(res[p][name], float) for p in pr.POLICIES}
            for p in pr.POLICIES:
                x = vals[p][idx]
                r = dict(subset=subset, metric=name, policy=P_SHORT[p], **_q(x))
                if name == "ttc_min":
                    r["share_lt_2s"] = float(np.mean(x < 2.0))
                if with_ci:
                    stat = (lambda a: np.nanmedian(a)) if kind == "median" else (lambda a: np.nanmean(a))
                    bs = np.array([stat(vals[p][d][m[d]]) for d in draws])
                    r.update(ci_stat=kind, ci_lo=float(np.nanpercentile(bs, 2.5)), ci_hi=float(np.nanpercentile(bs, 97.5)))
                rows.append(r)
            for a, b in (("P3_staged", "P2_conservative"), ("P3_staged", "P4_oracle")):
                stat = (lambda z: np.nanmedian(z)) if kind == "median" else (lambda z: np.nanmean(z))
                r = dict(subset=subset, metric=name, policy=f"{P_SHORT[a]}-{P_SHORT[b]}", mean=stat(vals[a][idx]) - stat(vals[b][idx]), n=len(idx))
                if with_ci:
                    bs = np.array([stat(vals[a][d][m[d]]) - stat(vals[b][d][m[d]]) for d in draws])
                    r.update(ci_stat=kind, ci_lo=float(np.nanpercentile(bs, 2.5)), ci_hi=float(np.nanpercentile(bs, 97.5)))
                rows.append(r)
    return pd.DataFrame(rows)


def figure(res, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"P1_naive": "#2a78d6", "P2_conservative": "#eb6834", "P3_staged": "#1baf7a", "P4_oracle": "#eda100"}
    ls = {"P1_naive": "-", "P2_conservative": "--", "P3_staged": "-", "P4_oracle": ":"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    go, st = y == 1, y == 0
    for ax, subset, mask, name, pols, xlabel in ((axes[0], "go", go, "delay", pr.POLICIES[1:], "AV crossing delay vs P1 (s), leader-go events"),
                                                   (axes[1], "stop", st, "max_decel", pr.POLICIES, "AV maximum deceleration (m/s²), leader-stop events")):
        for p in pols:
            x = np.sort(np.asarray(res[p][name], float)[mask])
            x = x[np.isfinite(x)]
            yy = np.arange(1, len(x) + 1) / len(x)
            ax.step(x, yy, where="post", color=col[p], ls=ls[p], lw=2, label=f"{P_SHORT[p]} {p.split('_')[1]}")
            ax.annotate(P_SHORT[p], (x[-1], 1.0), xytext=(4, -10 * list(pols).index(p)), textcoords="offset points", fontsize=8, color="#444")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Cumulative share of events", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        nf = int(np.isfinite(np.asarray(res[pols[0]][name], float)[mask]).sum())
        ax.set_title(f"n = {nf} of {int(mask.sum())} leader-{subset} events" if nf < mask.sum() else f"n = {int(mask.sum())} leader-{subset} events", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=300)
    plt.close(fig)


def run(args):
    os.makedirs(OUT, exist_ok=True)
    Z = dict(np.load(f"{RAW}/leader_arrays.npz"))
    tag = ""
    if args.clean_d:   # robustness variant: the leader never moves backwards (running minimum of d removes position jumps)
        Z["d"] = np.minimum.accumulate(Z["d"], axis=1)
        tag = "_cleand"
    ev, ib, y, _, _ = _decisions(DEFAULT["theta"])
    assert (Z["ev"] == ib).all()
    t_halt = np.array([pr.halt_time(Z["d"][i, : Z["n_obs"][i]], Z["v"][i, : Z["n_obs"][i]]) for i in range(len(ib))])
    assert np.isfinite(t_halt[y == 0]).all(), "every leader-stop event must have an observed halt"
    cyc = bench_eval.cycles(ev)[ib]
    draws = _draws(cyc, args.n_boot)
    dec = {th: _decisions(th)[3:] for th in GRID["theta"]}
    # ---- default
    yh, tc = dec[DEFAULT["theta"]]
    res = run_combo(Z, y, yh, tc, t_halt, DEFAULT["tau"], DEFAULT["a_av"], DEFAULT["a_l"])
    rows = []
    for p in pr.POLICIES:
        r = res[p]
        for i in range(len(ib)):
            rows.append(dict(policy=P_SHORT[p], ev=int(ib[i]), segment=ev["segment"].iloc[ib[i]], track_id=int(ev["track_id"].iloc[ib[i]]), label=int(y[i]),
                             t_c=float(tc[i]), yhat=int(yh[i]), t_halt=float(t_halt[i]), t_end=float(r["t_end"][i]), t_cross_av=float(r["t_cross"][i]),
                             delay=float(r["delay"][i]), v_min=float(r["v_min"][i]), max_decel=float(r["max_decel"][i]), g_min=float(r["g_min"][i]),
                             ttc_min=float(r["ttc_min"][i]), collision=bool(r["collision"][i]), hard_brake=bool(r["hard_brake"][i]),
                             unnecessary_brake=bool(r["unnecessary_brake"][i]), emergency=bool(r["emergency"][i]),
                             frames_used=int(r["frames"][i]), frames_extrapolated=int(max(r["frames"][i] - Z["n_obs"][i], 0))))
    E = pd.DataFrame(rows)
    E.to_csv(f"{OUT}/events_default{tag}.csv", index=False)
    S = summarise(res, y, draws)
    S.to_csv(f"{OUT}/summary_default{tag}.csv", index=False)
    ex = E.groupby("policy").agg(n=("ev", "size"), n_extrapolated=("frames_extrapolated", lambda s: int((s > 0).sum())),
                                 mean_extrapolated_s=("frames_extrapolated", lambda s: float(s[s > 0].mean() * pr.DT) if (s > 0).any() else 0.0),
                                 max_extrapolated_s=("frames_extrapolated", lambda s: float(s.max() * pr.DT)),
                                 go_not_crossed=("t_cross_av", lambda s: int(s[E.loc[s.index, "label"] == 1].isna().sum())))
    ex.to_csv(f"{OUT}/extrapolation{tag}.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(ex.to_string(), flush=True)
        print(S.round(4).to_string(index=False), flush=True)
    if args.clean_d:
        return
    figure(res, y)
    # ---- sensitivity
    sens = []
    for tau, a_av, a_l, th in itertools.product(GRID["tau"], GRID["a_av"], GRID["a_l"], GRID["theta"]):
        yh2, tc2 = dec[th]
        r2 = run_combo(Z, y, yh2, tc2, t_halt, tau, a_av, a_l)
        s2 = summarise(r2, y, draws, with_ci=True)
        keep = s2[((s2.subset == "go") & (s2.metric.isin(["delay", "unnecessary_brake"]))) | ((s2.subset == "stop") & (s2.metric.isin(["max_decel", "hard_brake", "collision", "g_min"])))]
        for r in keep.itertuples():
            sens.append(dict(tau=tau, a_av=a_av, a_l=a_l, theta=th, subset=r.subset, metric=r.metric, policy=r.policy, mean=r.mean, median=r.median,
                             q25=r.q25, q75=r.q75, ci_lo=r.ci_lo, ci_hi=r.ci_hi))
        print(f"sens tau {tau} a_av {a_av} a_l {a_l} theta {th} done", flush=True)
    pd.DataFrame(sens).to_csv(f"{OUT}/sensitivity.csv", index=False)
    # ---- headline
    g = S[(S.subset == "go") & (S.metric == "delay")].set_index("policy")
    st = S[(S.subset == "stop")].set_index(["metric", "policy"])
    print(f"HEADLINE go delay mean P2 {g.loc['P2','mean']:.4f} P3 {g.loc['P3','mean']:.4f} P4 {g.loc['P4','mean']:.4f}; "
          f"P3-P2 {g.loc['P3-P2','mean']:+.4f} [{g.loc['P3-P2','ci_lo']:+.4f}, {g.loc['P3-P2','ci_hi']:+.4f}] "
          f"(P2 mean is ~0 at the defaults, so no % change); P3-P4 {g.loc['P3-P4','mean']:+.4f} [{g.loc['P3-P4','ci_lo']:+.4f}, {g.loc['P3-P4','ci_hi']:+.4f}]", flush=True)
    print(f"HEADLINE stop P3 max decel median {st.loc[('max_decel','P3'),'median']:.4f} hard {st.loc[('hard_brake','P3'),'mean']:.4f}; "
          f"P1 collision {st.loc[('collision','P1'),'mean']:.4f} hard {st.loc[('hard_brake','P1'),'mean']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="traj,run")
    ap.add_argument("--shift-table", default="results/v2xseq/stopline_shift.csv")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--clean-d", action="store_true", help="run: default parameters only, with the leader distance made non-increasing")
    args = ap.parse_args()
    for t in args.tasks.split(","):
        print(f"=== {t}", flush=True)
        {"traj": traj, "run": run}[t](args)


if __name__ == "__main__":
    main()

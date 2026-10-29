"""Deployment export: the two onset models fitted on ALL 1,867 V2X-Seq leaders (no folds held out).

  export/t50_params.yaml  univariate logistic regression on d/v at yellow onset (d/v <= TT_MAX, same fit as the
                          t_50 of the D / Q2 sections: sklearn LogisticRegression default L2, C = 1):
                          P(go) = sigmoid(b0 + b1 * d/v), t50 = -b0 / b1
  export/mlp_dva.joblib   sklearn Pipeline(StandardScaler, MLPClassifier(32, 16), alpha = 1e-2) on (d, v, a) at onset,
                          the dzp.bench_models.MLPDVA configuration used for the after-onset update (L / N sections),
                          fitted on all 1,867 leaders at lead 0 (the 23-fold runs fit one model per training fold
                          and per lead)
  export/mlp_dva_t{0.0,0.5,1.0,1.5,2.0}.joblib
                          the same configuration per lead t after onset, on the (d, v, a) at onset + t, with the
                          training set of scripts/leadtime_dz.py (-> N section staged decision): every leader with
                          t_r known whose track is still observed and that has not crossed the line at t (NOT band
                          only; 1,867 / 1,535 / 1,274 / 1,043 / 851). mlp_dva_t0.0 is checked to equal mlp_dva.

  .venv/bin/python scripts/export_models.py
"""
import datetime
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from dzp import bench_eval
from dzp.bench_models import DVA, MLPDVA
from v2xseq_eval import load_tables

LEADS_DIR = "data/raw/v2xseq/results/leaders_v4_leads"
OUT = "export"
TT_MAX, DZ_LO, DZ_HI = 12.0, 2.5, 5.5
EXPECTED_T50 = 2.673873           # results/v2xseq/v11/q2_stopline_sensitivity.csv, 10 % row (same data, same fit)


def main():
    os.makedirs(OUT, exist_ok=True)
    ev, F, S = load_tables(LEADS_DIR)
    y = ev["label"].values.astype(int)
    assert len(ev) == 1867 and int(y.sum()) == 1292, (len(ev), int(y.sum()))
    tt = F["tt"].values
    band = (tt >= DZ_LO) & (tt <= DZ_HI)

    # ---- (1) d/v logistic regression -> t50
    keep = tt <= TT_MAX
    lr = LogisticRegression(max_iter=1000).fit(tt[keep][:, None], y[keep])
    b0, b1 = float(lr.intercept_[0]), float(lr.coef_[0, 0])
    t50 = -b0 / b1
    assert abs(t50 - EXPECTED_T50) < 1e-5, t50
    params = dict(
        model="univariate logistic regression on d/v at yellow onset",
        formula="P(go) = 1 / (1 + exp(-(b0 + b1 * tt))), tt = d / v (s); decision go iff tt < t50",
        b0=b0, b1=b1, t50_s=t50,
        fit="sklearn LogisticRegression(max_iter=1000), default L2 penalty C=1 (as the t_50 of docs/FINDINGS.md D / Q2)",
        training_data=dict(source="V2X-Seq-TFD single-infrastructure, through platoon leaders at yellow onset, calibrated stop lines (results/v2xseq/stopline_shift.csv)",
                           n_leaders=int(len(y)), n_go=int(y.sum()), n_used_tt_le_12s=int(keep.sum()), n_used_go=int(y[keep].sum()),
                           band_2_5_to_5_5_s=dict(n=int(band.sum()), n_go=int(y[band].sum()))),
        note="all 1,867 leaders, no folds held out; the 23-fold LOIO evaluation refits this per training fold (fold t50 range 2.634-2.703 s)",
        exported=str(datetime.date.today()), sklearn=sklearn.__version__, python=platform.python_version())
    with open(f"{OUT}/t50_params.yaml", "w") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

    # ---- (2) MLP(d, v, a) pipeline
    m = MLPDVA().fit(F, S, y)
    pipe = m.est
    sc, clf = pipe.steps[0][1], pipe.steps[1][1]
    assert isinstance(sc, StandardScaler) and isinstance(clf, MLPClassifier)
    assert clf.hidden_layer_sizes == (32, 16) and clf.alpha == 1e-2 and clf.random_state == 0 and m.cols == DVA, (clf.hidden_layer_sizes, clf.alpha)
    joblib.dump(pipe, f"{OUT}/mlp_dva.joblib")
    p = pipe.predict_proba(F[DVA].values)[:, 1]
    yh = (p >= 0.5).astype(int)
    info = dict(file="mlp_dva.joblib", object="sklearn Pipeline(StandardScaler, MLPClassifier)", input_columns=DVA,
                input_units="d: front-bumper distance to the stop line (m, calibrated line); v: speed (m/s); a: longitudinal acceleration (m/s^2); all at the prediction time",
                output="predict_proba[:, 1] = P(go)", hidden_layer_sizes=[32, 16], alpha=clf.alpha, max_iter=clf.max_iter, random_state=clf.random_state,
                n_iter_=int(clf.n_iter_), converged=bool(clf.n_iter_ < clf.max_iter), loss_=float(clf.loss_),
                training_data=dict(n_leaders=int(len(y)), n_go=int(y.sum()), lead_s=0.0),
                training_set_macro_f1=dict(all=float(bench_eval._macro_f1(y, yh)), band=float(bench_eval._macro_f1(y[band], yh[band]))),
                note="fitted on all 1,867 leaders at lead 0 s (no folds held out); the after-onset update in the L / N sections fits "
                     "one such model per training fold (23-fold LOIO) and per lead 0 / 0.5 / 1.0 / 1.5 / 2.0 s",
                exported=str(datetime.date.today()), sklearn=sklearn.__version__, python=platform.python_version())
    with open(f"{OUT}/mlp_dva.yaml", "w") as f:
        yaml.safe_dump(info, f, sort_keys=False, allow_unicode=True)
    print(f"t50 b0 {b0:.6f} b1 {b1:.6f} t50 {t50:.6f} (n {int(keep.sum())})", flush=True)
    print(f"mlp hidden {clf.hidden_layer_sizes} alpha {clf.alpha} iters {clf.n_iter_} loss {clf.loss_:.4f}; train macro-F1 all {info['training_set_macro_f1']['all']:.4f} "
          f"band {info['training_set_macro_f1']['band']:.4f}", flush=True)
    # round trip
    q = joblib.load(f"{OUT}/mlp_dva.joblib").predict_proba(F[DVA].values)[:, 1]
    assert np.allclose(p, q)
    print("round trip ok", flush=True)

    # ---- (3) per-lead MLPs, training set as scripts/leadtime_dz.py (all usable leaders at lead L, not band only)
    from staged_dz import LEADS, load
    ev2, FS, band0, crossed, usable = load(LEADS_DIR)
    y2 = ev2["label"].values.astype(int)
    per_lead = []
    for L in LEADS:
        FL = FS[L][0]
        m_tr = usable[L]                                  # ok0 & finite d & v known & not crossed (d > 0)
        assert int(m_tr.sum()) == {0.0: 1867, 0.5: 1535, 1.0: 1274, 1.5: 1043, 2.0: 851}[L], (L, int(m_tr.sum()))
        mL = MLPDVA().fit(FL.iloc[m_tr], FS[L][1][m_tr], y2[m_tr])
        clfL = mL.est.steps[1][1]
        assert clfL.hidden_layer_sizes == (32, 16) and clfL.alpha == 1e-2 and clfL.random_state == 0
        fn = f"mlp_dva_t{L:.1f}.joblib"
        joblib.dump(mL.est, f"{OUT}/{fn}")
        pL = mL.est.predict_proba(FL[DVA].values[m_tr])[:, 1]
        yhL = (pL >= 0.5).astype(int)
        bL = band0[m_tr]
        per_lead.append(dict(file=fn, lead_s=float(L), input_time=f"onset + {L:.1f} s (d, v, a at that time)", n_train=int(m_tr.sum()), n_train_go=int(y2[m_tr].sum()),
                             n_train_band=int(bL.sum()), n_iter_=int(clfL.n_iter_), converged=bool(clfL.n_iter_ < clfL.max_iter), loss_=float(clfL.loss_),
                             training_set_macro_f1=dict(all=float(bench_eval._macro_f1(y2[m_tr], yhL)), band=float(bench_eval._macro_f1(y2[m_tr][bL], yhL[bL])))))
        if L == 0.0:
            same = all(np.allclose(a, b) for a, b in zip(clf.coefs_, clfL.coefs_)) and all(np.allclose(a, b) for a, b in zip(clf.intercepts_, clfL.intercepts_))
            assert same and np.allclose(sc.mean_, mL.est.steps[0][1].mean_), "mlp_dva_t0.0 differs from mlp_dva"
            print("mlp_dva_t0.0 == mlp_dva (weights, scaler)", flush=True)
        print(f"lead {L:.1f}: n {int(m_tr.sum())} (go {int(y2[m_tr].sum())}) iters {clfL.n_iter_} loss {clfL.loss_:.4f}", flush=True)
    info["per_lead_models"] = per_lead
    info["per_lead_note"] = ("same configuration as mlp_dva.joblib; training set per lead as scripts/leadtime_dz.py (used by the N-section staged decision): "
                             "all leaders with t_r known, track observed and not yet across the line at onset + t, not band only; mlp_dva_t0.0 equals mlp_dva")
    with open(f"{OUT}/mlp_dva.yaml", "w") as f:
        yaml.safe_dump(info, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()

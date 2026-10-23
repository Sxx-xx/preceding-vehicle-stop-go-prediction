"""Stop-line shift per approach for V2X-Seq (see dzp.v2xseq module doc, 'line shift').

Input: a scripts/v2xseq_check.py run made WITHOUT a shift table (approaches.csv + stop_positions.json: per segment
and approach, the minimum front distance of every through vehicle that halted within -15..30 m of the line).
Per site, approaches are clustered by stop-line midpoint (25 m); shift_m = Q10(halt positions) - 1 m, so the
leading halted vehicles end up ~1 m upstream of the line. Approaches with fewer than MIN_HALTS halts get no entry
(the adapter then leaves the line where it is).

  .venv/bin/python scripts/v2xseq_calibrate.py --check data/raw/v2xseq/results/check_v3 --out results/v2xseq/stopline_shift.csv
  --lead-q 0.05 / 0.20 (Q2 sensitivity, docs/FINDINGS.md Q section) -> results/v2xseq/stopline_shift_q05.csv / _q20.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from v2xseq_review import _pos_clusters

MIN_HALTS = 20
LEAD_Q = 0.10
TARGET_M = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lead-q", type=float, default=LEAD_Q, help="halt-position quantile that is put TARGET_M upstream of the line")
    args = ap.parse_args()
    A = pd.read_csv(f"{args.check}/approaches.csv")
    H = pd.read_json(f"{args.check}/stop_positions.json")
    H = H.merge(A[["segment", "sid", "inter", "name", "mid_x", "mid_y"]], on=["segment", "sid", "inter"])
    rows = []
    for site, a in A.groupby("inter"):
        a = a.assign(key=_pos_clusters(a[["mid_x", "mid_y"]].values, a["name"].values))
        h = H[H["inter"] == site].merge(a[["segment", "sid", "key"]], on=["segment", "sid"])
        for key, g in a.groupby("key"):
            d = np.concatenate([np.array(x) for x in h.loc[h["key"] == key, "d"]]) if (h["key"] == key).any() else np.array([])
            q10 = float(np.quantile(d, args.lead_q)) if len(d) >= MIN_HALTS else np.nan
            rows.append(dict(site=site, name=g["name"].iloc[0], mid_x=g["mid_x"].mean(), mid_y=g["mid_y"].mean(), segs=len(g),
                             n_halt=len(d), q10_halt_m=q10, q50_halt_m=float(np.median(d)) if len(d) else np.nan,
                             shift_m=q10 - TARGET_M if np.isfinite(q10) else np.nan))
    T = pd.DataFrame(rows)
    T = T[T["shift_m"].notna()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    T.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    print(T.round(1).to_string(index=False))
    print("entries", len(T), "shift quantiles", np.round(np.quantile(T["shift_m"], [0.1, 0.25, 0.5, 0.75, 0.9]), 1))


if __name__ == "__main__":
    main()

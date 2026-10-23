import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from dzp import policy_replay as pr


def _leader(d0, v0, decel, n=401):
    """Leader from onset: constant speed (decel = 0) or braking at `decel` to a halt, 10 Hz."""
    t = np.arange(n) * pr.DT
    v = np.maximum(v0 - decel * t, 0.0)
    d = d0 - np.cumsum(np.r_[0.0, 0.5 * (v[1:] + v[:-1]) * pr.DT])
    return d[None, :], v[None, :]


def _run(d, v, label, t_c, yhat, policy):
    t_halt = np.array([pr.halt_time(d[0], v[0])])
    return pr.simulate(d, v, np.array([4.5]), np.array([label]), np.array([t_c]), np.array([yhat]), t_halt, policy)


def test_go_label_oracle_has_zero_delay():
    d, v = _leader(30.0, 12.0, 0.0)                      # leader passes at constant speed
    r1, r4 = _run(d, v, 1, 2.0, 1, "P1_naive"), _run(d, v, 1, 2.0, 1, "P4_oracle")
    assert np.isfinite(r1["t_cross"][0]) and r4["t_cross"][0] == r1["t_cross"][0]
    for p in ("P2_conservative", "P3_staged"):           # careful first -> never earlier than the naive AV
        assert _run(d, v, 1, 2.0, 1, p)["t_cross"][0] >= r1["t_cross"][0]


def test_stop_label_no_collision_for_any_policy():
    d, v = _leader(40.0, 12.0, 2.5)                      # leader brakes to a halt 11.2 m before the line
    for p in pr.POLICIES:
        r = _run(d, v, 0, 1.0, 0, p)
        assert not r["collision"][0] and r["g_min"][0] > 0
        assert np.isnan(r["t_cross"][0])                 # the AV stays behind the stopped leader

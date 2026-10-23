"""R. Follower-policy replay: a virtual automated vehicle (AV) behind a dilemma-zone leader, replayed at 10 Hz.

Leader: recorded trajectory from yellow onset (front distance to the stop line d_L > 0 upstream, speed v_L, length
L_L), extended beyond the track end at constant speed. The AV starts at onset with the leader's speed, a time gap
TAU0 behind: net gap g0 = TAU0 v_L + G_MIN, so d_AV = d_L + L_L + g0. g is always the net gap (leader rear bumper to
AV front bumper), collision = g <= 0.

Controller (IDM form, one step per DT): a = A_ACC [1 - (v/v0)^4 - (g*/g)^2], v0 = leader speed at onset, clipped to
[-a_av, +A_ACC]. Modes:
  normal   g* = G_MIN + tau v
  careful  g* = G_MIN + tau v + max(v^2/(2 a_av) - v_L^2/(2 a_l), 0)
  stop     brake with the constant deceleration that stops G_MIN behind the leader's predicted halt position
           (v_L^2/(2 a_l) ahead of the leader now): a = -v^2 / (2 (g + v_L^2/(2 a_l) - G_MIN)), clipped to
           [-a_av, 0], combined with the normal-mode IDM value by min().
Common to every policy: once the leader has crossed the line (d_L <= 0) the mode is normal; and an emergency
brake beyond -a_av, down to -A_EMERG, whenever the constant deceleration needed to match the leader's speed before
contact, (v^2 - v_L^2) / (2 (g - 0.5)), exceeds a_av (this is the 'collision avoidance common to all policies';
it does not use the leader's acceleration). The AV ignores its own time-to-red rule.

Policies (mode over time):
  P1_naive         normal always
  P2_conservative  careful until the leader crosses (then normal); if the leader halts it stays careful
  P3_staged        careful until the commit time t_c, then normal (decision go) or stop (decision stop)
  P4_oracle        from onset: normal if the label is go, stop if it is stop
Horizon per event: leader-stop events until t_halt + 3 s; leader-go events until the AV crosses the line (or T_CAP).
"""
from __future__ import annotations

import numpy as np

DT = 0.1
G_MIN = 2.0
A_ACC = 1.5
A_EMERG = 8.0
TAU0 = 2.0
V_HALT = 1.0
T_CAP = 40.0
POST_HALT_S = 3.0
UNNEC_A, UNNEC_RUN = -1.0, 5     # 'unnecessary braking': a < -1 m/s^2 for >= 0.5 s (5 frames)
HARD_A = -3.0
POLICIES = ("P1_naive", "P2_conservative", "P3_staged", "P4_oracle")
NORMAL, CAREFUL, STOP = 0, 1, 2


def extend_leader(d, v, n_frames):
    """Pad per-frame leader arrays (from onset) to n_frames with constant-speed extrapolation; returns (d, v, n_obs)."""
    d, v = np.asarray(d, float), np.asarray(v, float)
    n_obs = len(d)
    if n_obs >= n_frames:
        return d[:n_frames], v[:n_frames], min(n_obs, n_frames)
    k = np.arange(1, n_frames - n_obs + 1)
    return np.r_[d, d[-1] - v[-1] * DT * k], np.r_[v, np.full(len(k), v[-1])], n_obs


def halt_time(d, v):
    """First time (s from onset) with v < V_HALT and d > 0, or NaN."""
    i = np.where((v < V_HALT) & (d > 0))[0]
    return i[0] * DT if len(i) else np.nan


def _mode(policy, t, label, t_c, yhat, crossed):
    n = len(label)
    if policy == "P1_naive":
        m = np.full(n, NORMAL)
    elif policy == "P2_conservative":
        m = np.full(n, CAREFUL)
    elif policy == "P3_staged":
        m = np.where(t < t_c - 1e-9, CAREFUL, np.where(yhat == 1, NORMAL, STOP))
    elif policy == "P4_oracle":
        m = np.where(label == 1, NORMAL, STOP)
    else:
        raise ValueError(policy)
    return np.where(crossed, NORMAL, m)


def simulate(dL, vL, LL, label, t_c, yhat, t_halt, policy, tau=1.5, a_av=3.0, a_l=2.0):
    """Vectorised over events. dL, vL: (n, T) leader arrays from onset (already extended); LL: (n,) leader length;
    label: (n,) 1 go / 0 stop; t_c, yhat: (n,) P3 commit time and decision; t_halt: (n,) leader halt time (NaN if none).
    Returns a dict of per-event metrics (arrays of length n)."""
    dL, vL = np.asarray(dL, float), np.asarray(vL, float)
    n, T = dL.shape
    label, t_c, yhat, t_halt, LL = (np.asarray(x, float) for x in (label, t_c, yhat, t_halt, LL))
    v0 = np.maximum(vL[:, 0], 1.0)
    vA = vL[:, 0].copy()
    dA = dL[:, 0] + LL + TAU0 * vL[:, 0] + G_MIN
    t_end = np.where(label == 0, np.where(np.isfinite(t_halt), t_halt + POST_HALT_S, T_CAP), T_CAP)
    t_end = np.minimum(t_end, (T - 1) * DT)
    active = np.ones(n, bool)
    a_min = np.zeros(n); v_min = vA.copy(); g_min = np.full(n, np.inf); ttc_min = np.full(n, np.inf)
    t_cross = np.full(n, np.nan); unnec = np.zeros(n, bool); run = np.zeros(n, int)
    frames = np.zeros(n, int); crossed_once = np.zeros(n, bool); hard = np.zeros(n, bool); coll = np.zeros(n, bool)
    emerg = np.zeros(n, bool)
    for k in range(T):
        t = k * DT
        dl, vl = dL[:, k], vL[:, k]
        crossed_once |= dl <= 0
        g = dA - dl - LL
        gc = np.maximum(g, 0.01)
        mode = _mode(policy, t, label, t_c, yhat, crossed_once)
        extra = np.maximum(vA ** 2 / (2 * a_av) - vl ** 2 / (2 * a_l), 0.0)
        gs = G_MIN + tau * vA + np.where(mode == CAREFUL, extra, 0.0)
        a_idm = np.clip(A_ACC * (1 - (vA / v0) ** 4 - (gs / gc) ** 2), -a_av, A_ACC)
        s_av = g + vl ** 2 / (2 * a_l) - G_MIN
        a_stop = np.where(s_av > 0.05, -vA ** 2 / (2 * np.maximum(s_av, 0.05)), -a_av)
        a_stop = np.clip(a_stop, -a_av, 0.0)
        a_norm = np.clip(A_ACC * (1 - (vA / v0) ** 4 - ((G_MIN + tau * vA) / gc) ** 2), -a_av, A_ACC)
        a = np.where(mode == STOP, np.minimum(a_stop, a_norm), a_idm)
        need = np.where(vA > vl, (vA ** 2 - vl ** 2) / (2 * np.maximum(g - 0.5, 0.01)), 0.0)
        em = need > a_av
        a = np.where(em, -np.minimum(need, A_EMERG), a)
        a = np.where((vA <= 0) & (a < 0), 0.0, a)
        # metrics on active events (state at this frame, action a)
        act = active
        emerg |= act & em
        a_min = np.where(act, np.minimum(a_min, a), a_min)
        v_min = np.where(act, np.minimum(v_min, vA), v_min)
        g_min = np.where(act, np.minimum(g_min, g), g_min)
        ttc = np.where(vA > vl + 1e-6, g / np.maximum(vA - vl, 1e-6), np.inf)
        ttc_min = np.where(act, np.minimum(ttc_min, ttc), ttc_min)
        coll |= act & (g <= 0)
        hard |= act & (a < HARD_A)
        run = np.where(act & (a < UNNEC_A), run + 1, 0)
        unnec |= run >= UNNEC_RUN
        frames = np.where(act, k + 1, frames)
        # crossing / horizon
        newly = act & (dA <= 0) & np.isnan(t_cross)
        t_cross[newly] = t
        active &= ~((label == 1) & (dA <= 0))
        active &= t < t_end - 1e-9
        if not active.any():
            break
        v_new = np.maximum(vA + a * DT, 0.0)
        dA = dA - 0.5 * (vA + v_new) * DT
        vA = v_new
    return dict(t_cross=t_cross, a_min=a_min, max_decel=-a_min, v_min=v_min, g_min=g_min, ttc_min=ttc_min,
                collision=coll, hard_brake=hard, unnecessary_brake=unnec, emergency=emerg, frames=frames, t_end=t_end)

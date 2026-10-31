"""
Digital-twin fault flagging and isolation.

The twin runs in sim.py:
    nu_hat(k+1) = nu_hat(k) + dt * f_nom(nu_meas, eta_meas, B * T(n_cmd), vc_hat) + (dt / T_o) (nu_meas - nu_hat)
    vc_hat(k+1) = vc_hat(k) + dt * gamma * R (nu_meas - nu_hat)_(1:3)
with the nominal model and the assumption that every thruster delivers its commanded thrust.
Innovation e = nu_meas - nu_hat, low-passed (0.5 s).  Here:
  * slow baseline b (EWMA 10 s, frozen once flagged) removes residual steady bias,
  * residual r = e - b; per-axis thresholds theta = 1.25 x max |r| over fault-free CALIBRATION runs only,
  * persistence: the fault is DECLARED when any axis has exceeded its threshold continuously for 0.5 s;
    the reported detection delay is declaration time minus fault onset (it therefore includes the 0.5 s window),
  * isolation (single-fault hypothesis, 3 s after declaration): a_r = dr/dt + r/T_o = M^-1 b_i (-f u_i) -> thruster i,
    loss f.  A two-fault extension tests all pairs (least squares, f_i, f_j >= 0).
"""
import itertools
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ftc_rov"))
import rov_model as rm

TAU_OBS = 5.0
TAU_BASE = 10.0
PERSIST = 0.5
T_START = 5.0
MDIAG = np.concatenate([rm.nominal_params()["m"] + rm.nominal_params()["A1"],
                        rm.nominal_params()["I"] + rm.nominal_params()["A2"]])
BI = rm.B_ALLOC / MDIAG[:, None]            # acceleration signature of 1 N on each thruster


def residual(t, e, freeze_at=None):
    """e: (T, N, 6) innovation -> r (T, N, 6). freeze_at: (N,) time after which the baseline is frozen."""
    dt = t[1] - t[0]
    a = 1 - np.exp(-dt / TAU_BASE)
    b = np.zeros_like(e)
    N = e.shape[1]
    fr = np.full(N, np.inf) if freeze_at is None else freeze_at
    bk = e[0].copy()
    cnt = 0
    for k in range(len(t)):
        cnt += 1
        aa = max(a, 1.0 / cnt)                         # running mean during start-up
        upd = t[k] < fr
        bk = np.where(upd[:, None], bk + aa * (e[k] - bk), bk)
        b[k] = bk
    return e - b


def thresholds(runs, factor=1.25):
    """runs: list of (t, innov (T,N,6)) fault-free calibration runs -> per-axis thresholds"""
    mx = np.zeros(6)
    for t, e in runs:
        r = residual(t, e)
        mx = np.maximum(mx, np.abs(r[t >= T_START]).reshape(-1, 6).max(0))
    return factor * mx


def flag_time(t, r, thr, t_from=T_START):
    """DECLARATION time per run: first time at which any axis has exceeded thr continuously for PERSIST s.
    Also returns the onset of that window.  NaN if never."""
    dt = t[1] - t[0]
    need = int(round(PERSIST / dt)) + 1               # samples spanning >= PERSIST seconds
    over = (np.abs(r) > thr).any(2) & (t[:, None] >= t_from)
    N = r.shape[1]
    decl = np.full(N, np.nan); onset = np.full(N, np.nan)
    run = np.zeros(N, int)
    for k in range(len(t)):
        run = np.where(over[k], run + 1, 0)
        hit = (run >= need) & np.isnan(decl)
        decl[hit] = t[k]; onset[hit] = t[k - need + 1]
    return decl, onset


def detect(t, innov, thr):
    """two-pass: flag with running baseline, then recompute residual with the baseline frozen at the flag onset."""
    r0 = residual(t, innov)
    decl, onset = flag_time(t, r0, thr)
    r = residual(t, innov, freeze_at=np.where(np.isnan(onset), np.inf, onset))
    return decl, r


def _ar(t, r, tcmd, t_flag, n, win):
    w = (t >= t_flag) & (t <= t_flag + win)
    if w.sum() < 3:
        return None, None
    tt, rr = t[w], r[w, n]
    drdt = np.polyfit(tt - tt[0], rr, 1)[0]
    return drdt + rr.mean(0) / TAU_OBS, tcmd[w, n].mean(0)


def isolate(t, r, tcmd, t_flag, win=3.0):
    """single-fault hypothesis test -> (thruster index, estimated loss fraction)"""
    N = r.shape[1]
    iso = np.full(N, -1); loss = np.full(N, np.nan)
    for n in range(N):
        if np.isnan(t_flag[n]):
            continue
        ar, Tm = _ar(t, r, tcmd, t_flag[n], n, win)
        if ar is None:
            continue
        best, bi, bf = np.inf, -1, np.nan
        for i in range(8):
            b = BI[:, i]
            if abs(Tm[i]) < 0.2:
                continue
            c = (b @ ar) / (b @ b)
            f = -c / Tm[i]
            J = np.linalg.norm(ar - b * c) / (np.linalg.norm(ar) + 1e-12)
            if 0.02 < f < 1.3 and J < best:
                best, bi, bf = J, i, f
        iso[n], loss[n] = bi, bf
    return iso, loss


def isolate_pair(t, r, tcmd, t_flag, win=3.0):
    """two-fault hypothesis: best pair (i, j) with non-negative losses; returns (pair index i*10+j, f_i, f_j, J1, J2)"""
    N = r.shape[1]
    out = []
    for n in range(N):
        if np.isnan(t_flag[n]):
            out.append((-1, np.nan, np.nan, np.nan, np.nan)); continue
        ar, Tm = _ar(t, r, tcmd, t_flag[n], n, win)
        best = (np.inf, -1, np.nan, np.nan)
        J1 = np.inf
        for i in range(8):
            if abs(Tm[i]) < 0.2:
                continue
            b = BI[:, i]; c = (b @ ar) / (b @ b)
            J1 = min(J1, np.linalg.norm(ar - b * c) / (np.linalg.norm(ar) + 1e-12))
        for i, j in itertools.combinations(range(8), 2):
            if abs(Tm[i]) < 0.2 or abs(Tm[j]) < 0.2:
                continue
            A = np.stack([-BI[:, i] * Tm[i], -BI[:, j] * Tm[j]], 1)
            f, *_ = np.linalg.lstsq(A, ar, rcond=None)
            if (f < 0.02).any() or (f > 1.3).any():
                continue
            J = np.linalg.norm(ar - A @ f) / (np.linalg.norm(ar) + 1e-12)
            if J < best[0]:
                best = (J, i * 10 + j, f[0], f[1])
        out.append((best[1], best[2], best[3], J1, best[0]))
    return out

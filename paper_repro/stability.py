"""
Dynamics / stability analysis per thruster-health case (paper Sec. V-A: "Stability was confirmed through
root locus analysis and frequency response evaluation").

Closed loop = 6-DOF plant + T200 thrust lag + multi-loop PID (same law as sim.py) + per-case allocation K_F.
State (34): p(3), Euler(3), nu(6), position integrators(3), attitude integrators(3), thrust(8),
       Pade states of the 30 ms digital-control delay (8).
For every case:
  1. hover equilibrium at 10 m (time integration to steady state; integrators that saturate are frozen),
  2. numerical linearisation A = df/dx, B_r = df/dr  (r = x_ref, y_ref, z_ref, psi_ref),
  3. closed-loop poles, damping, dominant time constant, classification,
  4. root locus vs overall controller gain K (0.05 ... 20 x nominal) -> upper gain margin,
  5. closed-loop frequency responses x_ref->x, z_ref->z, psi_ref->psi -> -3 dB bandwidth.
"""
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import os, sys, warnings
import numpy as np
from scipy.integrate import solve_ivp
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(_ROOT, "ftc_rov"))
import rov_model as rm
import sim
from allocation import all_cases, analyse_case, Allocator, W_NOM

OUT = os.path.join(_ROOT, "paper_repro/data")
P1 = rm.batch_params(1)
PN = rm.nominal_params()
M1N = PN["m"] + PN["A1"]; M2N = PN["I"] + PN["A2"]
G = sim.GAINS
EI_LIM, EIA_LIM = 6.0, 0.5
GSWEEP = np.logspace(np.log10(0.05), np.log10(20), 45)
TDELAY = 0.03      # 50 Hz sample-and-hold (10 ms) + one-step computation (20 ms)
NX = 34
W = np.logspace(-3, 1.3, 220)


def euler_R(e):
    return rm.quat_to_R(rm.euler_to_quat(np.array([e[0]]), np.array([e[1]]), np.array([e[2]])))[0]


def euler_rate(e, w):
    ph, th = e[0], e[1]
    c, s, t = np.cos(ph), np.sin(ph), np.tan(th)
    T = np.array([[1, s * t, c * t], [0, c, -s], [0, s / np.cos(th), c / np.cos(th)]])
    return T @ w


class Closed:
    def __init__(self, healthy, gscale=1.0, base=None):
        self.h = np.asarray(healthy, bool)
        if base is not None:
            self.info, self.K = base.info, base.K
        else:
            self.info = analyse_case(self.h)
            self.K = Allocator(self.h[None], np.diag(self.info["W"])[None]).K[0]
        self.g = gscale
        self.mode = self.info["mode"]
        self.frozen = np.zeros(6, bool)          # saturated integrators (3 pos + 3 att)

    def f(self, x, r):
        g = self.g
        p, e, nu = x[0:3], x[3:6], x[6:12]
        ei, eia, T, zd = x[12:15], x[15:18], x[18:26], x[26:34]
        R = euler_R(e)
        # ---- plant
        tau = rm.B_ALLOC @ (T * self.h)
        q = rm.euler_to_quat(np.array([e[0]]), np.array([e[1]]), np.array([e[2]]))
        nd, _ = rm.accel(q, nu[None], tau[None], np.zeros((1, 3)), P1)
        pd = R @ nu[:3]
        ed = euler_rate(e, nu[3:])
        # ---- controller (perfect measurements, hover reference)
        err = np.array([r[0], r[1], r[2]]) - p
        ev = -pd
        acmd = g * (G["kp"] * err + G["kd"] * ev + G["ki"] * ei)
        ab = R.T @ acmd
        v1, w = nu[:3], nu[3:]
        Fb = M1N * ab            # drag feed-forward uses the reference velocity, which is zero at hover
        zWb = R[2, :]
        tilt = -np.cross(zWb, np.array([0, 0, 1.0]))
        yaw_free = self.mode in ("INFEASIBLE_HORIZONTAL", "INFEASIBLE_DEPTH", "YAW_FREE")
        eyaw = 0.0 if yaw_free else sim.wrap(r[3] - e[2])
        eatt = tilt + eyaw * zWb
        kp_a = np.array([G["kp_rp"], G["kp_rp"], G["kp_y"]]); kd_a = np.array([G["kd_rp"], G["kd_rp"], G["kd_y"]])
        ki_a = np.array([G["ki_rp"], G["ki_rp"], G["ki_y"]])
        Mb = M2N * g * (kp_a * eatt - kd_a * w + ki_a * eia) + PN["Dl"][3:] * w + PN["Dq"][3:] * np.abs(w) * w
        gh = rm.restoring(R[None], dict(W=np.array([PN["W"]]), B=np.array([PN["B"]]), rg=np.zeros((1, 3)),
                                         rb=PN["rb"][None]))[0]
        tau_d = np.concatenate([Fb, Mb]) - gh
        Tc = np.clip(self.K @ tau_d, -rm.T_REV_MAX, rm.T_FWD_MAX) * self.h
        # first-order Pade of the control delay: y = 2 z - u,  z' = (u - z) / (T/2)
        Tdel = 2 * zd - Tc
        zdd = (Tc - zd) / (TDELAY / 2)
        Td = (Tdel - T) / rm.TAU_MOTOR
        # integrators (heading-steered hover: only along producible axis d0, as in sim.py)
        e_int = err.copy()
        if self.mode == "HEADING_STEERED":
            ps = e[2]; d = self.info["d0"]
            d0w = np.array([np.cos(ps) * d[0] - np.sin(ps) * d[1], np.sin(ps) * d[0] + np.cos(ps) * d[1]])
            e_int[:2] = (err[:2] @ d0w) * d0w
        eid = e_int.copy(); eiad = eatt.copy()
        # saturation: no integration outward at the limit
        eid = np.where((np.abs(ei) >= EI_LIM - 1e-9) & (np.sign(ei) == np.sign(eid)), 0.0, eid)
        eiad = np.where((np.abs(eia) >= EIA_LIM - 1e-9) & (np.sign(eia) == np.sign(eiad)), 0.0, eiad)
        integ = np.concatenate([eid, eiad])
        integ[self.frozen] = 0.0
        return np.concatenate([pd, ed, nd[0], integ, Td, zdd])

    # ------------------------------------------------------------------
    def equilibrium(self, r, x0=None, T=150.0):
        if x0 is None:
            x0 = np.zeros(NX); x0[0:3] = r[:3]; x0[5] = r[3]
            gh = rm.restoring(np.eye(3)[None], dict(W=np.array([PN["W"]]), B=np.array([PN["B"]]),
                                                    rg=np.zeros((1, 3)), rb=PN["rb"][None]))[0]
            x0[18:26] = self.K @ (-gh) * self.h
            x0[26:34] = x0[18:26]
        x = x0.copy()
        from scipy.optimize import fsolve
        try:
            xn, info, ier, _ = fsolve(lambda z: self.f(z, r), x0, full_output=True, xtol=1e-12)
            okn = (ier == 1 or np.linalg.norm(self.f(xn, r)) < 1e-9) and np.all(np.abs(xn[12:15]) < EI_LIM) \
                and np.all(np.abs(xn[15:18]) < EIA_LIM) and np.linalg.norm(xn[:3] - r[:3]) < 0.5 and np.abs(xn[3:5]).max() < 1.2
        except Exception:
            okn = False
        if okn:
            x = xn
        elif np.linalg.norm(self._clip_rhs(x, r)) > 1e-8:
            # fixed-step RK4 (robust at the integrator-clip discontinuity)
            h = 0.01
            for _ in range(int(T / h)):
                k1 = self._clip_rhs(x, r); k2 = self._clip_rhs(x + 0.5 * h * k1, r)
                k3 = self._clip_rhs(x + 0.5 * h * k2, r); k4 = self._clip_rhs(x + h * k3, r)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                x[12:15] = np.clip(x[12:15], -EI_LIM, EI_LIM); x[15:18] = np.clip(x[15:18], -EIA_LIM, EIA_LIM)
        x[12:15] = np.clip(x[12:15], -EI_LIM, EI_LIM); x[15:18] = np.clip(x[15:18], -EIA_LIM, EIA_LIM)
        sat = np.concatenate([np.abs(x[12:15]) >= EI_LIM - 1e-6, np.abs(x[15:18]) >= EIA_LIM - 1e-6])
        self.frozen = sat
        res = np.linalg.norm(self.f(x, r))
        return x, res, sat

    def _clip_rhs(self, x, r):
        x = x.copy()
        x[12:15] = np.clip(x[12:15], -EI_LIM, EI_LIM); x[15:18] = np.clip(x[15:18], -EIA_LIM, EIA_LIM)
        return self.f(x, r)

    def jac(self, x, r, eps=1e-6):
        n = len(x)
        A = np.zeros((n, n)); Br = np.zeros((n, 4))
        for i in range(n):
            d = np.zeros(n); d[i] = eps * max(1.0, abs(x[i]))
            A[:, i] = (self.f(x + d, r) - self.f(x - d, r)) / (2 * d[i])
        for i in range(4):
            d = np.zeros(4); d[i] = eps
            Br[:, i] = (self.f(x, r + d) - self.f(x, r - d)) / (2 * eps)
        keep = np.ones(n, bool); keep[12:18] = ~self.frozen
        dead = np.nonzero(~self.h)[0] + 18
        keep[dead] = False
        keep[dead + 8] = False                                   # failed thrusters: state removed (always 0)
        return A[np.ix_(keep, keep)], Br[keep], keep


def freq_resp(A, B, C, w):
    n = A.shape[0]
    I = np.eye(n)
    return np.array([C @ np.linalg.solve(1j * wk * I - A, B) for wk in w])


def classify(ev, tol=1e-4):
    mr = ev.real.max()
    if mr > tol:
        return "UNSTABLE"
    if mr > -tol:
        return "MARGINAL"
    return "STABLE"


def analyse(ci):
    cases, masks = all_cases()
    h = masks[ci]
    r = np.array([0.0, 0.0, 10.0, 0.0])
    cl = Closed(h)
    out = dict(case_id=ci, mode=cl.mode, n_failed=int((~h).sum()))
    if cl.mode == "INFEASIBLE_DEPTH":
        out.update(eq_found=False, note="no vertical thrust: positive buoyancy cannot be trimmed -> no hover equilibrium")
        return out, None
    x, res, sat = cl.equilibrium(r)
    out.update(eq_found=bool(res < 1e-3), eq_residual=float(res), eq_tilt_deg=float(np.rad2deg(np.abs(x[3:5]).max())),
               eq_pos_err=float(np.linalg.norm(x[:3] - r[:3])), n_sat_integrators=int(sat.sum()))
    A, Br, keep = cl.jac(x, r)
    ev = np.linalg.eigvals(A)
    out["class"] = classify(ev)
    st = ev[ev.real < -1e-4]
    out["max_real"] = float(ev.real.max())
    out["n_zero_poles"] = int((np.abs(ev) < 1e-4).sum())
    out["dominant_tc_s"] = float(1 / np.abs(st.real).min()) if len(st) else np.nan
    osc = st[np.abs(st.imag) > 1e-3]
    out["min_damping"] = float((-osc.real / np.abs(osc)).min()) if len(osc) else 1.0
    # root locus
    loci = []
    gm_up = np.nan; gm_lo = np.nan
    for gsc in GSWEEP:
        # root locus: linearised about the nominal-gain equilibrium
        c2 = Closed(h, gsc, base=cl); c2.frozen = cl.frozen.copy()
        A2, _, _ = c2.jac(x, r)
        e2 = np.linalg.eigvals(A2)
        loci.append(e2)
        unst = e2.real.max() > 1e-4
        if gsc > 1 and unst and np.isnan(gm_up):
            gm_up = gsc
        if gsc < 1 and not unst:
            pass
        if gsc < 1 and unst:
            gm_lo = gsc
    out["gain_margin_up_dB"] = float(20 * np.log10(gm_up)) if np.isfinite(gm_up) else np.inf
    out["gain_margin_low"] = float(gm_lo) if np.isfinite(gm_lo) else 0.0
    # frequency responses
    idx_all = np.arange(NX)[keep]
    def C_of(state):
        c = np.zeros(len(idx_all)); c[np.nonzero(idx_all == state)[0][0]] = 1; return c
    fr = {}
    for nm, st_i, inp in [("surge_x", 0, 0), ("sway_y", 1, 1), ("depth_z", 2, 2), ("yaw_psi", 5, 3)]:
        Gw = freq_resp(A, Br[:, inp], C_of(st_i), W)
        fr[nm] = Gw
        mag = 20 * np.log10(np.abs(Gw) + 1e-12)
        dc = mag[0]
        below = np.nonzero(mag < dc - 3)[0]
        out[f"bw_{nm}_rad_s"] = float(W[below[0]]) if len(below) and abs(dc) < 3 else np.nan
        out[f"dc_gain_{nm}_dB"] = float(dc)
    detail = dict(ev=ev, loci=np.array(loci, dtype=object), fr=fr, x_eq=x, keep=keep)
    return out, detail


def _one(ci):
    import pickle
    path = f"{OUT}/stab/{ci:03d}.pkl"
    if os.path.exists(path):
        return ci
    o, d = analyse(ci)
    pickle.dump((o, d), open(path, "wb"))
    print(ci, o.get("mode"), o.get("class"), flush=True)
    return ci


def main():
    import pandas as pd, pickle
    from multiprocessing import Pool
    os.makedirs(f"{OUT}/stab", exist_ok=True)
    with Pool(2) as pool:
        list(pool.imap_unordered(_one, range(255), chunksize=1))
    res = [pickle.load(open(f"{OUT}/stab/{ci:03d}.pkl", "rb")) for ci in range(255)]
    rows = [r[0] for r in res]
    det = {i: r[1] for i, r in enumerate(res)}
    df = pd.DataFrame(rows)
    df["class"] = df["class"].fillna("NO_EQUILIBRIUM")
    df.to_csv(f"{OUT}/stability.csv", index=False)
    pickle.dump(det, open(f"{OUT}/stability_detail.pkl", "wb"))
    print(df.groupby(["n_failed", "class"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()

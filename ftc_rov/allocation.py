"""
Fault-tolerant control allocation for the 8-thruster BlueROV2 Heavy.

For every failure set F (254 of them: C(8,1)+...+C(8,7)) we build a dedicated
reconfiguration law:
  * the reduced thrust configuration matrix B_F (failed columns removed),
  * a controllability analysis (rank of horizontal / vertical sub-blocks),
  * a guidance mode (FULL, COUPLED-ATTITUDE, HEADING-STEERED, YAW-FREE, INFEASIBLE),
  * DOF priority weights W_F,
  * the weighted pseudo-inverse K_F (thrust = K_F * tau when no limit is hit),
  * a constrained allocator: redistributed pseudo-inverse (RPI) + projected-gradient
    polish, solving  min ||W(B u - tau)||^2 + lam||u||^2,  u_min <= u <= u_max,
    u_i = 0 for failed thrusters.
Thrust is converted to RPM through the T200 map.
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from rov_model import B_ALLOC, T_FWD_MAX, T_REV_MAX, HORIZ, VERT

LAM = 1e-3
U_MAX = T_FWD_MAX
U_MIN = -T_REV_MAX
# DOF weights [X, Y, Z, K, M, N]: depth > translation > yaw > roll/pitch (passively stable)
W_NOM = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.5])
REL_ALPHA = 0.0      # residual weight kept on a released (uncontrollable-with-depth) attitude axis

MODES = ["FULL", "COUPLED_ATTITUDE", "HEADING_STEERED", "YAW_FREE",
         "INFEASIBLE_HORIZONTAL", "INFEASIBLE_DEPTH"]


def all_cases():
    """List of healthy masks: index 0 = nominal, then all 254 failure sets ordered by size."""
    cases = [tuple()]
    for k in range(1, 8):
        cases += list(itertools.combinations(range(8), k))
    masks = np.ones((len(cases), 8), bool)
    for i, f in enumerate(cases):
        masks[i, list(f)] = False
    return cases, masks


def analyse_case(healthy):
    B = B_ALLOC * healthy[None, :]
    rank = np.linalg.matrix_rank(B, tol=1e-6)
    Bh = B[[0, 1, 5]][:, HORIZ]              # X, Y, N from horizontals
    Bv = B[[2, 3, 4]][:, VERT]               # Z, K, M from verticals
    rh = np.linalg.matrix_rank(Bh, tol=1e-6)
    rv = np.linalg.matrix_rank(Bv, tol=1e-6)
    d0 = np.array([1.0, 0.0])
    if rh == 2:
        U, s, Vt = np.linalg.svd(Bh)
        n = U[:, 2]                           # normal of reachable (X,Y,N) plane
        if np.hypot(n[0], n[1]) < 1e-6:       # cannot produce yaw moment at all
            mode = "YAW_FREE"
        else:
            d0 = np.array([-n[1], n[0]]) / np.hypot(n[0], n[1])
            mode = "HEADING_STEERED"
    elif rh == 3:
        mode = "FULL" if rv == 3 else "COUPLED_ATTITUDE"
    else:
        mode = "INFEASIBLE_HORIZONTAL"
    if rv == 0:
        mode = "INFEASIBLE_DEPTH"
    # can heave be produced with zero roll/pitch moment?  (zero-moment heave)
    zf = np.zeros(6); zf[2] = 1
    sol = np.linalg.lstsq(B, zf, rcond=None)[0]
    pure_heave = np.linalg.norm(B @ sol - zf) < 1e-6
    W = W_NOM.copy()
    if mode == "YAW_FREE":
        W[5] = 0.0
    Q = np.diag(W)
    if mode == "HEADING_STEERED":
        # lateral force (perpendicular to the zero-yaw thrust axis d0) cannot be produced
        # independently of yaw: release it; the cross-track job is done by steering the heading
        e = np.array([-d0[1], d0[0]])
        Q[0:2, 0:2] = Q[0:2, 0:2] @ (np.eye(2) - np.outer(e, e))
    e_unc = np.zeros(2)
    if rv == 2:
        # (Z,K,M) reachable plane from verticals; with depth prioritised, the moment
        # component along e_unc is dictated by heave -> released (not fought) by the allocator
        U, s, Vt = np.linalg.svd(Bv)
        nv = U[:, 2]
        if np.hypot(nv[1], nv[2]) > 1e-9:
            e_unc = nv[1:] / np.hypot(nv[1], nv[2])
            P = np.eye(2) - (1 - REL_ALPHA) * np.outer(e_unc, e_unc)
            Q[3:5, 3:5] = np.diag(W[3:5]) @ P
    elif rv == 1:
        Q[3:5, 3:5] = 0.0                     # single vertical: roll & pitch left to buoyancy
        e_unc = np.array([np.nan, np.nan])
    info = dict(rank=int(rank), rank_h=int(rh), rank_v=int(rv), mode=mode,
                d0=d0, W=W, Q=Q, e_unc=e_unc, pure_heave=bool(pure_heave))
    info.update(capability(healthy, info))
    return info


def capability(healthy, info):
    """Attainable-wrench summary used by the capability-aware time scaling."""
    free_v = (3, 4) if info["rank_v"] < 3 else ()
    e = np.eye(6)
    if info["mode"] == "HEADING_STEERED":
        d = np.zeros(6); d[:2] = info["d0"]
        cap_h = min(capacity_dir(healthy, d, free_v), capacity_dir(healthy, -d, free_v))
    elif info["mode"] in ("FULL", "COUPLED_ATTITUDE"):
        cap_h = min(capacity_dir(healthy, sg * e[i], free_v) for i in (0, 1) for sg in (1, -1))
    else:
        cap_h = 0.0
    cap_z = min(capacity_dir(healthy, e[2], free_v), capacity_dir(healthy, -e[2], free_v)) if info["rank_v"] > 0 else 0.0
    # lever arm of the heave line of action (tilt coupling when roll/pitch released)
    arm = 0.0
    if info["rank_v"] < 3 and info["rank_v"] > 0 and not info["pure_heave"]:
        r = np.array([[0.120, 0.218], [0.120, -0.218], [-0.120, 0.218], [-0.120, -0.218]])
        hv = healthy[4:]
        arm = float(np.linalg.norm(r[hv].mean(0)))
    action = {"FULL": "continue", "COUPLED_ATTITUDE": "continue (attitude coupled, slowed)",
              "HEADING_STEERED": "continue (heading-steered, slowed)",
              "YAW_FREE": "continue (yaw free)",
              "INFEASIBLE_HORIZONTAL": "abort: hold depth / controlled ascent",
              "INFEASIBLE_DEPTH": "abort: passive (buoyant) ascent"}[info["mode"]]
    return dict(cap_h=float(cap_h), cap_z=float(cap_z), arm=arm, action=action)


def capacity(healthy):
    """Max attainable generalized force in each +/- DOF with the others held at zero (LP)."""
    B = B_ALLOC
    lo = np.where(healthy, U_MIN, 0.0)
    hi = np.where(healthy, U_MAX, 0.0)
    cap = np.zeros(12)
    for d in range(6):
        Aeq = np.delete(B, d, 0)
        for sgn_i, sgn in enumerate([1, -1]):
            res = linprog(-sgn * B[d], A_eq=Aeq, b_eq=np.zeros(5),
                          bounds=list(zip(lo, hi)), method="highs")
            cap[2 * d + sgn_i] = sgn * B[d] @ res.x if res.success else 0.0
    return cap


def capacity_dir(healthy, d, free=()):
    """Max s such that B u = s d (+ anything on 'free' DOFs) within thruster limits."""
    B = B_ALLOC
    keep = [i for i in range(6) if i not in free]
    lo = np.where(healthy, U_MIN, 0.0); hi = np.where(healthy, U_MAX, 0.0)
    # variables [u(8), s]
    c = np.zeros(9); c[-1] = -1
    Aeq = np.hstack([B[keep], -np.asarray(d, float)[keep][:, None]])
    res = linprog(c, A_eq=Aeq, b_eq=np.zeros(len(keep)), bounds=list(zip(lo, hi)) + [(0, None)], method="highs")
    return res.x[-1] if res.success else 0.0


class Allocator:
    """Batched constrained allocator. One 'case' per batch row."""

    def __init__(self, healthy, Q):
        self.set_cases(healthy, Q)

    def set_cases(self, healthy, Q):
        """Q: (N,6) diagonal weights or (N,6,6) weighting matrices."""
        healthy = np.asarray(healthy, bool)
        Q = np.asarray(Q, float)
        if Q.ndim == 2:
            Q = np.stack([np.diag(q) for q in Q])
        self.h = healthy
        N = healthy.shape[0]
        Bm = B_ALLOC[None] * healthy[:, None, :]                  # N,6,8
        QtQ = np.einsum("nki,nkj->nij", Q, Q)
        self.BtW2 = np.einsum("nki,nkj->nij", Bm, QtQ)                 # N,8,6
        H = np.einsum("nik,nkj->nij", self.BtW2, Bm) + LAM * np.eye(8)
        # failed thrusters: decouple (identity rows) so solves keep u_i = 0
        dead = ~healthy
        H = H * (healthy[:, :, None] & healthy[:, None, :]) + np.eye(8) * dead[:, :, None]
        self.H = H
        self.Hinv = np.linalg.inv(H)
        self.K = np.einsum("nij,njk->nik", self.Hinv, self.BtW2)   # weighted pseudo-inverse N,8,6
        self.L = np.linalg.eigvalsh(H).max(1)
        self.lo = np.where(healthy, U_MIN, 0.0)
        self.hi = np.where(healthy, U_MAX, 0.0)

    def update_rows(self, rows, healthy_rows, Q_rows):
        sub = Allocator(healthy_rows, Q_rows)
        for name in ["h", "BtW2", "H", "Hinv", "K", "L", "lo", "hi"]:
            getattr(self, name)[rows] = getattr(sub, name)

    @staticmethod
    def _max_scale(a, b, lo, hi):
        """largest s in [0,1] with lo <= a + s b <= hi (elementwise, per row)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            s_hi = np.where(b > 1e-12, (hi - a) / b, np.inf)
            s_lo = np.where(b < -1e-12, (lo - a) / b, np.inf)
        return np.clip(np.minimum(s_hi, s_lo).min(1), 0.0, 1.0)

    def solve(self, tau, method="priority", n_iter_pg=6):
        if method == "priority":
            return self.solve_priority(tau)
        return self.solve_rpi(tau, n_iter_pg)

    def solve_priority(self, tau):
        """Two-level, direction-preserving priority allocation.
        Level 1: depth + roll + pitch (safety-critical, vertical block);
        Level 2: surge, sway, yaw scaled to the capacity that is left.
        Neither level ever leaks unintended moments through saturation."""
        tv = tau.copy(); tv[:, [0, 1, 5]] = 0.0
        th = tau - tv
        uv = np.einsum("nij,nj->ni", self.K, tv)
        uh = np.einsum("nij,nj->ni", self.K, th)
        zero = np.zeros_like(uv)
        sv = self._max_scale(zero, uv, self.lo, self.hi)
        uv = uv * sv[:, None]
        sh = self._max_scale(uv, uh, self.lo, self.hi)
        u = uv + sh[:, None] * uh
        sat = (sv < 0.999) | (sh < 0.999)
        self.last_scale = (sv, sh)
        return np.clip(u, self.lo, self.hi), sat

    def solve_rpi(self, tau, n_iter_pg=6):
        f = np.einsum("nij,nj->ni", self.BtW2, tau)
        u = np.einsum("nij,nj->ni", self.K, tau)
        sat_any = ((u < self.lo - 1e-9) | (u > self.hi + 1e-9)).any(1)
        if sat_any.any():
            idx = np.nonzero(sat_any)[0]
            u[idx] = self._rpi(idx, f[idx], u[idx])
            # projected-gradient polish (lets saturated actuators be released)
            H, lo, hi = self.H[idx], self.lo[idx], self.hi[idx]
            step = 1.0 / self.L[idx][:, None]
            ui = u[idx]
            for _ in range(n_iter_pg):
                g = np.einsum("nij,nj->ni", H, ui) - f[idx]
                ui = np.clip(ui - step * g, lo, hi)
            u[idx] = ui
        return u, sat_any

    def _rpi(self, idx, f, u0):
        H, lo, hi = self.H[idx], self.lo[idx], self.hi[idx]
        free = self.h[idx].copy()
        uf = np.zeros_like(u0)
        u = u0.copy()
        for _ in range(8):
            viol = free & ((u < lo - 1e-9) | (u > hi + 1e-9))
            if not viol.any():
                break
            uf = np.where(viol, np.clip(u, lo, hi), uf)
            free = free & ~viol
            fixed = ~free
            A = H * (free[:, :, None] & free[:, None, :]) + np.eye(8) * fixed[:, :, None]
            rhs = np.where(free, f - np.einsum("nij,nj->ni", H, np.where(fixed, uf, 0.0)), uf)
            u = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
            u = np.where(free, u, uf)
        return np.clip(u, lo, hi)

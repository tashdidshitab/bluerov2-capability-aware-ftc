"""
Fault-aware linear time-varying MPC (comparator for the revision).

Decision variables: the 8 thruster forces over the horizon (move blocking), hard box limits, failed thrusters
fixed at 0 (health from the same RPM-telemetry FDI as every other controller).
Model: 12-state (p, Euler, nu_1, nu_2) nonlinear 6-DOF model (rigid-body mass + added mass, linear+quadratic
damping, restoring), linearised by finite differences at every solve about the current state and the last input,
discretised by forward Euler. Offset-free: a 6-DOF acceleration disturbance d_hat is estimated from the one-step
prediction error and carried over the horizon (absorbs current, Coriolis and model error).
QP: condensed, box-constrained, solved for all vehicles at once with accelerated projected gradient (FISTA),
warm-started from the shifted previous solution.
"""
import numpy as np
import rov_model as rm

PN = rm.nominal_params()
MD = np.concatenate([PN["m"] + PN["A1"], PN["I"] + PN["A2"]])


def _euler_R(e):
    return rm.quat_to_R(rm.euler_to_quat(e[:, 0], e[:, 1], e[:, 2]))


def _f(x, u, d):
    """batched continuous dynamics, x (N,12) = [p, eul, nu1, nu2]"""
    N = x.shape[0]
    e = x[:, 3:6]; v = x[:, 6:9]; w = x[:, 9:12]
    R = _euler_R(e)
    pd = np.einsum("nij,nj->ni", R, v)
    ph, th = e[:, 0], e[:, 1]
    c, s, t = np.cos(ph), np.sin(ph), np.tan(np.clip(th, -1.3, 1.3))
    ct = np.cos(np.clip(th, -1.3, 1.3))
    ed = np.stack([w[:, 0] + s * t * w[:, 1] + c * t * w[:, 2], c * w[:, 1] - s * w[:, 2], (s * w[:, 1] + c * w[:, 2]) / ct], 1)
    nu = x[:, 6:12]
    g = rm.restoring(R, dict(W=np.full(N, PN["W"]), B=np.full(N, PN["B"]), rg=np.zeros((N, 3)), rb=np.tile(PN["rb"], (N, 1))))
    tau = u @ rm.B_ALLOC.T
    nud = (tau + g - PN["Dl"] * nu - PN["Dq"] * np.abs(nu) * nu) / MD + d
    return np.concatenate([pd, ed, nud], 1)


class FaultAwareMPC:
    def __init__(self, N, H=12, dt=0.15, block=2, iters=40, every=5,
                 q=(40, 40, 80, 30, 30, 10, 2, 2, 2, 0.5, 0.5, 0.5), r=1e-3, rd=5e-3, d_gain=0.3):
        self.N, self.H, self.dt, self.block, self.iters, self.every = N, H, dt, block, iters, every
        self.nb = H // block
        self.Q = np.asarray(q, float)
        self.r, self.rd, self.d_gain = r, rd, d_gain
        self.health = np.ones((N, 8), bool)
        self.U = np.zeros((N, self.nb, 8))
        self.d = np.zeros((N, 6))
        self.x_pred = None
        self.u_last = np.zeros((N, 8))
        self.sat = np.zeros(N, bool)
        self.solve_time = 0.0; self.n_solves = 0

    def set_health(self, rows, mask_rows):
        self.health[rows] = mask_rows
        self.U[rows] *= mask_rows[:, None, :]

    def _bounds(self):
        lo = np.where(self.health, -rm.T_REV_MAX, 0.0)
        hi = np.where(self.health, rm.T_FWD_MAX, 0.0)
        return lo, hi

    def solve(self, pm, em, num, motion, tv, kappa, ref, k, dt_ctrl):
        import time
        t0 = time.perf_counter()
        N, H, nb, dtm = self.N, self.H, self.nb, self.dt
        x0 = np.concatenate([pm, em, num], 1)
        # offset-free disturbance update from last one-step prediction (velocity part)
        if self.x_pred is not None:
            err = (num - self.x_pred[:, 6:12]) / (self.every * dt_ctrl)
            self.d = np.clip(self.d + self.d_gain * err, -2.0, 2.0)
        u0 = self.u_last
        f0 = _f(x0, u0, self.d)
        # Jacobian A_c by central differences, B_c exact
        A = np.zeros((N, 12, 12))
        for j in range(12):
            dx = np.zeros(12); dx[j] = 1e-4
            A[:, :, j] = (_f(x0 + dx, u0, self.d) - _f(x0 - dx, u0, self.d)) / 2e-4
        Bc = np.zeros((N, 12, 8)); Bc[:, 6:12, :] = (rm.B_ALLOC / MD[:, None])[None] * self.health[:, None, :]
        Ad = np.eye(12)[None] + dtm * A
        Bd = dtm * Bc
        cd = dtm * (f0 - np.einsum("nij,nj->ni", A, x0) - np.einsum("nij,nj->ni", Bc, u0))
        # prediction X_k = Phi_k x0 + sum_j Gam_kj U_blk(j) + c_k ,  k = 1..H
        Phi = np.zeros((N, H, 12, 12)); Gam = np.zeros((N, H, 12, nb * 8)); cvec = np.zeros((N, H, 12))
        Ak = np.tile(np.eye(12), (N, 1, 1)); Gk = np.zeros((N, 12, nb * 8)); ck = np.zeros((N, 12))
        for kk in range(H):
            blk = kk // self.block
            Gk = np.einsum("nij,njk->nik", Ad, Gk)
            Gk[:, :, blk * 8:(blk + 1) * 8] += Bd
            ck = np.einsum("nij,nj->ni", Ad, ck) + cd
            Ak = np.einsum("nij,njk->nik", Ad, Ak)
            Phi[:, kk] = Ak; Gam[:, kk] = Gk; cvec[:, kk] = ck
        free = np.einsum("nkij,nj->nki", Phi, x0) + cvec                         # (N,H,12)
        # references over the horizon (base trajectory preview + current online modification)
        tq = tv[:, None] + kappa[:, None] * dtm * np.arange(1, H + 1)[None]
        jj = np.minimum(np.round(tq / (motion.t[1] - motion.t[0])).astype(int), motion.T - 1)
        j0 = np.minimum(np.round(tv / (motion.t[1] - motion.t[0])).astype(int), motion.T - 1)
        pr = motion.p[jj] + (ref["p"] - motion.p[j0])[:, None, :]
        psr = em[:, 2:3] + np.angle(np.exp(1j * (motion.psi[jj] + (ref["psi"] - motion.psi[j0])[:, None] - em[:, 2:3])))
        R0 = _euler_R(em)
        vr_b = np.einsum("nji,nj->ni", R0, ref["v"])
        Xr = np.zeros((N, H, 12)); Xr[:, :, 0:3] = pr; Xr[:, :, 5] = psr; Xr[:, :, 6:9] = vr_b[:, None, :]
        Qb = np.tile(self.Q, H)
        G2 = Gam.reshape(N, H * 12, nb * 8)
        G2Q = G2 * Qb[None, :, None]
        Hm = np.matmul(G2.transpose(0, 2, 1), G2Q)                                 # (N,nU,nU)
        nU = nb * 8
        # input weights (magnitude + rate)
        D = np.eye(nU) - np.eye(nU, k=-8)
        Hm += self.r * np.eye(nU)[None] + self.rd * (D.T @ D)[None]
        gv = np.einsum("nki,nk->ni", G2Q, (free - Xr).reshape(N, H * 12))
        u_prev = np.zeros((N, nU)); u_prev[:, :8] = self.u_last
        gv -= self.rd * np.einsum("ij,nj->ni", D.T, np.concatenate([self.u_last, np.zeros((N, nU - 8))], 1))
        lo, hi = self._bounds()
        LO = np.tile(lo, (1, nb)); HI = np.tile(hi, (1, nb))
        # step size from power iteration
        v = np.ones((N, nU))
        for _ in range(8):
            v = np.einsum("nij,nj->ni", Hm, v); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        Lc = np.einsum("ni,nij,nj->n", v, Hm, v)[:, None] * 1.05 + 1e-9
        # warm start (shift)
        U = np.concatenate([self.U[:, 1:], self.U[:, -1:]], 1).reshape(N, nU)
        U = np.clip(U, LO, HI); Y = U.copy(); tk = 1.0
        for _ in range(self.iters):
            Un = np.clip(Y - (np.einsum("nij,nj->ni", Hm, Y) + gv) / Lc, LO, HI)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * tk * tk))
            Y = Un + ((tk - 1) / tn) * (Un - U)
            U, tk = Un, tn
        self.U = U.reshape(N, nb, 8)
        u = self.U[:, 0].copy()
        self.sat = ((np.abs(u - lo) < 1e-6) | (np.abs(u - hi) < 1e-6)).any(1) & self.health.any(1)
        # one-step prediction for the disturbance update (after 'every' control steps)
        hstep = self.every * dt_ctrl
        self.x_pred = x0 + hstep * _f(x0, u, self.d)
        self.u_last = u
        self.solve_time += time.perf_counter() - t0; self.n_solves += 1
        return u

"""
Batched closed-loop simulator: BlueROV2 Heavy + fault-tolerant controller (FT-PID) and comparators.

Per control step (50 Hz):
  1. sensors (noise, optional delay + kinematic predictor)
  2. reference at the vehicle's *virtual time* (capability-aware time scaling keeps
     the geometric path identical and only slows the timing law after a fault)
  3. FDI: RPM-telemetry residual + persistence -> isolated failure set -> new law
  4. guidance per failure class (heading-steering for under-actuated horizontal plane)
  5. position PID + reference feed-forward + light model terms -> body wrench
  6. attitude PID (tilt error valid at any tilt + heading about the vertical)
  7. allocation -> thrust -> RPM command   (or: fault-aware MPC computes thrusts directly)
  8. plant: T200 RPM dynamics + 6-DOF dynamics (Heun or RK4), optional model-form mismatch
  9. optional digital twin (observer + current estimate) running in parallel

Controller variants (SimConfig):
  ftc=False                         baseline PID + nominal allocation, no FDI-driven change
  alloc_reconfig / class_guidance   switch the reconfigured allocation / class guidance on/off
  alloc_method                      'priority' (proposed), 'rpi' (bounded least squares), 'clip' (WPI + clipping)
  time_scaling                      capability-aware time scaling
  supervisor                        abort + ascent for structurally infeasible classes
  mpc                               dict -> fault-aware LTV-MPC replaces PID + allocation
"""
import time as _time
import numpy as np
from scipy.spatial import cKDTree
import rov_model as rm
from allocation import Allocator, analyse_case, W_NOM

DT_CTRL = 0.02
N_SUB = 2
KAPPA_MIN = 0.35

GAINS = dict(kp=1.0, kd=2.0, ki=0.25,
             kp_rp=9.0, kd_rp=6.0, ki_rp=0.3,
             kp_y=2.25, kd_y=3.0, ki_y=0.1,
             a_max=0.6, los_k=0.4, los_ki=0.0, steer_rate=15.0, steer_rate_hover=10.0, hover_law='int')

ABORT_ASCENT = 0.25      # m/s controlled-ascent rate after an abort
EYAW_MAX = np.deg2rad(45.0)   # heading-error saturation in the attitude law
GRACE = 20.0            # s, settling allowance after the reference ends (all controllers)
ABORT_DEPTH = 1.0        # m, ascent ends here
HOTEL_W = 40.0           # W, electronics + sensors (energy accounting only)

_CASE_CACHE = {}


def case_info(healthy):
    key = tuple(bool(x) for x in healthy)
    if key not in _CASE_CACHE:
        _CASE_CACHE[key] = analyse_case(np.array(key))
    return _CASE_CACHE[key]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def euler_R(e):
    return rm.quat_to_R(rm.euler_to_quat(e[:, 0], e[:, 1], e[:, 2]))


def motion_demand(motion):
    vh = np.hypot(motion.v[:, 0], motion.v[:, 1]).max()
    ah = np.hypot(motion.a[:, 0], motion.a[:, 1]).max()
    vz = np.abs(motion.v[:, 2]).max()
    az = np.abs(motion.a[:, 2]).max()
    return vh, ah, vz, az


def kappa_for(info, demand, Pn):
    """Capability-aware time-scaling factor (1 = original timing)."""
    if info["mode"] in ("INFEASIBLE_HORIZONTAL", "INFEASIBLE_DEPTH", "YAW_FREE"):
        return 1.0
    vh, ah, vz, az = demand
    M1 = Pn["m"] + Pn["A1"]
    ks = np.linspace(1.0, KAPPA_MIN, 66)
    Fh = 15.0 * ks * vh + 180.0 * (ks * vh) ** 2 + M1[0] * ks ** 2 * ah + 3.0      # +3 N current allowance
    Fz = Pn["Dl"][2] * ks * vz + Pn["Dq"][2] * (ks * vz) ** 2 + M1[2] * ks ** 2 * az + abs(Pn["B"] - Pn["W"])
    ok = (Fh <= 0.6 * info["cap_h"]) & (Fz <= 0.6 * info["cap_z"])
    if info["rank_v"] < 3:
        # tilt budget (25 deg) under coupled roll/pitch: horizontal force acts 8.5 cm below CG,
        # heave acts at lever arm 'arm' from CG; restoring moment = B * BG * sin(tilt)
        Mres = Pn["B"] * abs(Pn["rb"][2]) * np.sin(np.deg2rad(25))
        ok &= (0.085 * Fh + info["arm"] * Fz) <= Mres + 0.2
    k = ks[ok].max() if ok.any() else KAPPA_MIN
    return float(k)


class SimConfig:
    def __init__(self, ftc=True, t_fail=20.0, noise=True, delay_steps=0, predictor=True,
                 unc=0.0, current_scale=1.0, seed=0, fdi=True, log_every=25, dt_ctrl=DT_CTRL, n_sub=N_SUB,
                 integrator="heun", alloc_method="priority", time_scaling=True, guidance=True,
                 release_axis=False, extra_fail=None,
                 ext_wrench=None, partial_mask=None, partial_gain=1.0,
                 twin=False, twin_tau=5.0, twin_gamma=0.005, log_extra=False,
                 alloc_reconfig=True, class_guidance=True, supervisor=False, mpc=None,
                 plant_mismatch=None, current_fn=None, rpi_iters=30, fail_times=None, init_perturb=None, ff_ref=True, int_outside=True, ei_max=6.0):
        self.__dict__.update(locals()); del self.__dict__["self"]
        if not guidance:
            self.class_guidance = False


# ------------------------------------------------------------------ model-form mismatch (plant only)
def mismatch_wrench(Tact, nu, q, vc, p, mm):
    """Extra body wrench from effects the controller model does not contain.
    inflow : thrust loss with axial inflow, T_eff = T (1 - inflow * clip(v_axial / 0.5, -1, 1))
    hull   : (eta_h, eta_v) thruster-hull interaction efficiency of horizontal / vertical thrusters
    torque : propeller reaction torque Q_i = torque * T_i about each thruster axis, alternating spin sense
    tether : dict(pull=F) catenary-like pull towards the surface anchor (0,0,0) + 20 N s/m lateral drag share
    returns tau_extra (N,6) to ADD to the nominal thruster wrench B @ Tact."""
    N = Tact.shape[0]
    R = rm.quat_to_R(q)
    vr = nu[:, :3] - np.einsum("nji,nj->ni", R, vc)
    T = Tact.copy()
    if mm.get("inflow"):
        va = vr @ rm.E_TH.T                                   # (N,8) inflow along each thruster axis
        T = T * (1 - mm["inflow"] * np.clip(va / 0.5, -1, 1))
    if mm.get("hull"):
        eh, ev = mm["hull"]
        T = T * np.r_[np.full(4, eh), np.full(4, ev)]
    tau = (T - Tact) @ rm.B_ALLOC.T
    if mm.get("torque"):
        spin = np.array([1, -1, -1, 1, 1, -1, -1, 1], float)
        tau[:, 3:] += (mm["torque"] * T * spin) @ rm.E_TH
    if mm.get("tether"):
        pull = mm["tether"].get("pull", 2.0)
        d = -p.copy(); d[:, 2] = -np.maximum(p[:, 2], 0.5) * 0.5          # towards anchor, upward share
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        Fw = pull * d - 2.0 * np.einsum("nij,nj->ni", R, vr) * np.array([1, 1, 0.3])
        tau[:, :3] += np.einsum("nji,nj->ni", R, Fw)
        tau[:, 3:] += np.cross(np.array([-0.2, 0.0, -0.1]), np.einsum("nji,nj->ni", R, Fw))   # tether attach point
    return tau


def simulate(motion, failed_masks, cfg, log_rpm=False):
    """failed_masks: (N,8) bool, True = thruster fails at cfg.t_fail.
    cfg.extra_fail: optional (t2, mask2) second (sequential) failure."""
    rng = np.random.default_rng(cfg.seed)
    N = failed_masks.shape[0]
    dt = cfg.dt_ctrl; h = dt / cfg.n_sub
    P = rm.batch_params(N, rng if cfg.unc > 0 else None, cfg.unc)
    Pn = rm.nominal_params()
    M1n = Pn["m"] + Pn["A1"]; M2n = Pn["I"] + Pn["A2"]
    demand = motion_demand(motion)
    dt_ref = motion.t[1] - motion.t[0]
    T_end = motion.t[-1]
    K_nom = int(T_end / dt)
    long_run = cfg.ftc and (cfg.time_scaling or cfg.supervisor)
    K_max = int((T_end / (KAPPA_MIN if long_run else 1.0) + GRACE) / dt) + 1

    # --- initial state
    p = np.tile(motion.p[0], (N, 1)).astype(float)
    q = rm.euler_to_quat(np.zeros(N), np.zeros(N), np.full(N, motion.psi[0]))
    nu = np.zeros((N, 6))
    if cfg.init_perturb is not None:                  # (N,12): dp, d(euler), dnu
        ip = np.asarray(cfg.init_perturb, float)
        p = p + ip[:, :3]
        q = rm.euler_to_quat(ip[:, 3], ip[:, 4], motion.psi[0] + ip[:, 5])
        nu = nu + ip[:, 6:12]
    n_rpm = np.zeros((N, 8))
    alive = np.ones((N, 8), bool)

    # --- controller state
    ei = np.zeros((N, 3)); ei_att = np.zeros((N, 3))
    det_mask = np.ones((N, 8), bool)
    Qm = np.tile(np.diag(W_NOM), (N, 1, 1))
    alloc = Allocator(det_mask, Qm)
    mode = np.array(["FULL"] * N, dtype=object)
    d0 = np.tile([1.0, 0.0], (N, 1))
    e_unc = np.zeros((N, 2))
    psi_d_state = np.full(N, motion.psi[0]); branch = np.zeros(N); turning = np.zeros(N, bool)
    F_lp = np.zeros((N, 2)); hov_turn = np.zeros(N, bool)
    kappa = np.ones(N); tv = np.zeros(N)
    n_exp = np.zeros((N, 8)); res_cnt = np.zeros((N, 8))
    t_detect = np.full((N, 8), np.nan)
    sat_prev = np.zeros(N, bool)
    mem = {}
    buf = []
    g_par = dict(W=np.full(N, Pn["W"]), B=np.full(N, Pn["B"]), rg=np.zeros((N, 3)), rb=np.tile(Pn["rb"], (N, 1)))
    aborted = np.zeros(N, bool); t_abort = np.full(N, np.nan); z_abort = np.full(N, np.nan)
    p_abort = np.zeros((N, 3)); abort_kind = np.zeros(N, int)     # 1 controlled ascent, 2 passive
    surfaced = np.zeros(N, bool)
    t_ref_end = np.full(N, np.nan); t_arrive = np.full(N, np.nan); settled = np.zeros(N, bool)
    ncmd_fail = np.full((N, 8), np.nan)
    mpc = None
    if cfg.mpc is not None:
        from mpc import FaultAwareMPC
        mpc = FaultAwareMPC(N, **cfg.mpc)
        Tcmd = np.zeros((N, 8))
    ctrl_time = 0.0

    # --- environment
    cs, cd, cz = motion.current
    cs = np.full(N, cs * cfg.current_scale); cd = np.full(N, float(cd))
    if cfg.unc > 0:
        cd = cd + rng.uniform(-0.5, 0.5, N)
        cs = cs * rng.uniform(0.5, 1.5, N)
    cvec = np.stack([np.cos(cd) * cs, np.sin(cd) * cs, np.full(N, cz)], 1)

    # --- logs
    nlog = K_max // cfg.log_every + 2
    f32 = np.float32
    L = dict(t=np.zeros(nlog), tv=np.zeros((nlog, N), f32), p=np.zeros((nlog, N, 3), f32),
             eul=np.zeros((nlog, N, 3), f32), pr=np.zeros((nlog, N, 3), f32), psir=np.zeros((nlog, N), f32),
             sat=np.zeros((nlog, N), bool), thrust=np.zeros((nlog, N, 8), f32), power=np.zeros((nlog, N), f32))
    if log_rpm:
        L["rpm"] = np.zeros((nlog, N, 8), f32); L["rpm_cmd"] = np.zeros((nlog, N, 8), f32)
    pmask = np.zeros((N, 8), bool) if cfg.partial_mask is None else np.asarray(cfg.partial_mask, bool)
    eff = np.ones((N, 8))
    if cfg.twin:
        Ptw = rm.batch_params(N)
        nu_hat = np.zeros((N, 6)); e_f = np.zeros((N, 6)); vc_hat = np.zeros((N, 3))
        L_obs = dt / cfg.twin_tau; a_f = 1 - np.exp(-dt / 0.5)
    if cfg.log_extra:
        for nm, shp in [("nu", 6), ("nu_meas", 6), ("nu_hat", 6), ("innov", 6), ("tau", 6), ("pw8", 8),
                        ("tcmd", 8), ("vr", 3), ("vw", 3), ("vc_hat", 3)]:
            L[nm] = np.zeros((nlog, N, shp), f32)
    energy = np.zeros(N); sat_count = np.zeros(N); act_steps = np.zeros(N)
    li = 0
    fail_k = int(round(cfg.t_fail / dt))
    fail2_k = int(round(cfg.extra_fail[0] / dt)) if cfg.extra_fail is not None else -1
    a_m = 1 - np.exp(-dt / rm.TAU_MOTOR); a_s = 1 - np.exp(-h / rm.TAU_MOTOR)
    kp_a = np.array([GAINS["kp_rp"], GAINS["kp_rp"], GAINS["kp_y"]])
    kd_a = np.array([GAINS["kd_rp"], GAINS["kd_rp"], GAINS["kd_y"]])
    ki_a = np.array([GAINS["ki_rp"], GAINS["ki_rp"], GAINS["ki_y"]])
    em = np.zeros((N, 3))
    n_cmd = np.zeros((N, 8))

    def reconfigure(rows):
        for i in rows:
            info = case_info(det_mask[i])
            if cfg.alloc_reconfig:
                Qm[i] = info["Q"] if cfg.release_axis else np.diag(info["W"])
                alpha = GAINS.get("hs_lat_w", 1.0)
                if info["mode"] == "HEADING_STEERED" and not cfg.release_axis and alpha < 1.0:
                    ed = np.array([-info["d0"][1], info["d0"][0]])
                    Qm[i][0:2, 0:2] = Qm[i][0:2, 0:2] @ (np.eye(2) - (1 - alpha) * np.outer(ed, ed))
                e_unc[i] = np.nan_to_num(info["e_unc"], nan=0.0) if cfg.release_axis else 0.0
            md = info["mode"]
            if not cfg.class_guidance:
                md = "FULL"
            mode[i] = md
            d0[i] = info["d0"]
            psi_d_state[i] = em[i, 2]; branch[i] = 0.0
            if cfg.time_scaling:
                kappa[i] = kappa_for(info, demand, Pn)
            if cfg.supervisor and info["mode"] in ("INFEASIBLE_HORIZONTAL", "INFEASIBLE_DEPTH") and not aborted[i]:
                aborted[i] = True; t_abort[i] = k * dt; z_abort[i] = p[i, 2]; p_abort[i] = pm[i]
                abort_kind[i] = 1 if info["mode"] == "INFEASIBLE_HORIZONTAL" else 2
                mode[i] = info["mode"]          # heading free during the ascent
        if cfg.alloc_reconfig:
            alloc.update_rows(rows, det_mask[rows], Qm[rows])
        if mpc is not None:
            mpc.set_health(rows, det_mask[rows])

    k = 0
    done = np.zeros(N, bool)
    while k < K_max:
        tk = k * dt
        if k >= K_nom and settled.all():
            break
        if cfg.fail_times is not None:                # per-thruster failure schedule (N,8), inf = never
            alive = alive & ~(tk >= cfg.fail_times)
        if k == fail_k:
            alive = ~failed_masks if cfg.fail_times is None else alive
            eff = np.where(pmask, cfg.partial_gain, 1.0)
            ncmd_fail = n_cmd.copy()
        if k == fail2_k:
            alive = alive & ~cfg.extra_fail[1]
        # ---------------- measurements
        eul_true = rm.quat_to_euler(q)
        if cfg.noise:
            pm = p + rng.normal(0, 1, (N, 3)) * [0.02, 0.02, 0.01]
            em = eul_true + rng.normal(0, np.deg2rad(0.3), (N, 3))
            num = nu + rng.normal(0, 1, (N, 6)) * [0.01, 0.01, 0.01, 0.005, 0.005, 0.005]
            nmeas = n_rpm + rng.normal(0, 20, (N, 8))
        else:
            pm, em, num, nmeas = p.copy(), eul_true.copy(), nu.copy(), n_rpm.copy()
        if cfg.delay_steps > 0:
            buf.append((pm, em, num))
            if len(buf) > cfg.delay_steps + 1:
                buf.pop(0)
            pm, em, num = buf[0]
            if cfg.predictor and len(buf) == cfg.delay_steps + 1:
                Rd = euler_R(em); dly = cfg.delay_steps * dt
                pm = pm + np.einsum("nij,nj->ni", Rd, num[:, :3]) * dly
                em = em + num[:, 3:] * dly
        Rm = euler_R(em)
        c0 = _time.perf_counter()

        # ---------------- FDI
        if cfg.ftc and cfg.fdi:
            resid = np.abs(nmeas - n_exp)
            thr = np.maximum(250.0, 0.3 * np.abs(n_exp))
            res_cnt = np.where(resid > thr, res_cnt + 1, 0)
            new_det = (res_cnt * dt >= 0.25) & det_mask
            if new_det.any():
                rows = np.nonzero(new_det.any(1))[0]
                t_detect[new_det] = tk
                det_mask[new_det] = False
                reconfigure(rows)
                ei[rows] *= 0.5
        elif cfg.ftc and not cfg.fdi and (k == fail_k or k == fail2_k):
            det_mask = ~failed_masks if k == fail_k else det_mask & ~cfg.extra_fail[1]
            t_detect = np.where(np.isnan(t_detect) & ~det_mask, tk, t_detect)
            reconfigure(np.arange(N))

        # ---------------- reference at virtual time
        tv = np.minimum(tv + kappa * dt * (k > 0) * ~aborted, T_end)
        j = np.minimum(np.round(tv / dt_ref).astype(int), motion.T - 1)
        done = (tv >= T_end - 1e-9) | surfaced
        nd = (~done).astype(float)
        ref = dict(p=motion.p[j].copy(), v=motion.v[j] * (kappa * nd)[:, None],
                   a=motion.a[j] * (kappa ** 2 * nd)[:, None],
                   psi=motion.psi[j].copy(), r=motion.r[j] * kappa * nd, tv=tv, kappa=kappa)
        if motion.reactive is not None:
            motion.reactive(k, dict(p=pm, p_true=p, eul=em, tv=tv), ref, mem, rng)
        if aborted.any():
            ab = aborted
            zt = np.maximum(ABORT_DEPTH, z_abort - ABORT_ASCENT * (tk - np.nan_to_num(t_abort)))
            ref["p"][ab, :2] = p_abort[ab, :2]
            ref["p"][ab, 2] = np.where(abort_kind[ab] == 1, zt[ab], pm[ab, 2])
            ref["v"][ab] = 0.0
            ref["v"][ab, 2] = np.where((abort_kind[ab] == 1) & (zt[ab] > ABORT_DEPTH), -ABORT_ASCENT, 0.0)
            ref["a"][ab] = 0.0; ref["r"][ab] = 0.0
            surfaced = surfaced | (ab & (p[:, 2] <= ABORT_DEPTH + 0.05))

        # ---------------- position loop
        e = ref["p"] - pm
        vw = np.einsum("nij,nj->ni", Rm, num[:, :3])
        ev = ref["v"] - vw
        e_int = e.copy()
        hs_rows = (mode == "HEADING_STEERED") & (np.hypot(ref["v"][:, 0], ref["v"][:, 1]) < 0.05)
        if hs_rows.any():
            # integrate only along the producible (zero-yaw) thrust axis d0 (world frame) + depth
            psi_m = em[:, 2]
            d0w = np.stack([np.cos(psi_m) * d0[:, 0] - np.sin(psi_m) * d0[:, 1],
                            np.sin(psi_m) * d0[:, 0] + np.cos(psi_m) * d0[:, 1]], 1)
            proj = (e[:, :2] * d0w).sum(1, keepdims=True) * d0w
            e_int[:, :2] = np.where(hs_rows[:, None], proj, e[:, :2])
        ei = np.where(sat_prev[:, None], ei, np.clip(ei + e_int * dt, -cfg.ei_max, cfg.ei_max))
        if cfg.int_outside:
            # saturate the transient (PD + reference) part only; the integral (steady disturbance estimate:
            # current-induced drag, buoyancy error) is added outside the acceleration limit
            acmd = np.clip(ref["a"] + GAINS["kp"] * e + GAINS["kd"] * ev, -GAINS["a_max"], GAINS["a_max"]) + GAINS["ki"] * ei
        else:
            acmd = ref["a"] + GAINS["kp"] * e + GAINS["kd"] * ev + GAINS["ki"] * ei
            acmd = np.clip(acmd, -GAINS["a_max"], GAINS["a_max"])

        # ---------------- guidance per failure class
        psi_d = ref["psi"].copy(); r_d = ref["r"].copy()
        hs = mode == "HEADING_STEERED"
        if hs.any():
            vref_h = np.hypot(ref["v"][:, 0], ref["v"][:, 1])
            eh = np.hypot(e[:, 0], e[:, 1])
            moving = vref_h > 0.05
            # transit: line-of-sight vector (path tangent + cross-track correction)
            # hover  : point the thrust axis at the station error, hysteresis 0.3 m on / 0.15 m off
            vwb = num[:, :3]
            Fdem = acmd[:, :2] * M1n[:2] + np.einsum("nij,nj->ni", Rm, Pn["Dl"][:3] * vwb + Pn["Dq"][:3] * np.abs(vwb) * vwb)[:, :2]
            F_lp[:] = F_lp + (dt / 10.0) * (Fdem - F_lp)
            if GAINS["hover_law"] == "lp":
                Fh = F_lp + 2.0 * e[:, :2]; hover_hold = ~moving & ~turning
            else:
                hov_turn[:] = np.where(eh > 0.3, True, np.where(eh < 0.15, False, hov_turn))
                Fh = e[:, :2] + GAINS["los_ki"] * ei[:, :2]; hover_hold = ~moving & ~hov_turn
            F = np.where(moving[:, None], ref["v"][:, :2] + GAINS["los_k"] * e[:, :2], Fh)
            Fn = np.hypot(F[:, 0], F[:, 1])
            ang = np.arctan2(F[:, 1], F[:, 0]) - np.arctan2(d0[:, 1], d0[:, 0])
            cur = psi_d_state + wrap(ang + branch * np.pi - psi_d_state)
            alt = psi_d_state + wrap(ang + (1 - branch) * np.pi - psi_d_state)
            flip = np.abs(alt - psi_d_state) + np.deg2rad(40) < np.abs(cur - psi_d_state)
            branch = np.where(hs & flip, 1 - branch, branch)
            cand = np.where(flip, alt, cur)
            mis = np.abs(cand - psi_d_state)
            turning = np.where(mis > np.deg2rad(20), True, np.where(mis < np.deg2rad(4), False, turning))
            upd = hs & (Fn > 0.08) & ~hover_hold | (hs & moving)
            rl = np.where(moving, np.deg2rad(GAINS["steer_rate"]), np.deg2rad(GAINS["steer_rate_hover"])) * dt
            step = np.clip(cand - psi_d_state, -rl, rl)
            psi_d_state = np.where(upd, psi_d_state + step, psi_d_state)
            if GAINS.get("hs_hover_yaw", False):
                hold = hs & hover_hold
                ys = np.clip(wrap(ref["psi"] - psi_d_state), -rl, rl)
                psi_d_state = np.where(hold, psi_d_state + ys, psi_d_state)
                step = np.where(hold, ys, step); upd = upd | hold
            psi_d = np.where(hs, psi_d_state, psi_d)
            r_d = np.where(hs, np.where(upd, step / dt, 0.0), r_d)
        yf = (mode == "YAW_FREE") | (mode == "INFEASIBLE_HORIZONTAL") | (mode == "INFEASIBLE_DEPTH")
        psi_d = np.where(yf, em[:, 2], psi_d); r_d = np.where(yf, 0.0, r_d)
        psi_d_state = np.where(hs, psi_d_state, em[:, 2])

        if mpc is None:
            # ---------------- body wrench
            ab_ = np.einsum("nji,nj->ni", Rm, acmd)
            w = num[:, 3:]
            # drag feed-forward from the REFERENCE velocity (body frame). Using the measured velocity here creates
            # positive velocity feedback whenever the plant drag (relative to the water) is smaller than the
            # feed-forward drag (along-track current), which runs away once the acceleration command saturates.
            v1 = np.einsum("nji,nj->ni", Rm, ref["v"]) if cfg.ff_ref else num[:, :3]
            F_b = M1n * ab_ + Pn["Dl"][:3] * v1 + Pn["Dq"][:3] * np.abs(v1) * v1
            # reduced-attitude (tilt) error, valid for any tilt and independent of yaw, + heading error about vertical
            zWb = Rm[:, 2, :]
            tilt_err = -np.cross(zWb, np.array([0.0, 0.0, 1.0]))
            eyaw = wrap(psi_d - em[:, 2])
            eatt = tilt_err + eyaw[:, None] * zWb
            # proportional term: tilt with the roll/pitch gain, heading error (a rotation about the world vertical,
            # mapped to the body through zWb) with the yaw gain and saturated, so that a large heading error on a
            # tilted vehicle cannot leak into roll/pitch with the (4x larger) roll/pitch gain
            eyaw_s = np.clip(eyaw, -EYAW_MAX, EYAW_MAX)
            p_att = GAINS["kp_rp"] * tilt_err + GAINS["kp_y"] * eyaw_s[:, None] * zWb
            e_rp = eatt[:, :2] - e_unc * (eatt[:, :2] * e_unc).sum(1, keepdims=True)
            eint = np.concatenate([e_rp, eatt[:, 2:]], 1)
            ei_att = np.where(sat_prev[:, None], ei_att, np.clip(ei_att + eint * dt, -0.5, 0.5))
            wd = r_d[:, None] * zWb
            M_b = M2n * (p_att + kd_a * (wd - w) + ki_a * ei_att) + Pn["Dl"][3:] * w + Pn["Dq"][3:] * np.abs(w) * w
            g_hat = rm.restoring(Rm, g_par)
            tau_d = np.concatenate([F_b, M_b], 1) - g_hat
            # ---------------- allocation -> RPM
            if cfg.alloc_method == "clip":
                Tcmd = np.clip(np.einsum("nij,nj->ni", alloc.K, tau_d), alloc.lo, alloc.hi)
                Tun = np.einsum("nij,nj->ni", alloc.K, tau_d)
                sat = ((Tun < alloc.lo - 1e-9) | (Tun > alloc.hi + 1e-9)).any(1)
            elif cfg.alloc_method == "rpi":
                Tcmd, sat = alloc.solve(tau_d, method="rpi", n_iter_pg=cfg.rpi_iters)
            else:
                Tcmd, sat = alloc.solve(tau_d, method=cfg.alloc_method)
        else:
            if k % mpc.every == 0:
                ref_m = dict(ref); ref_m["psi"] = psi_d
                Tcmd = mpc.solve(pm, em, num, motion, tv, kappa, ref_m, k, dt)
            sat = mpc.sat
        sat_prev = sat
        n_cmd = np.where(det_mask, rm.thrust_to_rpm(Tcmd), 0.0)
        n_exp = n_exp + a_m * (n_cmd - n_exp)
        ctrl_time += _time.perf_counter() - c0
        if cfg.twin:
            if k == 0:
                nu_hat = num.copy()
            innov = num - nu_hat
            e_f = e_f + a_f * (innov - e_f)
            tau_tw = rm.rpm_to_thrust(n_exp) @ rm.B_ALLOC.T          # twin believes all commanded thrust is delivered
            q_m = rm.euler_to_quat(em[:, 0], em[:, 1], em[:, 2])
            a_hat, R_m = rm.accel(q_m, num, tau_tw, vc_hat, Ptw)
            nu_hat = nu_hat + dt * a_hat + L_obs * innov
            vc_hat = vc_hat + dt * cfg.twin_gamma * np.einsum("nij,nj->ni", R_m, innov[:, :3])

        # ---------------- plant
        if motion.wave > 0:
            fw = motion.wave * np.array([np.sin(2 * np.pi * tk / 6.0), 0.7 * np.cos(2 * np.pi * tk / 7.3),
                                         0.3 * np.sin(2 * np.pi * tk / 5.1)])
        if cfg.current_fn is not None:
            vc = np.broadcast_to(cfg.current_fn(tk), (N, 3)).astype(float)
        else:
            vc = cvec * (1 + 0.2 * np.sin(2 * np.pi * tk / 60.0))
        for _ in range(cfg.n_sub):
            n_rpm = n_rpm + a_s * (np.where(alive, n_cmd, 0.0) - n_rpm)
            Tact = rm.rpm_to_thrust(n_rpm) * P["kth"] * alive * eff
            tau = Tact @ rm.B_ALLOC.T
            if cfg.ext_wrench is not None:
                tau = tau + cfg.ext_wrench(tk)
            if cfg.plant_mismatch is not None:
                tau = tau + mismatch_wrench(Tact, nu, q, vc, p, cfg.plant_mismatch)
            if motion.wave > 0:
                tau = tau.copy(); tau[:, :3] += np.einsum("nji,j->ni", rm.quat_to_R(q), fw)
            if cfg.integrator == "heun":
                nd1, R1 = rm.accel(q, nu, tau, vc, P)
                pd1 = np.einsum("nij,nj->ni", R1, nu[:, :3]); qd1 = rm.quat_dot(q, nu[:, 3:])
                nu2 = nu + h * nd1; q2 = q + h * qd1; q2 /= np.linalg.norm(q2, axis=1, keepdims=True)
                nd2, R2 = rm.accel(q2, nu2, tau, vc, P)
                pd2 = np.einsum("nij,nj->ni", R2, nu2[:, :3]); qd2 = rm.quat_dot(q2, nu2[:, 3:])
                nu = nu + 0.5 * h * (nd1 + nd2); p = p + 0.5 * h * (pd1 + pd2); q = q + 0.5 * h * (qd1 + qd2)
            else:
                def f(qq, vv):
                    ndd, Rr = rm.accel(qq, vv, tau, vc, P)
                    return np.einsum("nij,nj->ni", Rr, vv[:, :3]), rm.quat_dot(qq, vv[:, 3:]), ndd
                k1 = f(q, nu); k2 = f(q + 0.5 * h * k1[1], nu + 0.5 * h * k1[2])
                k3 = f(q + 0.5 * h * k2[1], nu + 0.5 * h * k2[2]); k4 = f(q + h * k3[1], nu + h * k3[2])
                p = p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                q = q + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                nu = nu + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            p[:, 2] = np.maximum(p[:, 2], 0.0)          # free surface
        # uniform completion rule: arrival within 1 m of the final reference point within GRACE s of reference end
        newend = done & np.isnan(t_ref_end) & ~surfaced
        t_ref_end[newend] = tk
        arr = done & ~surfaced & np.isnan(t_arrive) & (np.linalg.norm(p - ref["p"], axis=1) < 1.0) & ~aborted
        t_arrive[arr] = tk
        active = ~settled
        pw = rm.thruster_power(Tact).sum(1)
        energy += pw * dt * active
        sat_count += sat * active; act_steps += active
        if k % cfg.log_every == 0:
            L["t"][li] = tk; L["tv"][li] = tv; L["p"][li] = p; L["eul"][li] = rm.quat_to_euler(q)
            L["pr"][li] = ref["p"]; L["psir"][li] = psi_d; L["sat"][li] = sat
            L["thrust"][li] = Tact; L["power"][li] = pw
            if log_rpm:
                L["rpm"][li] = n_rpm; L["rpm_cmd"][li] = n_cmd
            if cfg.log_extra:
                L["nu"][li] = nu; L["nu_meas"][li] = num; L["tau"][li] = tau
                L["pw8"][li] = rm.thruster_power(Tact); L["tcmd"][li] = Tcmd
                L["vr"][li] = ref["v"]; L["vw"][li] = np.einsum("nij,nj->ni", rm.quat_to_R(q), nu[:, :3])
                if cfg.twin:
                    L["nu_hat"][li] = nu_hat; L["innov"][li] = e_f; L["vc_hat"][li] = vc_hat
            li += 1
        settled = settled | surfaced | np.isfinite(t_arrive) | (np.isfinite(t_ref_end) & (tk - np.nan_to_num(t_ref_end) >= GRACE))
        k += 1
    for key in list(L.keys()):
        if isinstance(L[key], np.ndarray) and L[key].shape[0] == nlog:
            L[key] = L[key][:li]
    L.update(energy_Wh=energy / 3600, sat_frac=sat_count / np.maximum(act_steps, 1), t_detect=t_detect,
             mode_final=mode, det_mask=det_mask, kappa=kappa, t_finish=act_steps * dt,
             aborted=aborted, t_abort=t_abort, z_abort=z_abort, z_final=p[:, 2].copy(), surfaced=surfaced,
             ncmd_fail=ncmd_fail, ctrl_time_per_vehicle_step=ctrl_time / max(k, 1) / N,
             mpc_solve_time=(mpc.solve_time / max(mpc.n_solves, 1)) if mpc is not None else np.nan,
             T_nominal=T_end, t_sim=k * dt, p_final=p.copy(), finished=tv >= T_end - 1e-9,
             t_arrive=t_arrive, t_ref_end=t_ref_end)
    return L


# ------------------------------------------------------------------ metrics
def deviation(motion, p, pr):
    et = np.linalg.norm(p - pr, axis=-1)
    if motion.name == "Wall Following":
        from trajectories import wall_y
        d = np.abs((wall_y(p[..., 0]) - p[..., 1]) - motion.extra["d0"])
        return np.sqrt(d ** 2 + (p[..., 2] - 8.0) ** 2), et
    if motion.name == "Terrain Following":
        from trajectories import seabed_z
        d = np.abs((seabed_z(p[..., 0]) - p[..., 2]) - 3.0)
        return np.sqrt(d ** 2 + p[..., 1] ** 2), et
    if motion.eval_mode == "path":
        tree = cKDTree(motion.path)
        return tree.query(p.reshape(-1, 3))[0].reshape(p.shape[:-1]), et
    return et, et


def metrics(motion, L, failed_masks, t_fail):
    """Path retention (as published) + explicit mission outcome (revision).
    path_ok     : RMS < 0.25 m and max < 1 m over the post-fault, pre-abort, mission-active window
    completed   : not aborted and within 1 m of the final reference point within GRACE s after the reference ends
    success     : completed and path_ok and max tilt < 30 deg
    deadline_*  : completed within 1.5 x / 2 x the nominal mission duration"""
    import warnings
    t = L["t"]
    p, pr, eul = L["p"].astype(float), L["pr"].astype(float), L["eul"].astype(float)
    dev, et = deviation(motion, p, pr)
    t_ab = np.where(np.isnan(L["t_abort"]), np.inf, L["t_abort"]) if "t_abort" in L else np.full(p.shape[1], np.inf)
    post = (t[:, None] >= t_fail) & (L["tv"] < motion.t[-1] - 1e-6) & (t[:, None] < t_ab[None])
    post[-1] |= ~post.any(0)

    def rms(x):
        return np.sqrt(np.where(post, x ** 2, 0).sum(0) / post.sum(0))

    def mx(x):
        return np.where(post, x, -np.inf).max(0)
    tilt = np.rad2deg(np.max(np.abs(eul[..., :2]), axis=-1))
    out = dict(
        rms_dev=rms(dev), max_dev=mx(dev), rms_track=rms(et),
        rms_depth=rms(p[..., 2] - pr[..., 2]),
        rms_yaw_deg=np.rad2deg(rms(wrap(eul[..., 2] - L["psir"]))),
        max_tilt_deg=mx(tilt), final_err=et[-1],
        energy_Wh=L["energy_Wh"], sat_frac=L["sat_frac"], kappa=L["kappa"], t_mission=L["t_finish"],
    )
    td = np.where(failed_masks, L["t_detect"] - t_fail, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["det_delay_mean"] = np.nanmean(td, 1)
        out["det_delay_max"] = np.nanmax(td, 1)
    out["n_failed"] = failed_masks.sum(1)
    out["n_detected"] = (np.isfinite(td) & failed_masks).sum(1)
    out["false_alarms"] = (np.isfinite(L["t_detect"]) & ~failed_masks).sum(1)
    if motion.name == "Obstacle Avoidance":
        from trajectories import OBST
        d = np.linalg.norm(p[..., None, :2] - OBST[None, None, :, :2], axis=3) - OBST[None, None, :, 2]
        out["min_clear"] = d.min((0, 2))
    if motion.name == "Docking Approach":
        out["dock_pos_err"] = np.linalg.norm(p[-1] - motion.extra["dock"], axis=1)
        out["dock_yaw_err_deg"] = np.rad2deg(np.abs(wrap(eul[-1, :, 2])))
    if motion.name == "Adaptive Exploration":
        from trajectories import SRC
        out["dist_to_source"] = np.linalg.norm(p[-1] - SRC, axis=1)
    out["path_ok"] = (out["rms_dev"] < 0.25) & (out["max_dev"] < 1.0)
    out["success_legacy"] = out["path_ok"] & (out["max_tilt_deg"] < 30)
    # ---- explicit mission outcome
    aborted = L["aborted"] if "aborted" in L else np.zeros(p.shape[1], bool)
    finished = L["finished"]
    # end-of-mission position error (reference holds the final planned point after the mission ends)
    if motion.reactive is None:
        end_err = np.linalg.norm(L["p_final"] - motion.p[-1], axis=1)
    else:
        end_err = et[-1]
    T_nom = motion.t[-1]
    out["aborted"] = aborted
    out["end_err"] = end_err
    t_arr = L["t_arrive"] if "t_arrive" in L else np.where(finished & (end_err < 1.0), L["t_finish"], np.nan)
    out["t_arrive"] = t_arr
    out["completed"] = (~aborted) & np.isfinite(t_arr)
    out["deadline_1p5"] = out["completed"] & (np.nan_to_num(t_arr, nan=np.inf) <= 1.5 * T_nom + 1e-6)
    out["deadline_2"] = out["completed"] & (np.nan_to_num(t_arr, nan=np.inf) <= 2.0 * T_nom + 1e-6)
    out["success"] = out["completed"] & out["path_ok"] & (out["max_tilt_deg"] < 30)
    out["duration_ratio"] = t_arr / T_nom
    out["energy_total_Wh"] = L["energy_Wh"] + HOTEL_W * L["t_finish"] / 3600
    return out

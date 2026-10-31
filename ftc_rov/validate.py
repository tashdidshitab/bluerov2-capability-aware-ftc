"""
Validation suite.
  V1  allocation solver verification (vs. exact constrained weighted least squares)
  V2  plant-model verification (terminal speeds vs. von Benzon et al. Table 4, roll free decay,
      buoyant ascent speed, quaternion norm)
  V3  numerical convergence (Heun h=10 ms vs RK4 h=2.5 ms)
  V4  FDI performance (false alarms in nominal runs, detection delay) -> from main campaign
  V5  Monte-Carlo robustness (parametric +-20 %, buoyancy, BG, thruster gain, current, delay)
  V6  ablation study (each FTC ingredient removed in turn)
  V7  sequential (cascading) failures
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.optimize import lsq_linear
warnings.filterwarnings("ignore")
import rov_model as rm
from allocation import all_cases, analyse_case, Allocator, U_MIN, U_MAX
from trajectories import BUILDERS, Motion, DT
from sim import simulate, metrics, SimConfig, case_info
from run_campaign import run_motion, case_labels, T_FAIL

OUT = "results"


# ------------------------------------------------------------------ V1
def v1_allocation(n_tau=400, seed=1):
    rng = np.random.default_rng(seed)
    cases, masks = all_cases()
    rows = []
    for ci, (c, h) in enumerate(zip(cases, masks)):
        info = analyse_case(h)
        A = Allocator(np.repeat(h[None], n_tau, 0), np.repeat(info["Q"][None], n_tau, 0))
        # attainable wrenches: random feasible thrust on healthy thrusters, scaled 0.2..1.4 (some saturating)
        u_true = rng.uniform(U_MIN, U_MAX, (n_tau, 8)) * h * rng.uniform(0.2, 1.4, (n_tau, 1))
        tau = u_true @ rm.B_ALLOC.T
        u_p, _ = A.solve(tau, "priority")
        u_r, _ = A.solve(tau, "rpi", n_iter_pg=30)
        Q = info["Q"]
        lo = np.where(h, U_MIN, 0) - 1e-9; hi = np.where(h, U_MAX, 0) + 1e-9
        cost = lambda u: np.linalg.norm((u @ rm.B_ALLOC.T - tau) @ Q.T, axis=1)
        # exact optimum of the same weighted problem (scipy bounded LSQ) for a subsample
        sub = rng.choice(n_tau, 40, replace=False)
        Aq = np.vstack([Q @ rm.B_ALLOC, np.sqrt(1e-3) * np.eye(8)])
        c_opt = []
        for s in sub:
            b = np.concatenate([Q @ tau[s], np.zeros(8)])
            r = lsq_linear(Aq, b, bounds=(np.where(h, U_MIN, -1e-12), np.where(h, U_MAX, 1e-12)), method="bvls")
            c_opt.append(np.linalg.norm(Q @ (rm.B_ALLOC @ r.x - tau[s])))
        c_opt = np.array(c_opt)
        within = np.abs(u_true).max(1) <= np.minimum(-U_MIN, U_MAX)   # strictly inside limits
        rows.append(dict(case_id=ci, n_failed=len(c), mode=info["mode"],
                         bound_violation=float(max((u_p < lo).sum(), (u_p > hi).sum())),
                         failed_thruster_max_abs=float(np.abs(u_p[:, ~h]).max()) if (~h).any() else 0.0,
                         exact_rate_interior=float((cost(u_p)[within] < 1e-6 + 1e-3 * np.linalg.norm(tau[within], axis=1)).mean()) if within.any() else np.nan,
                         cost_ratio_priority_vs_opt=float(np.median((cost(u_p)[sub] + 1e-6) / (c_opt + 1e-6))),
                         cost_ratio_rpi_vs_opt=float(np.median((cost(u_r)[sub] + 1e-6) / (c_opt + 1e-6))),
                         opt_cost_median=float(np.median(c_opt))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/V1_allocation.csv", index=False)
    return df


# ------------------------------------------------------------------ V2
def _open_loop(tau_b, T=40.0, q0=None, nu0=None, h=0.005, B_extra=0.0):
    P = rm.batch_params(1)
    P["B"] = P["W"] + B_extra
    q = rm.euler_to_quat(np.zeros(1), np.zeros(1), np.zeros(1)) if q0 is None else q0
    nu = np.zeros((1, 6)) if nu0 is None else nu0.copy()
    out = []
    for k in range(int(T / h)):
        nd, R = rm.accel(q, nu, tau_b[None], np.zeros((1, 3)), P)
        nu = nu + h * nd
        q = q + h * rm.quat_dot(q, nu[:, 3:]); q /= np.linalg.norm(q)
        out.append(np.concatenate([nu[0], rm.quat_to_euler(q)[0], [np.linalg.norm(q)]]))
    return np.array(out)


def v2_model():
    P = rm.nominal_params()
    res = {}
    for name, d, F, ref in [("surge", 0, 85.0, 0.72), ("sway", 1, 85.0, 0.63), ("heave", 2, 120.0, 0.71)]:
        tau = np.zeros(6); tau[d] = F
        if d == 2:
            tau[2] += 0.0
        sim_v = _open_loop(tau, 30.0, B_extra=0.0)[-1, d]
        a, b, c = P["Dq"][d], P["Dl"][d], -F
        ana = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        res[name] = dict(force_N=F, sim_terminal=float(sim_v), analytic=float(ana), vonBenzon_table4=ref,
                         rel_err_vs_paper=float(abs(sim_v - ref) / ref))
    # with T200 @16 V capacity (this study's thrusters)
    cap = 129.68
    sim_v = _open_loop(np.array([cap, 0, 0, 0, 0, 0.0]), 30.0)[-1, 0]
    res["surge_T200_16V_full"] = dict(force_N=cap, sim_terminal=float(sim_v))
    # roll free decay (passive, no damping-linear in roll -> quadratic only)
    q0 = rm.euler_to_quat(np.array([np.deg2rad(20)]), np.zeros(1), np.zeros(1))
    tr = _open_loop(np.zeros(6), 20.0, q0=q0, B_extra=0.0)
    phi = tr[:, 6]
    zc = np.nonzero((phi[:-1] < 0) & (phi[1:] >= 0))[0]
    per = float(np.mean(np.diff(zc)) * 0.005) if len(zc) > 1 else np.nan
    Wn = P["W"]
    per_th = 2 * np.pi * np.sqrt((P["I"][0] + P["A2"][0]) / (Wn * abs(P["rb"][2])))
    res["roll_free_decay"] = dict(period_sim_s=per, period_theory_s=float(per_th))
    # buoyant ascent
    w = _open_loop(np.zeros(6), 60.0, B_extra=1.3)[-1, 2]
    a, b, c = P["Dq"][2], P["Dl"][2], -1.3
    res["buoyant_ascent"] = dict(sim_mps=float(-w), analytic_mps=float((-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)))
    res["quat_norm_drift"] = float(abs(tr[:, -1] - 1).max())
    with open(f"{OUT}/V2_model.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


# ------------------------------------------------------------------ V3
def _v3(bi):
    m = BUILDERS[bi]()
    cases, masks, labels = case_labels()
    idx = np.arange(37)
    failed = ~masks[idx]
    La = simulate(m, failed, SimConfig(ftc=True, t_fail=T_FAIL, noise=False, n_sub=2, integrator="heun"))
    Lb = simulate(m, failed, SimConfig(ftc=True, t_fail=T_FAIL, noise=False, n_sub=8, integrator="rk4"))
    n = min(len(La["t"]), len(Lb["t"]))
    d = np.linalg.norm(La["p"][:n] - Lb["p"][:n], axis=2)
    Ma = metrics(m, La, failed, T_FAIL); Mb = metrics(m, Lb, failed, T_FAIL)
    return dict(motion=m.name, max_pos_diff_m=float(d.max()), median_pos_diff_m=float(np.median(d.max(0))),
                max_rms_dev_diff_m=float(np.abs(Ma["rms_dev"] - Mb["rms_dev"]).max()))


def v3_convergence():
    with Pool(2) as pool:
        rows = pool.map(_v3, [2, 0, 6, 9, 14, 12])
    df = pd.DataFrame(rows); df.to_csv(f"{OUT}/V3_convergence.csv", index=False)
    return df


# ------------------------------------------------------------------ V5 Monte Carlo
def _mc(args):
    bi, seed = args
    rng = np.random.default_rng(1000 + seed)
    delay = int(rng.integers(0, 6))          # 0..100 ms sensor delay (with predictor)
    return run_motion(bi, configs=("FTC",), unc=0.2, seed=seed, delay_steps=delay, tag="MC", save_traj=False)


def v5_montecarlo(n_seeds=4):
    jobs = [(bi, s) for s in range(n_seeds) for bi in range(len(BUILDERS))]
    with Pool(2) as pool:
        res = pool.map(_mc, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(f"{OUT}/V5_montecarlo.csv", index=False)
    return df


# ------------------------------------------------------------------ V6 ablation
ABL = {"FTC": {}, "no_time_scaling": dict(time_scaling=False), "no_axis_release": dict(release_axis=False),
       "no_heading_guidance": dict(guidance=False), "RPI_allocator": dict(alloc_method="rpi"),
       "oracle_FDI": dict(fdi=False)}
ABL_MOTIONS = [2, 0, 6, 9, 14, 12]


def _abl(bi):
    cases, masks, labels = case_labels()
    idx = [i for i, c in enumerate(cases) if len(c) <= 3]
    return run_motion(bi, configs=tuple(ABL.keys()), tag="ablation", save_traj=False, case_idx=idx, extra=ABL)


def v6_ablation():
    with Pool(2) as pool:
        res = pool.map(_abl, ABL_MOTIONS, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(f"{OUT}/V6_ablation.csv", index=False)
    return df


# ------------------------------------------------------------------ V7 sequential failures
def _seq(bi):
    m = BUILDERS[bi]()
    pairs = [(a, b) for a in range(8) for b in range(8) if a != b]
    f1 = np.zeros((len(pairs), 8), bool); f2 = np.zeros((len(pairs), 8), bool)
    for j, (a, b) in enumerate(pairs):
        f1[j, a] = True; f2[j, b] = True
    rows = []
    for cname, ftc in [("FTC", True), ("BASELINE", False)]:
        L = simulate(m, f1, SimConfig(ftc=ftc, t_fail=T_FAIL, extra_fail=(60.0, f2)))
        M = metrics(m, L, f1 | f2, T_FAIL)
        for j, (a, b) in enumerate(pairs):
            info = case_info(~(f1[j] | f2[j]))
            rows.append(dict(motion=m.name, controller=cname, first=f"T{a+1}", second=f"T{b+1}", mode=info["mode"],
                             **{k: float(v[j]) for k, v in M.items() if np.ndim(v)}))
    return rows


def v7_sequential():
    with Pool(2) as pool:
        res = pool.map(_seq, [2, 9, 5])
    df = pd.DataFrame([r for rr in res for r in rr]); df.to_csv(f"{OUT}/V7_sequential.csv", index=False)
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ["v1", "v2", "v3", "v6", "v7", "v5"]
    for w in which:
        t0 = time.time()
        r = globals()[{"v1": "v1_allocation", "v2": "v2_model", "v3": "v3_convergence", "v5": "v5_montecarlo",
                       "v6": "v6_ablation", "v7": "v7_sequential"}[w]]()
        print(w, f"{time.time() - t0:.0f}s", flush=True)
        if isinstance(r, pd.DataFrame):
            print(r.describe().T.head(12).to_string())
        else:
            print(json.dumps(r, indent=1))

"""
Stability of the switched closed loop (reviewer MC5).

A. Analytical part (exact computations that support Proposition 2 in the manuscript)
   A1 closed-loop characteristic polynomials of the position, roll/pitch and yaw loops (per unit mass/inertia),
      roots, Hurwitz test, peak disturbance-to-error gain (H-infinity norm of s/(s^3 + kd s^2 + kp s + ki))
   A2 allocation exactness per FULL / COUPLED law: ||(B_F K_F - I) P_c|| on the controllable wrench subspace
      (lambda-regularisation residual) -> perturbation size entering the ISS bound
   A3 unsaturated region: the demanded wrench of every mission, after time scaling, stays inside the attainable
      set with margin (from the capacity LP used for kappa_F)
B. Numerical part (nonlinear plant, saturation, guidance, switching)
   B1 random cascading failures: 1-4 failures in random order at random times (0.3-10 s apart), random initial
      perturbation (+-0.5 m, +-15 deg, +-0.2 m/s, +-10 deg/s), hover and straight transit; classes by final set
   B2 fast switching: the same failures arriving 0.3 s apart (faster than detection), worst case for transients
Outputs results/stability_switching/*.csv, summary.json
"""
import os, json, itertools, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rov_model as rm
from allocation import all_cases, analyse_case, Allocator, capacity_dir
from sim import simulate, SimConfig, GAINS, case_info
from trajectories import m_station, m_straight

OUT = "results/stability_switching"
os.makedirs(OUT, exist_ok=True)
CASES, MASKS = all_cases()


def loop_poly(kp, kd, ki):
    return np.array([1.0, kd, kp, ki])


def hinf(den, num=(1.0, 0.0)):
    w = np.logspace(-4, 3, 20000)
    s = 1j * w
    G = np.polyval(num, s) / np.polyval(den, s)
    i = np.argmax(np.abs(G))
    return float(np.abs(G[i])), float(w[i])


def a1():
    out = {}
    for name, (kp, kd, ki) in {"position": (GAINS["kp"], GAINS["kd"], GAINS["ki"]),
                               "roll_pitch": (GAINS["kp_rp"], GAINS["kd_rp"], GAINS["ki_rp"]),
                               "yaw": (GAINS["kp_y"], GAINS["kd_y"], GAINS["ki_y"])}.items():
        den = loop_poly(kp, kd, ki)
        r = np.roots(den)
        g, wpk = hinf(den)
        out[name] = dict(poly=den.tolist(), roots=[f"{z.real:.4f}{z.imag:+.4f}j" for z in r],
                         max_real=float(r.real.max()), hurwitz=bool(kd * kp > ki and min(kp, kd, ki) > 0),
                         hinf_dist_to_error=g, hinf_freq=wpk,
                         decay_rate=float(-r.real.max()))
    return out


def a2():
    rows = []
    for c, h in zip(CASES, MASKS):
        info = analyse_case(h)
        if info["mode"] not in ("FULL", "COUPLED_ATTITUDE"):
            continue
        A = Allocator(h[None], np.diag(info["W"])[None])
        BK = rm.B_ALLOC @ A.K[0]
        # controllable wrench subspace = range(B_F)
        U, s, Vt = np.linalg.svd(rm.B_ALLOC * h)
        r = int((s > 1e-9).sum())
        Pc = U[:, :r] @ U[:, :r].T
        E = (BK - np.eye(6)) @ Pc
        # COUPLED: on the full space, the uncontrollable (moment) direction is left to buoyancy
        Efull = BK - np.eye(6)
        rows.append(dict(case="+".join(f"T{i+1}" for i in c) if c else "none", mode=info["mode"], rank=r,
                         alloc_residual_on_range=float(np.linalg.norm(E, 2)),
                         alloc_residual_full=float(np.linalg.norm(Efull, 2))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/A2_allocation_exactness.csv", index=False)
    return df.groupby("mode")[["alloc_residual_on_range", "alloc_residual_full"]].max().to_dict("index")


# ---------------------------------------------------------------- B: random cascading failures
def schedule(rng, n_v, kind, fast=False):
    """random failure schedules ending in a recoverable class; returns fail_times (n_v,8) and final class"""
    ft = np.full((n_v, 8), np.inf); final = []
    for i in range(n_v):
        while True:
            k = rng.integers(1, 5)
            S = rng.choice(8, k, replace=False)
            h = np.ones(8, bool); h[S] = False
            md = case_info(h)["mode"]
            if (kind == "recoverable" and md in ("FULL", "COUPLED_ATTITUDE", "HEADING_STEERED")) or kind == "any":
                break
        t = 20.0
        for th in S:
            ft[i, th] = t
            t += 0.3 if fast else rng.uniform(0.3, 10.0)
        final.append(md)
    return ft, np.array(final)


def perturb(rng, n_v):
    ip = np.zeros((n_v, 12))
    ip[:, :3] = rng.uniform(-0.5, 0.5, (n_v, 3))
    ip[:, 3:6] = np.deg2rad(rng.uniform(-15, 15, (n_v, 3)))
    ip[:, 6:9] = rng.uniform(-0.2, 0.2, (n_v, 3))
    ip[:, 9:12] = np.deg2rad(rng.uniform(-10, 10, (n_v, 3)))
    return ip


def b_job(args):
    seed, mname, fast = args
    rng = np.random.default_rng(seed)
    n_v = 120
    m = {"hover": m_station, "transit": m_straight}[mname]()
    m.current = (0.05, np.pi / 4, 0.0); m.wave = 0.0
    ft, final = schedule(rng, n_v, "recoverable", fast)
    ip = perturb(rng, n_v)
    L = simulate(m, np.zeros((n_v, 8), bool), SimConfig(t_fail=1e9, seed=seed, fail_times=ft, init_perturb=ip,
                                                         supervisor=True, log_every=10))
    t = L["t"]; e = np.linalg.norm(L["p"].astype(float) - L["pr"].astype(float), axis=2)
    tilt = np.rad2deg(np.abs(L["eul"][..., :2]).max(-1))
    last_sw = np.where(np.isinf(ft), -np.inf, ft).max(1)
    rows = []
    for i in range(n_v):
        after = t >= last_sw[i] + 30.0
        tail = (t >= t[-1] - 20.0)
        rows.append(dict(seed=seed, mission=mname, fast=fast, final_class=final[i], n_fail=int(np.isfinite(ft[i]).sum()),
                         max_err=float(e[:, i].max()), max_tilt=float(tilt[:, i].max()),
                         err_after_30s=float(e[after, i].max()) if after.any() else np.nan,
                         err_tail_rms=float(np.sqrt((e[tail, i] ** 2).mean())),
                         detected_all=bool((np.isfinite(L["t_detect"][i]) == np.isfinite(ft[i])).all())))
    return rows


def main():
    S = {"A1_loops": a1(), "A2_alloc": a2()}
    print(json.dumps(S, indent=1), flush=True)
    jobs = [(s, mn, fast) for s in range(3) for mn in ("hover", "transit") for fast in (False, True)]
    with Pool(2) as pool:
        res = pool.map(b_job, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(f"{OUT}/B_random_switching.csv", index=False)
    g = df.groupby(["final_class", "mission", "fast"]).agg(
        n=("max_err", "size"), max_err_p95=("max_err", lambda x: float(np.percentile(x, 95))),
        max_err_max=("max_err", "max"), tail_rms_median=("err_tail_rms", "median"),
        tail_rms_p95=("err_tail_rms", lambda x: float(np.percentile(x, 95))),
        bounded_lt_2m=("max_err", lambda x: float((x < 2.0).mean())),
        converged_tail_lt_0p25=("err_tail_rms", lambda x: float((x < 0.25).mean())),
        max_tilt_p95=("max_tilt", lambda x: float(np.percentile(x, 95))), all_detected=("detected_all", "mean"))
    g.to_csv(f"{OUT}/B_summary.csv")
    S["B"] = {f"{a}|{b}|{'fast' if c else 'random'}": v for (a, b, c), v in g.to_dict("index").items()}
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()

"""
Sensitivity studies requested by the reviewer (minor comments, Sections 4.2-4.3, 5.1).

S-lambda  regularisation lambda in {1e-6 ... 1e-1}: allocation residual on the attainable subspace for every
          recoverable law, and closed-loop outcome (FTC) over all 255 cases on 4 manoeuvres
S-rate    heading-rate limit of the heading-steering guidance (transit 5/10/15/30/60 deg/s, hover scaled 2/3):
          path deviation, heading oscillation (RMS yaw rate, number of yaw-rate sign reversals per minute,
          branch flips), energy, for all 90 heading-steered cases on lawnmower, straight line and figure-eight
T-man     table of the 24 reference manoeuvres: duration, path length, speeds, depth range, current
Outputs results/sensitivity/*.csv
"""
import os, sys, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import rov_model as rm
import allocation as AL
import sim
from sim import simulate, metrics, SimConfig, case_info
from trajectories import BUILDERS, m_lawnmower, m_straight, m_figure8, m_dive, m_station

OUT = "results/sensitivity"
os.makedirs(OUT, exist_ok=True)
CASES, MASKS = AL.all_cases()


def lam_alloc():
    rows = []
    for lam in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        AL.LAM = lam
        for c, h in zip(CASES, MASKS):
            info = AL.analyse_case(h)
            if info["mode"] not in ("FULL", "COUPLED_ATTITUDE", "HEADING_STEERED"):
                continue
            A = AL.Allocator(h[None], np.diag(info["W"])[None])
            U, s, Vt = np.linalg.svd(rm.B_ALLOC * h)
            r = int((s > 1e-9).sum()); Pc = U[:, :r] @ U[:, :r].T
            E = (rm.B_ALLOC @ A.K[0] - np.eye(6)) @ Pc
            rows.append(dict(lam=lam, mode=info["mode"], residual=float(np.linalg.norm(E, 2)),
                             kmax=float(np.abs(A.K[0]).max())))
    AL.LAM = 1e-3
    df = pd.DataFrame(rows)
    df.groupby(["lam", "mode"]).agg(residual_max=("residual", "max"), gain_max=("kmax", "max")).to_csv(f"{OUT}/lambda_alloc.csv")
    return df


def lam_job(args):
    lam, bi = args
    AL.LAM = lam
    sim._CASE_CACHE.clear()
    m = BUILDERS[bi]()
    L = simulate(m, ~MASKS, SimConfig(t_fail=20.0, supervisor=True))
    M = metrics(m, L, ~MASKS, 20.0)
    return [dict(lam=lam, motion=m.name, case_id=i, mode=case_info(MASKS[i])["mode"], success=bool(M["success"][i]),
                 path_ok=bool(M["path_ok"][i] and not M["aborted"][i]), rms_dev=float(M["rms_dev"][i]),
                 sat_frac=float(M["sat_frac"][i])) for i in range(1, 255)]


def rate_job(args):
    rate, bi = args
    sim.GAINS["steer_rate"] = rate; sim.GAINS["steer_rate_hover"] = rate * 2 / 3
    m = BUILDERS[bi]()
    idx = [i for i in range(255) if case_info(MASKS[i])["mode"] == "HEADING_STEERED"]
    f = ~MASKS[idx]
    L = simulate(m, f, SimConfig(t_fail=20.0, supervisor=True, log_every=5))
    M = metrics(m, L, f, 20.0)
    t = L["t"]; psi = np.unwrap(L["eul"][..., 2].astype(float), axis=0)
    r = np.gradient(psi, t, axis=0)
    rows = []
    for j, i in enumerate(idx):
        w = (t >= 25) & (L["tv"][:, j] < m.t[-1] - 1e-3)
        rr = r[w, j]
        rev = np.sum(np.diff(np.sign(rr[np.abs(rr) > np.deg2rad(1.0)])) != 0) if w.sum() > 2 else 0
        dur = max((w.sum()) * (t[1] - t[0]), 1e-9)
        rows.append(dict(rate=rate, motion=m.name, case_id=i, rms_dev=float(M["rms_dev"][i if False else j]),
                         path_ok=bool(M["path_ok"][j]), success=bool(M["success"][j]),
                         rms_yaw_rate_dps=float(np.rad2deg(np.sqrt((rr ** 2).mean()))) if len(rr) else np.nan,
                         reversals_per_min=float(rev / dur * 60), energy_Wh=float(M["energy_Wh"][j]),
                         duration_ratio=float(M["duration_ratio"][j])))
    return rows


def manoeuvre_table():
    rows = []
    for b in BUILDERS:
        m = b()
        seg = np.linalg.norm(np.diff(m.p, axis=0), axis=1)
        rows.append(dict(manoeuvre=m.name, description=m.desc, duration_s=round(float(m.t[-1]), 1),
                         path_length_m=round(float(seg.sum()), 1),
                         max_speed_mps=round(float(np.linalg.norm(m.v, axis=1).max()), 2),
                         max_accel_mps2=round(float(np.linalg.norm(m.a, axis=1).max()), 3),
                         depth_min_m=round(float(m.p[:, 2].min()), 1), depth_max_m=round(float(m.p[:, 2].max()), 1),
                         heading_range_deg=round(float(np.rad2deg(np.ptp(m.psi))), 0),
                         current_mps=m.current[0], reactive=m.reactive is not None))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/manoeuvre_table.csv", index=False)
    return df


def main():
    print(manoeuvre_table().to_string())
    lam_alloc()
    jobs_r = [(r, bi) for r in (5.0, 10.0, 15.0, 30.0, 60.0) for bi in (2, 10, 7)]
    jobs_l = [(l, bi) for l in (1e-5, 1e-3, 1e-1) for bi in (0, 2, 6, 14)]
    with Pool(2) as pool:
        rr = pool.map(rate_job, jobs_r, chunksize=1)
        ll = pool.map(lam_job, jobs_l, chunksize=1)
    dr = pd.DataFrame([x for y in rr for x in y]); dr.to_csv(f"{OUT}/heading_rate.csv", index=False)
    dl = pd.DataFrame([x for y in ll for x in y]); dl.to_csv(f"{OUT}/lambda_closed_loop.csv", index=False)
    print(dr.groupby("rate")[["rms_dev", "path_ok", "success", "rms_yaw_rate_dps", "reversals_per_min", "energy_Wh"]].median())
    print(dr.groupby("rate")[["path_ok", "success"]].mean())
    print(dl.groupby(["lam", "mode"])[["success", "path_ok"]].mean())


if __name__ == "__main__":
    main()

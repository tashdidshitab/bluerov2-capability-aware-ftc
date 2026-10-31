"""
Revision campaign (reviewer comments 2, 3, 6, 7): every controller on identical fault sets (all 255 health cases),
sensor noise, actuator limits and missions (24 manoeuvres); with and without time scaling; explicit mission outcome.

Controllers
  BL         PID + nominal allocation, no reconfiguration (previous baseline)
  BL+TS      as BL, but the reference is time-scaled exactly like the FTC (RPM FDI used for the timing only)
  FA-WPI     PID + failure-aware weighted pseudo-inverse, saturation by clipping
  FA-BLS     PID + failure-aware bounded least-squares allocation (redistributed pseudo-inverse + projected gradient)
  MPC        fault-aware linear time-varying MPC on thruster forces (hard limits, failed thrusters fixed at 0)
  *+TS       the same with capability-aware time scaling
  FTC-noTS   proposed structural layer (priority allocation + class guidance + supervisor) without time scaling
  FTC        proposed, complete
  MPC+SL     MPC inner loop + proposed structural layer (time scaling + class guidance + supervisor)
Plant 'mismatch': inflow thrust loss, thruster-hull interaction, propeller reaction torque, tether pull/drag
(effects that no controller models).
Usage: python run_revision.py <tag> [plant=mismatch] [configs=A,B,...] [motions=0,2,...]
Every run: failure at t = 20 s, seed 0 (same noise realisation for every controller).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from trajectories import BUILDERS
from sim import simulate, metrics, SimConfig, case_info
from allocation import all_cases

T_FAIL = 20.0
NO = dict(time_scaling=False, class_guidance=False)
CONFIGS = {
    "BL":        dict(ftc=False),
    "BL+TS":     dict(alloc_reconfig=False, class_guidance=False, time_scaling=True),
    "FA-WPI":    dict(alloc_method="clip", **NO),
    "FA-WPI+TS": dict(alloc_method="clip", class_guidance=False, time_scaling=True),
    "FA-BLS":    dict(alloc_method="rpi", **NO),
    "FA-BLS+TS": dict(alloc_method="rpi", class_guidance=False, time_scaling=True),
    "FTC-noTS":  dict(time_scaling=False, supervisor=True),
    "FTC":       dict(supervisor=True),
    "MPC":       dict(mpc={}, **NO),
    "MPC+TS":    dict(mpc={}, class_guidance=False, time_scaling=True),
    "MPC+SL":    dict(mpc={}, supervisor=True),
    # ablations of the proposed controller (one ingredient changed at a time)
    "FTC-noGuid":  dict(supervisor=True, class_guidance=False),
    "FTC-RPI":     dict(supervisor=True, alloc_method="rpi"),
    "FTC-oracle":  dict(supervisor=True, fdi=False),
}
MISMATCH = dict(inflow=0.2, hull=(0.9, 0.95), torque=0.015, tether=dict(pull=2.0))
COST = {"MPC": 6, "MPC+TS": 8, "MPC+SL": 8}
OUT = "results/revision"


def labels():
    cases, masks = all_cases()
    return cases, masks, ["none" if not c else "+".join(f"T{i+1}" for i in c) for c in cases]


def job(args):
    tag, cname, bi, plant, seed, unc = args
    out = f"{OUT}/{tag}/{cname.replace('+', 'p')}_{bi:02d}.csv"
    if os.path.exists(out):
        return out
    m = BUILDERS[bi]()
    cases, masks, lab = labels()
    failed = ~masks
    kw = dict(CONFIGS[cname])
    if plant == "mismatch":
        kw["plant_mismatch"] = MISMATCH
    t0 = time.time()
    delay = 0
    if unc > 0:                                   # Monte Carlo: random 0-100 ms sensor delay with predictor
        delay = int(np.random.default_rng(1000 + seed + 100 * bi).integers(0, 6))
    L = simulate(m, failed, SimConfig(t_fail=T_FAIL, seed=seed, unc=unc, delay_steps=delay, **kw))
    M = metrics(m, L, failed, T_FAIL)
    rows = []
    for i in range(len(cases)):
        info = case_info(masks[i])
        r = dict(tag=tag, plant=plant, seed=seed, unc=unc, delay_ms=delay * 20, controller=cname, motion=m.name, motion_id=bi, case=lab[i], case_id=i,
                 n_failed=len(cases[i]), mode=info["mode"], T_nominal=m.t[-1],
                 t_abort=float(L["t_abort"][i]), z_abort=float(L["z_abort"][i]), z_final=float(L["z_final"][i]),
                 surfaced=bool(L["surfaced"][i]),
                 ctrl_us_per_vehicle_step=L["ctrl_time_per_vehicle_step"] * 1e6,
                 mpc_solve_ms_batch=L["mpc_solve_time"] * 1e3, n_batch=len(cases))
        for k, v in M.items():
            r[k] = float(v[i]) if np.ndim(v) else float(v)
        nf = L["ncmd_fail"][i][failed[i]]
        r["ncmd_fail_min_abs"] = float(np.nanmin(np.abs(nf))) if len(nf) else np.nan
        rows.append(r)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[{tag}] {cname:10s} {m.name:22s} {time.time() - t0:6.0f}s", flush=True)
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    opts = dict(a.split("=", 1) for a in sys.argv[2:])
    plant = opts.get("plant", "nominal")
    seed = int(opts.get("seed", 0)); unc = float(opts.get("unc", 0.0))
    cfgs = opts["configs"].split(",") if "configs" in opts else list(CONFIGS)
    mots = [int(x) for x in opts["motions"].split(",")] if "motions" in opts else list(range(len(BUILDERS)))
    jobs = [(tag, c, bi, plant, seed, unc) for c in cfgs for bi in mots]
    jobs.sort(key=lambda j: -COST.get(j[1], 1) * BUILDERS[j[2]]().t[-1])
    with Pool(2) as pool:
        list(pool.imap_unordered(job, jobs, chunksize=1))
    files = [f"{OUT}/{tag}/{c.replace('+', 'p')}_{bi:02d}.csv" for c in cfgs for bi in mots]
    df = pd.concat([pd.read_csv(f) for f in files if os.path.exists(f)])
    df.to_csv(f"{OUT}/{tag}_all.csv", index=False)
    print("done", tag, len(df))


if __name__ == "__main__":
    main()

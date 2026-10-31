"""
Digital-twin stress tests (revision, reviewer MC4).

Calibration and evaluation are disjoint:
  CAL  fault-free runs used ONLY to set the per-axis thresholds: 5 manoeuvres x 12 vehicles, random constant
       current 0-0.25 m/s (random direction) + slow sinusoidal drift, noise seed 100
  everything below is evaluation (different seeds, currents and scenarios)
Fault onset t0 = 10 s unless stated. Reported delay = DECLARATION time (end of the 0.5 s persistence window) - onset.

  S1 single partial loss 5..75 % on each thruster, still water, 3 noise seeds
  S2 same at 0.2 m/s current (45 deg) and 0.3 m/s cross-current (90 deg)
  S3 same under plant model-form mismatch (inflow loss, hull efficiency, reaction torque, tether)
  S4 simultaneous dual partial faults (all 28 pairs) 50+50 % and 25+25 %
  S5 fault-like disturbances, no fault: current step 0.1/0.2/0.3 m/s (8 directions, 2 s ramp), tether pull step
     5/10/20 N, 1 s 30 N bump, parameter mismatch +-10/20 %, model-form mismatch
  S6 fault-free exposure for the false-alarm rate: 5 manoeuvres x 24 vehicles, random current 0-0.25 m/s with
     slow drift (independent seed 300)
Outputs data/stress/*.npz, data/stress_rows.csv, data/stress_thresholds.npy
"""
import os, sys, copy, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ftc_rov")); sys.path.insert(0, HERE)
import experiments as EX
import twin_detect as TD
from sim import simulate, SimConfig
from trajectories import m_lawnmower

OUT = os.path.join(HERE, "data", "stress")
os.makedirs(OUT, exist_ok=True)
MISMATCH = dict(inflow=0.2, hull=(0.9, 0.95), torque=0.015, tether=dict(pull=2.0))
T0 = 10.0
MOTIONS = (EX.m_transit, m_lawnmower, EX.m_dive_turn, EX.m_speed_updown, EX.m_hover_perturb)


def with_current(m, speed, ang):
    m = copy.copy(m); m.current = (speed, ang, 0.0); return m


def random_current_fn(rng, NV):
    spd = rng.uniform(0, 0.25, NV); ang = rng.uniform(0, 2 * np.pi, NV)
    base = np.stack([spd * np.cos(ang), spd * np.sin(ang), np.zeros(NV)], 1)
    drift = rng.uniform(0, 0.05, NV)[:, None] * np.stack([np.cos(ang + 1), np.sin(ang + 1), np.zeros(NV)], 1)
    per = rng.uniform(120, 600, NV)[:, None]
    return lambda tk: base + drift * np.sin(2 * np.pi * tk / per)


def raw(name, motion, failed, **kw):
    """simulate once and cache innovation + commanded thrust"""
    f = f"{OUT}/{name}_raw.npz"
    if os.path.exists(f):
        z = np.load(f); return {k: z[k] for k in z.files}
    t0 = time.time()
    cfg = SimConfig(ftc=True, t_fail=kw.pop("t_fail", T0), twin=True, log_extra=True, log_every=EX.LOG_EVERY,
                    seed=kw.pop("seed", 11), **kw)
    L = simulate(motion, failed, cfg, log_rpm=False)
    out = dict(t=L["t"], innov=L["innov"].astype(np.float32), tcmd=L["tcmd"].astype(np.float32),
               t_det_rpm=np.nanmin(np.where(np.isfinite(L["t_detect"]), L["t_detect"], np.inf), 1),
               vc_hat=L["vc_hat"][-1])
    np.savez_compressed(f, **out)
    print(f"{name:40s} N={failed.shape[0]:3d} {time.time()-t0:5.0f}s", flush=True)
    return out


def evaluate(o, thr, partial=True):
    tf, r = TD.detect(o["t"], o["innov"].astype(float), thr)
    res = dict(t_flag=tf, peak=np.abs(r / thr).max((0, 2)))
    if partial:
        iso, est = TD.isolate(o["t"], r, o["tcmd"].astype(float), tf)
        res.update(iso=iso, est=est, r=r)
    return res


def main():
    rng = np.random.default_rng(100)
    # ---------------- calibration (fault-free only)
    cal = []
    for mf in MOTIONS:
        m = with_current(mf(), 0.0, 0.0)
        o = raw(f"CAL_{mf.__name__}", m, np.zeros((12, 8), bool), current_fn=random_current_fn(rng, 12), t_fail=1e9,
                seed=100, ext_wrench=(EX.perturbation if mf is EX.m_hover_perturb else None))
        cal.append((o["t"], o["innov"].astype(float)))
    THR = TD.thresholds(cal)
    np.save(os.path.join(HERE, "data", "stress_thresholds.npy"), THR)
    print("thresholds", np.round(THR, 4), flush=True)
    rows = []
    tr = EX.m_transit()

    def singles(tag, motion, losses, seeds, **kw):
        for loss in losses:
            for sd in seeds:
                o = raw(f"{tag}_L{int(round(loss*100)):02d}_s{sd}", motion, np.zeros((8, 8), bool),
                        partial_mask=np.eye(8, dtype=bool), partial_gain=1 - loss, seed=sd, **kw)
                ev = evaluate(o, THR)
                for i in range(8):
                    rows.append(dict(scenario=tag, kind="single", loss=loss, seed=sd, true=f"T{i+1}",
                                     flag_t=ev["t_flag"][i], delay=ev["t_flag"][i] - T0,
                                     pred=(f"T{ev['iso'][i]+1}" if ev["iso"][i] >= 0 else ("unisolated" if np.isfinite(ev["t_flag"][i]) else "none")),
                                     est=ev["est"][i], rpm_flag=bool(np.isfinite(o["t_det_rpm"][i])),
                                     exposure_s=0.0))
    L_ALL = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75)
    singles("S1_still", tr, L_ALL, (1, 2, 3))
    singles("S2_cur020_45", with_current(tr, 0.2, np.pi / 4), (0.10, 0.25, 0.50), (1, 2))
    singles("S2_cur030_90", with_current(tr, 0.3, np.pi / 2), (0.10, 0.25, 0.50), (1, 2))
    singles("S3_mismatch", tr, (0.10, 0.25, 0.50), (1, 2), plant_mismatch=MISMATCH)
    # S4 dual
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    pm = np.zeros((28, 8), bool)
    for n, (i, j) in enumerate(pairs):
        pm[n, [i, j]] = True
    for loss in (0.25, 0.5):
        o = raw(f"S4_dual_L{int(loss*100)}", tr, np.zeros((28, 8), bool), partial_mask=pm, partial_gain=1 - loss, seed=5)
        tf, r = TD.detect(o["t"], o["innov"].astype(float), THR)
        pr = TD.isolate_pair(o["t"], r, o["tcmd"].astype(float), tf)
        iso1, _ = TD.isolate(o["t"], r, o["tcmd"].astype(float), tf)
        for n, (i, j) in enumerate(pairs):
            pidx, fi, fj, J1, J2 = pr[n]
            rows.append(dict(scenario="S4_dual", kind="dual", loss=loss, seed=5, true=f"T{i+1}+T{j+1}",
                             flag_t=tf[n], delay=tf[n] - T0,
                             pred=(f"T{pidx//10+1}+T{pidx%10+1}" if pidx >= 0 else "none"), est=np.nanmean([fi, fj]),
                             pred_single=(f"T{iso1[n]+1}" if iso1[n] >= 0 else "none"), J1=J1, J2=J2,
                             rpm_flag=bool(np.isfinite(o["t_det_rpm"][n])), exposure_s=0.0))
    # S5 fault-like disturbances (no fault)
    ts = 30.0

    def nofault(tag, motion, NV, **kw):
        o = raw(tag, motion, np.zeros((NV, 8), bool), t_fail=1e9, **kw)
        ev = evaluate(o, THR, partial=False)
        T_exp = o["t"][-1] - TD.T_START
        for n in range(NV):
            rows.append(dict(scenario=tag, kind="nofault", loss=0.0, seed=kw.get("seed", 7), true="none",
                             flag_t=ev["t_flag"][n], delay=np.nan,
                             pred=("alarm" if np.isfinite(ev["t_flag"][n]) else "none"), est=np.nan,
                             rpm_flag=bool(np.isfinite(o["t_det_rpm"][n])), exposure_s=T_exp, peak=ev["peak"][n]))
    ang = np.arange(8) * np.pi / 4
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros(8)], 1)
    for amp in (0.1, 0.2, 0.3):
        fn = (lambda A, D: (lambda tk: A * np.clip((tk - ts) / 2.0, 0, 1) * D))(amp, dirs)
        nofault(f"S5_current_step_{int(amp*100):02d}", with_current(tr, 0, 0), 8, current_fn=fn, seed=7)
    for F in (5.0, 10.0, 20.0):
        W = np.zeros((8, 6)); W[:, 0] = F * np.cos(ang); W[:, 1] = F * np.sin(ang)
        nofault(f"S5_tether_step_{int(F):02d}N", tr, 8, ext_wrench=(lambda W_: (lambda tk: W_ * (tk >= ts)))(W), seed=7)
    W = np.zeros((8, 6)); W[:, 0] = 30 * np.cos(ang); W[:, 1] = 30 * np.sin(ang)
    nofault("S5_bump_30N_1s", tr, 8, ext_wrench=lambda tk: W * ((tk >= ts) & (tk < ts + 1.0)), seed=7)
    for unc in (0.1, 0.2):
        nofault(f"S5_param_mismatch_{int(unc*100)}", tr, 16, unc=unc, seed=8)
    nofault("S5_model_form_mismatch", tr, 8, plant_mismatch=MISMATCH, seed=8)
    # S6 fault-free exposure (independent seed / currents)
    rng6 = np.random.default_rng(300)
    for mf in MOTIONS:
        m = with_current(mf(), 0.0, 0.0)
        nofault(f"S6_nominal_{mf.__name__}", m, 24, current_fn=random_current_fn(rng6, 24), seed=300,
                ext_wrench=(EX.perturbation if mf is EX.m_hover_perturb else None))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "data", "stress_rows.csv"), index=False)
    print("done", len(df))


if __name__ == "__main__":
    main()

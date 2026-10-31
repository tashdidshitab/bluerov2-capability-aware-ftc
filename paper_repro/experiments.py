"""
Reproduction of the experiments of
  Shitab et al., "Digital Twin for Underwater Vehicles: MATLAB-Based Real-Time UROV Motion
  Simulation Frameworks for Predictive Monitoring", RAAICON 2025,
on the BlueROV2 Heavy fault-tolerant framework, for every thruster-health case
(nominal + all 254 failure combinations of 1..7 thrusters).

  E1  External perturbation (paper Sec. VI-A, Figs 7-8): roll / pitch / yaw moment pulses, 0.4 s
  E2  Lawnmower path tracking (Sec. VI-B, Fig 9)
  E3  Dive & Turn manoeuvre (Sec. VI-B, Fig 10): coordinated dive + 90 deg yaw
  E4  Predictive monitoring (Sec. VI-C, Figs 11-12): digital-twin observer (Eq. 4), predicted vs
      measured velocities, fault flagging; complete failures (all 254) + 50 % partial loss
      (paper's case, plus 25 / 75 %) of every thruster
  E5  Energy estimation (Sec. VI-D, Eqs 5-6): P_i = T_i v_i / eta_i, E = int sum P_i
  E6  "Speed Up and Down" (Figs 13-14): speed / depth / heading reference + 8-thruster PWM
"""
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(_ROOT, "ftc_rov"))
import rov_model as rm
from trajectories import Motion, DT, smoothstep, m_lawnmower
from sim import simulate, SimConfig, metrics, case_info
from allocation import all_cases

OUT = os.path.join(_ROOT, "paper_repro/data")
os.makedirs(OUT, exist_ok=True)
LOG_EVERY = 5          # 0.1 s logging


def labels():
    cases, masks = all_cases()
    lab = ["+".join(f"T{i+1}" for i in c) if c else "none" for c in cases]
    return cases, masks, lab


# ------------------------------------------------------------------ motions
def m_hover_perturb():
    t = np.arange(0, 40, DT)
    p = np.tile([0.0, 0.0, 10.0], (len(t), 1))
    return Motion("Perturbation (hover)", t, p, np.zeros_like(t), "Hover at 10 m, moment pulses",
                  current=(0.0, 0.0, 0.0), eval_mode="time")


PULSES = [(5.0, 3, +1.5), (10.0, 3, -1.5),      # roll  (axis 3 = K)
          (15.0, 4, +1.5), (20.0, 4, -1.5),     # pitch (axis 4 = M)
          (25.0, 5, +1.5), (30.0, 5, -1.5)]     # yaw   (axis 5 = N)
PULSE_LEN = 0.4


def perturbation(t):
    w = np.zeros(6)
    for t0, ax, mag in PULSES:
        if t0 <= t < t0 + PULSE_LEN:
            w[ax] = mag
    return w


def m_dive_turn():
    t = np.arange(0, 140, DT)
    z = 2.0 + 10.0 * smoothstep((t - 10) / 70)               # dive 2 -> 12 m (t = 10..80 s)
    psi = np.deg2rad(90) * smoothstep((t - 40) / 60)          # 90 deg turn (t = 40..100 s), overlaps the dive
    p = np.stack([np.zeros_like(t), np.zeros_like(t), z], 1)
    return Motion("Dive & Turn", t, p, psi, "Dive 2->12 m with coordinated 90 deg yaw", current=(0.0, 0.0, 0.0),
                  eval_mode="time")


def m_transit():
    t = np.arange(0, 60, DT)
    v = 0.3 * smoothstep((t - 1) / 6) * (1 - smoothstep((t - 52) / 6))
    x = np.cumsum(v) * DT
    p = np.stack([x, np.zeros_like(t), np.full_like(t, 10.0)], 1)
    return Motion("Transit (monitoring)", t, p, np.zeros_like(t), "0.3 m/s transit at 10 m", current=(0.0, 0.0, 0.0),
                  eval_mode="time")


# Fig. 13 speed schedule (knots of the paper's "Speed Up and Down" scenario)
SPEED_KNOTS = np.array([[0, 0.2], [60, 0.2], [120, 0.4], [150, 0.4], [210, 0.25], [240, 0.05], [300, 0.05]])


def m_speed_updown():
    t = np.arange(0, 300, DT)
    v = np.interp(t, SPEED_KNOTS[:, 0], SPEED_KNOTS[:, 1])
    v = v * smoothstep(t / 8.0)                                # start from rest
    from scipy.ndimage import gaussian_filter1d
    v = gaussian_filter1d(v, 100)
    x = np.cumsum(v) * DT
    p = np.stack([x, np.zeros_like(t), np.full_like(t, 5.0)], 1)
    return Motion("Speed Up and Down", t, p, np.zeros_like(t), "Speed 0.2-0.4-0.05 m/s, depth 5 m, heading 0",
                  current=(0.0, 0.0, 0.0), eval_mode="time")


EXPS = {
    "E1": dict(motion=m_hover_perturb, t_fail=2.0, ext=perturbation, ctrls=("FTC", "BASELINE")),
    "E2": dict(motion=m_lawnmower, t_fail=20.0, ext=None, ctrls=("FTC", "BASELINE")),
    "E3": dict(motion=m_dive_turn, t_fail=20.0, ext=None, ctrls=("FTC", "BASELINE")),
    "E4": dict(motion=m_transit, t_fail=10.0, ext=None, ctrls=("FTC",)),
    "E6": dict(motion=m_speed_updown, t_fail=20.0, ext=None, ctrls=("FTC", "BASELINE")),
}
KEEP = ["t", "tv", "p", "pr", "eul", "psir", "rpm", "rpm_cmd", "thrust", "nu", "nu_meas", "nu_hat", "innov",
        "tau", "pw8", "tcmd", "vr", "vw", "power"]


def run(eid):
    spec = EXPS[eid]
    m = spec["motion"]()
    cases, masks, lab = labels()
    failed = ~masks
    out = {}
    for c in spec["ctrls"]:
        t0 = time.time()
        cfg = SimConfig(ftc=(c == "FTC"), t_fail=spec["t_fail"], ext_wrench=spec["ext"], twin=True,
                        log_extra=True, log_every=LOG_EVERY, seed=11)
        L = simulate(m, failed, cfg, log_rpm=True)
        M = metrics(m, L, failed, spec["t_fail"])
        for k in KEEP:
            out[f"{c}__{k}"] = L[k]
        for k in ["t_detect", "kappa", "energy_Wh", "t_finish"]:
            out[f"{c}__{k}"] = L[k]
        for k, v in M.items():
            out[f"{c}__M_{k}"] = np.asarray(v, float)
        print(f"{eid} {c} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"{OUT}/{eid}.npz", **out)


def run_partial():
    """E4b: partial thrust loss (paper: 50 % on a lateral thruster) + nominal calibration / validation runs."""
    m = m_transit()
    rows = []
    for loss in (0.25, 0.5, 0.75):
        rows += [(i, loss) for i in range(8)]
    pm = np.zeros((len(rows), 8), bool)
    for j, (i, _) in enumerate(rows):
        pm[j, i] = True
    out = {"thruster": np.array([r[0] for r in rows]), "loss": np.array([r[1] for r in rows])}
    for loss in (0.25, 0.5, 0.75):
        sel = out["loss"] == loss
        cfg = SimConfig(ftc=True, t_fail=10.0, twin=True, log_extra=True, log_every=LOG_EVERY, seed=11,
                        partial_mask=pm[sel], partial_gain=1 - loss)
        L = simulate(m, np.zeros((sel.sum(), 8), bool), cfg, log_rpm=True)
        for k in KEEP:
            out[f"L{int(loss*100)}__{k}"] = L[k]
        out[f"L{int(loss*100)}__t_detect_rpm"] = L["t_detect"]
    np.savez_compressed(f"{OUT}/E4_partial.npz", **out)
    # nominal (fault-free) runs for threshold calibration (seeds 100-109) and validation (seeds 200-219, all motions)
    cal = {}
    for tag, seeds, motions in [("cal", range(100, 110), [m_transit]),
                                ("val", range(200, 220), [m_transit, m_lawnmower, m_dive_turn, m_speed_updown, m_hover_perturb])]:
        inn = []
        for mf in motions:
            mm = mf()
            for sd in seeds:
                L = simulate(mm, np.zeros((1, 8), bool), SimConfig(ftc=True, t_fail=1e9, twin=True, log_extra=True,
                                                                    log_every=LOG_EVERY, seed=sd), log_rpm=False)
                inn.append((mm.name, sd, L["t"], L["innov"][:, 0]))
        cal[tag] = inn
    import pickle
    pickle.dump(cal, open(f"{OUT}/E4_nominal_runs.pkl", "wb"))
    print("partial + nominal done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["E1", "E3", "E4", "E2", "E6", "partial"]
    from multiprocessing import Pool
    with Pool(2) as pool:
        pool.map(lambda_run := run_any, which) if False else None
    for w in which:
        (run_partial if w == "partial" else run)(w) if w == "partial" else None


def run_any(w):
    return run_partial() if w == "partial" else run(w)

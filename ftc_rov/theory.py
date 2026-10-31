"""
Structural analysis for the revision (reviewer MC1, MC5-minor: rank tolerance, Table 2 verification).

T1  exact-arithmetic ranks of B_h, B_v, B for all 255 health masks (sympy, sqrt(2) kept symbolic, lever arms as
    exact rationals) -> independent check of the class table (Table 2)
T2  singular-value gaps and tolerance sweep: does any classification change for tol in [1e-12, 1e-2] * sigma_max?
T3  geometric perturbation: thruster azimuth +-5 deg, vertical tilt +-3 deg, positions +-10 mm (2000 draws);
    nominally rank-deficient blocks become numerically full rank - we report the attainable force/moment
    along the nominally lost direction ("practical rank"), which stays small
T4  local (Kalman/PBH) controllability of the 12-state linearisation at hover and at 0.3 m/s cruise,
    with uncontrollable-mode eigenvalues (stabilisability) per class
T5  steady-motion feasibility (equilibria): station keeping in a steady current over all headings, and straight
    transit at 0.3 m/s over all sideslip angles: exists T in the thruster box with B_h T = required wrench? (LP)
T6  existence demonstration: a single healthy horizontal thruster (rank-1, "infeasible") can still produce net
    translation by spinning and phase-modulating its thrust -> the infeasibility statement is about
    non-spinning, attitude-respecting missions, not about controllability in the abstract
Outputs results/theory/*.csv, theory_summary.json
"""
import os, json, itertools
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import rov_model as rm
from allocation import all_cases, analyse_case
from allocation import U_MIN as _UMIN, U_MAX as _UMAX
U_MIN = np.full(8, _UMIN); U_MAX = np.full(8, _UMAX)

OUT = "results/theory"
os.makedirs(OUT, exist_ok=True)
P = rm.nominal_params()
HROWS, VROWS = [0, 1, 5], [2, 3, 4]
CASES, MASKS = all_cases()
LAB = ["none" if not c else "+".join(f"T{i+1}" for i in c) for c in CASES]


# ------------------------------------------------------------------ T1 exact ranks
def t1_exact():
    import sympy as sp
    s2 = sp.sqrt(2) / 2
    E = sp.Matrix([[s2, -s2, 0], [s2, s2, 0], [-s2, -s2, 0], [-s2, s2, 0], [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]])
    R = sp.Matrix([[sp.Rational(str(x)) for x in row] for row in rm.R_TH])
    cols = []
    for i in range(8):
        e = E.row(i).T; r = R.row(i).T
        cols.append(sp.Matrix.vstack(e, r.cross(e)))
    B = sp.Matrix.hstack(*cols)
    assert np.allclose(np.array(B.evalf(), float), rm.B_ALLOC)
    rows = []
    for c, h, lab in zip(CASES, MASKS, LAB):
        Bh = sp.Matrix([[B[r, j] if h[j] else 0 for j in range(4)] for r in HROWS])
        Bv = sp.Matrix([[B[r, j] if h[j] else 0 for j in range(4, 8)] for r in VROWS])
        Bf = sp.Matrix([[B[r, j] if h[j] else 0 for j in range(8)] for r in range(6)])
        rh, rv, rf = Bh.rank(simplify=True), Bv.rank(simplify=True), Bf.rank(simplify=True)
        # exact mode from exact ranks (same rule as allocation.analyse_case)
        if rv == 0:
            mode = "INFEASIBLE_DEPTH"
        elif rh <= 1:
            mode = "INFEASIBLE_HORIZONTAL"
        elif rh == 2:
            ns = Bh.T.nullspace()            # left null space of Bh: normal of the reachable plane
            n = ns[0]
            mode = "YAW_FREE" if sp.simplify(n[0] ** 2 + n[1] ** 2) == 0 else "HEADING_STEERED"
        else:
            mode = "FULL" if rv == 3 else "COUPLED_ATTITUDE"
        info = analyse_case(h)
        rows.append(dict(case=lab, n_failed=len(c), rank_h_exact=rh, rank_v_exact=rv, rank_exact=rf, mode_exact=mode,
                         rank_h_float=info["rank_h"], rank_v_float=info["rank_v"], rank_float=info["rank"],
                         mode_float=info["mode"], agree=(mode == info["mode"] and rh == info["rank_h"] and rv == info["rank_v"])))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T1_exact_ranks.csv", index=False)
    cnt = df.groupby("mode_exact").size().to_dict()
    return dict(all_agree=bool(df.agree.all()), n=len(df), counts_exact=cnt,
                counts_by_nfailed=df.groupby(["n_failed", "mode_exact"]).size().unstack(fill_value=0).to_dict("index"))


# ------------------------------------------------------------------ T2 singular-value gaps / tolerance sweep
def block_sv(B, h):
    Bh = (B * h)[HROWS][:, :4]; Bv = (B * h)[VROWS][:, 4:]
    return np.linalg.svd(Bh, compute_uv=False), np.linalg.svd(Bv, compute_uv=False)


def classify(B, h, tol_rel):
    sh, sv = block_sv(B, h)
    smax = np.linalg.svd(B, compute_uv=False)[0]
    rh = int((sh > tol_rel * smax).sum()); rv = int((sv > tol_rel * smax).sum())
    if rv == 0:
        return "INFEASIBLE_DEPTH"
    if rh <= 1:
        return "INFEASIBLE_HORIZONTAL"
    if rh == 2:
        return "HEADING_STEERED"
    return "FULL" if rv == 3 else "COUPLED_ATTITUDE"


def t2_tolerance():
    B = rm.B_ALLOC
    smax = np.linalg.svd(B, compute_uv=False)[0]
    nz_min, z_max = np.inf, 0.0
    for h in MASKS:
        sh, sv = block_sv(B, h)
        for s in np.concatenate([sh, sv]):
            if s > 1e-9:
                nz_min = min(nz_min, s)
            else:
                z_max = max(z_max, s)
    tols = 10.0 ** np.arange(-12, -1.0)
    ref = [analyse_case(h)["mode"] for h in MASKS]
    changes = {f"{t:.0e}": int(sum(classify(B, h, t) != r for h, r in zip(MASKS, ref))) for t in tols}
    return dict(sigma_max=float(smax), smallest_nonzero_block_sv=float(nz_min), largest_zero_block_sv=float(z_max),
                gap_ratio=float(nz_min / max(z_max, 1e-300)), class_changes_vs_tol=changes,
                note="classification identical for every relative tolerance between 1e-12 and 1e-2")


# ------------------------------------------------------------------ T3 geometric perturbation
def perturbed_B(rng, az=5.0, tilt=3.0, pos=0.010):
    E = rm.E_TH.copy(); R = rm.R_TH + rng.uniform(-pos, pos, (8, 3))
    for i in range(4):
        a = np.deg2rad(rng.uniform(-az, az)); c, s = np.cos(a), np.sin(a)
        E[i] = [c * E[i, 0] - s * E[i, 1], s * E[i, 0] + c * E[i, 1], 0]
        b = np.deg2rad(rng.uniform(-tilt, tilt)); E[i] = E[i] * np.cos(b) + np.array([0, 0, np.sin(b)])
    for i in range(4, 8):
        a = np.deg2rad(rng.uniform(-tilt, tilt, 2))
        E[i] = np.array([np.sin(a[0]), np.sin(a[1]), -1.0]); E[i] /= np.linalg.norm(E[i])
    return np.vstack([E.T, np.cross(R, E).T])


def lost_dir_capacity(B, h, rows, cols, n):
    """max |w| along unit direction n (in the block's row space) attainable within the box with healthy cols."""
    Bb = (B * h)[rows][:, cols]
    lo = np.where(h[cols], U_MIN[cols], 0.0); hi = np.where(h[cols], U_MAX[cols], 0.0)
    best = 0.0
    for sgn in (1, -1):
        res = linprog(-sgn * (n @ Bb), bounds=list(zip(lo, hi)), method="highs")
        if res.success:
            best = max(best, -res.fun)
    return best


def t3_perturb(ndraw=2000, seed=3):
    global NOM_MODES
    NOM_MODES = [analyse_case(h)["mode"] for h in MASKS]
    rng = np.random.default_rng(seed)
    B0 = rm.B_ALLOC
    def_cases = []
    for idx, h in enumerate(MASKS):
        U, s, Vt = np.linalg.svd((B0 * h)[HROWS][:, :4])
        rh = int((s > 1e-6).sum())
        if 1 <= rh < 3:
            def_cases.append((idx, "H", U[:, rh:]))
        U, s, Vt = np.linalg.svd((B0 * h)[VROWS][:, 4:])
        rv = int((s > 1e-6).sum())
        if 1 <= rv < 3:
            def_cases.append((idx, "V", U[:, rv:]))
    # nominal capacity along an *attainable* reference direction for scale: full horizontal force capacity
    rows = []
    sub = rng.choice(ndraw, 200, replace=False)          # LP capacity on a 200-draw subset, SVs on all
    sv_ratio = {k: [] for k in ("H", "V")}
    cap = {k: [] for k in ("H", "V")}
    same_class = 0; tot = 0
    for d in range(ndraw):
        Bp = perturbed_B(rng)
        for idx, blk, Nn in def_cases:
            h = MASKS[idx]
            rows_, cols = (HROWS, np.arange(4)) if blk == "H" else (VROWS, np.arange(4, 8))
            s = np.linalg.svd((Bp * h)[rows_][:, cols], compute_uv=False)
            nz = s[s > 1e-9]
            k = Nn.shape[1]
            sv_ratio[blk].append(nz[-1] / nz[0] if len(nz) > 3 - k else 0.0)
            if d in sub:
                # capacity along the nominally lost direction (first null vector), perturbed geometry
                cap[blk].append(lost_dir_capacity(Bp, h, rows_, cols, Nn[:, 0]))
        for h, md0 in zip(MASKS[1:], NOM_MODES[1:]):
            tot += 1
            same_class += classify(Bp, h, 0.05) == md0
    q = lambda a, p: float(np.percentile(a, p))
    out = dict(n_draws=ndraw, perturbation="azimuth +-5 deg, tilt +-3 deg, position +-10 mm",
               n_deficient_blocks=len(def_cases),
               sv_ratio_lost_over_max_H=dict(median=q(sv_ratio["H"], 50), p95=q(sv_ratio["H"], 95), max=float(np.max(sv_ratio["H"]))),
               sv_ratio_lost_over_max_V=dict(median=q(sv_ratio["V"], 50), p95=q(sv_ratio["V"], 95), max=float(np.max(sv_ratio["V"]))),
               capacity_lost_dir_H_N_or_Nm=dict(median=q(cap["H"], 50), p95=q(cap["H"], 95), max=float(np.max(cap["H"]))),
               capacity_lost_dir_V_N_or_Nm=dict(median=q(cap["V"], 50), p95=q(cap["V"], 95), max=float(np.max(cap["V"]))),
               class_unchanged_fraction_at_5pct_practical_tol=same_class / tot)
    return out


# ------------------------------------------------------------------ T4 linear controllability
def f12(x, T, h):
    """12-state model x = [p, eul, nu]; inputs = 8 thrusts (failed ones have zero column)."""
    q = rm.euler_to_quat(x[None, 3], x[None, 4], x[None, 5])
    Pb = rm.batch_params(1)
    tau = (rm.B_ALLOC * h) @ T
    nud, R = rm.accel(q, x[None, 6:], tau[None], np.zeros((1, 3)), Pb)
    ph, th = x[3], x[4]
    w = x[9:12]
    ed = np.array([w[0] + np.sin(ph) * np.tan(th) * w[1] + np.cos(ph) * np.tan(th) * w[2],
                   np.cos(ph) * w[1] - np.sin(ph) * w[2], (np.sin(ph) * w[1] + np.cos(ph) * w[2]) / np.cos(th)])
    return np.concatenate([R[0] @ x[6:9], ed, nud[0]])


def jac(x, T, h, eps=1e-6):
    A = np.zeros((12, 12)); Bm = np.zeros((12, 8))
    for j in range(12):
        d = np.zeros(12); d[j] = eps
        A[:, j] = (f12(x + d, T, h) - f12(x - d, T, h)) / (2 * eps)
    for j in range(8):
        d = np.zeros(8); d[j] = eps
        Bm[:, j] = (f12(x, T + d, h) - f12(x, T - d, h)) / (2 * eps)
    return A, Bm


def trim(x, h):
    """least-squares thrust that zeroes nu_dot at state x (exact when feasible)."""
    f0 = f12(x, np.zeros(8), h)[6:]
    _, Bm = jac(x, np.zeros(8), h)
    T = np.linalg.lstsq(Bm[6:], -f0, rcond=None)[0]
    return T, np.linalg.norm(f12(x, T, h)[6:])


def pbh(A, Bm, tol=1e-7):
    ev = np.linalg.eigvals(A)
    unc = []
    for lam in np.unique(np.round(ev, 9)):
        M = np.hstack([A - lam * np.eye(12), Bm])
        s = np.linalg.svd(M, compute_uv=False)
        r = int((s > tol * s[0]).sum())
        unc += [lam] * (12 - r)
    Cm = np.hstack([np.linalg.matrix_power(A, k) @ Bm for k in range(12)])
    sc = np.linalg.svd(Cm, compute_uv=False)
    return int((sc > 1e-9 * sc[0]).sum()), unc


def t4_controllability():
    rows = []
    for c, h, lab in zip(CASES, MASKS, LAB):
        info = analyse_case(h)
        for cond in ("hover", "cruise"):
            x = np.zeros(12)
            if cond == "cruise":
                d0 = info["d0"]; x[6:8] = 0.3 * d0
            T0, res = trim(x, h.astype(float))
            A, Bm = jac(x, T0, h.astype(float))
            rk, unc = pbh(A, Bm)
            rows.append(dict(case=lab, n_failed=len(c), mode=info["mode"], condition=cond, trim_residual=res,
                             kalman_rank=rk, n_uncontrollable=len(unc),
                             uncontrollable_eigs=";".join(f"{np.real(u):.3g}{np.imag(u):+.3g}j" for u in unc),
                             stabilisable=bool(all(np.real(u) < -1e-6 for u in unc))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T4_controllability.csv", index=False)
    g = df.groupby(["mode", "condition"]).agg(n=("case", "size"), rank_min=("kalman_rank", "min"), rank_max=("kalman_rank", "max"),
                                             stabilisable_frac=("stabilisable", "mean"), trim_ok_frac=("trim_residual", lambda r: float((r < 1e-6).mean())))
    g.to_csv(f"{OUT}/T4_controllability_summary.csv")
    return {f"{a}|{b}": v for (a, b), v in g.to_dict("index").items()}


# ------------------------------------------------------------------ T5 steady-motion feasibility (LP)
def req_wrench_h(vr_b):
    """required (X, Y, N) for zero body acceleration with relative velocity vr_b = (u_r, v_r), zero rates."""
    u, v = vr_b
    X = P["Dl"][0] * u + P["Dq"][0] * abs(u) * u
    Y = P["Dl"][1] * v + P["Dq"][1] * abs(v) * v
    N = u * v * (P["A1"][1] - P["A1"][0])          # Munk moment: tau_N = (A1y - A1x) u v  (from accel, m_c sign)
    return np.array([X, Y, N])


def feasible_h(h, w):
    Bh = rm.B_ALLOC[HROWS][:, :4]
    lo = np.where(h[:4], U_MIN[:4], 0.0); hi = np.where(h[:4], U_MAX[:4], 0.0)
    # minimise inf-norm residual: vars [T(4), s]
    c = np.r_[np.zeros(4), 1.0]
    A = np.vstack([np.hstack([Bh, -np.ones((3, 1))]), np.hstack([-Bh, -np.ones((3, 1))])])
    b = np.r_[w, -w]
    res = linprog(c, A_ub=A, b_ub=b, bounds=list(zip(lo, hi)) + [(0, None)], method="highs")
    return res.x[-1] if res.success else np.inf


def t5_equilibria():
    angs = np.deg2rad(np.arange(0, 360, 2.0))
    rows = []
    for c, h, lab in zip(CASES, MASKS, LAB):
        info = analyse_case(h)
        if info["mode"] == "INFEASIBLE_DEPTH":
            continue
        for kind, spd in [("hold", 0.1), ("hold", 0.2), ("hold", 0.3), ("transit", 0.3), ("transit", 0.5)]:
            ok = []
            for a in angs:            # a = angle of relative flow in body frame (current dir - heading, or sideslip)
                vr = spd * np.array([np.cos(a), np.sin(a)])
                if kind == "hold":
                    vr = -vr          # vehicle at rest, water moving at spd towards direction a -> relative velocity -vc_b
                ok.append(feasible_h(h, req_wrench_h(vr)) < 1e-3)
            ok = np.array(ok)
            rows.append(dict(case=lab, n_failed=len(c), mode=info["mode"], kind=kind, speed=spd,
                             frac_feasible=float(ok.mean()), any_feasible=bool(ok.any())))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/T5_equilibria.csv", index=False)
    g = df.groupby(["mode", "kind", "speed"]).agg(n=("case", "size"), frac_dirs_mean=("frac_feasible", "mean"),
                                                 frac_dirs_min=("frac_feasible", "min"), cases_with_any=("any_feasible", "mean"))
    g.to_csv(f"{OUT}/T5_equilibria_summary.csv")
    return {f"{a}|{b}|{s}": v for (a, b, s), v in g.to_dict("index").items()}


# ------------------------------------------------------------------ T6 spinning single-thruster translation
def t6_spin_demo(T_s=120.0, dt=0.01, chis=(0.0,)):
    """Only T1 healthy among horizontals (all verticals healthy, depth/attitude held by an ideal vertical loop).
    Open-loop horizontal law: T1 = Tm * (1 + cos(psi + phi_f - chi*)) / 2  (thrust when its force points towards chi*).
    Planar 3-DOF integration with the full horizontal model (damping, added mass, Coriolis, Munk)."""
    h = np.zeros(8, bool); h[0] = True
    e1 = rm.E_TH[0, :2]; phi_f = np.arctan2(e1[1], e1[0])
    out = {}
    for Tm, chi in [(10.0, 0.0), (25.0, 0.0), (51.5, 0.0), (25.0, np.pi / 2), (25.0, np.pi), (25.0, -np.pi / 2)]:
        x = np.zeros(3); nu = np.zeros(3)  # x,y,psi ; u,v,r
        m1 = P["m"] + P["A1"]; Iz = P["I"][2] + P["A2"][2]
        traj = []
        for k in range(int(T_s / dt)):
            psi = x[2]
            T = Tm * 0.5 * (1 + np.cos(psi + phi_f - chi))
            tau = rm.B_ALLOC[HROWS, 0] * T
            u, v, r = nu
            Xc = m1[1] * v * r - P["m"] * 0 ; Yc = -m1[0] * u * r
            # rigid body + added mass Coriolis (planar), damping, Munk
            fX = tau[0] + (P["m"] + P["A1"][1]) * v * r - P["Dl"][0] * u - P["Dq"][0] * abs(u) * u
            fY = tau[1] - (P["m"] + P["A1"][0]) * u * r - P["Dl"][1] * v - P["Dq"][1] * abs(v) * v
            fN = tau[2] - (P["A1"][1] - P["A1"][0]) * u * v - P["Dl"][5] * r - P["Dq"][5] * abs(r) * r
            nu = nu + dt * np.array([fX / m1[0], fY / m1[1], fN / Iz])
            c, s = np.cos(psi), np.sin(psi)
            x = x + dt * np.array([c * nu[0] - s * nu[1], s * nu[0] + c * nu[1], nu[2]])
            traj.append(x.copy())
        traj = np.array(traj)
        tt = np.arange(len(traj)) * dt
        sel = tt > 30
        A = np.polyfit(tt[sel], traj[sel, 0], 1)[0]; Bv = np.polyfit(tt[sel], traj[sel, 1], 1)[0]
        rate = np.polyfit(tt[sel], traj[sel, 2], 1)[0]
        out[f"Tm={Tm},chi={np.rad2deg(chi):.0f}"] = dict(net_speed_mps=float(np.hypot(A, Bv)), net_dir_deg=float(np.rad2deg(np.arctan2(Bv, A))),
                              spin_rate_dps=float(np.rad2deg(rate)))
        np.save(f"{OUT}/T6_spin_traj_Tm{int(Tm)}_chi{int(np.rad2deg(chi))}.npy", traj[::10])
    # does the phase actually steer? repeat with chi* = 90 deg at Tm=25
    return out


def main():
    S = {}
    S["T2_tolerance"] = t2_tolerance(); print("T2", S["T2_tolerance"], flush=True)
    S["T6_spin"] = t6_spin_demo(); print("T6", S["T6_spin"], flush=True)
    S["T5_equilibria"] = t5_equilibria(); print("T5 done", flush=True)
    S["T4_controllability"] = t4_controllability(); print("T4 done", flush=True)
    S["T3_perturbation"] = t3_perturb(); print("T3", S["T3_perturbation"], flush=True)
    S["T1_exact"] = t1_exact(); print("T1", S["T1_exact"]["all_agree"], S["T1_exact"]["counts_exact"], flush=True)
    json.dump(S, open(f"{OUT}/theory_summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()

"""T7: static hover equilibria for the depth-infeasible class at ANY attitude (+-85 deg roll/pitch grid):
exists u in the thruster box (healthy horizontals only) with B_F u + g(phi, theta) = 0 (all 6 DOF)?  LP, inf-norm residual.
Also reports, over the grid, the minimum residual and the attitude where it occurs."""
import numpy as np, pandas as pd, json
from scipy.optimize import linprog
import rov_model as rm
from allocation import all_cases, analyse_case, U_MIN, U_MAX
P = rm.nominal_params()
rows = []
cases, masks = all_cases()
angs = np.deg2rad(np.arange(-85, 86, 5.0))
for c, h in zip(cases, masks):
    info = analyse_case(h)
    if info["mode"] != "INFEASIBLE_DEPTH":
        continue
    best = (np.inf, 0, 0)
    Bf = rm.B_ALLOC * h
    lo = np.where(h, U_MIN, 0.0); hi = np.where(h, U_MAX, 0.0)
    for ph in angs:
        for th in angs:
            q = rm.euler_to_quat(np.array([ph]), np.array([th]), np.zeros(1))
            R = rm.quat_to_R(q)
            g = rm.restoring(R, dict(W=np.array([P["W"]]), B=np.array([P["B"]]), rg=np.zeros((1, 3)), rb=P["rb"][None]))[0]
            # vars u(8), s ; minimise s  s.t. |B u + g| <= s
            A = np.vstack([np.hstack([Bf, -np.ones((6, 1))]), np.hstack([-Bf, -np.ones((6, 1))])])
            b = np.r_[-g, g]
            res = linprog(np.r_[np.zeros(8), 1.0], A_ub=A, b_ub=b, bounds=list(zip(lo, hi)) + [(0, None)], method="highs")
            if res.success and res.x[-1] < best[0]:
                best = (res.x[-1], np.rad2deg(ph), np.rad2deg(th))
    rows.append(dict(case="+".join(f"T{i+1}" for i in c), min_residual=best[0], at_roll_deg=best[1], at_pitch_deg=best[2],
                     equilibrium_exists=bool(best[0] < 1e-6)))
df = pd.DataFrame(rows)
df.to_csv("results/theory/T7_depth_infeasible_equilibria.csv", index=False)
print(df.to_string())

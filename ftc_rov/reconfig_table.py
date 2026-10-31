"""
Builds the per-failure-case reconfiguration law table (the 'algorithm per failure'):
for each of the 254 failure sets: controllability ranks, guidance mode, released axes,
weighted pseudo-inverse K_F (8x6), attainable wrench capacity, recommended action,
and the steady-state RPM redistribution of the surviving thrusters for two canonical
operating points (hover at depth; cruise 0.3 m/s forward at constant depth).
"""
import json
import numpy as np
import pandas as pd
import rov_model as rm
from allocation import all_cases, analyse_case, capacity, Allocator


def steady_wrench(u=0.0):
    P = rm.nominal_params()
    tau = np.zeros(6)
    tau[0] = P["Dl"][0] * u + P["Dq"][0] * u * abs(u)
    tau[2] = P["B"] - P["W"]          # push down to cancel positive buoyancy (NED z down)
    return tau


def build(out_dir="results"):
    cases, masks = all_cases()
    rows, laws = [], {}
    ops = {"hover": steady_wrench(0.0), "cruise": steady_wrench(0.3)}
    for c, h in zip(cases, masks):
        info = analyse_case(h)
        cap = capacity(h)
        A = Allocator(h[None], np.diag(info["W"])[None])
        row = dict(case="+".join(f"T{i+1}" for i in c) if c else "none", n_failed=len(c),
                   rank=info["rank"], rank_h=info["rank_h"], rank_v=info["rank_v"], mode=info["mode"],
                   action=info["action"], d0_x=info["d0"][0], d0_y=info["d0"][1],
                   cap_h=info["cap_h"], cap_z=info["cap_z"])
        for d, nm in enumerate(["X", "Y", "Z", "K", "M", "N"]):
            row[f"cap_+{nm}"] = cap[2 * d]; row[f"cap_-{nm}"] = cap[2 * d + 1]
        for op, tau in ops.items():
            tau = tau.copy()
            if info["mode"] == "HEADING_STEERED" and op == "cruise":
                # vehicle steers so that its producible thrust axis d0 points along the path
                fx = tau[0]; tau[0], tau[1] = fx * info["d0"][0], fx * info["d0"][1]
            T, sat = A.solve(tau[None])
            rpm = rm.thrust_to_rpm(T[0]) * h
            res = rm.B_ALLOC @ T[0] - tau
            for i in range(8):
                row[f"{op}_rpm_T{i+1}"] = round(float(rpm[i]))
            row[f"{op}_wrench_residual"] = float(np.linalg.norm(res[[0, 1, 2, 5]]))
            row[f"{op}_power_W"] = float(rm.thruster_power(T[0]).sum())
        rows.append(row)
        laws[row["case"]] = dict(mode=info["mode"], action=info["action"], healthy=h.astype(int).tolist(),
                                 Q=np.round(np.diag(info["W"]), 4).tolist(), K=np.round(A.K[0], 5).tolist(),
                                 d0=info["d0"].tolist(), uncontrollable_moment_axis=np.nan_to_num(info["e_unc"]).tolist())
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/reconfiguration_table.csv", index=False)
    with open(f"{out_dir}/reconfiguration_laws.json", "w") as f:
        json.dump(laws, f, indent=1)
    return df


if __name__ == "__main__":
    df = build()
    print(df.groupby(["n_failed", "mode"]).size())
    print(df[["case", "mode"] + [f"cruise_rpm_T{i}" for i in range(1, 9)]].head(12).to_string())

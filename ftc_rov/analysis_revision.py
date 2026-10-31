"""
Revision analysis: tables + figures from results/revision/*_all.csv (and Monte Carlo / seed replicates).

Definitions (failure runs only, n_failed >= 1):
  success      completed AND path_ok AND max tilt < 30 deg        (primary outcome)
  completed    not aborted, within 1 m of the final reference point within 20 s after the reference ends
  path_kept    path_ok (RMS < 0.25 m, max < 1 m, post-fault) AND not aborted
  aborted      supervisor abort (structurally infeasible class detected)
Uncertainty: the 24 x 254 scenario set is exhaustively enumerated (not sampled); intervals are therefore
reported for generalisation to other manoeuvres / fault sets of the same families by a two-way (crossed)
cluster bootstrap that resamples manoeuvres and failure sets independently (B = 2000), paired across controllers.
Paired binary comparisons: exact McNemar test on discordant (manoeuvre, failure set) pairs.
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
warnings.filterwarnings("ignore")

R = "results/revision"
OUT = "results/revision_tables"
os.makedirs(OUT, exist_ok=True)
ORDER = ["BL", "BL+TS", "FA-WPI", "FA-WPI+TS", "FA-BLS", "FA-BLS+TS", "MPC", "MPC+TS", "FTC-noTS", "FTC", "MPC+SL",
         "FTC-noGuid", "FTC-RPI", "FTC-oracle"]
B = 2000


def load(tag):
    f = f"{R}/{tag}_all.csv"
    if not os.path.exists(f):
        import glob
        fs = glob.glob(f"{R}/{tag}/*.csv")
        if not fs:
            return None
        return pd.concat([pd.read_csv(x) for x in fs])
    return pd.read_csv(f)


def prep(df):
    df = df.copy()
    for c in ("aborted", "completed", "path_ok", "success", "deadline_1p5", "deadline_2"):
        df[c] = df[c].astype(float) > 0.5
    df["path_kept"] = df.path_ok & ~df.aborted
    df["tilt_ok"] = df.max_tilt_deg < 30
    df["legacy"] = df.path_ok & df.tilt_ok
    return df


def crossed_bootstrap(piv, rng, B=B):
    """piv: array (n_motion, n_case) of values (NaN allowed). returns bootstrap means (B,)"""
    nm, nc = piv.shape
    out = np.empty(B)
    for b in range(B):
        im = rng.integers(0, nm, nm); ic = rng.integers(0, nc, nc)
        out[b] = np.nanmean(piv[np.ix_(im, ic)])
    return out


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def matrix(df, ctrl, col):
    d = df[df.controller == ctrl].pivot_table(index="motion_id", columns="case_id", values=col, aggfunc="first")
    return d


def table_main(df, rng, ctrls):
    d = df[df.n_failed > 0]
    rows = []
    for c in ctrls:
        x = d[d.controller == c]
        if x.empty:
            continue
        piv = matrix(d, c, "success").values.astype(float)
        bs = crossed_bootstrap(piv, rng)
        lo, hi = ci(bs)
        comp = x[x.completed]
        rows.append(dict(controller=c, n_runs=len(x), success=x.success.mean(), success_ci_lo=lo, success_ci_hi=hi,
                         completed=x.completed.mean(), path_kept=x.path_kept.mean(), tilt_ok=x.tilt_ok.mean(),
                         aborted=x.aborted.mean(), deadline_1p5=x.deadline_1p5.mean(), deadline_2=x.deadline_2.mean(),
                         duration_ratio_median_completed=comp.duration_ratio.median() if len(comp) else np.nan,
                         energy_total_Wh_median_completed=comp.energy_total_Wh.median() if len(comp) else np.nan,
                         energy_thruster_Wh_median_completed=comp.energy_Wh.median() if len(comp) else np.nan,
                         rms_dev_median=x.rms_dev.median(), sat_frac_median=x.sat_frac.median(),
                         ctrl_us_per_vehicle_step=x.ctrl_us_per_vehicle_step.median(),
                         mpc_solve_ms_batch=x.mpc_solve_ms_batch.median()))
    return pd.DataFrame(rows)


def table_by(df, ctrls, by, col="success"):
    d = df[(df.n_failed > 0) & df.controller.isin(ctrls)]
    t = d.groupby([by, "controller"])[col].mean().unstack()
    return t[[c for c in ctrls if c in t.columns]]


def paired(df, rng, ref="FTC", ctrls=None):
    d = df[df.n_failed > 0]
    rows = []
    A = matrix(d, ref, "success").astype(float)
    for c in ctrls:
        if c == ref or c not in d.controller.unique():
            continue
        Bm = matrix(d, c, "success").astype(float).reindex(index=A.index, columns=A.columns)
        diff = (A - Bm).values
        bs = crossed_bootstrap(diff, rng)
        a = A.values.ravel(); b = Bm.values.ravel(); ok = ~np.isnan(a) & ~np.isnan(b)
        n10 = int(((a == 1) & (b == 0) & ok).sum()); n01 = int(((a == 0) & (b == 1) & ok).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
        # rank-biserial on RMS deviation (lower is better), paired
        ra = matrix(d, ref, "rms_dev").reindex(index=A.index, columns=A.columns).values.ravel()
        rb = matrix(d, c, "rms_dev").reindex(index=A.index, columns=A.columns).values.ravel()
        okr = np.isfinite(ra) & np.isfinite(rb)
        dd = rb[okr] - ra[okr]; dd = dd[dd != 0]
        if len(dd):
            rk = pd.Series(np.abs(dd)).rank().values
            rbc = (rk[dd > 0].sum() - rk[dd < 0].sum()) / rk.sum()
        else:
            rbc = 0.0
        ea = matrix(d, ref, "energy_total_Wh").reindex(index=A.index, columns=A.columns).values.ravel()
        eb = matrix(d, c, "energy_total_Wh").reindex(index=A.index, columns=A.columns).values.ravel()
        ca = matrix(d, ref, "completed").reindex(index=A.index, columns=A.columns).values.ravel().astype(float)
        cb = matrix(d, c, "completed").reindex(index=A.index, columns=A.columns).values.ravel().astype(float)
        both = (ca > 0.5) & (cb > 0.5) & np.isfinite(ea) & np.isfinite(eb)
        e_ratio = float(np.median(eb[both] / ea[both])) if both.any() else np.nan
        rows.append(dict(comparator=c, delta_success=np.nanmean(diff), energy_ratio_joint_completed=e_ratio, n_joint_completed=int(both.sum()), delta_ci_lo=ci(bs)[0], delta_ci_hi=ci(bs)[1],
                         ftc_better=n10, comparator_better=n01, mcnemar_p=p,
                         risk_ratio=np.nanmean(a) / max(np.nanmean(b), 1e-9), rank_biserial_rms=rbc))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(7)
    S = {}
    frames = [x for x in (load("main"), load("mpc"), load("mpcsub")) if x is not None]
    nom = prep(pd.concat(frames))
    ctrls = [c for c in ORDER if c in nom.controller.unique()]
    subset_only = ["MPC", "MPC+TS", "MPC+SL"]
    full_ctrls = [c for c in ctrls if c not in subset_only]
    T1 = table_main(nom[nom.controller.isin(full_ctrls)], rng, full_ctrls)
    T1.to_csv(f"{OUT}/T1_main.csv", index=False)
    table_by(nom, full_ctrls, "n_failed").to_csv(f"{OUT}/T2_success_by_nfailed.csv")
    table_by(nom, full_ctrls, "mode").to_csv(f"{OUT}/T3_success_by_class.csv")
    table_by(nom, full_ctrls, "mode", "path_kept").to_csv(f"{OUT}/T3b_pathkept_by_class.csv")
    table_by(nom, full_ctrls, "mode", "completed").to_csv(f"{OUT}/T3c_completed_by_class.csv")
    table_by(nom, full_ctrls, "motion").to_csv(f"{OUT}/T3d_success_by_motion.csv")
    P = paired(nom, rng, "FTC", full_ctrls); P.to_csv(f"{OUT}/T4_paired_vs_FTC.csv", index=False)
    # infeasible: legacy (transient) path criterion vs completion, controllers without supervisor
    inf = nom[(nom["mode"].isin(["INFEASIBLE_HORIZONTAL", "INFEASIBLE_DEPTH"])) & (nom.n_failed > 0)]
    T5 = inf.groupby(["mode", "controller"]).agg(legacy_path_ok=("path_ok", "mean"), completed=("completed", "mean"),
                                                   success=("success", "mean"), aborted=("aborted", "mean"),
                                                   surfaced=("surfaced", "mean"),
                                                   t_abort_after_fault=("t_abort", lambda x: float(np.nanmedian(x - 20.0)))).reset_index()
    T5.to_csv(f"{OUT}/T5_infeasible.csv", index=False)
    # MPC+SL subset comparison (same 8 manoeuvres for every controller)
    if "MPC+SL" in nom.controller.unique():
        sub = sorted(nom[nom.controller == "MPC+SL"].motion_id.unique())
        S["subset_motions"] = [int(x) for x in sub]
        ns = nom[nom.motion_id.isin(sub)]
        T6 = table_main(ns, rng, [c for c in ctrls if c in ns.controller.unique()])
        T6.to_csv(f"{OUT}/T6_subset_with_MPCSL.csv", index=False)
        paired(ns, rng, "FTC", [c for c in ctrls if c in ns.controller.unique()]).to_csv(f"{OUT}/T6b_subset_paired.csv", index=False)
        paired(ns, rng, "MPC+SL", ["MPC", "MPC+TS"]).to_csv(f"{OUT}/T6c_subset_paired_vs_MPCSL.csv", index=False)
    # mismatch
    mm = load("mismatch")
    if mm is not None:
        mm = prep(mm)
        mmp = load("mismatch_mpc")
        if mmp is not None:
            mm = pd.concat([mm, prep(mmp)])
        rows = []
        for c in mm.controller.unique():
            mots = mm[mm.controller == c].motion_id.unique()
            a = nom[(nom.controller == c) & nom.motion_id.isin(mots) & (nom.n_failed > 0)]
            b = mm[(mm.controller == c) & (mm.n_failed > 0)]
            rows.append(dict(controller=c, n_motions=len(mots), success_nominal=a.success.mean(), success_mismatch=b.success.mean(),
                             completed_nominal=a.completed.mean(), completed_mismatch=b.completed.mean(),
                             path_kept_nominal=a.path_kept.mean(), path_kept_mismatch=b.path_kept.mean(),
                             healthy_rms_nominal=nom[(nom.controller == c) & (nom.n_failed == 0) & nom.motion_id.isin(mots)].rms_dev.median(),
                             healthy_rms_mismatch=mm[(mm.controller == c) & (mm.n_failed == 0)].rms_dev.median()))
        pd.DataFrame(rows).to_csv(f"{OUT}/T7_mismatch.csv", index=False)
        table_by(mm, list(mm.controller.unique()), "mode").to_csv(f"{OUT}/T7b_mismatch_by_class.csv")
    # noise seeds
    seeds = [prep(x) for x in (load("seed1"),) if x is not None]
    if seeds:
        smots = seeds[0].motion_id.unique()
        base = nom[nom.controller.isin(["FTC", "BL"]) & nom.motion_id.isin(smots)].assign(seed=0)
        allS = pd.concat([base] + seeds)
        allS[allS.n_failed > 0].groupby(["controller", "seed"])[["success", "completed", "path_kept"]].mean().to_csv(f"{OUT}/T8_noise_seeds.csv")
        # per-run agreement across seeds
        pv = allS[(allS.n_failed > 0)].pivot_table(index=["controller", "motion_id", "case_id"], columns="seed", values="success")
        agree = pv.apply(lambda r: r.nunique() == 1, axis=1).groupby(level=0).mean()
        agree.to_csv(f"{OUT}/T8b_seed_agreement.csv")
    # Monte Carlo
    mcs = [prep(x) for x in (load(f"mc{s}") for s in range(2)) if x is not None]
    if mcs:
        mc = pd.concat(mcs)
        mc.to_csv(f"{OUT}/MC_all_runs.csv", index=False)
        rows = []
        for md, g in mc[mc.n_failed > 0].groupby("mode"):
            # bootstrap over failure sets (clusters), seeds and manoeuvres pooled inside the cluster
            cl = g.groupby("case_id").success.mean().values
            bs = [rng.choice(cl, len(cl)).mean() for _ in range(B)]
            nm = nom[(nom.controller == "FTC") & (nom["mode"] == md) & (nom.n_failed > 0)]
            rows.append(dict(mode=md, runs=len(g), success_nominal=nm.success.mean(), success_mc=g.success.mean(),
                             ci_lo=ci(bs)[0], ci_hi=ci(bs)[1], completed_mc=g.completed.mean(), path_kept_mc=g.path_kept.mean(),
                             rms_median=g.rms_dev.median()))
        pd.DataFrame(rows).to_csv(f"{OUT}/T9_montecarlo.csv", index=False)
        mc.groupby(["seed"]).delay_ms.describe().to_csv(f"{OUT}/T9b_mc_delays.csv")
    # detection delay vs commanded RPM at failure (single failures, FTC)
    sf = nom[(nom.controller == "FTC") & (nom.n_failed == 1)]
    sf = sf.assign(rpm_bin=pd.cut(sf.ncmd_fail_min_abs, [-1, 250, 500, 1000, 4000]))
    sf.groupby("rpm_bin").det_delay_max.describe().to_csv(f"{OUT}/T10_delay_vs_rpm.csv")
    allf = nom[(nom.controller == "FTC") & (nom.n_failed > 0)]
    S["fdi"] = dict(failures=int(allf.n_failed.sum()), detected=int(allf.n_detected.sum()),
                    false_alarms=int(allf.false_alarms.sum()),
                    delay_median=float(allf.det_delay_mean.median()), delay_p95=float(np.nanpercentile(allf.det_delay_max, 95)),
                    undetected_runs=int((allf.n_detected < allf.n_failed).sum()))
    # ablation (same manoeuvres)
    ab = load("ablation")
    if ab is not None:
        ab = prep(ab); am = ab.motion_id.unique()
        A2 = pd.concat([nom[(nom.controller == "FTC") & nom.motion_id.isin(am)], ab])
        ta = table_main(A2, rng, ["FTC"] + [c for c in ORDER if c in ab.controller.unique()])
        ta.to_csv(f"{OUT}/T11_ablation.csv", index=False)
        table_by(A2, ["FTC"] + list(ab.controller.unique()), "mode").to_csv(f"{OUT}/T11b_ablation_by_class.csv")
        pa = paired(A2, rng, "FTC", list(ab.controller.unique())); pa.to_csv(f"{OUT}/T11c_ablation_paired.csv", index=False)
    # abort outcomes (FTC)
    fa = nom[(nom.controller == "FTC") & nom.aborted & (nom.n_failed > 0)]
    S["aborts"] = {md: dict(n=int(len(g)), surfaced=float(g.surfaced.mean()), t_abort_med=float(np.nanmedian(g.t_abort - 20)),
                       z_final_med=float(g.z_final.median()), z_final_max=float(g.z_final.max())) for md, g in fa.groupby("mode")}
    # saturation: why priority and BLS give the same closed loop
    S["saturation"] = nom[nom.n_failed > 0].groupby("controller").sat_frac.describe()[["mean", "50%", "75%", "max"]].to_dict("index")
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print(T1.round(3).to_string()); print(P.round(4).to_string()); print(json.dumps(S["fdi"]))


if __name__ == "__main__":
    main()

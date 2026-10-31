"""Figures for the manuscript (results/revision*, results/theory, results/stability_switching, twin stress)."""
import os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG = os.path.join(_ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "axes.edgecolor": MUTED, "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
CTRL_C = {"BL": "#9a9893", "BL+TS": "#5f5e5a", "FA-WPI": C[3], "FA-WPI+TS": "#b37a00", "FA-BLS": C[4], "FA-BLS+TS": "#b8436d",
          "MPC": C[2], "MPC+TS": "#0d7a52", "FTC-noTS": "#8fb7ea", "FTC": C[0], "MPC+SL": C[6]}
ORDER = ["BL", "BL+TS", "FA-WPI", "FA-WPI+TS", "FA-BLS", "FA-BLS+TS", "MPC", "MPC+TS", "FTC-noTS", "FTC"]
MODES = ["FULL", "COUPLED_ATTITUDE", "HEADING_STEERED", "INFEASIBLE_HORIZONTAL", "INFEASIBLE_DEPTH"]
MODE_LAB = dict(zip(MODES, ["Full", "Coupled", "Heading-steered", "Infeasible-H", "Infeasible-D"]))
sys.path.insert(0, os.path.join(_ROOT, "ftc_rov"))
from analysis_revision import load, prep


def nominal():
    fr = [x for x in (load("main"), load("mpc"), load("mpcsub")) if x is not None]
    d = prep(pd.concat(fr))
    nm = d.groupby("controller").motion_id.nunique()
    return d[d.controller.isin(nm[nm == nm.max()].index)]          # full-campaign controllers only


def fig_outcomes(df):
    d = df[df.n_failed > 0]
    ctrls = [c for c in ORDER if c in d.controller.unique()]
    cats = [("success", "Success (completed, path kept, tilt < 30°)", C[0]),
            ("deg", "Completed, path or tilt criterion missed", "#8fb7ea"),
            ("abort", "Aborted by supervisor (mission-infeasible)", C[3]),
            ("fail", "Not completed, not aborted", "#c9c8c3")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for k, c in enumerate(ctrls):
        x = d[d.controller == c]
        v = dict(success=x.success.mean(), deg=(x.completed & ~x.success).mean(), abort=x.aborted.mean(),
                 fail=(~x.completed & ~x.aborted).mean())
        b = 0
        for key, lab, col in cats:
            ax.bar(k, v[key] * 100, bottom=b, color=col, edgecolor="white", lw=0.6, label=lab if k == 0 else None)
            b += v[key] * 100
        ax.text(k, v["success"] * 100 / 2, f"{v['success']*100:.0f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(len(ctrls))); ax.set_xticklabels(ctrls, rotation=35, ha="right")
    ax.set_ylabel("% of failure runs (24 manoeuvres × 254 sets)"); ax.set_ylim(0, 100); ax.grid(axis="x")
    ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(f"{FIG}/fig_outcomes.png"); plt.close(fig)


def fig_success_nfailed(df):
    d = df[df.n_failed > 0]
    ctrls = [c for c in ["BL", "BL+TS", "FA-BLS+TS", "MPC", "FTC"] if c in d.controller.unique()]
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, col, title in [(axs[0], "success", "(a) mission success"), (axs[1], "path_kept", "(b) path kept (not aborted)")]:
        for c in ctrls:
            x = d[d.controller == c]
            ks = sorted(x.n_failed.unique())
            m = [x[x.n_failed == k][col].mean() * 100 for k in ks]
            lo, hi = [], []
            for k in ks:
                piv = x[x.n_failed == k].pivot_table(index="motion_id", columns="case_id", values=col).values.astype(float)
                bs = [np.nanmean(piv[rng.integers(0, piv.shape[0], piv.shape[0])]) for _ in range(400)]
                lo.append(np.percentile(bs, 2.5) * 100); hi.append(np.percentile(bs, 97.5) * 100)
            lw = 2.0 if c == "FTC" else 1.1
            ax.plot(ks, m, "-o", ms=2.5, lw=lw, color=CTRL_C[c], label=("FA-WPI+TS / FA-BLS+TS" if c == "FA-BLS+TS" else c))
            ax.fill_between(ks, lo, hi, color=CTRL_C[c], alpha=0.12, lw=0)
        ax.set_xlabel("Failed thrusters"); ax.set_ylabel("% of runs"); ax.set_ylim(0, 102); ax.set_title(title, loc="left", fontsize=8.5)
    axs[1].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_success_nfailed.png"); plt.close(fig)


def fig_fairness(df):
    d = df[df.n_failed > 0]
    pairs = [("BL", "BL+TS"), ("FA-WPI", "FA-WPI+TS"), ("FA-BLS", "FA-BLS+TS"), ("MPC", "MPC+TS"), ("FTC-noTS", "FTC")]
    pairs = [p for p in pairs if p[0] in d.controller.unique() and p[1] in d.controller.unique()]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, col, title in [(axs[0], "success", "(a) success"), (axs[1], "tilt_ok", "(b) max tilt < 30°"),
                           (axs[2], "deadline_1p5", "(c) completed within 1.5 × nominal time")]:
        for k, (a, b) in enumerate(pairs):
            va = d[d.controller == a][col].mean() * 100; vb = d[d.controller == b][col].mean() * 100
            ax.bar(k - 0.18, va, 0.34, color="#c9c8c3"); ax.bar(k + 0.18, vb, 0.34, color=CTRL_C[b])
            ax.text(k - 0.18, va + 1, f"{va:.0f}", ha="center", fontsize=6); ax.text(k + 0.18, vb + 1, f"{vb:.0f}", ha="center", fontsize=6)
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p[0].replace("-noTS", "") for p in pairs], rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0, 105); ax.set_title(title, loc="left", fontsize=8); ax.grid(axis="x")
    from matplotlib.patches import Patch
    axs[0].legend(handles=[Patch(color="#c9c8c3", label="nominal timing"), Patch(color="#5f5e5a", label="time-scaled (colour per controller)")],
                  fontsize=6.0, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_fairness.png"); plt.close(fig)


def fig_heatmap(df, ctrls=("FTC", "FA-BLS+TS", "MPC+TS")):
    d = df[df.n_failed > 0]
    ctrls = [c for c in ctrls if c in d.controller.unique()]
    seq = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#0d3d78"])
    fig, axs = plt.subplots(1, len(ctrls), figsize=(3.4 * len(ctrls), 5.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, ctrls):
        x = d[d.controller == c]
        piv = x.groupby(["motion_id", "n_failed"]).success.mean().unstack() * 100
        names = x.groupby("motion_id").motion.first().reindex(piv.index)
        im = ax.imshow(piv.values, cmap=seq, vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(piv.values):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.6, color="white" if v > 60 else INK)
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([int(k) for k in piv.columns])
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names.values, fontsize=6.5)
        ax.set_xlabel("Failed thrusters"); ax.grid(False); ax.set_title(c, loc="left", fontsize=8.5)
    cb = fig.colorbar(im, ax=axs, shrink=0.6); cb.set_label("Mission success (% of failure sets)")
    fig.savefig(f"{FIG}/fig_heatmap.png"); plt.close(fig)


def fig_theory():
    T = os.path.join(_ROOT, "ftc_rov/results/theory")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    for chi, col in [(0, C[0]), (90, C[1]), (180, C[2]), (-90, C[6])]:
        f = f"{T}/T6_spin_traj_Tm25_chi{chi}.npy"
        if os.path.exists(f):
            tr = np.load(f)
            ax.plot(tr[:, 1], tr[:, 0], color=col, lw=0.8, label=f"commanded direction {chi}°")
    ax.set_aspect("equal", "datalim"); ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
    ax.set_title("(a) one horizontal thruster (T1), spinning,\nphase-modulated thrust: net drift ≈ 1.8 cm/s", loc="left", fontsize=8)
    ax.legend(fontsize=6)
    ax = axs[1]
    e5 = pd.read_csv(f"{T}/T5_equilibria_summary.csv")
    e5 = e5[(e5.kind == "transit") & (e5.speed == 0.3)]
    vals = [e5[e5["mode"] == m].frac_dirs_mean.values[0] * 100 if (e5["mode"] == m).any() else np.nan for m in MODES[:4]]
    ax.bar(range(4), vals, color=[C[0], C[2], C[3], "#9a9893"])
    for k, v in enumerate(vals):
        ax.text(k, v + 2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels([MODE_LAB[m] for m in MODES[:4]], rotation=20)
    ax.set_ylabel("% of flow directions"); ax.set_ylim(0, 110); ax.grid(axis="x")
    ax.set_title("(b) steady level transit at 0.3 m/s exists\n(attainable drag wrench, 2° grid)", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_theory.png"); plt.close(fig)


def fig_twin():
    S = pd.read_csv(os.path.join(_ROOT, "paper_repro/data/stress_rows.csv"))
    S["ok"] = (S.pred == S.true) & (S.delay > 0)
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    sg = S[(S.kind == "single") & (S.scenario != "S2_cur030_90")]
    labs = ["none"] + [f"T{i}" for i in range(1, 9)] + ["alarm"]
    cm = np.zeros((9, 10))
    for _, r in sg.iterrows():
        p = r.pred if r.pred in labs else "alarm"
        if r.pred != "none" and not (r.delay > 0):
            p = "alarm"                      # declared before onset = false alarm
        cm[int(r.true[1:]), labs.index(p)] += 1
    nf = S[(S.kind == "nofault")]
    cm[0, 0] = (nf.pred == "none").sum(); cm[0, 9] = (nf.pred != "none").sum()
    ax = axs[0]
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap=LinearSegmentedColormap.from_list("s", ["#ffffff", C[0]]), vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        if v:
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(10)); ax.set_xticklabels(["none"] + [f"T{i}" for i in range(1, 9)] + ["false\nalarm"], fontsize=5.5)
    ax.set_yticks(range(9)); ax.set_yticklabels(["no fault"] + [f"T{i}" for i in range(1, 9)], fontsize=6)
    ax.set_xlabel("declared"); ax.set_ylabel("true"); ax.grid(False); ax.set_title("(a) confusion matrix (all losses)", loc="left", fontsize=8)
    ax = axs[1]
    sgl = S[S.kind == "single"]
    for sc, col, lab in [("S1_still", C[0], "still water"), ("S2_cur020_45", C[2], "0.2 m/s @45°"),
                         ("S3_mismatch", C[6], "model-form mismatch"), ("S2_cur030_90", C[3], "0.3 m/s @90° (outside cal.)")]:
        for grp, ls in (("H", "-"), ("V", "--")):
            x = sgl[(sgl.scenario == sc) & sgl.true.isin([f"T{i}" for i in ((1, 2, 3, 4) if grp == "H" else (5, 6, 7, 8))])]
            g = x.groupby("loss").ok.mean() * 100
            ax.plot(g.index * 100, g.values, ls, marker="o", ms=2.2, color=col, lw=1, label=lab if grp == "H" else None)
    ax.plot([], [], "k-", lw=1, label="T1–T4"); ax.plot([], [], "k--", lw=1, label="T5–T8")
    ax.set_xlabel("thrust loss [%]"); ax.set_ylabel("correctly isolated [%]"); ax.set_ylim(-3, 105)
    ax.set_title("(b) isolation rate", loc="left", fontsize=8); ax.legend(fontsize=4.8, ncol=1, loc="lower right", bbox_to_anchor=(1.0, 0.04), framealpha=0.95)
    ax = axs[2]
    det = sgl[sgl.ok & (sgl.scenario != "S2_cur030_90")]
    ls_ = sorted(det.loss.unique())
    data = [det[det.loss == l].delay.values for l in ls_]
    ax.boxplot(data, positions=range(len(data)), widths=0.6, showfliers=True, patch_artist=True, flierprops=dict(ms=2),
               boxprops=dict(facecolor="#cfe0f6", edgecolor=C[0]), medianprops=dict(color=C[0]))
    ax.set_xticks(range(len(data))); ax.set_xticklabels([f"{int(l*100)}\n(n={len(d)})" for l, d in zip(ls_, data)], fontsize=6)
    ax.set_yscale("log"); ax.axhline(0.5, color=MUTED, ls=":", lw=0.8); ax.text(len(data) - 0.5, 0.55, "persistence 0.5 s", fontsize=5.5, ha="right", color=INK2)
    ax.set_xlabel("thrust loss [%]"); ax.set_ylabel("declaration delay [s]"); ax.set_title("(c) delay (correct isolations)", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_twin_stress.png"); plt.close(fig)


def fig_switching():
    f = os.path.join(_ROOT, "ftc_rov/results/stability_switching/B_random_switching.csv")
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for k, md in enumerate(["FULL", "COUPLED_ATTITUDE", "HEADING_STEERED"]):
        for j, fast in enumerate([False, True]):
            x = d[(d.final_class == md) & (d.fast == fast)]
            ax.scatter(np.full(len(x), k + (j - 0.5) * 0.3) + np.random.default_rng(k).uniform(-0.08, 0.08, len(x)),
                       x.err_tail_rms, s=5, color=[C[0], C[2], C[3]][k], alpha=0.5 if fast else 0.9,
                       marker="^" if fast else "o", label=("0.3 s apart" if fast else "random 0.3-10 s") if k == 0 else None)
    ax.set_yscale("log"); ax.axhline(0.25, color=MUTED, ls=":", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([MODE_LAB[m] for m in ["FULL", "COUPLED_ATTITUDE", "HEADING_STEERED"]])
    ax.set_ylabel("RMS error, last 20 s [m]"); ax.legend(fontsize=6.5)
    ax.set_title("Random cascading failures from perturbed states (final class)", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_switching.png"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["theory"]
    if any(w in which for w in ("outcomes", "nfailed", "fairness", "heatmap")):
        df = nominal()
        if "outcomes" in which: fig_outcomes(df)
        if "nfailed" in which: fig_success_nfailed(df)
        if "fairness" in which: fig_fairness(df)
        if "heatmap" in which: fig_heatmap(df)
    if "theory" in which: fig_theory()
    if "twin" in which: fig_twin()
    if "switching" in which: fig_switching()
    print("ok", which)

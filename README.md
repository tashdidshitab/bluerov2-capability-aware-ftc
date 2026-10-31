# Capability-aware fault-tolerant control of the BlueROV2 Heavy across all 254 thruster-failure combinations

This repository contains the code and data for the manuscript *"Capability-aware fault-tolerant control of an eight-thruster underwater vehicle across all 254 thruster-failure combinations"*. It contains the source code, configuration (all settings are in the scripts; seeds listed below), the 254 generated reconfiguration laws, the raw per-run metrics of every experiment, and the scripts that regenerate every table and figure.

## Layout

| Path | Content |
|---|---|
| `ftc_rov/rov_model.py` | 6-DOF BlueROV2 Heavy model (Fossen; von Benzon et al. 2022 parameters), T200 RPM model, power model, model-form mismatch effects |
| `ftc_rov/allocation.py` | Structural classification, weighted pseudo-inverse laws, priority / RPI / clipping allocators, capability LPs |
| `ftc_rov/trajectories.py` | Exact generators of the 24 reference manoeuvres (Table 5) |
| `ftc_rov/sim.py` | Batched closed-loop simulator: controller (Eqs. 7–14), guidance, time scaling, supervisor, RPM FDI, digital twin, mission metrics (Section 5.3) |
| `ftc_rov/mpc.py` | Fault-aware LTV-MPC comparator (FISTA QP) |
| `ftc_rov/run_revision.py` | Controller configurations (`CONFIGS`), mismatch plant (`MISMATCH`), campaign driver |
| `ftc_rov/analysis_revision.py` | Statistics: crossed two-way cluster bootstrap (B = 2000, seed 7), exact McNemar, rank-biserial |
| `ftc_rov/theory.py`, `theory_t7.py` | Proposition 1 computations (exact ranks, tolerance, perturbation, controllability, equilibria, spinning demo) |
| `ftc_rov/switching_stability.py` | Proposition 2 computations and randomised cascading-failure tests |
| `ftc_rov/sensitivity.py` | λ and heading-rate sensitivity; manoeuvre table |
| `ftc_rov/bench/` | Single-precision C implementation of one control step + benchmark (Table 13) |
| `ftc_rov/results/reconfiguration_laws.json`, `reconfiguration_table.csv` | The 254 laws (class, d₀, weights, K_F, κ_F, supervisor action) and steady RPM table (Table 4) |
| `ftc_rov/results/revision/*_all.csv` | **Raw per-run metrics** of every experiment (one row per manoeuvre × health case × controller) |
| `ftc_rov/results/revision_tables/` | Statistical tables produced by `analysis_revision.py` |
| `ftc_rov/results/theory/`, `stability_switching/`, `sensitivity/` | Raw outputs of the theory, switching and sensitivity studies |
| `paper_repro/twin_detect.py`, `twin_stress.py` | Digital-twin detector (timing logic of Section 6.6) and stress test |
| `paper_repro/stability.py` | 34-state closed-loop linearisation per failure set (Table 12) |
| `paper_repro/data/` | `stress_rows.csv` (one row per twin test), thresholds, `stability.csv` |

## Requirements

Python 3.13 with the packages in `requirements.txt`; gcc for the benchmark. Everything runs on a standard 2-core machine; total CPU time for the full pipeline is about 10–12 h.

## One command

```
make all          # or individual targets, see below
```

## Table and figure map

| Item | Command (`make` target) | Output |
|---|---|---|
| Table 2, Fig. 2, Section 3.3 | `theory` | `ftc_rov/results/theory/T1_exact_ranks.csv`, `T2…T3` in `theory_summary.json` |
| Proposition 1 (b)–(d), Fig. 3 | `theory`, then `figures` | `T4_controllability*.csv`, `T5_equilibria*.csv`, `T7_depth_infeasible_equilibria.csv`, `T6_spin_traj_*.npy` |
| Table 4 | `laws` | `ftc_rov/results/reconfiguration_table.csv` |
| Table 5 | `sensitivity` | `ftc_rov/results/sensitivity/manoeuvre_table.csv` |
| Table 6, Figs 5–8 | `campaign`, `tables`, `figures` | `results/revision/main_all.csv`, `revision_tables/T1_main.csv`, `T2…T5`, `T4_paired_vs_FTC.csv` |
| Table 7 (MPC) | `mpc`, `tables` | `results/revision/mpcsub_all.csv`, `revision_tables/T6_subset_with_MPCSL.csv`, `T6b`, `T6c` |
| Table 8 (classes) | `campaign`, `tables` | `revision_tables/T3*_by_class.csv`, `T5_infeasible.csv` |
| Table 9 (robustness) | `robustness`, `tables` | `mc0_all.csv`, `mismatch_all.csv`, `seed1_all.csv`; `revision_tables/T7*, T8*, T9*` |
| Table 10 (ablation) | `ablation`, `tables` | `ablation_all.csv`; `revision_tables/T11*` |
| Table 11, Fig. 9 (digital twin) | `twin`, `figures` | `paper_repro/data/stress_rows.csv` |
| Table 12 | `stability` | `paper_repro/data/stability.csv` |
| Proposition 2, Fig. 10 | `switching`, `figures` | `ftc_rov/results/stability_switching/` |
| Table 13 | `bench` | printed JSON (median, percentiles, worst case, memory) |
| Table 14 | `sensitivity` | `results/sensitivity/lambda_*.csv`, `heading_rate.csv` |
| Detection statistics (Section 6.4) | `tables` | `revision_tables/summary.json` (`fdi`), `T10_delay_vs_rpm.csv` |

## Seeds and stochastic settings

- Nominal campaign, ablation, MPC, mismatch: sensor-noise seed 0 (identical realisation for every controller); failure at t = 20 s, detected online.
- Noise replication: seed 1.
- Monte Carlo (`unc=0.2`, seed 0): per vehicle, uniform ±20 % on inertia, added mass and damping (mass ±5 %), net buoyancy 0–3 N, metacentric height 0.5–2 cm, ±10 % thruster gain, current speed × 0.5–1.5 and direction ±0.5 rad; sensor delay 0–5 control steps drawn per manoeuvre with `numpy.random.default_rng(1000 + seed + 100 × manoeuvre_index)`.
- Bootstrap: `numpy.random.default_rng(7)`, B = 2000.
- Digital twin: calibration seed 100 (thresholds = 1.25 × maximum fault-free residual); evaluation seeds 1–3 (S1–S3), 5 (S4), 7–8 (S5), 300 (S6).
- Switching tests: seeds 0–2, 120 vehicles per (seed, mission, switching speed).

## Column dictionary of `results/revision/*_all.csv`

`controller, motion, motion_id, case, case_id, n_failed, mode` identify the run. Outcome columns follow Section 5.3: `path_ok` (RMS < 0.25 m and max < 1 m over the active post-failure window), `completed` (not aborted, within 1 m of the final reference point within 20 s after the reference ends), `success` (completed, path_ok, max tilt < 30°), `aborted`, `t_abort`, `surfaced`, `z_final`, `deadline_1p5`, `deadline_2`, `t_arrive`, `duration_ratio`. Accuracy and safety: `rms_dev`, `max_dev`, `max_tilt_deg`, `rms_depth`, `rms_yaw_deg`. Energy: `energy_Wh` (thruster), `energy_total_Wh` (thruster + 40 W hotel). Detection: `det_delay_mean`, `det_delay_max`, `n_detected`, `false_alarms`, `ncmd_fail_min_abs`. Cost: `ctrl_us_per_vehicle_step`, `mpc_solve_ms_batch`.

## Version

Release v2.0 (October 2026). A DOI will be minted when the archive is deposited in a public repository.

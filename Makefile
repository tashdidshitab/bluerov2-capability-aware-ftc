# Reproduce the results of the manuscript "Capability-aware fault-tolerant control of an eight-thruster underwater vehicle
# across all 254 thruster-failure combinations".  Run from this directory.  CPU times are for 2 cores.
PY ?= python3
SUB = 0,2,4,6,11,14,15,19
ABL = 0,2,6,9,14,12

all: laws theory campaign robustness mpc ablation stability twin switching sensitivity bench tables figures

laws:            ## 254 reconfiguration laws -> ftc_rov/results/reconfiguration_*.{csv,json}            (~1 min)
	cd ftc_rov && $(PY) reconfig_table.py
theory:          ## Proposition 1, Section 3.3: exact ranks, tolerance, perturbation, controllability, equilibria (~1 h)
	cd ftc_rov && $(PY) theory.py && $(PY) theory_t7.py
campaign:        ## Table 6, Figs 5-8: 8 controllers x 24 manoeuvres x 255 health cases (seed 0)           (~2.5 h)
	cd ftc_rov && $(PY) run_revision.py main configs=FTC,BL,BL+TS,FA-WPI,FA-WPI+TS,FA-BLS,FA-BLS+TS,FTC-noTS
robustness:      ## Table 9: Monte Carlo (unc 0.2, seed 0), mismatch plant, noise seed 1                    (~2 h)
	cd ftc_rov && $(PY) run_revision.py mc0 configs=FTC unc=0.2 seed=0
	cd ftc_rov && $(PY) run_revision.py mismatch plant=mismatch configs=FTC,BL,FA-BLS+TS motions=$(SUB)
	cd ftc_rov && $(PY) run_revision.py seed1 configs=FTC,BL seed=1 motions=$(SUB)
mpc:             ## Table 7: fault-aware MPC variants on the 8-manoeuvre subset                             (~1.5 h)
	cd ftc_rov && $(PY) run_revision.py mpcsub configs=MPC,MPC+TS,MPC+SL motions=$(SUB)
ablation:        ## Table 10                                                                                (~20 min)
	cd ftc_rov && $(PY) run_revision.py ablation configs=FTC-noGuid,FTC-RPI,FTC-oracle motions=$(ABL)
stability:       ## Table 12: 34-state linearisation per failure set                                         (~50 min)
	cd paper_repro && $(PY) stability.py
twin:            ## Table 11, Fig. 9: digital-twin stress test (calibration seed 100, evaluation seeds 1-3,5,7,8,300) (~15 min)
	cd paper_repro && $(PY) twin_stress.py
switching:       ## Proposition 2, Fig. 10: loop polynomials, allocation residual, randomised cascading failures (~30 min)
	cd ftc_rov && $(PY) switching_stability.py
sensitivity:     ## Table 5 (manoeuvres), Table 14 (lambda, heading-rate limit)                                (~40 min)
	cd ftc_rov && $(PY) sensitivity.py
bench:           ## Table 13: C real-time benchmark
	cd ftc_rov/bench && gcc -O2 -std=gnu99 -o controller_step controller_step.c -lm && ./controller_step 2000000
tables:          ## statistics (crossed bootstrap, McNemar, rank-biserial) -> ftc_rov/results/revision_tables; manuscript tables
	cd ftc_rov && $(PY) analysis_revision.py
	@true
figures:         ## Figs 3, 5-10
	cd ftc_rov && $(PY) figures_revision.py theory outcomes nfailed fairness heatmap twin switching
.PHONY: all laws theory campaign robustness mpc ablation stability twin switching sensitivity bench tables figures

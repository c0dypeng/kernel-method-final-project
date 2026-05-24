#!/usr/bin/env bash
# Small-scale benchmark: 3 algos x 2 envs x 3 seeds = 18 runs.
# Smaller budgets than run_all.sh — produces real (if noisy) training curves
# fast enough to fit in a single overnight session, so you have plottable
# numbers for slides before the full benchmark finishes.
#
# Outputs to results/small_scale/  (separate from smoke and full-benchmark).
#
# Budgets:
#   Pendulum-v1:    20,000 env steps   (SAC + MBPO),  6 iterations  (PILCO)
#   HalfCheetah-v4: 50,000 env steps   (SAC + MBPO),  5 iterations  (PILCO)
#
# Expected wallclock on a single RTX 2080 Ti:
#   SAC  Pendulum (20K)        ~2 min/seed     x 3 =  6 min
#   SAC  HalfCheetah (50K)     ~10 min/seed    x 3 = 30 min
#   MBPO Pendulum (20K)        ~12 min/seed    x 3 = 36 min
#   MBPO HalfCheetah (50K)     ~50 min/seed    x 3 = 2.5 hr
#   PILCO Pendulum  (6 iter)   ~10 min/seed    x 3 = 30 min
#   PILCO HalfCheetah (5 iter) ~20 min/seed    x 3 = 60 min
# Grand total: ~5-6 hr. Fits comfortably in a single working day.
#
# Caveats (also noted in the dialog where you picked this scale):
#   - 50K HalfCheetah is too short to clearly show MBPO's sample-efficiency
#     advantage over SAC. The curves will all still be climbing at the end.
#   - 20K Pendulum is enough for SAC to fully solve it.
#   - PILCO results should look similar at both scales (PILCO is iteration-
#     bounded, not env-step-bounded).
#
# To run a subset, set ALGOS, ENVS, or SEEDS env vars:
#   ALGOS="sac pilco" SEEDS="0 1" ./run_small_scale.sh

set -euo pipefail

# NB: outer loop is SEEDS, inner loop is ALGOS. This way if the run gets
# interrupted you have *complete seeds* (good for plotting partial results)
# rather than complete algorithms-but-only-on-some-seeds. Order within each
# seed is PILCO -> SAC -> MBPO: PILCO first because it's the riskiest
# (most likely to crash on HalfCheetah), SAC second as the baseline that
# always works, MBPO last as the heaviest cell.
ALGOS=${ALGOS:-"pilco sac mbpo"}
ENVS=${ENVS:-"Pendulum-v1 HalfCheetah-v4"}
SEEDS=${SEEDS:-"0 1 2"}
RESULTS_DIR=${RESULTS_DIR:-"results/small_scale"}
LOGS_DIR=${LOGS_DIR:-"logs/small_scale"}

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" plots/small_scale

for seed in $SEEDS; do
  for env in $ENVS; do
    for algo in $ALGOS; do
      tag="${algo}__${env}__seed${seed}"
      echo
      echo "===================================================================="
      echo "  SMALL-SCALE RUN: $tag"
      echo "===================================================================="

      case "$env" in
        Pendulum-v1)    sac_steps=20000; mbpo_steps=20000; pilco_iter=6 ;;
        HalfCheetah-v4) sac_steps=50000; mbpo_steps=50000; pilco_iter=5 ;;
        *) echo "unknown env $env" >&2; exit 1 ;;
      esac

      case "$algo" in
        sac)
          python -m runners.run_sac \
              --env "$env" --steps "$sac_steps" --seed "$seed" \
              --results-dir "$RESULTS_DIR" \
              2>&1 | tee "$LOGS_DIR/${tag}.log" || echo "[!] ${tag} FAILED"
          ;;
        mbpo)
          python -m runners.run_mbpo \
              --env "$env" --steps "$mbpo_steps" --seed "$seed" \
              --results-dir "$RESULTS_DIR" \
              2>&1 | tee "$LOGS_DIR/${tag}.log" || echo "[!] ${tag} FAILED"
          ;;
        pilco)
          python -m runners.run_pilco \
              --env "$env" --iterations "$pilco_iter" --seed "$seed" \
              --results-dir "$RESULTS_DIR" \
              2>&1 | tee "$LOGS_DIR/${tag}.log" || echo "[!] ${tag} FAILED (expected for PILCO on HalfCheetah)"
          ;;
        *) echo "unknown algo $algo" >&2; exit 1 ;;
      esac
    done
  done
done

echo
echo "===================================================================="
echo "  SMALL-SCALE SWEEP DONE — generating plots"
echo "===================================================================="
python -m runners.plot_results --results-dir "$RESULTS_DIR" --out-dir plots/small_scale
echo "Done. CSVs in $RESULTS_DIR/, plots in plots/small_scale/."

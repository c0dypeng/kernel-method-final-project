#!/usr/bin/env bash
# Orchestration script: runs all (algo, env, seed) combinations sequentially.
# Total: 3 algos x 2 envs x 3 seeds = 18 runs.
# Full benchmark budget: Pendulum 100K env steps, HalfCheetah 1M env steps.
# Expected wallclock on a single RTX 2080 Ti:
#   SAC Pendulum (100K)        ~10 min/seed   x 3 = 30 min
#   SAC HalfCheetah (1M)       ~2.5 hr/seed   x 3 = 7.5 hr
#   MBPO Pendulum (100K)       ~1 hr/seed     x 3 = 3 hr     <- model+SAC inner loop is heavy
#   MBPO HalfCheetah (1M)      ~15 hr/seed    x 3 = 45 hr    <- the long pole
#   PILCO Pendulum (10 iter)   ~15 min/seed   x 3 = 45 min
#   PILCO HalfCheetah (8 iter, sparse, expected fail) ~30 min/seed x 3 = 90 min
# Grand total: ~60 hr. Plan for ~3 days of wall time on a single GPU, or
# parallelize across two GPUs (see README "Two-GPU tip") to cut roughly in half.
#
# To run a subset, set ALGOS, ENVS, or SEEDS env vars:
#   ALGOS="sac pilco" SEEDS="0 1" ./run_all.sh

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

mkdir -p results plots logs

for seed in $SEEDS; do
  for env in $ENVS; do
    for algo in $ALGOS; do
      tag="${algo}__${env}__seed${seed}"
      echo
      echo "===================================================================="
      echo "  RUN: $tag"
      echo "===================================================================="

      case "$env" in
        Pendulum-v1)    sac_steps=100000;  mbpo_steps=100000;  pilco_iter=10 ;;
        HalfCheetah-v4) sac_steps=1000000; mbpo_steps=1000000; pilco_iter=8 ;;
        *) echo "unknown env $env" >&2; exit 1 ;;
      esac

      case "$algo" in
        sac)
          python -m runners.run_sac --env "$env" --steps "$sac_steps" --seed "$seed" \
              2>&1 | tee "logs/${tag}.log" || echo "[!] ${tag} FAILED"
          ;;
        mbpo)
          python -m runners.run_mbpo --env "$env" --steps "$mbpo_steps" --seed "$seed" \
              2>&1 | tee "logs/${tag}.log" || echo "[!] ${tag} FAILED"
          ;;
        pilco)
          python -m runners.run_pilco --env "$env" --iterations "$pilco_iter" --seed "$seed" \
              2>&1 | tee "logs/${tag}.log" || echo "[!] ${tag} FAILED (expected for PILCO on HalfCheetah)"
          ;;
        *) echo "unknown algo $algo" >&2; exit 1 ;;
      esac
    done
  done
done

echo
echo "===================================================================="
echo "  ALL DONE — generating plots"
echo "===================================================================="
python -m runners.plot_results
echo "Done. Results in results/, plots in plots/."

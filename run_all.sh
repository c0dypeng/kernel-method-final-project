#!/usr/bin/env bash
# Orchestration script: runs all (algo, env, seed) combinations sequentially.
# Total: 3 algos x 2 envs x 3 seeds = 18 runs.
# Expected wallclock on a single RTX 2080 Ti:
#   SAC Pendulum (50K)        ~5  min/seed   x 3 = 15 min
#   SAC HalfCheetah (200K)    ~30 min/seed   x 3 = 90 min
#   MBPO Pendulum (50K)       ~30 min/seed   x 3 = 90 min   <- model+SAC inner loop is heavy
#   MBPO HalfCheetah (200K)   ~3 hr/seed     x 3 = 9 hr
#   PILCO Pendulum (8 iter)   ~10 min/seed   x 3 = 30 min
#   PILCO HalfCheetah (5 iter, sparse, expected fail) ~20 min/seed x 3 = 60 min
# Grand total: ~13 hr. Run overnight.
#
# To run a subset, set ALGOS, ENVS, or SEEDS env vars:
#   ALGOS="sac pilco" SEEDS="0 1" ./run_all.sh

set -euo pipefail

ALGOS=${ALGOS:-"sac mbpo pilco"}
ENVS=${ENVS:-"Pendulum-v1 HalfCheetah-v4"}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p results plots logs

for algo in $ALGOS; do
  for env in $ENVS; do
    for seed in $SEEDS; do
      tag="${algo}__${env}__seed${seed}"
      echo
      echo "===================================================================="
      echo "  RUN: $tag"
      echo "===================================================================="

      case "$env" in
        Pendulum-v1)    sac_steps=50000;  mbpo_steps=50000;  pilco_iter=8 ;;
        HalfCheetah-v4) sac_steps=200000; mbpo_steps=200000; pilco_iter=5 ;;
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

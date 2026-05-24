# Kernel-Method Final Project — PILCO vs MBPO vs SAC

A controlled comparison of three reinforcement learning algorithms across two continuous-control benchmarks, packaged for one-command reproduction on an NVIDIA GPU host via Docker.

| Algorithm | Family | Key idea | Implementation |
|---|---|---|---|
| **SAC** (Haarnoja et al. 2018) | Model-free, off-policy | Soft-actor-critic with stochastic policy + entropy bonus | [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) |
| **MBPO** (Janner et al. 2019) | Model-based | Ensemble of dynamics models + short imagined rollouts feeding SAC | [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib) |
| **PILCO** (Deisenroth, Fox & Rasmussen 2015) | Model-based, Bayesian | GP dynamics + analytic policy gradients via moment matching | [c0dypeng/PILCO-modern](https://github.com/c0dypeng/PILCO-modern) |

| Environment | obs / act dim | Budget | Why we picked it |
|---|---|---|---|
| `Pendulum-v1` | 3 / 1 | 100K env steps | Low-dim, PILCO's home turf, fast to train |
| `HalfCheetah-v4` | 17 / 6 | 1M env steps | MBPO/SAC benchmark, PILCO is expected to fail here |

## Expected results (qualitative — actual numbers from your runs go in `plots/`)

- **Pendulum-v1:** PILCO learns fastest in env-steps (a few hundred transitions), MBPO catches up by ~5K steps, SAC by ~10K. All three reach near-optimal (~−150) reward.
- **HalfCheetah-v4:** MBPO > SAC > PILCO. MBPO leads on sample efficiency, SAC overtakes at the longer horizon, PILCO is **expected to flatline near zero return** — the GP doesn't scale to 17D dynamics.

This last finding is the *interesting* one and motivates the whole class of modern model-based RL methods that use neural-net dynamics models (MBPO et al.) instead of GPs.

## Quick start (on the Ubuntu GPU host)

```bash
git clone https://github.com/c0dypeng/kernel-method-final-project.git
cd kernel-method-final-project
docker build -t kmfp .                              # 20-40 min first time
docker run --gpus all -it --rm kmfp python smoke_test.py
```

Expected: `Summary: 15/15 passed` (takes ~4–5 min). The last three checks actually run SAC, MBPO, and PILCO for short Pendulum runs so the full pipeline is verified end-to-end for all three algorithms.

Smoke-test eval values are saved to `results/smoke/`:
```
results/smoke/sac__Pendulum-v1__seed0.csv     # SAC,   2000 env steps, ~4 eval points
results/smoke/mbpo__Pendulum-v1__seed0.csv    # MBPO,   500 env steps, ~3 eval points
results/smoke/pilco__Pendulum-v1__seed0.csv   # PILCO,   2 iterations, ~3 eval points
```

These are kept separate from full-benchmark results in `results/*.csv` so the two never overwrite each other. You can plot the smoke-run curves immediately after smoke passes (good for slide-ready figures if the full sweep isn't done yet):
```bash
python -m runners.plot_results --results-dir results/smoke --out-dir plots/smoke
```

There are **three scales** of experiments, in order of size:

| Script | Pendulum | HalfCheetah | Wallclock | Output dir |
|---|---|---|---|---|
| `smoke_test.py` | 2K SAC, 500 MBPO, 2 PILCO iter | not run | ~5 min | `results/smoke/` |
| `./run_small_scale.sh` | **20K** SAC/MBPO, **6** PILCO iter | **50K** SAC/MBPO, **5** PILCO iter | ~5–6 hr | `results/small_scale/` |
| `./run_all.sh` (full) | 100K SAC/MBPO, 10 PILCO iter | 1M SAC/MBPO, 8 PILCO iter | ~60 hr | `results/` |

Each scale writes to its own subdirectory so results never overwrite each other. Recommendation: run `smoke_test.py` first (5 min) to confirm everything works, then `./run_small_scale.sh` overnight (~5–6 hr) for slide-ready training curves, then `./run_all.sh` over a few days for the final report numbers.

```bash
# Full benchmark (~60 hours):
docker run --gpus all -it --rm -v "$(pwd)":/workspace kmfp ./run_all.sh

# Small-scale (~5-6 hours, plottable curves for slides):
docker run --gpus all -it --rm -v "$(pwd)":/workspace kmfp ./run_small_scale.sh
```

Or run individual cells:

```bash
docker run --gpus all -it --rm -v "$(pwd)":/workspace kmfp \
    python -m runners.run_sac --env Pendulum-v1 --steps 50000 --seed 0
```

Outputs:
- `results/<algo>__<env>__seed<N>.csv` — unified schema: `env_steps, mean_return, std_return, wallclock_s`
- `logs/<algo>__<env>__seed<N>.log` — full stdout from each run
- `plots/<env>_comparison.png` — per-env figures with seed-mean ± min/max bands
- `plots/summary.csv` — final-return table

## Prereqs on the host

- NVIDIA GPU + driver supporting CUDA 12.1 (driver ≥ 530).
- Docker.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so `--gpus all` works.

Verify with:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## What's in the container

| Component | Version | Purpose |
|---|---|---|
| Python | 3.10 | |
| PyTorch | 2.1.2 (CUDA 12.1) | SAC + MBPO |
| stable-baselines3 | 2.3.2 | SAC |
| mbrl-lib | 0.2.0 | MBPO |
| TensorFlow | 2.15.1 | PILCO |
| GPflow | 2.9.2 | PILCO |
| gymnasium | 0.29.1 | All three |
| mujoco | 3.1.6 | HalfCheetah |

PyTorch and TensorFlow each bring their own CUDA runtime wheels — no conflict.

## Evaluation protocol

To make the algorithms directly comparable, **every runner uses the exact same evaluation function** (`evaluate_policy` in `runners/common.py`) and logs to the same CSV schema:

```
env_steps, mean_return, std_return, wallclock_s
```

- **env_steps** = cumulative real-environment transitions consumed (the universally-meaningful x-axis for sample efficiency).
- **mean_return / std_return** = mean and std of episode returns from `n_eval_episodes` deterministic-policy rollouts in a fresh env (no exploration noise). Default is 5 episodes per eval.

Eval cadence:
- **SAC**: every `total_steps/20` env steps (so 20 eval points per run).
- **MBPO**: same — `total_steps/20`. The runner explicitly hooks into the MBPO inner loop to run an eval at fixed intervals (mbrl-lib's built-in CSV is only logged once per epoch with `num_eval_episodes=1`, which is too sparse and has no error bars; we override it).
- **PILCO**: once per outer iteration (~10 evals total per run).

This is what makes the training curves comparable — same metric, same number of eval episodes, same deterministic policy.

### Plot axes

Because PILCO learns from a tiny replay buffer (~200–500 env steps total) and SAC/MBPO need 50K–200K env steps, the x-axis spans 2–3 orders of magnitude across the three algorithms. The comparison plot uses a **log x-axis** so all three curves are readable in the same figure. Without the log axis, PILCO collapses to a single line pressed against the y-axis.

### Summary table

`plots/summary.csv` reports the **mean of the last 3 eval points** per seed (then mean ± std across seeds). This is more robust than a single end-of-run point — particularly for PILCO where the GP can produce spurious final-iteration spikes.

## Why PILCO is expected to fail on HalfCheetah

PILCO's dynamics model is a Gaussian Process. GP inference is O(N³) in dataset size and the moment-matching policy evaluation is O(D²·N) per timestep where D is state dimension. With:
- 17-dim state × 6-dim action = 23-dim GP inputs
- ~125 transitions after 5 iterations × 25 steps each
- 6 outputs to model (one GP per state dimension)

…the kernel matrix is already poorly conditioned and the gradient signal through the moment-matched policy is noisy. Even with the sparse FITC variant (which we use here — see `ENV_CONFIGS["HalfCheetah-v4"]` in `runners/run_pilco.py`), PILCO essentially performs random search in this regime. **This is the expected outcome and the whole reason the field moved to MBPO-style methods.**

## Project layout

```
kernel-method-final-project/
├── Dockerfile, .dockerignore, .gitignore, README.md
├── smoke_test.py                  # 12-check sanity test (~2 min)
├── run_all.sh                     # full sweep: 3 algos × 2 envs × 3 seeds
├── runners/
│   ├── common.py                  # shared eval + CSV logging
│   ├── run_sac.py                 # SAC via stable-baselines3
│   ├── run_mbpo.py                # MBPO via mbrl-lib (DictConfig built programmatically)
│   ├── run_pilco.py               # PILCO using PILCO-modern algorithm code
│   └── plot_results.py            # comparison plots + summary CSV
├── pilco_src/                     # PILCO algorithm package (from c0dypeng/PILCO-modern)
│   ├── setup.py
│   └── pilco/{__init__.py, controllers.py, rewards.py, models/}
├── results/                       # CSV outputs (gitignored)
├── logs/                          # stdout per run (gitignored)
└── plots/                         # PNG figures + summary.csv (committed after runs)
```

## Iterating on code

Mount your checkout to skip rebuilds:

```bash
docker run --gpus all -it --rm -v "$(pwd)":/workspace kmfp bash
# inside container:
python -m runners.run_sac --env Pendulum-v1 --steps 5000 --seed 0   # quick test
```

## Two-GPU tip

If you have two GPUs and want to keep one free, pin to GPU 1:
```bash
docker run --gpus '"device=1"' -it --rm -v "$(pwd)":/workspace kmfp ./run_all.sh
```

## License

Code in this repo: MIT. PILCO sources under `pilco_src/` retain the original MIT license from nrontsis/PILCO (see `PILCO_LICENSE`). MBPO config follows the original BSD license from facebookresearch/mbrl-lib.

## References

- M.P. Deisenroth, D. Fox, C.E. Rasmussen, *Gaussian Processes for Data-Efficient Learning in Robotics and Control*, IEEE TPAMI 2015.
- T. Haarnoja et al., *Soft Actor-Critic*, ICML 2018.
- M. Janner, J. Fu, M. Zhang, S. Levine, *When to Trust Your Model: Model-Based Policy Optimization*, NeurIPS 2019.
- L. Pineda et al., *MBRL-Lib: A Modular Library for Model-Based RL*, 2021.

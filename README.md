# Kernel-Method Final Project — PILCO vs MBPO vs SAC

A controlled comparison of three reinforcement learning algorithms across two continuous-control benchmarks, packaged for one-command reproduction on an NVIDIA GPU host via Docker.

| Algorithm | Family | Key idea | Implementation |
|---|---|---|---|
| **SAC** (Haarnoja et al. 2018) | Model-free, off-policy | Soft-actor-critic with stochastic policy + entropy bonus | [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) |
| **MBPO** (Janner et al. 2019) | Model-based | Ensemble of dynamics models + short imagined rollouts feeding SAC | [facebookresearch/mbrl-lib](https://github.com/facebookresearch/mbrl-lib) |
| **PILCO** (Deisenroth, Fox & Rasmussen 2015) | Model-based, Bayesian | GP dynamics + analytic policy gradients via moment matching | [c0dypeng/PILCO-modern](https://github.com/c0dypeng/PILCO-modern) |

| Environment | obs / act dim | Why we picked it |
|---|---|---|
| `Pendulum-v1` | 3 / 1 | Low-dim, PILCO's home turf, fast to train |
| `HalfCheetah-v4` | 17 / 6 | MBPO/SAC benchmark, PILCO is expected to fail here |

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

Expected: `Summary: 12/12 passed`.

Then run the whole benchmark (~13 hours on a single 2080 Ti — see `run_all.sh` for breakdown):

```bash
docker run --gpus all -it --rm -v "$(pwd)":/workspace kmfp ./run_all.sh
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

To make the algorithms directly comparable, every runner logs to the same CSV schema (`env_steps`, `mean_return`, `std_return`, `wallclock_s`) where:

- **env_steps** = cumulative real-environment transitions consumed (the universally-meaningful x-axis for sample efficiency).
- **mean_return / std_return** = mean and std of episode returns from `n_eval_episodes` deterministic-policy rollouts in a fresh env (no exploration noise).

For SAC the eval frequency is `total_steps/20`. For MBPO it's whatever cadence mbrl-lib uses internally, parsed from its CSV. For PILCO the eval happens once per outer iteration.

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

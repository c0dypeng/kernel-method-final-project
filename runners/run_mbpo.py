"""MBPO runner — calls facebookresearch/mbrl-lib's mbpo.train() directly.

Previous versions of this file reimplemented mbrl-lib's training loop from
scratch to hook in periodic eval. That accumulated 9+ bugs. mbrl-lib already
ships periodic eval at epoch boundaries and writes CSVs; we just call its
train function and post-process the output into the unified schema.

Usage:
    python -m runners.run_mbpo --env Pendulum-v1   --steps 100000  --seed 0
    python -m runners.run_mbpo --env HalfCheetah-v4 --steps 1000000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf, DictConfig

# Compatibility shim: mbrl-lib was written against classic `gym`, our project
# uses `gymnasium`. Map `gym.spaces.Box` to gymnasium's equivalent so mbrl's
# `_target_: gym.spaces.Box` lookups resolve.
try:
    import gym as _classic_gym  # noqa: F401
except ImportError:
    import gymnasium.spaces as _spaces
    _shim = types.ModuleType("gym")
    _shim.env = types.ModuleType("gym.env")
    _shim.env.Box = _spaces.Box
    _shim.spaces = types.ModuleType("gym.spaces")
    _shim.spaces.Box = _spaces.Box
    sys.modules.setdefault("gym", _shim)
    sys.modules.setdefault("gym.env", _shim.env)
    sys.modules.setdefault("gym.spaces", _shim.spaces)

import mbrl.algorithms.mbpo as mbpo


def no_termination_fn(act, next_obs):
    """Termination predicate for Pendulum and HalfCheetah (both timeout-only).
    Returns shape (N, 1) BoolTensor per mbrl's TermFnType contract.
    """
    done = torch.zeros(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)
    return done[:, None]


def build_cfg(env_id: str, total_steps: int, seed: int, device: str, work_dir: str) -> DictConfig:
    """Construct an mbrl-lib DictConfig matching the library's mbpo.yaml +
    overrides/mbpo_<env>.yaml structure, with values inlined (no hydra
    interpolation strings — those don't resolve outside hydra runtime)."""

    if env_id == "Pendulum-v1":
        sac_hidden = 256
        # Pendulum: small 3D state, length-1 rollouts are fine.
        rollout_schedule = [20, 150, 1, 1]
    elif env_id == "HalfCheetah-v4":
        sac_hidden = 512
        # MBPO paper's HalfCheetah config (Janner et al. 2019, App. Table 3):
        # rollout length ramps 1 -> 15 over epochs 20-100.
        rollout_schedule = [20, 100, 1, 15]
    else:
        raise ValueError(f"unsupported env: {env_id}")

    overrides = {
        "env": f"gym___{env_id}",
        "term_fn": "no_termination",
        "num_steps": total_steps,
        "epoch_length": 1000,
        "num_elites": 5,
        "patience": 5,
        "model_lr": 1e-3,
        "model_wd": 1e-5,
        "model_batch_size": 256,
        "validation_ratio": 0.2,
        "freq_train_model": 250,
        "effective_model_rollouts_per_step": 400,
        "rollout_schedule": rollout_schedule,
        "num_sac_updates_per_step": 10,
        "sac_updates_every_steps": 1,
        "num_epochs_to_retain_sac_buffer": 1,
        "sac_gamma": 0.99,
        "sac_tau": 0.005,
        "sac_alpha": 0.2,
        "sac_policy": "Gaussian",
        "sac_target_update_interval": 1,
        "sac_automatic_entropy_tuning": True,
        "sac_target_entropy": -1,
        "sac_hidden_size": sac_hidden,
        "sac_lr": 3e-4,
        "sac_batch_size": 256,
    }

    dynamics_model = {
        "_target_": "mbrl.models.GaussianMLP",
        "device": device,
        "num_layers": 4,
        "in_size": "???",
        "out_size": "???",
        "ensemble_size": 7,
        "hid_size": 200,
        "deterministic": False,
        "propagation_method": "random_model",
        "learn_logvar_bounds": False,
    }

    algorithm = {
        "name": "mbpo",
        "normalize": True,
        "normalize_double_precision": True,
        "target_is_delta": True,
        "learned_rewards": True,
        "freq_train_model": overrides["freq_train_model"],
        "real_data_ratio": 0.0,
        "sac_samples_action": True,
        "initial_exploration_steps": 5000,
        "random_initial_explore": False,
        "num_eval_episodes": 5,
        "agent": {
            "_target_": "mbrl.third_party.pytorch_sac_pranz24.sac.SAC",
            "num_inputs": "???",
            "action_space": {
                "_target_": "gym.spaces.Box",
                "low": "???",
                "high": "???",
                "shape": "???",
            },
            "args": {
                "gamma": overrides["sac_gamma"],
                "tau": overrides["sac_tau"],
                "alpha": overrides["sac_alpha"],
                "policy": overrides["sac_policy"],
                "target_update_interval": overrides["sac_target_update_interval"],
                "automatic_entropy_tuning": overrides["sac_automatic_entropy_tuning"],
                "target_entropy": overrides["sac_target_entropy"],
                "hidden_size": overrides["sac_hidden_size"],
                "device": device,
                "lr": overrides["sac_lr"],
            },
        },
    }

    return OmegaConf.create({
        "seed": seed,
        "device": device,
        "log_frequency_agent": 1000,
        "save_video": False,
        "experiment": "default",
        "debug_mode": False,
        "root_dir": work_dir,
        "algorithm": algorithm,
        "dynamics_model": dynamics_model,
        "overrides": overrides,
    })


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def mbrl_csv_to_unified(work_dir: Path, dst_csv: Path, start_time: float) -> None:
    """mbrl-lib writes results.csv with columns like:
       env_step, episode_reward, ...
    Convert into our schema (env_steps, mean_return, std_return, wallclock_s).
    """
    candidates = sorted(work_dir.glob("**/results.csv"))
    if not candidates:
        candidates = sorted(work_dir.glob("**/*.csv"))
    if not candidates:
        raise RuntimeError(f"mbrl produced no CSV in {work_dir}")
    src = candidates[0]
    print(f"[mbpo] mbrl CSV source: {src}")

    rows: dict[int, list[float]] = {}
    with open(src) as f:
        reader = csv.DictReader(f)
        for r in reader:
            step_key = next((k for k in ("env_step", "step") if k in r), None)
            ret_key = next((k for k in ("episode_reward", "episode_return", "eval_return") if k in r), None)
            if step_key is None or ret_key is None:
                continue
            try:
                step = int(float(r[step_key]))
                ret = float(r[ret_key])
            except (ValueError, TypeError):
                continue
            rows.setdefault(step, []).append(ret)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "mean_return", "std_return", "wallclock_s"])
        n = max(len(rows), 1)
        for i, step in enumerate(sorted(rows.keys())):
            arr = np.asarray(rows[step])
            w.writerow([
                step,
                float(arr.mean()),
                float(arr.std()),
                float((time.time() - start_time) * (i + 1) / n),
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["Pendulum-v1", "HalfCheetah-v4"])
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cuda")
    # --eval-freq / --n-eval-episodes are accepted for CLI compatibility with
    # the other runners but ignored: mbrl-lib controls eval cadence internally
    # via epoch_length (1000 env steps) and num_eval_episodes (5).
    p.add_argument("--eval-freq", type=int, default=None, help="(ignored)")
    p.add_argument("--n-eval-episodes", type=int, default=5, help="(ignored)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[mbpo] env={args.env} steps={args.steps} seed={args.seed} device={args.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    work_dir = Path(tempfile.mkdtemp(prefix=f"mbpo_{args.env}_seed{args.seed}_"))
    print(f"[mbpo] mbrl-lib work_dir={work_dir}")

    cfg = build_cfg(args.env, args.steps, args.seed, args.device, str(work_dir))

    env = make_env(args.env, args.seed)
    test_env = make_env(args.env, 10_000 + args.seed)

    start_time = time.time()
    mbpo.train(
        env=env,
        test_env=test_env,
        termination_fn=no_termination_fn,
        cfg=cfg,
        silent=False,
        work_dir=str(work_dir),
    )

    dst_csv = Path(args.results_dir) / f"mbpo__{args.env}__seed{args.seed}.csv"
    mbrl_csv_to_unified(work_dir, dst_csv, start_time)

    # Keep raw mbrl outputs alongside our unified CSV.
    raw_dst = Path(args.results_dir) / "raw" / f"mbpo__{args.env}__seed{args.seed}"
    raw_dst.mkdir(parents=True, exist_ok=True)
    for csv_file in work_dir.glob("**/*.csv"):
        shutil.copy(csv_file, raw_dst / csv_file.name)

    env.close()
    test_env.close()
    print(f"[mbpo] done -> {dst_csv}")


if __name__ == "__main__":
    main()

"""MBPO runner via facebookresearch/mbrl-lib.

Rather than calling mbrl's CLI, we build the DictConfig programmatically so we
can interleave evaluation with training. mbrl-lib's `mbpo.train` runs end-to-end
without a hook for external eval, so we instead read its own progress.csv at the
end and resample onto our uniform schema.

Usage:
    python -m runners.run_mbpo --env Pendulum-v1   --steps 50000  --seed 0
    python -m runners.run_mbpo --env HalfCheetah-v4 --steps 200000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf, DictConfig

import mbrl.util.common as common_util
import mbrl.util.mujoco
import mbrl.algorithms.mbpo as mbpo

from runners.common import EvalLogger


# ---------------------------------------------------------------------------
# Termination functions — mbrl needs to know when an episode ends from state.
# Pendulum never terminates (timeout-only). HalfCheetah likewise (timeout-only).
# Both fit the "no_termination" pattern.
# ---------------------------------------------------------------------------
def no_termination_fn(act, next_obs):
    """Always-False termination — matches mbrl.util.mujoco.no_termination."""
    return torch.zeros(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)


# ---------------------------------------------------------------------------
# Per-env mbrl config. Two flavors: small for Pendulum, full for HalfCheetah.
# ---------------------------------------------------------------------------
def build_cfg(env_id: str, total_steps: int, seed: int, device: str) -> DictConfig:
    # Common skeleton (mirrors mbrl-lib's mbpo.yaml + a halfcheetah-shaped override).
    common = {
        "seed": seed,
        "device": device,
        "log_frequency_agent": 1000,
        "save_video": False,
        "experiment": "default",
        "debug_mode": False,
        "root_dir": "${hydra:run.dir}",
    }

    # Dynamics model — Gaussian MLP ensemble, exactly mbrl-lib's default for MBPO.
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
        "activation_fn_cfg": {"_target_": "torch.nn.SiLU"},
    }

    if env_id == "Pendulum-v1":
        overrides = {
            "env": "gym___Pendulum-v1",
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
            "rollout_schedule": [20, 150, 1, 1],
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
            "sac_hidden_size": 256,  # smaller net for the simpler env
            "sac_lr": 3e-4,
            "sac_batch_size": 256,
        }
    elif env_id == "HalfCheetah-v4":
        overrides = {
            "env": "gym___HalfCheetah-v4",
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
            "rollout_schedule": [20, 150, 1, 1],
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
            "sac_hidden_size": 512,
            "sac_lr": 3e-4,
            "sac_batch_size": 256,
        }
    else:
        raise ValueError(f"Unsupported env: {env_id}")

    algorithm = {
        "name": "mbpo",
        "normalize": True,
        "normalize_double_precision": True,
        "target_is_delta": True,
        "learned_rewards": True,
        "freq_train_model": "${overrides.freq_train_model}",
        "real_data_ratio": 0.0,
        "sac_samples_action": True,
        "initial_exploration_steps": 5000,
        "random_initial_explore": False,
        "num_eval_episodes": 1,
        "agent": {
            "_target_": "mbrl.third_party.pytorch_sac_pranz24.sac.SAC",
            "num_inputs": "???",
            "action_space": {
                "_target_": "gym.env.Box",
                "low": "???",
                "high": "???",
                "shape": "???",
            },
            "args": {
                "gamma": "${overrides.sac_gamma}",
                "tau": "${overrides.sac_tau}",
                "alpha": "${overrides.sac_alpha}",
                "policy": "${overrides.sac_policy}",
                "target_update_interval": "${overrides.sac_target_update_interval}",
                "automatic_entropy_tuning": "${overrides.sac_automatic_entropy_tuning}",
                "target_entropy": "${overrides.sac_target_entropy}",
                "hidden_size": "${overrides.sac_hidden_size}",
                "device": "${device}",
                "lr": "${overrides.sac_lr}",
            },
        },
    }

    cfg = OmegaConf.create({
        **common,
        "algorithm": algorithm,
        "dynamics_model": dynamics_model,
        "overrides": overrides,
    })
    return cfg


def make_env(env_id: str, seed: int):
    """Build the env mbrl-lib expects. It wants `gym` (the classic API) but
    works with gymnasium under shim. We hand it a gymnasium env and let mbrl's
    wrapper handle the rest."""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def parse_mbpo_csv_into_unified(mbpo_log: Path, dst_csv: Path, start_time: float) -> None:
    """mbrl-lib writes `results.csv` with columns episode/step/env_step/episode_reward.
    Convert to our unified schema (env_steps, mean_return, std_return, wallclock_s).
    """
    if not mbpo_log.exists():
        raise RuntimeError(f"expected mbrl log at {mbpo_log} not found")

    with open(mbpo_log) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # mbrl logs one row per eval episode. Group by env_step bin and average.
    bins: dict[int, list[float]] = {}
    for r in rows:
        # Common column names: env_step, episode_reward (variants exist across mbrl versions).
        step_key = next((k for k in ("env_step", "step") if k in r), None)
        ret_key = next((k for k in ("episode_reward", "episode_return", "eval_return") if k in r), None)
        if step_key is None or ret_key is None:
            continue
        try:
            step = int(float(r[step_key]))
            ret = float(r[ret_key])
        except (ValueError, TypeError):
            continue
        bins.setdefault(step, []).append(ret)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "mean_return", "std_return", "wallclock_s"])
        for step in sorted(bins.keys()):
            arr = np.asarray(bins[step])
            wallclock = time.time() - start_time
            w.writerow([step, float(arr.mean()), float(arr.std()), wallclock])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["Pendulum-v1", "HalfCheetah-v4"])
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[mbpo] env={args.env} steps={args.steps} seed={args.seed} device={args.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = build_cfg(args.env, args.steps, args.seed, args.device)

    env = make_env(args.env, args.seed)
    test_env = make_env(args.env, args.seed + 10_000)

    # mbrl writes its logs into work_dir. Use a per-run temp dir, copy out at the end.
    work_dir = Path(tempfile.mkdtemp(prefix=f"mbpo_{args.env}_seed{args.seed}_"))
    print(f"[mbpo] work_dir={work_dir}")

    start_time = time.time()
    mbpo.train(
        env=env,
        test_env=test_env,
        termination_fn=no_termination_fn,
        cfg=cfg,
        silent=False,
        work_dir=str(work_dir),
    )

    # Convert mbrl's CSV to our unified schema.
    # mbrl writes `results.csv` containing eval episodes.
    candidates = list(work_dir.glob("**/results.csv"))
    if not candidates:
        # Newer mbrl versions name it differently — fall back to any *.csv.
        candidates = list(work_dir.glob("**/*.csv"))
    if not candidates:
        raise RuntimeError(f"mbrl produced no CSV files in {work_dir}")
    mbrl_csv = candidates[0]
    print(f"[mbpo] mbrl CSV: {mbrl_csv}")

    dst_csv = Path(args.results_dir) / f"mbpo__{args.env}__seed{args.seed}.csv"
    parse_mbpo_csv_into_unified(mbrl_csv, dst_csv, start_time)

    # Also keep the raw mbrl outputs for debugging.
    raw_dst = Path(args.results_dir) / "raw" / f"mbpo__{args.env}__seed{args.seed}"
    raw_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(mbrl_csv, raw_dst / mbrl_csv.name)

    print(f"[mbpo] done -> {dst_csv}")


if __name__ == "__main__":
    main()

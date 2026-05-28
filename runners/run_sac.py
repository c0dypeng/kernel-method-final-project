"""SAC runner — thin wrapper around stable-baselines3.

Uses SB3's built-in EvalCallback for periodic deterministic-policy eval.
Post-processes SB3's evaluations.npz into our unified CSV schema.

Usage:
    python -m runners.run_sac --env Pendulum-v1   --steps 100000  --seed 0
    python -m runners.run_sac --env HalfCheetah-v4 --steps 1000000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["Pendulum-v1", "HalfCheetah-v4"])
    p.add_argument("--steps", type=int, required=True, help="total env steps to train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--eval-freq", type=int, default=None,
                   help="evaluate every N env steps (default: max(steps/20, 1))")
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def npz_to_unified_csv(npz_path: Path, csv_path: Path, start_time: float) -> None:
    """Convert SB3's evaluations.npz to our unified CSV schema.

    SB3 writes per-eval rows with:
      timesteps      shape (N,)            env_steps at each eval
      results        shape (N, n_episodes) per-eval per-episode returns
      ep_lengths     shape (N, n_episodes) per-eval per-episode lengths
    """
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    results = data["results"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env_steps", "mean_return", "std_return", "wallclock_s"])
        for i, ts in enumerate(timesteps):
            row_returns = results[i]
            w.writerow([
                int(ts),
                float(row_returns.mean()),
                float(row_returns.std()),
                # SB3 doesn't log wallclock per eval; back-fill linearly from
                # start. Slight inaccuracy but good enough for plotting.
                float((time.time() - start_time) * (i + 1) / len(timesteps)),
            ])


def main():
    args = parse_args()
    eval_freq = args.eval_freq or max(args.steps // 20, 1)

    csv_path = Path(args.results_dir) / f"sac__{args.env}__seed{args.seed}.csv"
    print(f"[sac] env={args.env} steps={args.steps} seed={args.seed} device={args.device}")
    print(f"[sac] eval_freq={eval_freq} n_eval_episodes={args.n_eval_episodes} -> {csv_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env = gym.make(args.env)
    train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)

    # SB3's EvalCallback runs on its own env, deterministic=True, and writes
    # an .npz file we can post-process into the unified schema.
    eval_env = gym.make(args.env)
    eval_env.reset(seed=10_000 + args.seed)
    eval_env.action_space.seed(10_000 + args.seed)

    eval_log_dir = Path(args.results_dir) / "raw" / f"sac__{args.env}__seed{args.seed}"
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=None,         # don't save checkpoints
        log_path=str(eval_log_dir),        # writes evaluations.npz here
        eval_freq=eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # SB3 SAC defaults match the SAC paper (Haarnoja et al. 2018) for the
    # core algorithmic knobs: lr=3e-4, batch=256, tau=0.005, gamma=0.99,
    # ent_coef="auto", target_entropy="auto", net_arch=[256, 256].
    model = SAC("MlpPolicy", train_env, seed=args.seed, device=args.device, verbose=0)

    start_time = time.time()
    model.learn(total_timesteps=args.steps, callback=eval_cb, log_interval=10_000)

    npz_path = eval_log_dir / "evaluations.npz"
    if not npz_path.exists():
        raise RuntimeError(f"SB3 didn't produce {npz_path}")
    npz_to_unified_csv(npz_path, csv_path, start_time)

    print(f"[sac] done -> {csv_path}")


if __name__ == "__main__":
    main()

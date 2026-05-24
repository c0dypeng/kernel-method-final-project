"""SAC runner via stable-baselines3.

Usage:
    python -m runners.run_sac --env Pendulum-v1   --steps 50000  --seed 0
    python -m runners.run_sac --env HalfCheetah-v4 --steps 200000 --seed 0

Writes a CSV to results/sac__<env>__seed<seed>.csv with the schema in common.CSV_COLUMNS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from runners.common import EvalLogger, evaluate_policy, env_factory_for


class PeriodicEvalCallback(BaseCallback):
    """Evaluate the current policy every `eval_freq` env steps and log to CSV."""

    def __init__(self, env_id: str, logger: EvalLogger, eval_freq: int,
                 n_eval_episodes: int, eval_seed: int, verbose: int = 0):
        super().__init__(verbose)
        self.env_id = env_id
        self.logger = logger
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self._next_eval_at = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval_at:
            self._next_eval_at = self.num_timesteps + self.eval_freq
            self._run_eval()
        return True

    def _on_training_end(self) -> None:
        # Final eval at the end of training (if not already at a multiple).
        self._run_eval()

    def _run_eval(self) -> None:
        def act_fn(obs):
            # deterministic=True returns the mean of the tanh-Gaussian, removing exploration noise.
            action, _ = self.model.predict(obs, deterministic=True)
            return action

        mean_ret, std_ret = evaluate_policy(
            env_factory=env_factory_for(self.env_id),
            act_fn=act_fn,
            n_eval_episodes=self.n_eval_episodes,
            seed=self.eval_seed,
        )
        self.logger.log(int(self.num_timesteps), mean_ret, std_ret)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["Pendulum-v1", "HalfCheetah-v4"])
    p.add_argument("--steps", type=int, required=True, help="total env steps to train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--eval-freq", type=int, default=None,
                   help="evaluate every N env steps (default: steps/20)")
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()
    eval_freq = args.eval_freq or max(args.steps // 20, 1)

    csv_path = Path(args.results_dir) / f"sac__{args.env}__seed{args.seed}.csv"
    print(f"[sac] env={args.env} steps={args.steps} seed={args.seed}")
    print(f"[sac] device={args.device}  eval_freq={eval_freq}  -> {csv_path}")

    logger = EvalLogger.open(csv_path)

    # Determinism: SB3 seeds the env, action space, and PyTorch RNG.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_env = gym.make(args.env)
    train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)

    # Hyperparams: SB3 defaults are tuned for MuJoCo and Pendulum already.
    # Using gradient_steps=1 keeps "update per env step" at SAC's standard ratio.
    model = SAC(
        "MlpPolicy",
        train_env,
        seed=args.seed,
        device=args.device,
        verbose=0,
    )

    cb = PeriodicEvalCallback(
        env_id=args.env,
        logger=logger,
        eval_freq=eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed=10_000 + args.seed,
    )
    model.learn(total_timesteps=args.steps, callback=cb, log_interval=10_000)

    print(f"[sac] done -> {csv_path}")


if __name__ == "__main__":
    main()

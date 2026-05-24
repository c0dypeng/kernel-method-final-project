"""Shared utilities for all three RL runners.

Common evaluation protocol so PILCO / MBPO / SAC results are directly comparable:
- evaluate the current policy on `n_eval_episodes` deterministic-policy rollouts
- log (env_steps, mean_return, std_return) to a CSV with a fixed schema

PILCO doesn't naturally count env steps the same way (it learns from a small
fixed buffer), so its runner records env_steps = cumulative real transitions
collected up to that point.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


CSV_COLUMNS = ["env_steps", "mean_return", "std_return", "wallclock_s"]


@dataclass
class EvalLogger:
    csv_path: Path
    start_time: float

    @classmethod
    def open(cls, csv_path: str | Path) -> "EvalLogger":
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Write header
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        return cls(csv_path=csv_path, start_time=time.time())

    def log(self, env_steps: int, mean_return: float, std_return: float) -> None:
        wallclock = time.time() - self.start_time
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([env_steps, mean_return, std_return, wallclock])
        print(
            f"  [eval] env_steps={env_steps:>8d}  "
            f"return={mean_return:>8.2f} ± {std_return:>6.2f}  "
            f"({wallclock:.1f}s)",
            flush=True,
        )


def evaluate_policy(
    env_factory: Callable,
    act_fn: Callable[[np.ndarray], np.ndarray],
    n_eval_episodes: int = 5,
    seed: int = 0,
    max_episode_steps: Optional[int] = None,
) -> tuple[float, float]:
    """Run `n_eval_episodes` episodes with the given action function.

    Args:
        env_factory: zero-arg callable returning a fresh gymnasium env.
        act_fn: maps observation (np.ndarray, shape (obs_dim,)) -> action (np.ndarray).
                Must be deterministic; SAC/MBPO callers should pass the mean action.
        n_eval_episodes: number of episodes to average over.
        seed: starting seed; episode i uses seed+i.
        max_episode_steps: cap per episode (None = use env default).

    Returns:
        (mean_return, std_return) across the episodes.
    """
    returns = []
    for ep in range(n_eval_episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        step = 0
        while True:
            action = act_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            # Clip to action space — defensive against rare PILCO overruns
            low, high = env.action_space.low, env.action_space.high
            action = np.clip(action, low, high)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            step += 1
            if terminated or truncated:
                break
            if max_episode_steps is not None and step >= max_episode_steps:
                break
        returns.append(ep_return)
        env.close()
    returns = np.asarray(returns)
    return float(returns.mean()), float(returns.std())


def env_factory_for(env_id: str) -> Callable:
    """Return a zero-arg callable that constructs a fresh gymnasium env by id."""
    import gymnasium as gym

    def _make():
        return gym.make(env_id)

    return _make


def env_shapes(env_id: str) -> tuple[int, int]:
    """Return (obs_dim, act_dim) for the given env."""
    import gymnasium as gym

    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()
    return obs_dim, act_dim

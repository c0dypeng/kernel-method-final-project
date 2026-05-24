"""MBPO runner via facebookresearch/mbrl-lib.

Drives MBPO in chunks and runs the same `evaluate_policy` between chunks that
the SAC and PILCO runners use, so all three produce identical CSV schemas.

This is a more invasive integration than calling `mbpo.train` directly — we
have to reach into mbrl-lib's internals to run training for N steps and then
hand control back to us for an eval. mbrl-lib doesn't expose a public "step N
times" API, so we replicate the relevant portion of `train()` here. The
algorithm is unchanged; only the outer loop differs.

Usage:
    python -m runners.run_mbpo --env Pendulum-v1   --steps 50000  --seed 0
    python -m runners.run_mbpo --env HalfCheetah-v4 --steps 200000 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf, DictConfig

# mbrl's SAC agent config has _target_: gym.env.Box (typo in upstream, but harmless if
# `gym` is importable). Provide a compatibility shim so a pure-gymnasium env works.
try:
    import gym as _classic_gym  # noqa: F401
except ImportError:
    # Build a minimal `gym.env.Box` -> `gymnasium.spaces.Box` shim and inject it.
    import types
    import gymnasium.spaces as _spaces
    _shim = types.ModuleType("gym")
    _shim.env = types.ModuleType("gym.env")
    _shim.env.Box = _spaces.Box
    _shim.spaces = types.ModuleType("gym.spaces")
    _shim.spaces.Box = _spaces.Box
    sys.modules.setdefault("gym", _shim)
    sys.modules.setdefault("gym.env", _shim.env)
    sys.modules.setdefault("gym.spaces", _shim.spaces)

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.third_party.pytorch_sac_pranz24.utils as sac_utils
import mbrl.util.common as common_util
import mbrl.util.replay_buffer
from mbrl.algorithms.mbpo import (
    MBPO_LOG_FORMAT,
    rollout_model_and_populate_sac_buffer,
    maybe_replace_sac_buffer,
)

from runners.common import EvalLogger, evaluate_policy, env_factory_for


# ---------------------------------------------------------------------------
# Termination function.
# mbrl-lib expects (act, next_obs) -> torch.BoolTensor of shape (N, 1).
# Both Pendulum and HalfCheetah are "no termination, timeout-only" envs.
# ---------------------------------------------------------------------------
def no_termination_fn(act, next_obs):
    done = torch.zeros(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)
    return done[:, None]


# ---------------------------------------------------------------------------
# Per-env mbrl config. Skeleton mirrors mbrl-lib's mbpo.yaml + a HalfCheetah-shape override.
# ---------------------------------------------------------------------------
def build_cfg(env_id: str, total_steps: int, seed: int, device: str, work_dir: str) -> DictConfig:
    common = {
        "seed": seed,
        "device": device,
        "log_frequency_agent": 1000,
        "save_video": False,
        "experiment": "default",
        "debug_mode": False,
        "root_dir": work_dir,
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
        # NB: do NOT pass `activation_fn_cfg` here. mbrl-lib's GaussianMLP
        # passes the cfg through hydra.utils.instantiate which materializes
        # the activation module into the cfg tree — but OmegaConf can't store
        # live nn.Module instances and raises. Letting it default (ReLU)
        # avoids this entirely. The original SiLU choice matters very little
        # for these benchmarks.
    }

    if env_id == "Pendulum-v1":
        sac_hidden = 256
    elif env_id == "HalfCheetah-v4":
        sac_hidden = 512
    else:
        raise ValueError(f"Unsupported env: {env_id}")

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
        "sac_hidden_size": sac_hidden,
        "sac_lr": 3e-4,
        "sac_batch_size": 256,
    }

    # NB: mbrl-lib's hydra YAML uses ${overrides.sac_xxx} interpolation
    # strings, but those only resolve when hydra builds the whole config
    # tree at once. When we construct the OmegaConf programmatically and
    # mbrl reads cfg.algorithm.agent.args.gamma via hydra.utils.instantiate,
    # the interpolation triggers and OmegaConf 2.1.2 raises
    # InterpolationKeyError because the parent context isn't set the way
    # hydra would set it. Inline the values from the `overrides` dict
    # directly — same effect, no resolution path required.
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
        "num_eval_episodes": 1,
        # NB: no _target_ / num_inputs / action_space here. We construct the
        # SAC agent directly in train_mbpo_with_eval() to bypass hydra's
        # OmegaConf -> ListConfig round-trip which corrupts the Box space.
        # Only `args` is carried in the config so the SAC hyperparams are
        # still accessible via cfg.algorithm.agent.args.<name>.
        "agent": {
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
        **common,
        "algorithm": algorithm,
        "dynamics_model": dynamics_model,
        "overrides": overrides,
    })


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Custom MBPO train loop with explicit eval hooks.
# This replicates the inner loop of mbrl.algorithms.mbpo.train but yields control
# to us after every `eval_every` env steps so we can run our unified eval.
# ---------------------------------------------------------------------------
def train_mbpo_with_eval(env, cfg: DictConfig, eval_callback,
                         eval_every: int, total_steps: int):
    """Run MBPO; call `eval_callback(env_step)` every `eval_every` env steps."""
    rng = np.random.default_rng(cfg.seed)
    torch_rng = torch.Generator(device=cfg.device)
    torch_rng.manual_seed(cfg.seed)

    # ---- Build dynamics model + agent (mirrors mbpo.train setup) ----
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Replay buffers
    replay_buffer = mbrl.util.replay_buffer.ReplayBuffer(
        capacity=cfg.overrides.num_steps + cfg.algorithm.initial_exploration_steps,
        obs_shape=obs_shape,
        action_shape=act_shape,
        rng=rng,
    )
    common_util.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )

    # SAC agent — bypass hydra.utils.instantiate entirely.
    #
    # mbrl-lib's stock approach is `hydra.utils.instantiate(cfg.algorithm.agent)`
    # which then recursively instantiates the nested action_space sub-config
    # as `gym.spaces.Box(low=..., high=..., shape=...)`. But OmegaConf turns
    # plain Python lists into ListConfig objects during round-trip, and
    # `gym.spaces.Box.__init__` does `low < high` element-wise comparisons
    # that crash with TypeError on ListConfig values.
    #
    # Easier: import the SAC class directly and construct it with the real
    # env.action_space (which is already a valid Space). The `args` parameter
    # just needs attribute access for gamma/tau/etc., which OmegaConf
    # provides natively.
    from mbrl.third_party.pytorch_sac_pranz24.sac import SAC as _SACAgent
    sac_args = cfg.algorithm.agent.args  # OmegaConf DictConfig, attribute access works
    agent = _SACAgent(
        num_inputs=obs_shape[0],
        action_space=env.action_space,
        args=sac_args,
    )

    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, no_termination_fn,
        generator=torch_rng,
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
    )

    # Rollout-length schedule
    rollout_length = int(cfg.overrides.rollout_schedule[2])

    # Stats for SAC training
    sac_buffer = None
    env_steps = 0
    obs, _ = env.reset(seed=cfg.seed + 1)
    episode_reward = 0.0
    episode_step = 0
    next_eval_at = 0

    print(f"[mbpo] starting main loop; total_steps={total_steps}, eval_every={eval_every}")

    while env_steps < total_steps:
        # ---- Train dynamics model every freq_train_model env steps ----
        if env_steps % cfg.overrides.freq_train_model == 0:
            common_util.train_model_and_save_model_and_data(
                dynamics_model, model_trainer, cfg.overrides, replay_buffer,
                work_dir=None,
            )

            # Refresh rollout length per schedule
            epoch = env_steps // cfg.overrides.epoch_length
            min_ep, max_ep, min_len, max_len = cfg.overrides.rollout_schedule
            if epoch <= min_ep:
                rollout_length = int(min_len)
            elif epoch >= max_ep:
                rollout_length = int(max_len)
            else:
                rollout_length = int(min_len + (epoch - min_ep) / (max_ep - min_ep) * (max_len - min_len))

            new_sac_buffer_capacity = int(
                cfg.overrides.effective_model_rollouts_per_step *
                cfg.overrides.epoch_length *
                cfg.overrides.num_epochs_to_retain_sac_buffer *
                rollout_length
            )
            # mbrl 0.2.0 signature: (sac_buffer, obs_shape, act_shape, new_capacity, seed)
            sac_buffer = maybe_replace_sac_buffer(
                sac_buffer,
                obs_shape,
                act_shape,
                new_sac_buffer_capacity,
                cfg.seed,
            )

        # ---- Imagined rollouts -> SAC buffer ----
        rollout_model_and_populate_sac_buffer(
            model_env, replay_buffer, agent, sac_buffer,
            cfg.algorithm.sac_samples_action, rollout_length,
            cfg.overrides.effective_model_rollouts_per_step,
        )

        # ---- One real-env step under the SAC policy ----
        action = agent.act(obs, sample=cfg.algorithm.sac_samples_action, batched=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # mbrl 0.2.0 ReplayBuffer.add takes (obs, action, next_obs, reward, terminated, truncated)
        replay_buffer.add(obs, action, next_obs, reward, terminated, truncated)
        episode_reward += float(reward)
        episode_step += 1
        env_steps += 1

        # ---- SAC updates ----
        if (env_steps % cfg.overrides.sac_updates_every_steps == 0 and
                len(sac_buffer) >= cfg.overrides.sac_batch_size):
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                agent.update_parameters(
                    sac_buffer, cfg.overrides.sac_batch_size, env_steps,
                )

        if done:
            obs, _ = env.reset()
            episode_reward = 0.0
            episode_step = 0
        else:
            obs = next_obs

        # ---- Eval hook ----
        if env_steps >= next_eval_at:
            next_eval_at = env_steps + eval_every
            eval_callback(env_steps, agent)

    eval_callback(env_steps, agent)
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["Pendulum-v1", "HalfCheetah-v4"])
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--eval-freq", type=int, default=None,
                   help="evaluate every N env steps (default: steps/20)")
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    eval_freq = args.eval_freq or max(args.steps // 20, 1)

    csv_path = Path(args.results_dir) / f"mbpo__{args.env}__seed{args.seed}.csv"
    print(f"[mbpo] env={args.env} steps={args.steps} seed={args.seed} device={args.device}")
    print(f"[mbpo] eval_freq={eval_freq}  -> {csv_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger = EvalLogger.open(csv_path)
    work_dir = tempfile.mkdtemp(prefix=f"mbpo_{args.env}_seed{args.seed}_")
    cfg = build_cfg(args.env, args.steps, args.seed, args.device, work_dir)

    env = make_env(args.env, args.seed)

    def eval_callback(env_step: int, agent):
        """Run the same evaluate_policy() that SAC/PILCO use so the CSVs match."""
        def act_fn(obs):
            # `agent.act(obs, sample=False)` returns the deterministic mean action.
            return agent.act(obs, sample=False, batched=False)

        mean_r, std_r = evaluate_policy(
            env_factory=env_factory_for(args.env),
            act_fn=act_fn,
            n_eval_episodes=args.n_eval_episodes,
            seed=10_000 + args.seed,
        )
        logger.log(env_step, mean_r, std_r)

    try:
        train_mbpo_with_eval(
            env, cfg, eval_callback,
            eval_every=eval_freq,
            total_steps=args.steps,
        )
    finally:
        env.close()

    print(f"[mbpo] done -> {csv_path}")


if __name__ == "__main__":
    main()

"""PILCO runner using the algorithm code from c0dypeng/PILCO-modern.

PILCO learns from a tiny replay buffer (the GP becomes O(N^3) so N stays small).
Eval protocol: between PILCO iterations, evaluate the deterministic policy on
`n_eval_episodes` real-env rollouts and log to the unified CSV.

For HalfCheetah we use the sparse GP variant (SMGPR) because dense GPs explode
at >300 transitions. This is **expected to fail** on cheetah — the original
PILCO authors never claimed it works on 17D+6D systems. We run it anyway to
document the data-efficiency / scalability tradeoff that motivates MBPO.

Usage:
    python -m runners.run_pilco --env Pendulum-v1   --iterations 8  --seed 0
    python -m runners.run_pilco --env HalfCheetah-v4 --iterations 5 --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import gymnasium as gym

from runners.common import EvalLogger, evaluate_policy, env_factory_for


# ---------------------------------------------------------------------------
# Suppress TF chatter so the eval log stays readable.
# ---------------------------------------------------------------------------
tf.get_logger().setLevel("ERROR")
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------------------------------------------------------
# Per-env PILCO hyperparams.
# Pendulum-v1: 3-dim obs (cos/sin/dot), 1-dim continuous torque [-2, 2].
# HalfCheetah-v4: 17-dim obs, 6-dim action [-1, 1]. Expected failure mode.
# ---------------------------------------------------------------------------
ENV_CONFIGS = {
    "Pendulum-v1": dict(
        max_action=2.0,
        horizon=40,
        num_basis_functions=30,
        sparse=False,
        num_induced_points=None,
        initial_random_rollouts=4,
        timesteps_per_rollout=40,
        subsampling=3,
        likelihood_var=0.001,
        # Pendulum-v1 obs is [cos(theta), sin(theta), theta_dot].
        # Upright equilibrium: cos(theta)=1, sin(theta)=0, theta_dot=0.
        # Hanging-down equilibrium: cos(theta)=-1, sin(theta)=0, theta_dot=0.
        # Matches PILCO-modern/examples/pendulum_swing_up.py:43-46 exactly.
        reward_target=np.array([1.0, 0.0, 0.0]),
        reward_W=np.diag([2.0, 2.0, 0.3]),
        # Fixed start-state distribution: pole hanging straight down, at rest.
        # This is what PILCO is "planning from"; using the first random-reset
        # observation instead (random theta in [-pi, pi]) makes PILCO solve a
        # different problem every seed. Matches pendulum_swing_up.py:45.
        m_init_override=np.array([[-1.0, 0.0, 0.0]]),
        S_init_override=np.diag([0.01, 0.05, 0.01]),
    ),
    "HalfCheetah-v4": dict(
        max_action=1.0,
        horizon=15,
        num_basis_functions=40,
        sparse=True,
        num_induced_points=100,
        initial_random_rollouts=3,
        timesteps_per_rollout=25,
        subsampling=1,
        likelihood_var=0.01,
        # HalfCheetah-v4 obs is 17-dim (rootz, rootangle, joints, ...). The
        # task reward in gym is forward_velocity - ctrl_cost, which the
        # saturating ExponentialReward(t, W) cost can't directly encode.
        # We use a proxy: target index 8 (rootx velocity = forward speed)
        # at a large positive value, weight only that dim. This is a known
        # limitation of PILCO on locomotion envs and part of why the
        # comparison documents PILCO failing here.
        reward_target=np.concatenate([np.zeros(8), [5.0], np.zeros(8)]),
        reward_W=np.diag(np.concatenate([np.zeros(8), [1.0], np.zeros(8)])),
        # No fixed m_init for HalfCheetah — it's already roughly deterministic
        # at the standing pose, so using the first-rollout observation is fine.
        m_init_override=None,
        S_init_override=None,
    ),
}


def env_obs_act_dims(env_id: str) -> tuple[int, int]:
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()
    return obs_dim, act_dim


def rollout_real_env(env, pilco_or_none, timesteps: int, subsampling: int,
                    random_policy: bool, max_action: float, action_dim: int,
                    seed: int | None = None):
    """Roll out one episode in the real env, returning (X, Y, cumulative_reward, n_steps).

    X has shape (T, obs_dim + act_dim), Y has shape (T, obs_dim) with Y_t = x_{t+1} - x_t.
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float64)

    X, Y = [], []
    ep_return = 0.0
    n_steps = 0
    for _ in range(timesteps):
        if random_policy or pilco_or_none is None:
            action = env.action_space.sample()
        else:
            action = pilco_or_none.compute_action(obs[None, :])[0].numpy()
            action = np.clip(action, -max_action, max_action)
        action = np.asarray(action, dtype=np.float64).reshape(action_dim)

        next_obs = obs
        for _sub in range(subsampling):
            step_out = env.step(action.astype(np.float32))
            if len(step_out) == 5:
                next_obs, r, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_obs, r, done, _ = step_out
            next_obs = np.asarray(next_obs, dtype=np.float64)
            ep_return += float(r)
            n_steps += 1
            if done:
                break

        X.append(np.hstack([obs, action]))
        Y.append(next_obs - obs)
        obs = next_obs
        if done:
            break
    return np.asarray(X), np.asarray(Y), ep_return, n_steps


def build_pilco(X, Y, cfg, state_dim, control_dim, m_init, S_init):
    from pilco.models import PILCO
    from pilco.controllers import RbfController
    from pilco.rewards import ExponentialReward
    from gpflow import set_trainable

    controller = RbfController(
        state_dim=state_dim,
        control_dim=control_dim,
        num_basis_functions=cfg["num_basis_functions"],
        max_action=cfg["max_action"],
    )
    # ExponentialReward with explicit target `t` and weighting `W` per env —
    # matches the PILCO-modern mountain_car.py pattern. Defaults (t=0, W=I)
    # would tell PILCO the goal is "drive every state to zero," which is not
    # what either of our envs actually wants.
    reward = ExponentialReward(
        state_dim=state_dim,
        t=cfg["reward_target"].reshape(1, state_dim),
        W=cfg["reward_W"],
    )

    pilco = PILCO(
        (X, Y),
        num_induced_points=cfg["num_induced_points"] if cfg["sparse"] else None,
        controller=controller,
        horizon=cfg["horizon"],
        reward=reward,
        m_init=m_init,
        S_init=S_init,
    )

    # Numerical-stability trick from the PILCO examples + the paper (§5.3.2).
    for model in pilco.mgpr.models:
        model.likelihood.variance.assign(cfg["likelihood_var"])
        set_trainable(model.likelihood.variance, False)
    return pilco


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=["Pendulum-v1", "HalfCheetah-v4"])
    p.add_argument("--iterations", type=int, default=8, help="PILCO outer iterations")
    p.add_argument("--n-eval-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = ENV_CONFIGS[args.env]
    print(f"[pilco] env={args.env} iterations={args.iterations} seed={args.seed} sparse={cfg['sparse']}")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    csv_path = Path(args.results_dir) / f"pilco__{args.env}__seed{args.seed}.csv"
    logger = EvalLogger.open(csv_path)

    obs_dim, act_dim = env_obs_act_dims(args.env)
    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    # --- Initial random rollouts ---
    print(f"[pilco] collecting {cfg['initial_random_rollouts']} random rollouts "
          f"({cfg['timesteps_per_rollout']} steps each)")
    X, Y = None, None
    total_env_steps = 0
    first_obs = None
    for j in range(cfg["initial_random_rollouts"]):
        Xi, Yi, _, n = rollout_real_env(
            env, None, cfg["timesteps_per_rollout"], cfg["subsampling"],
            random_policy=True, max_action=cfg["max_action"], action_dim=act_dim,
            seed=args.seed + j,
        )
        total_env_steps += n
        if first_obs is None:
            # First observation of the first rollout (state portion of X[0]).
            first_obs = Xi[0, :obs_dim].copy()
        X = Xi if X is None else np.vstack([X, Xi])
        Y = Yi if Y is None else np.vstack([Y, Yi])

    # PILCO start-state distribution.
    # If the env config provides explicit overrides (Pendulum-v1 does — see
    # pendulum_swing_up.py:45-46), use those. Otherwise fall back to the
    # first-observation-of-first-rollout pattern from mountain_car.py:31,
    # which works for near-deterministic-reset envs like HalfCheetah.
    if cfg.get("m_init_override") is not None:
        m_init = cfg["m_init_override"].astype(np.float64)
        S_init = cfg["S_init_override"].astype(np.float64)
    else:
        m_init = first_obs.reshape(1, obs_dim)
        S_init = 0.1 * np.eye(obs_dim)

    # --- Build PILCO and evaluate the *random* policy as a baseline ---
    pilco = build_pilco(X, Y, cfg, state_dim=obs_dim, control_dim=act_dim,
                        m_init=m_init, S_init=S_init)

    def act_fn(obs):
        return pilco.compute_action(np.asarray(obs, dtype=np.float64)[None, :])[0].numpy()

    mean_r, std_r = evaluate_policy(
        env_factory_for(args.env), act_fn,
        n_eval_episodes=args.n_eval_episodes, seed=20_000 + args.seed,
    )
    logger.log(total_env_steps, mean_r, std_r)

    # --- PILCO main loop ---
    for it in range(args.iterations):
        print(f"\n[pilco] **** ITERATION {it+1}/{args.iterations} **** "
              f"(data: {X.shape[0]} transitions, env_steps={total_env_steps})")

        t0 = time.time()
        try:
            # restarts=2 matches nrontsis/PILCO's pendulum_swing_up.py defaults
            # and the paper's emphasis on multi-restart policy optimization to
            # escape local optima. (Note: PILCO.optimize_models silently
            # ignores maxiter — upstream bug — but restarts is honored.)
            pilco.optimize_models(maxiter=50, restarts=2)
        except Exception as e:
            print(f"[pilco] optimize_models failed: {type(e).__name__}: {e}")
        try:
            pilco.optimize_policy(maxiter=50, restarts=2)
        except Exception as e:
            print(f"[pilco] optimize_policy failed: {type(e).__name__}: {e}")
        print(f"[pilco] optimization wall time: {time.time()-t0:.1f}s")

        # Real-env rollout under the learned policy — appended to the dataset.
        Xn, Yn, ep_return, n = rollout_real_env(
            env, pilco, cfg["timesteps_per_rollout"], cfg["subsampling"],
            random_policy=False, max_action=cfg["max_action"], action_dim=act_dim,
            seed=args.seed + 1000 + it,
        )
        total_env_steps += n
        X = np.vstack([X, Xn])
        Y = np.vstack([Y, Yn])
        try:
            pilco.mgpr.set_data((X, Y))
        except Exception as e:
            print(f"[pilco] set_data failed: {type(e).__name__}: {e}")

        # Evaluate.
        mean_r, std_r = evaluate_policy(
            env_factory_for(args.env), act_fn,
            n_eval_episodes=args.n_eval_episodes, seed=20_000 + args.seed,
        )
        logger.log(total_env_steps, mean_r, std_r)

    env.close()
    print(f"\n[pilco] done -> {csv_path}")


if __name__ == "__main__":
    main()

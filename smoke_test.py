"""Smoke test for the comparison-project container.

Runs in <2 minutes. Verifies:
  1. All three stacks import (torch+SB3, mbrl, TF+GPflow).
  2. GPU is visible to both PyTorch and TensorFlow.
  3. Pendulum-v1 and HalfCheetah-v4 both load.
  4. A tiny SAC training step works (5K env steps, ~30 sec on GPU).
  5. PILCO can build a model and take one optimization step on synthetic data.
  6. mbrl-lib can construct its dynamics-model ensemble.

Exit 0 = ready to run experiments. Exit 1 = something's broken.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

RESULTS = []


def check(name):
    def wrap(fn):
        try:
            fn()
            RESULTS.append((name, True, None))
            print(f"  PASS  {name}")
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name}")
            traceback.print_exc()
        return fn
    return wrap


print("=" * 60)
print("kernel-method-final-project smoke test")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
print("\n[1] Imports")


@check("torch + stable-baselines3")
def _():
    import torch
    import stable_baselines3
    assert torch.__version__.startswith("2.1"), f"torch {torch.__version__}"


@check("mbrl-lib")
def _():
    import mbrl
    import mbrl.algorithms.mbpo
    import mbrl.models


@check("tensorflow + gpflow")
def _():
    import tensorflow as tf
    import gpflow
    assert tf.__version__.startswith("2.15"), f"tf {tf.__version__}"


@check("gymnasium + mujoco")
def _():
    import gymnasium as gym
    import mujoco
    assert gym.__version__.startswith("0.29"), f"gym {gym.__version__}"


@check("PILCO algorithm package")
def _():
    from pilco.models import PILCO
    from pilco.controllers import RbfController
    from pilco.rewards import ExponentialReward


# ---------------------------------------------------------------------------
# 2. GPU visibility (both frameworks)
# ---------------------------------------------------------------------------
print("\n[2] GPU visibility")

ALLOW_NO_GPU = os.environ.get("KMFP_ALLOW_NO_GPU") == "1"


@check("PyTorch GPU")
def _():
    import torch
    if not torch.cuda.is_available():
        if ALLOW_NO_GPU:
            print("    WARN  no CUDA (KMFP_ALLOW_NO_GPU=1, ignored)")
            return
        raise AssertionError("torch.cuda.is_available() is False — set KMFP_ALLOW_NO_GPU=1 to skip")
    n = torch.cuda.device_count()
    print(f"    info: {n} CUDA device(s); current={torch.cuda.get_device_name(0)}")


@check("TensorFlow GPU")
def _():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        if ALLOW_NO_GPU:
            print("    WARN  TF sees no GPU (KMFP_ALLOW_NO_GPU=1, ignored)")
            return
        raise AssertionError("TF sees no GPU — set KMFP_ALLOW_NO_GPU=1 to skip")
    print(f"    info: {len(gpus)} GPU(s) visible to TF")


# ---------------------------------------------------------------------------
# 3. Env construction
# ---------------------------------------------------------------------------
print("\n[3] Env construction")


@check("Pendulum-v1 reset+step")
def _():
    import gymnasium as gym
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    env.close()


@check("HalfCheetah-v4 reset+step")
def _():
    import gymnasium as gym
    env = gym.make("HalfCheetah-v4")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (17,)
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    env.close()


# ---------------------------------------------------------------------------
# 4. Tiny SAC training step
# ---------------------------------------------------------------------------
print("\n[4] Algorithm sanity")


@check("SAC trains on Pendulum-v1 for 1000 steps")
def _():
    import gymnasium as gym
    from stable_baselines3 import SAC
    env = gym.make("Pendulum-v1")
    model = SAC("MlpPolicy", env, seed=0, verbose=0,
                learning_starts=200, train_freq=8)
    model.learn(total_timesteps=1000)
    env.close()


@check("MBPO dynamics-model ensemble can be constructed")
def _():
    import torch
    from mbrl.models import GaussianMLP
    model = GaussianMLP(
        in_size=4, out_size=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_layers=4, ensemble_size=7, hid_size=200, deterministic=False,
    )
    x = torch.randn(8, 4, device=next(model.parameters()).device)
    out = model(x)
    assert out is not None


# Smoke-test eval outputs live in results/smoke/ — kept separate from the
# full-benchmark `results/*.csv` so the two never overwrite each other and
# both can produce training curves.
SMOKE_RESULTS_DIR = Path("results") / "smoke"


@check("SAC end-to-end: trains for 2000 Pendulum env steps, saves CSV")
def _():
    """Verifies SAC pipeline + unified CSV emission. Saves to results/smoke/
    so the eval values persist (teammate request: 紀錄evaluation的數值 so
    we can draw training curves from smoke runs too)."""
    import os
    import subprocess

    SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "runners.run_sac",
        "--env", "Pendulum-v1",
        "--steps", "2000",
        "--seed", "0",
        "--results-dir", str(SMOKE_RESULTS_DIR),
        "--eval-freq", "500",
        "--n-eval-episodes", "2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(f"run_sac exited {result.returncode}\nstderr tail:\n{tail}")
    csv_path = SMOKE_RESULTS_DIR / "sac__Pendulum-v1__seed0.csv"
    assert csv_path.exists(), f"missing CSV at {csv_path}"
    lines = csv_path.read_text().splitlines()
    assert len(lines) >= 3, f"CSV has too few eval rows ({len(lines)-1}): {lines}"
    assert lines[0].startswith("env_steps"), f"bad header: {lines[0]}"
    print(f"    info: {len(lines)-1} eval rows saved to {csv_path}")


@check("MBPO end-to-end: trains for ~500 Pendulum env steps, saves CSV")
def _():
    """Verifies the full MBPO pipeline (dynamics model + SAC inner loop +
    eval callback + unified CSV) runs without crashing. ~30-60 sec on GPU.
    Output saved to results/smoke/ for training-curve inspection."""
    import os
    import subprocess

    SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "runners.run_mbpo",
        "--env", "Pendulum-v1",
        "--steps", "500",
        "--seed", "0",
        "--results-dir", str(SMOKE_RESULTS_DIR),
        "--eval-freq", "200",
        "--n-eval-episodes", "1",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(f"run_mbpo exited {result.returncode}\nstderr tail:\n{tail}")
    csv_path = SMOKE_RESULTS_DIR / "mbpo__Pendulum-v1__seed0.csv"
    assert csv_path.exists(), f"missing CSV at {csv_path}"
    lines = csv_path.read_text().splitlines()
    assert len(lines) >= 2, f"CSV has no eval rows: {lines}"
    assert lines[0].startswith("env_steps"), f"bad header: {lines[0]}"
    print(f"    info: {len(lines)-1} eval rows saved to {csv_path}")


@check("PILCO end-to-end: runs 2 iterations on Pendulum, saves CSV")
def _():
    """Verifies the full PILCO pipeline (GP fit + policy optimization +
    real-env rollout + unified CSV) runs without crashing. ~1-2 min.
    Output saved to results/smoke/ for training-curve inspection."""
    import os
    import subprocess

    SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "runners.run_pilco",
        "--env", "Pendulum-v1",
        "--iterations", "2",
        "--seed", "0",
        "--results-dir", str(SMOKE_RESULTS_DIR),
        "--n-eval-episodes", "2",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # PILCO's L-BFGS GP optimization is CPU-heavy; allow up to 5 min.
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(f"run_pilco exited {result.returncode}\nstderr tail:\n{tail}")
    csv_path = SMOKE_RESULTS_DIR / "pilco__Pendulum-v1__seed0.csv"
    assert csv_path.exists(), f"missing CSV at {csv_path}"
    lines = csv_path.read_text().splitlines()
    # PILCO logs: 1 baseline eval (random policy) + 1 per iteration = 3 rows for 2 iter
    assert len(lines) >= 3, f"CSV has too few eval rows ({len(lines)-1}): {lines}"
    assert lines[0].startswith("env_steps"), f"bad header: {lines[0]}"
    print(f"    info: {len(lines)-1} eval rows saved to {csv_path}")


@check("PILCO trains on tiny synthetic data (1 GP + 1 policy iter)")
def _():
    import numpy as np
    from pilco.models import PILCO
    from pilco.controllers import LinearController
    from pilco.rewards import ExponentialReward
    np.random.seed(0)
    state_dim, control_dim = 3, 1
    N = 10
    X = np.random.randn(N, state_dim + control_dim)
    Y = np.random.randn(N, state_dim) * 0.1
    pilco = PILCO(
        (X, Y),
        controller=LinearController(state_dim, control_dim),
        horizon=5,
        reward=ExponentialReward(state_dim),
        m_init=np.zeros((1, state_dim)),
        S_init=0.1 * np.eye(state_dim),
    )
    pilco.optimize_models(maxiter=1, restarts=1)
    pilco.optimize_policy(maxiter=1, restarts=1)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = [(n, err) for n, ok, err in RESULTS if not ok]
print(f"Summary: {passed}/{len(RESULTS)} passed")
if failed:
    print(f"\nFailures ({len(failed)}):")
    for name, err in failed:
        print(f"  - {name}: {err}")
    sys.exit(1)
print("All smoke tests passed.")
sys.exit(0)

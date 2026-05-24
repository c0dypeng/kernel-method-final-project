"""Generate the comparison plots from results/*.csv.

Output:
  plots/pendulum_v1_comparison.png    — return vs env steps for the three algos
  plots/halfcheetah_v4_comparison.png — same on the bigger env

Each line is the seed-mean, shaded band is min/max across seeds.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILENAME_RE = re.compile(r"^(?P<algo>pilco|mbpo|sac)__(?P<env>[^_]+(?:-v\d+))__seed(?P<seed>\d+)\.csv$")


def load_csv(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(float(r["env_steps"])), float(r["mean_return"])))
            except (KeyError, ValueError):
                continue
    if not rows:
        return None, None
    rows.sort()
    xs = np.asarray([x for x, _ in rows])
    ys = np.asarray([y for _, y in rows])
    return xs, ys


def aggregate(results_dir: Path):
    """Group CSVs by (env, algo); return {env: {algo: [(xs, ys), (xs, ys), ...]}}."""
    grouped: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(results_dir.glob("*.csv")):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        xs, ys = load_csv(p)
        if xs is None:
            continue
        grouped[m["env"]][m["algo"]].append((xs, ys, int(m["seed"])))
    return grouped


def common_grid(seed_runs, n_points=50):
    """Resample (xs, ys) onto a common evenly-spaced env_steps grid via linear interp.

    Use NaN outside each seed's actual data range — so a short-running seed
    doesn't drag the mean/band on the right via np.interp's default clamping.
    """
    all_max = max(xs[-1] for xs, _, _ in seed_runs)
    all_min = max(xs[0] for xs, _, _ in seed_runs)  # latest "first" point across seeds
    grid = np.linspace(all_min, all_max, n_points)
    Y = np.full((len(seed_runs), n_points), np.nan)
    for i, (xs, ys, _) in enumerate(seed_runs):
        Y[i] = np.interp(grid, xs, ys, left=np.nan, right=np.nan)
    return grid, Y


COLORS = {"sac": "#1f77b4", "mbpo": "#ff7f0e", "pilco": "#2ca02c"}
LABELS = {"sac": "SAC", "mbpo": "MBPO", "pilco": "PILCO"}


def plot_env(env: str, algo_runs: dict[str, list], out_path: Path):
    # PILCO produces ~200-500 env steps; SAC/MBPO go to 50K-200K. A linear
    # x-axis squashes PILCO into a vertical line. Use log scale + nanmean/min/max.
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ("sac", "mbpo", "pilco"):
        runs = algo_runs.get(algo, [])
        if not runs:
            continue
        grid, Y = common_grid(runs)
        # Use nan-aware reductions so partial coverage doesn't poison the curve.
        with np.errstate(all="ignore"):
            mean = np.nanmean(Y, axis=0)
            lo = np.nanmin(Y, axis=0)
            hi = np.nanmax(Y, axis=0)
        valid = ~np.isnan(mean)
        n = len(runs)
        ax.plot(grid[valid], mean[valid], label=f"{LABELS[algo]} (n={n})",
                color=COLORS[algo], lw=2)
        ax.fill_between(grid[valid], lo[valid], hi[valid],
                        color=COLORS[algo], alpha=0.2)

    ax.set_title(f"{env}: return vs environment steps (log x-axis)")
    ax.set_xlabel("environment steps (log scale)")
    ax.set_ylabel("mean episode return (eval)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    grouped = aggregate(Path(args.results_dir))
    if not grouped:
        print(f"[plot] no CSVs found in {args.results_dir}")
        return

    out_dir = Path(args.out_dir)
    for env, algo_runs in grouped.items():
        out_path = out_dir / f"{env.lower().replace('-', '_')}_comparison.png"
        plot_env(env, algo_runs, out_path)

    # Summary table — mean of the LAST 3 evals per seed, which is more robust
    # than a single end-of-run point (especially for PILCO where the final
    # iteration can spike due to numerical issues in the GP optimization).
    summary_path = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    K = 3
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "algo", "n_seeds", f"last{K}_return_mean", f"last{K}_return_std"])
        for env, algo_runs in sorted(grouped.items()):
            for algo in ("sac", "mbpo", "pilco"):
                runs = algo_runs.get(algo, [])
                if not runs:
                    continue
                # Per seed: mean of the last K eval rows. Across seeds: mean ± std.
                per_seed = np.asarray([ys[-K:].mean() for _, ys, _ in runs])
                w.writerow([
                    env, algo, len(runs),
                    f"{per_seed.mean():.2f}",
                    f"{per_seed.std():.2f}",
                ])
    print(f"[plot] wrote {summary_path}")


if __name__ == "__main__":
    main()

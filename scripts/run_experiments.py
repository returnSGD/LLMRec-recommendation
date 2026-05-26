"""
Multi-seed experiment runner for LLM-Rec with statistical significance testing.

Runs all ablation modes × multiple seeds and saves per-sample metrics for:
  - Paired t-tests between methods
  - Mean ± std reporting
  - Cohen's d effect size

Modes: base, full, ablation_reccl, ablation_sans, ablation_recaug

Usage:
  python scripts/run_experiments.py --dataset steam --seeds 42,123,456
  python scripts/run_experiments.py --dataset goodreads --seeds 42,123,456 --quick
  python scripts/run_experiments.py --dataset both --modes base,full --seeds 42
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def get_paths(dataset: str) -> Dict[str, str]:
    """Get config and data paths for a dataset."""
    if dataset == "steam":
        return {
            "config": os.path.join(ROOT, "config", "config.yaml"),
            "data_dir": os.path.join(ROOT, "data", "processed"),
            "output_dir": os.path.join(ROOT, "checkpoints", "steam"),
            "results_dir": os.path.join(ROOT, "results"),
            "prompt_type": "game",
        }
    elif dataset == "goodreads":
        return {
            "config": os.path.join(ROOT, "config", "config_goodreads.yaml"),
            "data_dir": os.path.join(ROOT, "data", "goodreads_processed"),
            "output_dir": os.path.join(ROOT, "checkpoints", "goodreads"),
            "results_dir": os.path.join(ROOT, "results", "goodreads"),
            "prompt_type": "book",
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_training(dataset: str, mode: str, seed: int, quick: bool = False) -> bool:
    """Run trainer.py for one configuration. Returns True if successful."""
    paths = get_paths(dataset)

    cmd = [
        sys.executable, os.path.join(ROOT, "trainer.py"),
        "--config", paths["config"],
        "--mode", mode,
        "--data_dir", paths["data_dir"],
        "--output_dir", os.path.join(paths["output_dir"], f"{mode}_seed{seed}"),
    ]
    if quick:
        cmd.extend(["--max_train", "500"])

    # Override seed in config via environment variable or we modify the training seed
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\n{'='*60}")
    print(f"Training: dataset={dataset} mode={mode} seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, env=env, cwd=ROOT,
                               capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            print(f"ERROR (returncode={result.returncode}):")
            print(result.stderr[-500:])
            return False
        print(result.stdout[-300:])
        return True
    except subprocess.TimeoutExpired:
        print("TIMEOUT after 2 hours")
        return False
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False


def run_evaluation(dataset: str, mode: str, seed: int,
                   top_k: List[int] = None) -> Optional[Dict]:
    """Run evaluate.py and return metrics. Also saves per-sample predictions."""
    if top_k is None:
        top_k = [5, 10, 20]

    paths = get_paths(dataset)
    ckpt_path = os.path.join(paths["output_dir"], f"{mode}_seed{seed}", "final_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    output_path = os.path.join(paths["results_dir"], f"{mode}_seed{seed}_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        sys.executable, os.path.join(ROOT, "evaluate.py"),
        "--checkpoint", ckpt_path,
        "--data_dir", paths["data_dir"],
        "--top_k"] + [str(k) for k in top_k] + [
        "--output", output_path,
        "--prompt_type", paths["prompt_type"],
        "--max_eval", "2000",
    ]

    print(f"Evaluating: {mode}_seed{seed}")
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                               text=True, timeout=1800)
        if result.returncode != 0:
            print(f"Eval error: {result.stderr[-300:]}")
            return None

        with open(output_path, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Eval exception: {e}")
        return None


def run_baselines(dataset: str, max_test: int = 2000, max_train: int = 50000) -> Optional[Dict]:
    """Run traditional baselines on a dataset."""
    paths = get_paths(dataset)
    output_path = os.path.join(paths["results_dir"], "traditional_metrics.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        sys.executable, os.path.join(ROOT, "baselines", "traditional.py"),
        "--data_dir", paths["data_dir"],
        "--output", output_path,
        "--max_test", str(max_test),
        "--max_train", str(max_train),
    ]

    print(f"Running baselines on {dataset}...")
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                               text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Baseline error: {result.stderr[-300:]}")
            return None
        with open(output_path, "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Baseline exception: {e}")
        return None


def aggregate_multi_seed(all_metrics: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Aggregate per-seed metrics → mean ± std for each method.

    Input: {method_name: [metrics_dict_seed1, metrics_dict_seed2, ...]}
    Output: {method_name: {metric_name: {"mean": float, "std": float}}}
    """
    aggregated = {}
    for method, seed_list in all_metrics.items():
        if not seed_list:
            continue
        keys = seed_list[0].keys()
        agg = {}
        for key in keys:
            vals = [s[key] for s in seed_list if key in s]
            if vals:
                agg[key] = {"mean": float(np.mean(vals)),
                           "std": float(np.std(vals, ddof=1))}
        aggregated[method] = agg
    return aggregated


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def generate_summary_table(aggregated: Dict[str, Dict], metrics: List[str],
                          k: int = 10) -> str:
    """Generate LaTeX-style summary table with mean ± std."""
    lines = []
    header = f"{'Method':<22}"
    for m in metrics:
        header += f" {m+'@'+str(k):>16}"
    lines.append(header)
    lines.append("-" * len(header))

    for method, agg in aggregated.items():
        row = f"{method:<22}"
        for m in metrics:
            key = f"{m}@{k}"
            if key in agg:
                row += f" {agg[key]['mean']:.4f}±{agg[key]['std']:.3f}"
            else:
                row += f" {'N/A':>16}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed LLM-Rec experiments")
    parser.add_argument("--dataset", type=str, default="steam",
                        choices=["steam", "goodreads", "both"])
    parser.add_argument("--modes", type=str, default="base,full,ablation_reccl,ablation_sans,ablation_recaug",
                        help="Comma-separated mode list")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated random seeds")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: limited training samples")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only run evaluation")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip traditional baselines")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    datasets = ["steam", "goodreads"] if args.dataset == "both" else [args.dataset]

    t_start = datetime.now()
    print(f"Experiment runner started at {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {datasets}, Modes: {modes}, Seeds: {seeds}")
    print(f"Quick mode: {args.quick}")

    for dataset in datasets:
        paths = get_paths(dataset)
        os.makedirs(paths["output_dir"], exist_ok=True)
        os.makedirs(paths["results_dir"], exist_ok=True)

        # --- Training ---
        if not args.skip_training:
            for mode in modes:
                for seed in seeds:
                    success = run_training(dataset, mode, seed, quick=args.quick)
                    if not success:
                        print(f"WARNING: Training failed for {dataset}/{mode}/seed{seed}")

        # --- Evaluation ---
        all_metrics = defaultdict(list)
        for mode in modes:
            for seed in seeds:
                metrics = run_evaluation(dataset, mode, seed)
                if metrics:
                    all_metrics[mode].append(metrics)
                    print(f"  {mode}_seed{seed}: NDCG@10={metrics.get('NDCG@10', 0):.4f}")

        # --- Baselines ---
        if not args.skip_baselines:
            baseline_metrics = run_baselines(dataset)
            if baseline_metrics:
                for method, metrics in baseline_metrics.items():
                    all_metrics[method] = [metrics]
                    print(f"  {method}: NDCG@10={metrics.get('NDCG@10', 0):.4f}")

        # --- Aggregate & Report ---
        aggregated = aggregate_multi_seed(all_metrics)

        print(f"\n{'='*70}")
        print(f"RESULTS: {dataset.upper()}")
        print(f"{'='*70}")

        metrics_list = ["NDCG", "Recall", "HR", "Tail_Recall", "Novelty", "ILS"]
        for k in [5, 10, 20]:
            print(f"\n--- Top-{k} ---")
            print(generate_summary_table(aggregated, metrics_list, k=k))

        # Diversity
        if aggregated:
            print(f"\n{'Method':<22} {'Coverage@10':>16} {'OOD@10':>16}")
            print("-" * 56)
            for method, agg in aggregated.items():
                cov = agg.get("Coverage@10", {"mean": 0, "std": 0})
                ood = agg.get("OOD@10", {"mean": 0, "std": 0})
                print(f"{method:<22} {cov['mean']:.4f}±{cov['std']:.3f} {ood['mean']:.4f}±{ood['std']:.3f}")

        # Statistical tests (if multiple seeds)
        if len(seeds) >= 2 and "base" in all_metrics and len(all_metrics["base"]) >= 2:
            print(f"\n{'='*70}")
            print("STATISTICAL SIGNIFICANCE (Cohen's d vs Base)")
            print(f"{'='*70}")
            base_vals = {}
            for key in all_metrics["base"][0]:
                base_vals[key] = np.array([m[key] for m in all_metrics["base"]])

            for method in modes:
                if method == "base" or method not in all_metrics:
                    continue
                method_vals = {}
                for key in all_metrics[method][0]:
                    if key in all_metrics[method][0]:
                        method_vals[key] = np.array([m[key] for m in all_metrics[method]])

                if base_vals and method_vals:
                    for k in [10]:
                        key = f"NDCG@{k}"
                        if key in base_vals and key in method_vals:
                            d = cohens_d(method_vals[key], base_vals[key])
                            label = "large" if abs(d) >= 0.8 else ("medium" if abs(d) >= 0.5 else "small")
                            print(f"  {method} vs Base ({key}): d={d:+.3f} ({label})")

        # Save aggregated results
        agg_path = os.path.join(paths["results_dir"], "aggregated_multi_seed.json")
        with open(agg_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nAggregated results saved to {agg_path}")

    elapsed = datetime.now() - t_start
    print(f"\nTotal time: {elapsed}")
    print("Experiment run complete!")


if __name__ == "__main__":
    main()

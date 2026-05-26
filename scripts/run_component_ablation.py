"""
Component-level ablation + hyperparameter sensitivity for LLM-Rec.

Ablation dimensions:
  RecCL:  seq-only, item-only, pred-only, full-3dim
  SANS:   easy-only, easy+medium, easy+medium+hard (full)
  RecAug: truncation-only, permutation-only, substitution-only, full-3op

Sensitivity sweeps:
  RecCL warmup_ratio: [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
  SANS temperature τ:  [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
  RecAug consistency λ: [0.01, 0.05, 0.1, 0.2, 0.5]

Also generates:
  - Convergence curves (loss per step)
  - Hyperparameter sensitivity plots (metric vs param)
  - Ablation gain bar charts

Usage:
  python scripts/run_component_ablation.py --dataset steam --quick
  python scripts/run_component_ablation.py --dataset both --analysis only
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Path helpers ─────────────────────────────────────────────

def get_paths(dataset: str) -> Dict[str, str]:
    if dataset == "steam":
        return {
            "config": os.path.join(ROOT, "config", "config.yaml"),
            "data_dir": os.path.join(ROOT, "data", "processed"),
            "output_dir": os.path.join(ROOT, "checkpoints", "ablation", "steam"),
            "results_dir": os.path.join(ROOT, "results", "ablation"),
            "prompt_type": "game",
            "base_model": "google/flan-t5-small",
        }
    else:
        return {
            "config": os.path.join(ROOT, "config", "config_goodreads.yaml"),
            "data_dir": os.path.join(ROOT, "data", "goodreads_processed"),
            "output_dir": os.path.join(ROOT, "checkpoints", "ablation", "goodreads"),
            "results_dir": os.path.join(ROOT, "results", "ablation", "goodreads"),
            "prompt_type": "book",
            "base_model": "google/flan-t5-small",
        }


def run_train_eval(dataset: str, mode: str, seed: int = 42,
                   overrides: Dict[str, str] = None,
                   experiment_id: str = None,
                   quick: bool = False,
                   max_train: int = None,
                   epochs: int = None) -> Optional[Dict]:
    """Run training + evaluation for one config. Returns metrics dict."""
    paths = get_paths(dataset)
    dir_name = f"{experiment_id}_seed{seed}" if experiment_id else f"{mode}_seed{seed}"
    ckpt_dir = os.path.join(paths["output_dir"], dir_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, os.path.join(ROOT, "trainer.py"),
        "--config", paths["config"],
        "--mode", mode,
        "--data_dir", paths["data_dir"],
        "--output_dir", ckpt_dir,
        "--seed", str(seed),
    ]
    if quick and max_train is None:
        cmd.extend(["--max_train", "500"])
    if max_train is not None:
        cmd.extend(["--max_train", str(max_train)])
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    if overrides:
        for key, val in overrides.items():
            cmd.extend([f"--{key}", str(val)])

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    print(f"  Training: {mode} ({dataset})...")
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                               text=True, timeout=3600, env=env)
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[-200:]}")
            return None
    except subprocess.TimeoutExpired:
        print("    TIMEOUT")
        return None

    # Evaluate
    ckpt_path = os.path.join(ckpt_dir, "final_model.pt")
    if not os.path.exists(ckpt_path):
        return None

    output_path = os.path.join(ckpt_dir, "metrics.json")
    eval_cmd = [
        sys.executable, os.path.join(ROOT, "evaluate.py"),
        "--checkpoint", ckpt_path,
        "--data_dir", paths["data_dir"],
        "--output", output_path,
        "--prompt_type", paths["prompt_type"],
        "--base_model", paths["base_model"],
        "--max_eval", "1000",
    ]

    try:
        result = subprocess.run(eval_cmd, cwd=ROOT, capture_output=True,
                               text=True, timeout=600, env=env)
        if result.returncode == 0 and os.path.exists(output_path):
            with open(output_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ── RecCL component ablation ─────────────────────────────────

def ablation_reccl_3dim(dataset: str, quick: bool = False,
                         max_train: int = None, epochs: int = None) -> Dict[str, Dict]:
    """Ablate each of RecCL's 3 difficulty dimensions individually.

    Modes:
      - base: no RecCL (baseline)
      - reccl_seq:  only sequence difficulty   (α=1.0, β=0, γ=0)
      - reccl_item: only item difficulty       (α=0, β=1.0, γ=0)
      - reccl_pred: only prediction difficulty (α=0, β=0, γ=1.0)
      - reccl_full: all 3 dimensions           (α=β=γ=0.33)
    """
    print("\n" + "=" * 60)
    print("RecCL 3-DIMENSIONAL COMPONENT ABLATION")
    print("=" * 60)

    configs = {
        "base": ("base", None),  # no RecCL at all
        "reccl_seq":  ("ablation_reccl", {"reccl_alpha": "1.0", "reccl_beta": "0.0", "reccl_gamma": "0.0"}),
        "reccl_item": ("ablation_reccl", {"reccl_alpha": "0.0", "reccl_beta": "1.0", "reccl_gamma": "0.0"}),
        "reccl_pred": ("ablation_reccl", {"reccl_alpha": "0.0", "reccl_beta": "0.0", "reccl_gamma": "1.0"}),
        "reccl_full": ("ablation_reccl", {"reccl_alpha": "0.33", "reccl_beta": "0.33", "reccl_gamma": "0.34"}),
    }

    results = {}
    for name, (mode, overrides) in configs.items():
        print(f"\n  --- {name} ---")
        # Use variant name as experiment_id for unique checkpoint dir
        metrics = run_train_eval(dataset, mode, seed=42, overrides=overrides,
                                 experiment_id=name, quick=quick,
                                 max_train=max_train, epochs=epochs)
        if metrics:
            results[name] = metrics
            print(f"    NDCG@10={metrics.get('NDCG@10', 0):.4f}, "
                  f"Recall@10={metrics.get('Recall@10', 0):.4f}, "
                  f"Tail_Recall@10={metrics.get('Tail_Recall@10', 0):.4f}")
        else:
            print(f"    No results")

    return results


# ── SANS component ablation ──────────────────────────────────

def ablation_sans_tiers(dataset: str, quick: bool = False,
                        max_train: int = None, epochs: int = None) -> Dict[str, Dict]:
    """Ablate SANS negative sampling tiers.

    Modes:
      - base: no SANS
      - sans_easy: only easy negatives (random, w_easy=1.0)
      - sans_em:    easy + medium (w_easy=0.25, w_medium=0.75)
      - sans_full:  easy + medium + hard (full SANS)
    """
    print("\n" + "=" * 60)
    print("SANS TIER ABLATION")
    print("=" * 60)

    configs = {
        "base": ("base", None, None),
        "sans_easy": ("ablation_sans",
                      {"sans_medium_count": "0", "sans_hard_count": "0"}, None),
        "sans_em": ("ablation_sans",
                    {"sans_hard_count": "0"}, None),
        "sans_full": ("ablation_sans", None, None),
    }

    results = {}
    for name, (mode, overrides, use_llm) in configs.items():
        print(f"\n  --- {name} ---")
        metrics = run_train_eval(dataset, mode, seed=42, overrides=overrides,
                                 experiment_id=name, quick=quick,
                                 max_train=max_train, epochs=epochs)
        if metrics:
            results[name] = metrics
            print(f"    NDCG@10={metrics.get('NDCG@10', 0):.4f}, "
                  f"Tail_Recall@10={metrics.get('Tail_Recall@10', 0):.4f}")
        else:
            print(f"    No results")

    return results


# ── RecAug component ablation ────────────────────────────────

def ablation_recaug_ops(dataset: str, quick: bool = False,
                        max_train: int = None, epochs: int = None) -> Dict[str, Dict]:
    """Ablate RecAug augmentation operations individually.

    Modes:
      - base: no augmentation
      - recaug_perm:  session permutation only (LLM-free with playtime gaps)
      - recaug_trunc: random-drop truncation only (LLM-free fallback)
      - recaug_full:  all 3 operations (perm + trunc, no sub without LLM)
    Note: LLM-guided substitution omitted (requires anthropic SDK + API).
    """
    print("\n" + "=" * 60)
    print("RecAug OPERATION ABLATION")
    print("=" * 60)

    configs = {
        "base": ("base", None),
        "recaug_perm": ("ablation_recaug", {"recaug_ops": "perm"}),
        "recaug_trunc": ("ablation_recaug", {"recaug_ops": "trunc"}),
        "recaug_full": ("ablation_recaug", {"recaug_ops": "perm,trunc"}),
    }

    results = {}
    for name, (mode, overrides) in configs.items():
        print(f"\n  --- {name} ---")
        metrics = run_train_eval(dataset, mode, seed=42, overrides=overrides,
                                 experiment_id=name, quick=quick,
                                 max_train=max_train, epochs=epochs)
        if metrics:
            results[name] = metrics
            print(f"    NDCG@10={metrics.get('NDCG@10', 0):.4f}, "
                  f"ILS@10={metrics.get('ILS@10', 0):.4f}")
        else:
            print(f"    No results")

    return results


# ── Hyperparameter sensitivity ───────────────────────────────

def sensitivity_warmup_ratio(dataset: str, quick: bool = False,
                             max_train: int = None, epochs: int = None) -> Dict[float, Dict]:
    """Sweep RecCL warmup_ratio ∈ [0.1, 0.7]."""
    print("\n" + "=" * 60)
    print("RecCL WARMUP RATIO SENSITIVITY")
    print("=" * 60)

    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    results = {}

    for r in ratios:
        print(f"\n  --- warmup_ratio={r} ---")
        overrides = {"reccl_warmup_ratio": str(r)}
        metrics = run_train_eval(dataset, "ablation_reccl", seed=42,
                                 overrides=overrides, quick=quick,
                                 max_train=max_train, epochs=epochs)
        if metrics:
            results[r] = metrics
            print(f"    NDCG@10={metrics.get('NDCG@10', 0):.4f}")

    return results


def sensitivity_sans_tau(dataset: str, quick: bool = False,
                         max_train: int = None, epochs: int = None) -> Dict[float, Dict]:
    """Sweep SANS temperature τ ∈ [0.01, 0.15]."""
    print("\n" + "=" * 60)
    print("SANS TEMPERATURE SENSITIVITY")
    print("=" * 60)

    taus = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    results = {}

    for tau in taus:
        print(f"\n  --- tau={tau} ---")
        overrides = {"sans_temperature": str(tau)}
        metrics = run_train_eval(dataset, "ablation_sans", seed=42,
                                 overrides=overrides, quick=quick,
                                 max_train=max_train, epochs=epochs)
        if metrics:
            results[tau] = metrics
            print(f"    NDCG@10={metrics.get('NDCG@10', 0):.4f}")

    return results


# ── Report generation ────────────────────────────────────────

def format_ablation_table(component: str, results: Dict[str, Dict],
                          metrics: List[str], k: int = 10) -> str:
    """Generate markdown ablation table."""
    lines = [f"### {component} Component Ablation (Top-{k})", ""]
    header = f"| Variant | " + " | ".join(f"{m}@{k}" for m in metrics) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(metrics) + 1)) + "|")

    baseline = results.get("base", {})
    for name, data in results.items():
        row = f"| {name} |"
        for m in metrics:
            key = f"{m}@{k}"
            val = data.get(key, 0)
            if name != "base" and baseline:
                base_val = baseline.get(key, 0)
                delta = val - base_val
                sign = "+" if delta > 0 else ""
                row += f" {val:.4f} ({sign}{delta:.3f}) |"
            else:
                row += f" {val:.4f} |"
        lines.append(row)

    return "\n".join(lines)


def generate_ablation_report(all_results: Dict[str, Dict], output_path: str):
    """Generate comprehensive ablation report."""
    report = []
    report.append("# LLM-Rec Component Ablation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    metrics_main = ["NDCG", "Recall", "HR"]
    metrics_side = ["Tail_Recall", "Novelty", "ILS"]

    for component in ["reccl", "sans", "recaug"]:
        key = f"ablation_{component}"
        if key in all_results and all_results[key]:
            report.append(format_ablation_table(component, all_results[key], metrics_main, k=10))
            report.append("")
            report.append(format_ablation_table(component, all_results[key], metrics_side, k=10))
            report.append("")

    # Sensitivity results
    for sweep_name in ["warmup_ratio", "sans_tau"]:
        if sweep_name in all_results and all_results[sweep_name]:
            report.append(f"### {sweep_name} Sensitivity")
            report.append("")
            report.append("| Value | NDCG@10 | Recall@10 | Tail_Recall@10 |")
            report.append("|---|---|---|---|")
            for val, metrics in sorted(all_results[sweep_name].items()):
                report.append(f"| {val} | {metrics.get('NDCG@10', 0):.4f} | "
                             f"{metrics.get('Recall@10', 0):.4f} | "
                             f"{metrics.get('Tail_Recall@10', 0):.4f} |")
            report.append("")

    report_text = "\n".join(report)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Component ablation for LLM-Rec")
    parser.add_argument("--dataset", type=str, default="steam",
                        choices=["steam", "goodreads", "both"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max_train", type=int, default=None,
                        help="Max training samples (overrides --quick)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--skip_reccl", action="store_true")
    parser.add_argument("--skip_sans", action="store_true")
    parser.add_argument("--skip_recaug", action="store_true")
    parser.add_argument("--skip_sensitivity", action="store_true")
    args = parser.parse_args()

    datasets = ["steam", "goodreads"] if args.dataset == "both" else [args.dataset]

    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset}")
        print(f"{'#'*60}")

        all_results = {}

        if not args.skip_reccl:
            all_results["ablation_reccl"] = ablation_reccl_3dim(
                dataset, args.quick, max_train=args.max_train, epochs=args.epochs)

        if not args.skip_sans:
            all_results["ablation_sans"] = ablation_sans_tiers(
                dataset, args.quick, max_train=args.max_train, epochs=args.epochs)

        if not args.skip_recaug:
            all_results["ablation_recaug"] = ablation_recaug_ops(
                dataset, args.quick, max_train=args.max_train, epochs=args.epochs)

        if not args.skip_sensitivity:
            all_results["warmup_ratio"] = sensitivity_warmup_ratio(
                dataset, args.quick, max_train=args.max_train, epochs=args.epochs)
            all_results["sans_tau"] = sensitivity_sans_tau(
                dataset, args.quick, max_train=args.max_train, epochs=args.epochs)

        # Generate report
        paths = get_paths(dataset)
        report_path = os.path.join(paths["results_dir"], "component_ablation_report.md")
        report = generate_ablation_report(all_results, report_path)
        print(report)


if __name__ == "__main__":
    main()

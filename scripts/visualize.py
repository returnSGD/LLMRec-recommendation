"""
Visualization for LLM-Rec experiments.

Generates:
  1. Convergence curves (train/val loss vs step)
  2. Hyperparameter sensitivity plots (metric vs param)
  3. Ablation gain bar charts
  4. Framework architecture diagram (via matplotlib)
  5. Per-metric comparison across methods

Usage:
  python scripts/visualize.py --results_dir results/
  python scripts/visualize.py --results_dir results/ --format pdf
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def setup_plots():
    """Configure matplotlib for publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


# ── Plot 1: Convergence Curves ───────────────────────────────

def plot_convergence_curves(loss_logs: Dict[str, List[float]],
                           output_path: str,
                           title: str = "Training Convergence"):
    """Plot loss vs step for multiple methods.

    Args:
        loss_logs: {method_name: [loss_at_step1, loss_at_step2, ...]}
    """
    plt = setup_plots()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"base": "#2196F3", "reccl": "#4CAF50", "sans": "#FF9800",
              "recaug": "#9C27B0", "full": "#F44336"}

    for method, losses in loss_logs.items():
        if not losses:
            continue
        color = colors.get(method.replace("LLM+", "").lower(), "#607D8B")
        steps = range(1, len(losses) + 1)
        ax.plot(steps, losses, label=method, color=color, linewidth=1.2, alpha=0.9)

        # Smoothed (moving average) overlay
        if len(losses) > 20:
            window = max(5, len(losses) // 50)
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(range(window, len(losses)+1), smoothed, color=color,
                   linewidth=2.5, alpha=0.6, linestyle="--")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(framealpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Convergence plot saved to {output_path}")


# ── Plot 2: Hyperparameter Sensitivity ───────────────────────

def plot_hyperparameter_sensitivity(sweep_results: Dict[str, Dict[float, Dict[str, float]]],
                                    output_dir: str):
    """Plot metric(s) vs hyperparameter value for each sweep.

    Args:
        sweep_results: {sweep_name: {param_value: {metric: value, ...}}}
    """
    plt = setup_plots()

    for sweep_name, values in sweep_results.items():
        if not values:
            continue
        sorted_vals = sorted(values.items())
        xs = [v for v, _ in sorted_vals]
        metrics = list(sorted_vals[0][1].keys())

        # Filter to interesting metrics
        plot_metrics = [m for m in metrics if any(k in m for k in
                       ["NDCG@10", "Recall@10", "HR@10", "Tail_Recall@10", "Novelty@10"])]

        fig, ax = plt.subplots(figsize=(7, 5))
        for m in plot_metrics[:4]:
            ys = [metrics_dict.get(m, 0) for _, metrics_dict in sorted_vals]
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, label=m.split("@")[0])

        ax.set_xlabel(sweep_name.replace("_", " ").title())
        ax.set_ylabel("Metric Value @10")
        ax.set_title(f"Hyperparameter Sensitivity: {sweep_name}")
        ax.legend(framealpha=0.8)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"sensitivity_{sweep_name}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Sensitivity plot saved to {out_path}")


# ── Plot 3: Ablation Gain Bar Chart ──────────────────────────

def plot_ablation_gains(ablation_results: Dict[str, Dict[str, Dict[str, float]]],
                        output_dir: str):
    """Bar chart showing per-component gains over baseline.

    Args:
        ablation_results: {
            "reccl": {"base": {...}, "seq_only": {...}, ...},
            "sans": {...}, "recaug": {...}
        }
    """
    plt = setup_plots()

    for component, variants in ablation_results.items():
        if "base" not in variants or len(variants) < 2:
            continue

        base = variants["base"]
        others = {k: v for k, v in variants.items() if k != "base"}

        # Compute gains for key metrics
        key_metrics = ["NDCG@10", "Recall@10", "HR@10", "Tail_Recall@10"]
        gains = {}
        for metric in key_metrics:
            if metric not in base:
                continue
            base_val = base[metric]
            gains[metric] = {}
            for name, data in others.items():
                if metric in data:
                    gains[metric][name] = (data[metric] - base_val) / max(abs(base_val), 1e-6) * 100

        if not gains:
            continue

        # Plot
        n_metrics = len(gains)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric, metric_gains) in zip(axes, gains.items()):
            names = list(metric_gains.keys())
            values = [metric_gains[n] for n in names]
            colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]

            bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Relative Gain (%)")
            ax.set_title(metric)

            # Annotate bars
            for bar, val in zip(bars, values):
                x_pos = bar.get_width()
                ha = "left" if x_pos > 0 else "right"
                offset = 0.3 if x_pos > 0 else -0.3
                ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2,
                       f"{val:+.1f}%", ha=ha, va="center", fontsize=8)

        fig.suptitle(f"{component.upper()} Component Ablation Gains", fontsize=13)
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"ablation_gains_{component}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Ablation gain chart saved to {out_path}")


# ── Plot 4: Method Comparison Radar ──────────────────────────

def plot_radar_comparison(methods_data: Dict[str, Dict[str, float]],
                         output_path: str,
                         title: str = "Method Comparison"):
    """Radar chart comparing methods across multiple metrics."""
    plt = setup_plots()

    radar_metrics = ["NDCG@10", "Recall@10", "HR@10", "Tail_Recall@10",
                     "Novelty@10"]
    # ILS and Coverage are on different scales

    # Normalize each metric to [0, 1] across methods
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Get min/max for normalization per metric
    normed = {}
    for m in radar_metrics:
        vals = [d.get(m, 0) for d in methods_data.values() if m in d]
        if not vals:
            continue
        vmin, vmax = min(vals), max(vals)
        for method, data in methods_data.items():
            if method not in normed:
                normed[method] = []
            val = data.get(m, 0)
            normed[method].append((val - vmin) / max(vmax - vmin, 1e-10))

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    colors = {"LLM-Base": "#2196F3", "LLM+RecCL": "#4CAF50",
              "LLM+SANS": "#FF9800", "LLM+RecAug": "#9C27B0",
              "LLM+All": "#F44336", "Random": "#9E9E9E",
              "Popularity": "#795548", "ItemKNN": "#607D8B",
              "SeqNgram": "#00BCD4"}

    for method, values in normed.items():
        if len(values) == n_metrics:
            vals = values + values[:1]  # close polygon
            color = colors.get(method, "#333333")
            ax.fill(angles, vals, alpha=0.1, color=color)
            ax.plot(angles, vals, "o-", linewidth=1.5, label=method, color=color, markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.split("@")[0] for m in radar_metrics], fontsize=8)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), framealpha=0.8, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Radar plot saved to {output_path}")


# ── Plot 5: Framework Architecture Diagram ───────────────────

def plot_framework_diagram(output_path: str):
    """Generate a simplified framework architecture diagram."""
    plt = setup_plots()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("LLM-Rec Sample Engineering Framework", fontsize=14, fontweight="bold", pad=20)

    # Define box positions and labels
    boxes = [
        # (x, y, w, h, label, color, text_color)
        # Input
        (0.3, 4.5, 2.0, 1.8, "User-Item\nSequences", "#E3F2FD", "black"),
        (0.3, 1.8, 2.0, 1.8, "Item Metadata\nCatalog", "#E8F5E9", "black"),

        # Sample Engineering (3 components)
        (3.5, 4.8, 2.2, 1.4, "RecCL\nCurriculum\nLearning", "#FFF3E0", "black"),
        (3.5, 3.0, 2.2, 1.4, "SANS\nNegative\nSampling", "#F3E5F5", "black"),
        (3.5, 1.2, 2.2, 1.4, "RecAug\nData\nAugmentation", "#E0F7FA", "black"),

        # Encoder
        (6.8, 2.8, 2.0, 2.8, "T5 Encoder-Decoder\n(FLAN-T5)\n+ LoRA", "#FCE4EC", "black"),

        # Outputs
        (9.8, 4.5, 2.0, 1.8, "Generative\nRecommendation\nLoss (CE+InfoNCE)", "#FFF9C4", "black"),
        (9.8, 1.8, 2.0, 1.8, "Item Ranking\n& Metrics", "#C8E6C9", "black"),
    ]

    # Draw boxes
    import matplotlib.patches as mpatches
    for x, y, w, h, label, color, tc in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.15",
                                       facecolor=color,
                                       edgecolor="#555555",
                                       linewidth=1.5,
                                       alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
               fontsize=9, color=tc, fontweight="medium")

    # Draw arrows (simplified)
    arrows = [
        (2.35, 5.4, 3.45, 5.5),   # input → reccl
        (2.35, 3.8, 3.45, 3.7),   # input → sans
        (2.35, 2.7, 3.45, 2.6),   # input → recaug
        (5.75, 5.5, 6.75, 5.0),   # reccl → encoder
        (5.75, 3.7, 6.75, 4.2),   # sans → encoder
        (5.75, 1.9, 6.75, 3.0),   # recaug → encoder
        (8.85, 5.4, 9.75, 5.4),   # encoder → loss
        (8.85, 3.0, 9.75, 3.0),   # encoder → metrics
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color="#555555",
                                  lw=1.5, connectionstyle="arc3,rad=0.1"))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Framework diagram saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-Rec visualization suite")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory with experiment result JSONs")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                        help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output format")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating visualizations...")

    # 1. Framework diagram
    plot_framework_diagram(os.path.join(args.output_dir, f"framework_diagram.{args.format}"))

    # 2. Try to load experiment results for data-driven plots
    result_files = {
        "base": os.path.join(args.results_dir, "baseline_metrics.json"),
        "reccl": os.path.join(args.results_dir, "reccl_metrics.json"),
        "sans": os.path.join(args.results_dir, "sans_metrics.json"),
        "recaug": os.path.join(args.results_dir, "recaug_metrics.json"),
        "full": os.path.join(args.results_dir, "full_metrics.json"),
    }

    methods_data = {}
    for method, fpath in result_files.items():
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                methods_data[f"LLM+{method.upper() if method != 'base' else 'LLM-Base'}"] = json.load(f)

    # Traditional baselines
    trad_path = os.path.join(args.results_dir, "traditional_metrics.json")
    if os.path.exists(trad_path):
        with open(trad_path, "r") as f:
            trad = json.load(f)
        for method, metrics in trad.items():
            methods_data[method] = metrics

    if methods_data:
        # Radar comparison
        plot_radar_comparison(methods_data,
                            os.path.join(args.output_dir, f"radar_comparison.{args.format}"))

    # 3. Ablation results
    ablation_dir = os.path.join(args.results_dir, "ablation")
    if os.path.exists(ablation_dir):
        ablation_files = {
            "reccl": os.path.join(ablation_dir, "reccl_ablation.json"),
            "sans": os.path.join(ablation_dir, "sans_ablation.json"),
            "recaug": os.path.join(ablation_dir, "recaug_ablation.json"),
        }
        ablation_data = {}
        for comp, fpath in ablation_files.items():
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    ablation_data[comp] = json.load(f)

        if ablation_data:
            plot_ablation_gains(ablation_data, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()

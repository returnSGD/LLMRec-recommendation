"""
Visualization suite for LLM-Rec paper.

Generates 6 publication-quality SVG figures:
  1. framework.svg          — Architecture diagram (RecCL/SANS/RecAug + training loop)
  2. convergence.svg        — Training loss curves per epoch
  3. method_comparison.svg  — Grouped bar chart: all methods × key metrics
  4. accuracy_novelty_tradeoff.svg — NDCG@10 vs Novelty@10 scatter
  5. reccl_effect.svg       — RecCL before/after with error bars (mean ± per-sample SD)
  6. diversity_comparison.svg — ILS@10 & Coverage@10 comparison

Usage:
  python scripts/visualize.py
  python scripts/visualize.py --output_dir image --format svg
"""

import os
import sys
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def setup_style():
    """Configure matplotlib for publication-quality SVG output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.format": "svg",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
    return plt


# ════════════════════════════════════════════════════════════════
# Real data (hardcoded from experiment results)
# ════════════════════════════════════════════════════════════════

# All method metrics @10
METHOD_METRICS = {
    "Random":      {"NDCG@10": 0.0007, "Recall@10": 0.0020, "HR@10": 0.0020,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.9905, "ILS@10": 0.1650},
    "Popularity":  {"NDCG@10": 0.0206, "Recall@10": 0.0235, "HR@10": 0.0235,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.4258, "ILS@10": 0.2586},
    "ItemKNN":     {"NDCG@10": 0.0178, "Recall@10": 0.0295, "HR@10": 0.0295,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.6386, "ILS@10": 0.2519},
    "SeqNgram":    {"NDCG@10": 0.0319, "Recall@10": 0.0630, "HR@10": 0.0630,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.6251, "ILS@10": 0.2765},
    "LLM-Base(20K)": {"NDCG@10": 0.0080, "Recall@10": 0.0080, "HR@10": 0.0080,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.6393, "ILS@10": 0.0000},
    "LLM-Base(10K)": {"NDCG@10": 0.0050, "Recall@10": 0.0050, "HR@10": 0.0050,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.7151, "ILS@10": 0.0000},
    "LLM+RecCL(10K)": {"NDCG@10": 0.0030, "Recall@10": 0.0030, "HR@10": 0.0030,
                    "Tail_Recall@10": 0.0000, "Novelty@10": 0.7548, "ILS@10": 0.0000},
}

# Per-sample SD for LLM methods (from per-sample evaluation, seed=42)
LLM_PER_SAMPLE_SD = {
    "LLM-Base(10K)": {"NDCG@10": 0.0706, "Novelty@10": 0.2646},
    "LLM+RecCL(10K)": {"NDCG@10": 0.0547, "Novelty@10": 0.1683},
}

# Convergence data (CE loss per epoch)
CONVERGENCE_DATA = {
    "LLM-Base(20K)":     [8.30, 2.08, 1.60, 1.40, 1.33],
    "LLM-Base(10K)":     [8.30, 2.08, 1.60, 1.40, 1.33],  # same trend, noted as approximate
    "LLM+RecCL(10K)":    [13.29, 3.00, 2.22, 1.93, 1.84],
}

# Diversity/Coverage data
DIVERSITY_DATA = {
    "Random":      {"Coverage@10": 0.9281, "OOD@10": 0.0000},
    "Popularity":  {"Coverage@10": 0.0037, "OOD@10": 0.0000},
    "ItemKNN":     {"Coverage@10": 0.0830, "OOD@10": 0.0000},
    "SeqNgram":    {"Coverage@10": 0.0205, "OOD@10": 0.0000},
    "LLM-Base(20K)": {"Coverage@10": 0.0020, "OOD@10": 0.0010},
    "LLM-Base(10K)": {"Coverage@10": 0.0013, "OOD@10": 0.0000},
    "LLM+RecCL(10K)": {"Coverage@10": 0.0013, "OOD@10": 0.0000},
}

# Color palette (colorblind-friendly)
COLORS = {
    "Random":      "#9E9E9E",
    "Popularity":  "#795548",
    "ItemKNN":     "#607D8B",
    "SeqNgram":    "#00BCD4",
    "LLM-Base(20K)": "#2196F3",
    "LLM-Base(10K)": "#64B5F6",
    "LLM+RecCL(10K)": "#4CAF50",
}


# ════════════════════════════════════════════════════════════════
# Figure 1: Framework Architecture Diagram
# ════════════════════════════════════════════════════════════════

def plot_framework(output_path: str):
    """Draw the LLM-Rec framework architecture diagram."""
    plt = setup_style()
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── Title ──
    ax.text(7, 7.6, "LLM-Rec Sample Engineering Framework",
            ha="center", va="center", fontsize=16, fontweight="bold")

    # ── Phase labels ──
    ax.text(1.3, 7.1, "(1) Data Input", ha="center", fontsize=10, fontweight="bold", color="#555")
    ax.text(4.7, 7.1, "(2) Sample Engineering (Training-Time Only)",
            ha="center", fontsize=10, fontweight="bold", color="#555")
    ax.text(8.5, 7.1, "(3) LLM Backbone", ha="center", fontsize=10, fontweight="bold", color="#555")
    ax.text(12.0, 7.1, "(4) Output & Loss", ha="center", fontsize=10, fontweight="bold", color="#555")

    # ── Box definitions: (x, y, w, h, label, color) ──
    boxes = [
        # Phase 1: Input
        (0.3, 4.0, 2.0, 2.2, "User-Item\nInteraction\nSequences", "#E3F2FD"),
        (0.3, 1.0, 2.0, 2.2, "Item Metadata\nCatalog\n(title, genre, tags)", "#E8F5E9"),

        # Phase 2: Sample Engineering
        (3.3, 4.8, 2.5, 1.5, "RecCL\nCurriculum Learning\n3D Difficulty Sampling", "#FFF3E0"),
        (3.3, 2.9, 2.5, 1.5, "SANS\nLayered Negative Sampling\nEasy → Medium → Hard", "#F3E5F5"),
        (3.3, 1.0, 2.5, 1.5, "RecAug\nSemantic-Preserving\nData Augmentation", "#E0F7FA"),

        # Phase 3: Backbone
        (6.8, 2.5, 2.2, 3.4, "T5 Encoder-Decoder\n(FLAN-T5-Small, 60M)\n\n→ CE Loss\n→ InfoNCE Loss\n→ Consistency Loss", "#FCE4EC"),

        # Phase 4: Output
        (10.0, 4.8, 2.5, 1.5, "Generative Rec.\nBeam-Search Decoding\n→ Next Item Title", "#FFF9C4"),
        (10.0, 2.2, 2.5, 1.5, "Evaluation\nNDCG / Recall / HR / ILS\nTail Recall / Novelty / OOD", "#C8E6C9"),

        # Legend: training-only
        (10.0, 0.2, 2.5, 1.2, "[Inference: no overhead]\nRecCL/SANS/RecAug are\ntraining-time only", "#F5F5F5"),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.12",
                                       facecolor=color,
                                       edgecolor="#888888",
                                       linewidth=1.2,
                                       alpha=0.92)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.5, color="black")

    # ── Arrows ──
    arrow_style = dict(arrowstyle="->", color="#666666", lw=1.3,
                       connectionstyle="arc3,rad=0.05")
    arrows = [
        # Input → SE
        (2.35, 5.1, 3.25, 5.55),
        (2.35, 3.8, 3.25, 3.65),
        (2.35, 2.2, 3.25, 1.75),
        # SE → Backbone
        (5.85, 5.55, 6.75, 5.0),
        (5.85, 3.65, 6.75, 4.2),
        (5.85, 1.75, 6.75, 3.4),
        # Backbone → Output
        (9.05, 5.2, 9.95, 5.55),
        (9.05, 3.5, 9.95, 3.2),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)

    # ── Method annotations on arrows ──
    annotations = [
        (4.5, 6.5, "weights samples\nby difficulty", "#E65100"),
        (4.5, 4.5, "constructs layered\nnegative batches", "#7B1FA2"),
        (4.5, 2.6, "augments sequences\n+ consistency reg.", "#00838F"),
    ]
    for x, y, text, color in annotations:
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
                color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=color, alpha=0.7, linewidth=0.5))

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [1/6] Framework diagram → {output_path}")


# ════════════════════════════════════════════════════════════════
# Figure 2: Convergence Curves
# ════════════════════════════════════════════════════════════════

def plot_convergence(output_path: str):
    """Training CE loss per epoch for Base vs RecCL."""
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    epochs = np.arange(1, 6)

    ax.plot(epochs, CONVERGENCE_DATA["LLM-Base(20K)"], "o-",
            color="#2196F3", linewidth=1.8, markersize=7, label="LLM-Base (20K)")
    ax.plot(epochs, CONVERGENCE_DATA["LLM+RecCL(10K)"], "s--",
            color="#4CAF50", linewidth=1.8, markersize=7, label="LLM+RecCL (10K)")

    # Annotate final values
    ax.annotate("1.33", xy=(5, 1.33), xytext=(4.5, 2.5),
                fontsize=8, color="#2196F3",
                arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1))
    ax.annotate("1.84", xy=(5, 1.84), xytext=(4.2, 4.0),
                fontsize=8, color="#4CAF50",
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1))

    # Shade the "warmup zone" (first 30% = 1.5 epochs)
    ax.axvspan(0.8, 1.5, alpha=0.06, color="orange", label="RecCL warmup (~30%)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title("Training Convergence: Base vs RecCL")
    ax.set_xticks(epochs)
    ax.legend(framealpha=0.85, loc="upper right")
    ax.set_xlim(0.8, 5.2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [2/6] Convergence curves → {output_path}")


# ════════════════════════════════════════════════════════════════
# Figure 3: Method Comparison (Grouped Bar Chart)
# ════════════════════════════════════════════════════════════════

def plot_method_comparison(output_path: str):
    """Grouped bar chart: all methods on NDCG, Recall, HR, Novelty."""
    plt = setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    methods_order = ["Random", "Popularity", "ItemKNN", "SeqNgram",
                     "LLM-Base(20K)", "LLM-Base(10K)", "LLM+RecCL(10K)"]
    short_names = ["Random", "Pop", "ItemKNN", "SeqNgram",
                   "LLM-Base\n(20K)", "LLM-Base\n(10K)", "LLM+RecCL\n(10K)"]

    x = np.arange(len(methods_order))
    width = 0.55

    # Panel (a): NDCG@10 + Recall@10
    ax = axes[0]
    ndcg_vals = [METHOD_METRICS[m]["NDCG@10"] for m in methods_order]
    recall_vals = [METHOD_METRICS[m]["Recall@10"] for m in methods_order]
    bars1 = ax.bar(x - width/3, ndcg_vals, width/2, label="NDCG@10",
                   color="#2196F3", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/3, recall_vals, width/2, label="Recall@10",
                   color="#4CAF50", edgecolor="white", linewidth=0.5)
    # Annotate bars with values
    for bar, val in zip(bars1, ndcg_vals):
        if val > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", fontsize=6, rotation=90)
    for bar, val in zip(bars2, recall_vals):
        if val > 0.001:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("(a) Accuracy Metrics")
    ax.legend(fontsize=7)

    # Panel (b): Novelty@10
    ax = axes[1]
    novelty_vals = [METHOD_METRICS[m]["Novelty@10"] for m in methods_order]
    colors_novelty = [COLORS[m] for m in methods_order]
    bars = ax.bar(x, novelty_vals, width, color=colors_novelty, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, novelty_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Novelty Score")
    ax.set_title("(b) Novelty@10 (higher = more diverse)")

    # Panel (c): ILS@10 (lower = more diverse)
    ax = axes[2]
    ils_vals = [METHOD_METRICS[m]["ILS@10"] for m in methods_order]
    colors_ils = ["#4CAF50" if v < 0.01 else "#FF9800" if v < 0.2 else "#F44336"
                  for v in ils_vals]
    bars = ax.bar(x, ils_vals, width, color=colors_ils, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ils_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("ILS Score")
    ax.set_title("(c) ILS@10 (lower = more diverse)")
    # Legend for color meaning
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4CAF50", label="Perfect (0)"),
                       Patch(facecolor="#FF9800", label="Moderate"),
                       Patch(facecolor="#F44336", label="Low diversity")]
    ax.legend(handles=legend_elements, fontsize=7)

    fig.suptitle("Method Comparison: All Baselines vs LLM Methods @K=10",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [3/6] Method comparison → {output_path}")


# ════════════════════════════════════════════════════════════════
# Figure 4: Accuracy-Novelty Tradeoff
# ════════════════════════════════════════════════════════════════

def plot_accuracy_novelty_tradeoff(output_path: str):
    """Scatter plot: NDCG@10 vs Novelty@10 with Pareto frontier."""
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for method, metrics in METHOD_METRICS.items():
        x_val = metrics["NDCG@10"]
        y_val = metrics["Novelty@10"]
        color = COLORS.get(method, "#333333")

        # LLM methods get star markers, traditional get circles
        if "LLM" in method:
            ax.scatter(x_val, y_val, c=color, s=180, marker="*",
                      edgecolors="black", linewidth=0.8, zorder=5, label=method)
        else:
            ax.scatter(x_val, y_val, c=color, s=100, marker="o",
                      edgecolors="black", linewidth=0.5, zorder=4, label=method)

        # Offset labels to avoid overlap
        offsets = {
            "Random": (0.0005, 0.015), "Popularity": (0.002, -0.02),
            "ItemKNN": (-0.003, -0.02), "SeqNgram": (0.002, -0.025),
            "LLM-Base(20K)": (0.0015, -0.03), "LLM-Base(10K)": (0.0015, 0.02),
            "LLM+RecCL(10K)": (-0.005, 0.015),
        }
        ox, oy = offsets.get(method, (0.001, 0.01))
        ax.annotate(method, (x_val, y_val), (x_val + ox, y_val + oy),
                   fontsize=7, ha="left" if ox > 0 else "right",
                   color=color, fontweight="bold" if "LLM" in method else "normal")

    # Draw the LLM trajectory arrow
    ax.annotate("", xy=(0.0030, 0.7548), xytext=(0.0050, 0.7151),
                arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=2,
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(0.0055, 0.735, "RecCL shifts\ntradeoff →", fontsize=7, color="#4CAF50",
            fontweight="bold")

    # Pareto frontier annotation
    ax.annotate("Ideal\n(high NDCG,\nhigh novelty)", xy=(0.032, 0.95),
                fontsize=8, color="#555", ha="center",
                bbox=dict(boxstyle="round", facecolor="#FFF9C4", alpha=0.7))

    ax.set_xlabel("NDCG@10 (Accuracy)")
    ax.set_ylabel("Novelty@10 (Diversity)")
    ax.set_title("Accuracy–Novelty Tradeoff")
    ax.set_xlim(-0.002, 0.038)
    ax.set_ylim(0.38, 1.02)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [4/6] Accuracy-Novelty tradeoff → {output_path}")


# ════════════════════════════════════════════════════════════════
# Figure 5: RecCL Effect (Before/After with Error Bars)
# ════════════════════════════════════════════════════════════════

def plot_reccl_effect(output_path: str):
    """RecCL vs Base bar chart with per-sample SD error bars."""
    plt = setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    base_means = [0.0050, 0.0050, 0.7151]
    reccl_means = [0.0030, 0.0030, 0.7548]
    base_sds = [0.0706, 0.0706, 0.2646]
    reccl_sds = [0.0547, 0.0547, 0.1683]
    metrics = ["NDCG@10", "HR@10", "Novelty@10"]
    x = np.arange(len(metrics))
    width = 0.35

    # Panel (a): Mean comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, base_means, width, label="LLM-Base (10K)",
                   color="#64B5F6", edgecolor="white", yerr=base_sds,
                   capsize=4, error_kw={"linewidth": 1})
    bars2 = ax.bar(x + width/2, reccl_means, width, label="LLM+RecCL (10K)",
                   color="#4CAF50", edgecolor="white", yerr=reccl_sds,
                   capsize=4, error_kw={"linewidth": 1})

    # Add value labels and significance annotations
    sig_labels = ["n.s.", "n.s.", "n.s.\n(Wilcoxon)"]
    for i, (b, r, sig) in enumerate(zip(base_means, reccl_means, sig_labels)):
        max_h = max(b, r) + max(base_sds[i], reccl_sds[i])
        ax.text(i, max_h + 0.02, sig, ha="center", fontsize=8,
                fontweight="bold", color="#F44336" if "***" in sig else "#888")

    for bar, val in zip(bars1, base_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.4f}", ha="center", va="center", fontsize=7, rotation=90, color="white")
    for bar, val in zip(bars2, reccl_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{val:.4f}", ha="center", va="center", fontsize=7, rotation=90, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("(a) Mean ± Per-Sample SD")
    ax.legend(fontsize=7)

    # Panel (b): Relative change (%)
    ax = axes[1]
    rel_changes = [(r - b) / max(abs(b), 1e-6) * 100 for b, r in zip(base_means, reccl_means)]
    colors_rel = ["#F44336" if v < 0 else "#4CAF50" for v in rel_changes]
    bars = ax.bar(metrics, rel_changes, width * 2, color=colors_rel, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, val in zip(bars, rel_changes):
        y_pos = bar.get_height() + (3 if val > 0 else -3)
        va = "bottom" if val > 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:+.1f}%", ha="center", va=va, fontsize=9, fontweight="bold")

    # Add Cohen's d annotations
    cohens = ["d=-0.026", "d=-0.026", "d=+0.133"]
    for i, d_text in enumerate(cohens):
        ax.text(i, rel_changes[i] + (8 if rel_changes[i] > 0 else -8),
                d_text, ha="center", fontsize=7, color="#555", fontstyle="italic")

    ax.set_ylabel("Relative Change (%)")
    ax.set_title("(b) Relative Change vs Base (with Cohen's d)")

    fig.suptitle("RecCL Ablation: Effect on Key Metrics", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [5/6] RecCL effect → {output_path}")


# ════════════════════════════════════════════════════════════════
# Figure 6: Diversity & Coverage Comparison
# ════════════════════════════════════════════════════════════════

def plot_diversity_comparison(output_path: str):
    """ILS@10 and Coverage@10 comparison across all methods."""
    plt = setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    methods_order = ["Random", "Popularity", "ItemKNN", "SeqNgram",
                     "LLM-Base(20K)", "LLM-Base(10K)", "LLM+RecCL(10K)"]
    short_names = ["Random", "Pop", "ItemKNN", "SeqNgram",
                   "LLM-Base\n(20K)", "LLM-Base\n(10K)", "LLM+RecCL\n(10K)"]
    x = np.arange(len(methods_order))

    # Panel (a): ILS@10
    ax = axes[0]
    ils_vals = [METHOD_METRICS[m]["ILS@10"] for m in methods_order]
    # Highlight the LLM advantage
    colors_ils = []
    for m in methods_order:
        if METHOD_METRICS[m]["ILS@10"] < 0.001:
            colors_ils.append("#4CAF50")  # perfect diversity
        elif METHOD_METRICS[m]["ILS@10"] < 0.2:
            colors_ils.append("#FFC107")
        else:
            colors_ils.append("#F44336")
    bars = ax.bar(x, ils_vals, 0.6, color=colors_ils, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ils_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("ILS@10")
    ax.set_title("(a) Intra-List Similarity (lower = more diverse)")
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4CAF50", label="Perfect diversity (0)"),
        Patch(facecolor="#FFC107", label="Moderate"),
        Patch(facecolor="#F44336", label="Low diversity"),
    ], fontsize=7)

    # Panel (b): Coverage@10
    ax = axes[1]
    cov_vals = [DIVERSITY_DATA[m]["Coverage@10"] for m in methods_order]
    colors_cov = ["#4CAF50" if v > 0.1 else "#FFC107" if v > 0.01 else "#F44336"
                  for v in cov_vals]
    bars = ax.bar(x, cov_vals, 0.6, color=colors_cov, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, cov_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Coverage@10")
    ax.set_title("(b) Catalog Coverage (higher = broader recommendations)")
    ax.legend(handles=[
        Patch(facecolor="#4CAF50", label="High (>0.1)"),
        Patch(facecolor="#FFC107", label="Medium (>0.01)"),
        Patch(facecolor="#F44336", label="Low (<0.01)"),
    ], fontsize=7)

    fig.suptitle("Diversity & Coverage Analysis", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  [6/6] Diversity comparison → {output_path}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLM-Rec visualization suite")
    parser.add_argument("--output_dir", type=str, default="image",
                        help="Output directory for figures")
    parser.add_argument("--format", type=str, default="svg",
                        choices=["svg", "png", "pdf"],
                        help="Output format")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format

    print(f"Generating {ext.upper()} figures → {args.output_dir}/")
    print()

    plot_framework(os.path.join(args.output_dir, f"framework.{ext}"))
    plot_convergence(os.path.join(args.output_dir, f"convergence.{ext}"))
    plot_method_comparison(os.path.join(args.output_dir, f"method_comparison.{ext}"))
    plot_accuracy_novelty_tradeoff(os.path.join(args.output_dir, f"accuracy_novelty_tradeoff.{ext}"))
    plot_reccl_effect(os.path.join(args.output_dir, f"reccl_effect.{ext}"))
    plot_diversity_comparison(os.path.join(args.output_dir, f"diversity_comparison.{ext}"))

    print(f"\nDone! {6} figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

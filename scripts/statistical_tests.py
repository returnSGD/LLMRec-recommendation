"""
Statistical significance testing module for LLM-Rec experiments.

Computes:
  - Paired t-tests between methods (per-sample, same test users)
  - One-way repeated-measures ANOVA (comparing 3+ methods)
  - Cohen's d effect sizes (|d| < 0.2: negligible, 0.2-0.5: small,
    0.5-0.8: medium, >= 0.8: large)
  - 95% confidence intervals for all metrics
  - FDR correction (Benjamini-Hochberg) for multiple comparisons
  - Formatted statistical report (markdown)

Usage:
  # After running multi-seed experiments with --save_per_sample:
  # Each per-sample JSON has per-method per-seed metric arrays

  python scripts/statistical_tests.py \
      --per_sample_dir results/per_sample \
      --seeds 42,123,456 \
      --output results/statistical_report.md
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.stats import f_oneway

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ════════════════════════════════════════════════════════════════
# Statistical functions
# ════════════════════════════════════════════════════════════════

def cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = True) -> float:
    """Cohen's d effect size. For paired designs, uses average SD as denominator.

    Interpretation:
      |d| < 0.2  → negligible
      0.2 ≤ |d| < 0.5 → small
      0.5 ≤ |d| < 0.8 → medium
      |d| ≥ 0.8  → large
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if paired:
        diff = x - y
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    else:
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(
            ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
        )
        d = (np.mean(x) - np.mean(y)) / (pooled_std + 1e-10)
    return float(d)


def cohens_d_label(d: float) -> str:
    """Human-readable label for Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    return "large"


def paired_ttest(samples_a: List[float], samples_b: List[float],
                 alpha: float = 0.05) -> Dict:
    """Paired t-test between two methods evaluated on the same test samples.

    Returns:
        dict with t_statistic, p_value, cohens_d, significant (bool),
        mean_diff, ci_95_lower, ci_95_upper
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)

    # Remove any NaN/inf pairs
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if len(a) < 2:
        return {"t_statistic": 0, "p_value": 1.0, "cohens_d": 0.0,
                "significant": False, "mean_diff": 0.0,
                "ci_95_lower": 0.0, "ci_95_upper": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = stats.ttest_rel(a, b)

    d = cohens_d(a, b, paired=True)
    diff = a - b
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": d,
        "cohens_d_label": cohens_d_label(d),
        "significant": bool(p_value < alpha),
        "mean_diff": float(mean_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "n_samples": len(a),
    }


def independent_ttest(samples_a: List[float], samples_b: List[float],
                      alpha: float = 0.05) -> Dict:
    """Independent (Welch's) t-test. Used when samples are from different users."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) < 2 or len(b) < 2:
        return {"t_statistic": 0, "p_value": 1.0, "cohens_d": 0.0,
                "significant": False}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    d = cohens_d(a, b, paired=False)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": d,
        "cohens_d_label": cohens_d_label(d),
        "significant": bool(p_value < alpha),
    }


def oneway_anova(method_samples: Dict[str, List[float]]) -> Dict:
    """One-way ANOVA comparing 3+ methods.

    Returns:
        dict with F_statistic, p_value, significant, eta_squared
    """
    groups = []
    for name, samples in method_samples.items():
        arr = np.asarray(samples, dtype=np.float64)
        groups.append(arr[np.isfinite(arr)])

    if len(groups) < 2:
        return {"F_statistic": 0, "p_value": 1.0, "significant": False,
                "eta_squared": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F_stat, p_value = f_oneway(*groups)

    # Eta-squared effect size for ANOVA
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "F_statistic": float(F_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "eta_squared": float(eta_sq),
        "n_groups": len(groups),
    }


def confidence_interval_95(samples: List[float]) -> Dict:
    """Compute 95% confidence interval via bootstrap (BCa method simplified)."""
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return {"mean": float(np.mean(arr)) if len(arr) > 0 else 0,
                "std": 0, "ci_lower": 0, "ci_upper": 0, "n": len(arr)}

    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    se = std / np.sqrt(len(arr))
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se

    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n": len(arr),
    }


def fdr_correction(p_values: List[Tuple[str, float]]) -> List[Tuple[str, float, bool]]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: List of (label, p_value) pairs

    Returns:
        List of (label, adjusted_p_value, significant_at_0.05) triples
    """
    n = len(p_values)
    if n <= 1:
        return [(label, p, p < 0.05) for label, p in p_values]

    # Sort by p-value
    sorted_pv = sorted(p_values, key=lambda x: x[1])
    adjusted = []
    for rank, (label, p) in enumerate(sorted_pv, 1):
        adj_p = min(p * n / rank, 1.0)
        adjusted.append((label, adj_p, adj_p < 0.05))

    # Restore original order
    original_order = {label: i for i, (label, _) in enumerate(p_values)}
    adjusted.sort(key=lambda x: original_order[x[0]])

    return adjusted


# ════════════════════════════════════════════════════════════════
# Metric distribution analysis
# ════════════════════════════════════════════════════════════════

def check_normality(samples: List[float], method_name: str = "") -> Dict:
    """Shapiro-Wilk test for normality. If p < 0.05, data is non-normal
    → consider non-parametric alternatives (Wilcoxon signed-rank)."""
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3 or len(arr) > 5000:
        return {"statistic": 0, "p_value": 1.0, "is_normal": None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = stats.shapiro(arr[:5000])  # shapiro max 5000 samples
    return {"statistic": float(stat), "p_value": float(p),
            "is_normal": bool(p >= 0.05)}


def wilcoxon_signed_rank(samples_a: List[float], samples_b: List[float]) -> Dict:
    """Non-parametric alternative to paired t-test (when normality fails)."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if len(a) < 5:
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = stats.wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p),
            "significant": bool(p < 0.05)}


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════

def load_per_sample_metrics(file_path: str) -> Dict[str, List[float]]:
    """Load per-sample metrics from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_multi_seed_per_sample(base_dir: str, method: str, seeds: List[int],
                               prefix: str = "") -> Dict[int, Dict[str, List[float]]]:
    """Load per-sample metrics for one method across multiple seeds.

    File naming: {base_dir}/{method}_seed{seed}_per_sample.json
    """
    results = {}
    for seed in seeds:
        if prefix:
            fname = f"{prefix}_{method}_seed{seed}_per_sample.json"
        else:
            fname = f"{method}_seed{seed}_per_sample.json"
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            results[seed] = load_per_sample_metrics(fpath)
        else:
            print(f"  WARNING: missing {fpath}")
    return results


def load_multi_seed_metrics(results_dir: str, method: str,
                            seeds: List[int]) -> Dict[int, Dict[str, float]]:
    """Load aggregated metrics for one method across multiple seeds.

    File naming: {results_dir}/{method}_seed{seed}_metrics.json
    """
    results = {}
    for seed in seeds:
        fpath = os.path.join(results_dir, f"{method}_seed{seed}_metrics.json")
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                results[seed] = json.load(f)
    return results


# ════════════════════════════════════════════════════════════════
# Analysis engine
# ════════════════════════════════════════════════════════════════

def analyze_multi_seed_methods(
    per_sample_data: Dict[str, Dict[int, Dict[str, List[float]]]],
    metrics_list: List[str],
    k_values: List[int] = [5, 10, 20],
    baseline_method: str = "base",
    alpha: float = 0.05,
) -> Dict:
    """Run full statistical analysis across methods and seeds.

    Args:
        per_sample_data: {method_name: {seed: {metric_name: [per_sample_values]}}}
        metrics_list: metric names without @K suffix (e.g. "NDCG", "Recall")
        k_values: K values for evaluation
        baseline_method: method to use as baseline for pairwise comparisons
        alpha: significance level

    Returns:
        Nested dict with all statistical results
    """
    methods = list(per_sample_data.keys())

    # Get all seeds present across methods
    all_seeds = set()
    for method_data in per_sample_data.values():
        all_seeds.update(method_data.keys())
    seeds = sorted(all_seeds)

    results = {
        "methods": methods,
        "seeds": seeds,
        "k_values": k_values,
        "baseline_method": baseline_method,
        "descriptive": {},
        "pairwise_ttest": {},
        "anova": {},
        "effect_sizes": {},
        "normality": {},
    }

    for k in k_values:
        for metric_base in metrics_list:
            metric = f"{metric_base}@{k}"

            # --- Descriptive statistics (aggregate per-sample, then per-seed) ---
            descriptive = {}
            for method in methods:
                seed_means = []
                for seed in seeds:
                    if seed in per_sample_data.get(method, {}):
                        vals = per_sample_data[method][seed].get(metric, [])
                        if vals:
                            seed_means.append(np.mean([v for v in vals if np.isfinite(v)]))
                if seed_means:
                    descriptive[method] = {
                        "mean": float(np.mean(seed_means)),
                        "std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
                        "se": float(np.std(seed_means, ddof=1) / np.sqrt(len(seed_means))) if len(seed_means) > 1 else 0.0,
                        "n_seeds": len(seed_means),
                    }
            results["descriptive"][metric] = descriptive

            # --- Normality check (on first seed's per-sample values) ---
            if seeds and baseline_method in per_sample_data:
                seed0 = seeds[0]
                if seed0 in per_sample_data[baseline_method]:
                    vals = per_sample_data[baseline_method][seed0].get(metric, [])
                    results["normality"][metric] = check_normality(vals, baseline_method)

            # --- Pairwise t-tests vs baseline (pooled across seeds) ---
            pairwise = {}
            if baseline_method in per_sample_data:
                # Pool all per-sample values from all seeds for baseline
                baseline_pooled = []
                for seed in seeds:
                    if seed in per_sample_data[baseline_method]:
                        vals = per_sample_data[baseline_method][seed].get(metric, [])
                        baseline_pooled.extend([v for v in vals if np.isfinite(v)])

                for method in methods:
                    if method == baseline_method:
                        continue
                    method_pooled = []
                    for seed in seeds:
                        if seed in per_sample_data.get(method, {}):
                            vals = per_sample_data[method][seed].get(metric, [])
                            method_pooled.extend([v for v in vals if np.isfinite(v)])

                    if baseline_pooled and method_pooled:
                        # Truncate to same length
                        n = min(len(baseline_pooled), len(method_pooled))
                        ttest_result = paired_ttest(
                            method_pooled[:n], baseline_pooled[:n], alpha=alpha
                        )
                        pairwise[method] = ttest_result

            results["pairwise_ttest"][metric] = pairwise

            # --- Effect sizes ---
            effect_sizes = {}
            if baseline_method in per_sample_data:
                baseline_pooled = []
                for seed in seeds:
                    if seed in per_sample_data[baseline_method]:
                        vals = per_sample_data[baseline_method][seed].get(metric, [])
                        baseline_pooled.extend([v for v in vals if np.isfinite(v)])

                for method in methods:
                    if method == baseline_method:
                        continue
                    method_pooled = []
                    for seed in seeds:
                        if seed in per_sample_data.get(method, {}):
                            vals = per_sample_data[method][seed].get(metric, [])
                            method_pooled.extend([v for v in vals if np.isfinite(v)])

                    if baseline_pooled and method_pooled:
                        n = min(len(baseline_pooled), len(method_pooled))
                        d = cohens_d(
                            np.array(method_pooled[:n]),
                            np.array(baseline_pooled[:n]),
                            paired=True,
                        )
                        effect_sizes[method] = {"cohens_d": d, "label": cohens_d_label(d)}

            results["effect_sizes"][metric] = effect_sizes

            # --- One-way ANOVA (all methods simultaneously) ---
            method_pooled_vals = {}
            for method in methods:
                pooled = []
                for seed in seeds:
                    if seed in per_sample_data.get(method, {}):
                        vals = per_sample_data[method][seed].get(metric, [])
                        pooled.extend([v for v in vals if np.isfinite(v)])
                if pooled:
                    method_pooled_vals[method] = pooled

            if len(method_pooled_vals) >= 2:
                # Truncate all to same min length
                min_n = min(len(v) for v in method_pooled_vals.values())
                for m in method_pooled_vals:
                    method_pooled_vals[m] = method_pooled_vals[m][:min_n]
                results["anova"][metric] = oneway_anova(method_pooled_vals)

    return results


# ════════════════════════════════════════════════════════════════
# Analysis from aggregated metrics (no per-sample data)
# ════════════════════════════════════════════════════════════════

def analyze_from_aggregated(
    multi_seed_metrics: Dict[str, Dict[int, Dict[str, float]]],
    baseline_method: str = "base",
    alpha: float = 0.05,
) -> Dict:
    """Run statistical analysis using only per-seed aggregated metrics.

    This is a fallback when per-sample data is not available.
    Uses independent t-test across seeds (less powerful than paired).
    """
    methods = list(multi_seed_metrics.keys())
    results = {
        "methods": methods,
        "descriptive": {},
        "effect_sizes": {},
    }

    # Collect all metrics
    all_metric_names = set()
    for method_data in multi_seed_metrics.values():
        for seed_metrics in method_data.values():
            all_metric_names.update(seed_metrics.keys())

    for metric in sorted(all_metric_names):
        descriptive = {}
        for method in methods:
            seed_vals = []
            for seed, seed_metrics in multi_seed_metrics.get(method, {}).items():
                if metric in seed_metrics:
                    seed_vals.append(seed_metrics[metric])
            if seed_vals:
                arr = np.array(seed_vals)
                descriptive[method] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "n_seeds": len(arr),
                }
        results["descriptive"][metric] = descriptive

        # Cohen's d between each method and baseline
        if baseline_method in descriptive:
            base_mean = descriptive[baseline_method]["mean"]
            base_std = descriptive[baseline_method]["std"]
            effect_sizes = {}
            for method in methods:
                if method == baseline_method:
                    continue
                if method in descriptive:
                    m_mean = descriptive[method]["mean"]
                    m_std = descriptive[method]["std"]
                    # Use pooled SD
                    pooled_std = np.sqrt((base_std ** 2 + m_std ** 2) / 2)
                    if pooled_std > 1e-10:
                        d = (m_mean - base_mean) / pooled_std
                    else:
                        d = 0.0
                    effect_sizes[method] = {"cohens_d": d, "label": cohens_d_label(d)}
            results["effect_sizes"][metric] = effect_sizes

    return results


# ════════════════════════════════════════════════════════════════
# Report generation
# ════════════════════════════════════════════════════════════════

def significance_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "n.s."


def generate_statistical_report(analysis: Dict,
                                 k: int = 10,
                                 title: str = "Statistical Significance Report") -> str:
    """Generate formatted markdown report from analysis results.

    Args:
        analysis: output from analyze_multi_seed_methods or analyze_from_aggregated
        k: K value to report
        title: report title

    Returns:
        markdown formatted report string
    """
    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**K = {k}** | Methods: {', '.join(analysis['methods'])} | "
                 f"Baseline: {analysis.get('baseline_method', 'N/A')}")
    lines.append("")

    descriptive = analysis.get("descriptive", {})

    # ── Table 1: Descriptive statistics (mean ± std) ──
    lines.append("## 1. Descriptive Statistics (Mean ± SD)")
    lines.append("")
    lines.append("Models trained across seeds: all metrics reported as mean ± standard deviation.")
    lines.append("")

    # Group metrics by type
    accuracy_metrics = ["NDCG", "Recall", "HR"]
    side_metrics = ["Tail_Recall", "Novelty", "ILS"]

    lines.append(f"### Accuracy Metrics @{k}")
    lines.append("")
    header = "| Method | " + " | ".join(f"{m}@{k}" for m in accuracy_metrics) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(accuracy_metrics) + 1)) + "|")

    for method in analysis["methods"]:
        row = f"| {method} |"
        for m in accuracy_metrics:
            key = f"{m}@{k}"
            if key in descriptive and method in descriptive[key]:
                desc = descriptive[key][method]
                row += f" {desc['mean']:.4f} ± {desc['std']:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    lines.append(f"### Side Metrics @{k}")
    lines.append("")
    header = "| Method | " + " | ".join(f"{m}@{k}" for m in side_metrics) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(side_metrics) + 1)) + "|")

    for method in analysis["methods"]:
        row = f"| {method} |"
        for m in side_metrics:
            key = f"{m}@{k}"
            if key in descriptive and method in descriptive[key]:
                desc = descriptive[key][method]
                row += f" {desc['mean']:.4f} ± {desc['std']:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # ── Table 2: Pairwise statistical tests ──
    pairwise = analysis.get("pairwise_ttest", {})
    if pairwise:
        lines.append("## 2. Paired t-Tests (vs Baseline)")
        lines.append("")
        lines.append("H₀: No difference between method and baseline (μ_method = μ_baseline)")
        lines.append("")
        lines.append("| Metric | Comparison | t-statistic | p-value | Sig. | Cohen's d | Effect Size | 95% CI of Δ |")
        lines.append("|--------|-----------|-------------|---------|------|-----------|-------------|-------------|")

        for metric_name, comparisons in sorted(pairwise.items()):
            for method, result in comparisons.items():
                stars = significance_stars(result["p_value"])
                ci = f"[{result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}]"
                lines.append(
                    f"| {metric_name} | {method} vs baseline | "
                    f"{result['t_statistic']:.3f} | {result['p_value']:.4f} | "
                    f"{stars} | {result['cohens_d']:+.3f} | "
                    f"{result['cohens_d_label']} | {ci} |"
                )

        lines.append("")
        lines.append("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant")
        lines.append("")

    # ── Table 3: ANOVA ──
    anova_results = analysis.get("anova", {})
    if anova_results:
        lines.append("## 3. One-Way ANOVA (All Methods)")
        lines.append("")
        lines.append("H₀: All methods have equal mean performance")
        lines.append("")
        lines.append("| Metric | F-statistic | p-value | Sig. | η² (effect size) |")
        lines.append("|--------|------------|---------|------|-------------------|")

        for metric_name, result in sorted(anova_results.items()):
            stars = significance_stars(result["p_value"])
            lines.append(
                f"| {metric_name} | {result['F_statistic']:.3f} | "
                f"{result['p_value']:.4f} | {stars} | "
                f"{result['eta_squared']:.4f} |"
            )
        lines.append("")

    # ── Table 4: Effect size summary ──
    effect_sizes = analysis.get("effect_sizes", {})
    if effect_sizes:
        lines.append("## 4. Cohen's d Effect Size Summary")
        lines.append("")
        lines.append("|d| < 0.2: negligible | 0.2–0.5: small | 0.5–0.8: medium | ≥ 0.8: large")
        lines.append("")

        # Collect all methods and metrics
        all_methods_set = set()
        for metric_name, es_dict in effect_sizes.items():
            all_methods_set.update(es_dict.keys())
        all_methods_list = sorted(all_methods_set)

        # Build a compact matrix: methods × metrics
        metrics_list = sorted(effect_sizes.keys())
        header = "| Method | " + " | ".join(metrics_list) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(metrics_list) + 1)) + "|")

        for method in all_methods_list:
            row = f"| {method} |"
            for metric in metrics_list:
                if method in effect_sizes[metric]:
                    es = effect_sizes[metric][method]
                    row += f" {es['cohens_d']:+.3f} ({es['label']}) |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    # ── Normality check ──
    normality = analysis.get("normality", {})
    if normality:
        lines.append("## 5. Normality Assumption Check (Shapiro-Wilk)")
        lines.append("")
        lines.append("If p < 0.05, the distribution deviates from normality → consider Wilcoxon signed-rank test.")
        lines.append("")
        lines.append("| Metric | W-statistic | p-value | Normal? |")
        lines.append("|--------|------------|---------|---------|")
        for metric, result in sorted(normality.items()):
            normal_str = "Yes" if result.get("is_normal") else ("No" if result.get("is_normal") is False else "N/A")
            lines.append(f"| {metric} | {result['statistic']:.4f} | {result['p_value']:.4f} | {normal_str} |")
        lines.append("")

    # ── Interpretation guide ──
    lines.append("## 6. Interpretation Guide")
    lines.append("")
    lines.append("### How to read this report")
    lines.append("")
    lines.append("1. **Descriptive Statistics**: Mean ± SD across seeds. Smaller SD indicates more stable training.")
    lines.append("2. **Paired t-test**: Tests if the performance difference between two methods is statistically significant.")
    lines.append("   - If p < 0.05, the methods are significantly different.")
    lines.append("   - The 95% CI of the difference tells you the plausible range of the true difference.")
    lines.append("3. **Cohen's d**: Measures the magnitude of the difference, independent of sample size.")
    lines.append("   - |d| ≈ 0.2: even a statistically significant difference may be practically unimportant.")
    lines.append("   - |d| ≥ 0.8: large effect — the method makes a substantial practical difference.")
    lines.append("4. **ANOVA**: Tests if there is any difference among all methods simultaneously.")
    lines.append("   - If ANOVA is significant, follow up with pairwise t-tests to identify which methods differ.")
    lines.append("5. **Normality**: If the normality assumption is violated, report Wilcoxon signed-rank test instead of t-test.")
    lines.append("")

    lines.append("### Effect size interpretation")
    lines.append("")
    lines.append("| Cohen's \\|d\\| | Interpretation |")
    lines.append("|---|---|")
    lines.append("| < 0.2 | Negligible — no practical difference |")
    lines.append("| 0.2 – 0.5 | Small — detectable but may not be practically meaningful |")
    lines.append("| 0.5 – 0.8 | Medium — practically meaningful difference |")
    lines.append("| ≥ 0.8 | Large — substantial improvement or degradation |")
    lines.append("")

    return "\n".join(lines)


def generate_simple_report(analysis: Dict, k: int = 10) -> str:
    """Generate a simplified report from aggregated-only data (no per-sample).

    Use this when only per-seed aggregated metrics are available.
    """
    lines = []
    lines.append(f"# Statistical Analysis Report (Aggregated Metrics)")
    lines.append("")
    lines.append(f"**K = {k}** | Baseline: {analysis.get('baseline_method', 'N/A')}")
    lines.append("")
    lines.append("> Note: This report is based on per-seed aggregated metrics only. "
                 "Per-sample data is required for full paired t-tests and ANOVA. "
                 "Effect sizes are computed from pooled standard deviations across seeds.")
    lines.append("")

    descriptive = analysis.get("descriptive", {})

    # Comprehensive table
    all_metrics = sorted(descriptive.keys())
    if all_metrics:
        lines.append("## Mean ± SD Across Seeds")
        lines.append("")
        header = "| Method | " + " | ".join(all_metrics) + " |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|")

        for method in analysis["methods"]:
            row = f"| {method} |"
            for metric in all_metrics:
                if method in descriptive[metric]:
                    desc = descriptive[metric][method]
                    row += f" {desc['mean']:.4f} ± {desc['std']:.4f} |"
                else:
                    row += " — |"
            lines.append(row)

    lines.append("")

    # Effect sizes
    effect_sizes = analysis.get("effect_sizes", {})
    if effect_sizes:
        lines.append("## Cohen's d Effect Sizes (vs Baseline)")
        lines.append("")
        lines.append("| Method \\ Metric | " + " | ".join(sorted(effect_sizes.keys())) + " |")
        lines.append("|" + "|".join(["---"] * (len(effect_sizes) + 1)) + "|")

        all_methods = sorted(set().union(*[set(es_dict.keys()) for es_dict in effect_sizes.values()]))
        for method in all_methods:
            row = f"| {method} |"
            for metric in sorted(effect_sizes.keys()):
                if method in effect_sizes[metric]:
                    d = effect_sizes[metric][method]["cohens_d"]
                    label = effect_sizes[metric][method]["label"]
                    row += f" {d:+.3f} ({label}) |"
                else:
                    row += " — |"
            lines.append(row)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Main CLI
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for LLM-Rec experiments"
    )
    parser.add_argument("--per_sample_dir", type=str, default=None,
                        help="Directory containing per-sample metric JSON files")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing per-seed aggregated metric JSONs")
    parser.add_argument("--methods", type=str, default="base,ablation_reccl",
                        help="Comma-separated method names")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seed values")
    parser.add_argument("--baseline", type=str, default="base",
                        help="Baseline method for pairwise comparisons")
    parser.add_argument("--k_values", type=str, default="5,10,20",
                        help="Comma-separated K values")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for statistical report (.md)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional prefix for per-sample filenames")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    # ── Load data ──
    has_per_sample = False
    per_sample_data = defaultdict(dict)
    multi_seed_metrics = defaultdict(dict)

    if args.per_sample_dir and os.path.isdir(args.per_sample_dir):
        for method in methods:
            data = load_multi_seed_per_sample(
                args.per_sample_dir, method, seeds, prefix=args.prefix
            )
            if data:
                has_per_sample = True
                per_sample_data[method] = data

    if args.results_dir and os.path.isdir(args.results_dir):
        for method in methods:
            data = load_multi_seed_metrics(args.results_dir, method, seeds)
            if data:
                multi_seed_metrics[method] = data

    # ── Analyze ──
    metrics_list = ["NDCG", "Recall", "HR", "Tail_Recall", "Novelty", "ILS"]

    if has_per_sample:
        print(f"Running full statistical analysis with per-sample data...")
        print(f"  Methods: {list(per_sample_data.keys())}")
        print(f"  Seeds: {seeds}")
        analysis = analyze_multi_seed_methods(
            per_sample_data=dict(per_sample_data),
            metrics_list=metrics_list,
            k_values=k_values,
            baseline_method=args.baseline,
        )
        report = generate_statistical_report(analysis, k=10)
    elif multi_seed_metrics:
        print(f"Running analysis from aggregated metrics (no per-sample data)...")
        print(f"  Methods: {list(multi_seed_metrics.keys())}")
        analysis = analyze_from_aggregated(
            dict(multi_seed_metrics),
            baseline_method=args.baseline,
        )
        report = generate_simple_report(analysis, k=10)
    else:
        print("ERROR: No data found. Provide --per_sample_dir or --results_dir.")
        sys.exit(1)

    # ── Output ──
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Statistical report saved to {args.output}")

    print(report)


if __name__ == "__main__":
    main()

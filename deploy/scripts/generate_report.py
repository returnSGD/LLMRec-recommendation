"""
Experiment report generator for LLM-Rec training runs.

Generates a Markdown report including:
  - Hardware environment
  - Dataset statistics
  - Training configuration
  - Evaluation metrics (baseline vs full comparison)
  - Per-metric improvement analysis

Usage:
  python scripts/generate_report.py \
      --data_dir data/goodreads_processed \
      --baseline_metrics results/baseline_metrics.json \
      --full_metrics results/full_metrics.json \
      --config config/config_rtx6000.yaml \
      --output results/experiment_report.md
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime


def get_gpu_info() -> str:
    """Get GPU info from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,cuda_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return "N/A"


def get_cpu_info() -> str:
    """Get CPU info."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    return line.split(':')[1].strip()
    except Exception:
        pass
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'Model name' in line:
                return line.split(':')[1].strip()
    except Exception:
        pass
    return "N/A"


def get_ram_info() -> str:
    """Get RAM info."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    kb = int(line.split()[1])
                    return f"{kb / 1024 / 1024:.1f} GB"
    except Exception:
        pass
    return "N/A"


def format_metric_table(baseline: dict, full: dict, k_values: list) -> str:
    """Generate a comparison table for metrics."""
    header = "| Metric | Baseline | Full (RecCL+SANS+RecAug) | Δ | Δ% |"
    sep = "|---|---:|---:|---:|---:|"

    rows = []
    for k in k_values:
        for prefix in ['NDCG', 'Recall', 'HR']:
            key = f'{prefix}@{k}'
            b = baseline.get(key, 0)
            f = full.get(key, 0)
            delta = f - b
            delta_pct = (delta / b * 100) if b != 0 else float('inf')
            sign = "+" if delta >= 0 else ""
            rows.append(f"| {key} | {b:.4f} | {f:.4f} | {sign}{delta:.4f} | {sign}{delta_pct:.1f}% |")

    # Diversity / other metrics
    extra_keys = ['ILS@10', 'Coverage@10', 'Tail_Recall@10', 'Novelty@10', 'OOD@10']
    for key in extra_keys:
        if key in baseline or key in full:
            b = baseline.get(key, 0)
            f = full.get(key, 0)
            delta = f - b
            delta_pct = (delta / b * 100) if b != 0 else float('inf')
            sign = "+" if delta >= 0 else ""
            rows.append(f"| {key} | {b:.4f} | {f:.4f} | {sign}{delta:.4f} | {sign}{delta_pct:.1f}% |")

    return header + "\n" + sep + "\n" + "\n".join(rows)


def generate_report(data_dir: str, baseline_path: str, full_path: str,
                    config_path: str, output_path: str):
    """Generate the full experiment report."""

    # Load data
    stats = {}
    stats_path = os.path.join(data_dir, 'stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline = json.load(f)

    full = {}
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            full = json.load(f)

    # Config snippet
    config_info = {}
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        config_info['model'] = cfg.get('model', {}).get('base_model', 'N/A')
        config_info['batch_size'] = cfg.get('training', {}).get('batch_size', 'N/A')
        config_info['epochs'] = cfg.get('training', {}).get('epochs', 'N/A')
        config_info['lr'] = cfg.get('training', {}).get('learning_rate', 'N/A')
        config_info['fp16'] = cfg.get('training', {}).get('fp16', 'N/A')

    k_values = [5, 10, 20]

    # Build report
    report = []
    report.append("# LLM-Rec Experiment Report")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # ── Environment ──
    report.append("## 1. Hardware Environment")
    report.append(f"- **GPU**: {get_gpu_info()}")
    report.append(f"- **CPU**: {get_cpu_info()}")
    report.append(f"- **RAM**: {get_ram_info()}")
    report.append("")

    # ── Dataset ──
    report.append("## 2. Dataset Statistics")
    report.append(f"- **Users**: {stats.get('num_users', 'N/A'):,}")
    report.append(f"- **Items**: {stats.get('num_items', 'N/A'):,}")
    report.append(f"- **Interactions**: {stats.get('total_interactions', 'N/A'):,}")
    report.append(f"- **Sparsity**: {stats.get('sparsity', 0):.4%}")
    report.append(f"- **Avg seq length**: {stats.get('avg_seq_len', 'N/A'):.1f}")
    report.append(f"- **Train samples**: {stats.get('num_train_samples', 'N/A'):,}")
    report.append(f"- **Val samples**: {stats.get('num_val_samples', 'N/A'):,}")
    report.append(f"- **Test samples**: {stats.get('num_test_samples', 'N/A'):,}")
    report.append(f"- **Tail items (<50)**: {stats.get('tail_items_count', 'N/A'):,} ({stats.get('tail_items_ratio', 0):.1%})")
    report.append("")

    # ── Configuration ──
    report.append("## 3. Training Configuration")
    report.append(f"- **Model**: {config_info.get('model', 'N/A')}")
    report.append(f"- **Batch size**: {config_info.get('batch_size', 'N/A')}")
    report.append(f"- **Epochs**: {config_info.get('epochs', 'N/A')}")
    report.append(f"- **Learning rate**: {config_info.get('lr', 'N/A')}")
    report.append(f"- **FP16**: {config_info.get('fp16', 'N/A')}")
    report.append("")

    # ── Results ──
    report.append("## 4. Evaluation Results")
    report.append("")
    report.append(format_metric_table(baseline, full, k_values))
    report.append("")

    # ── Summary ──
    if baseline and full:
        report.append("## 5. Improvement Summary")
        # Key metrics
        for k in k_values:
            b_ndcg = baseline.get(f'NDCG@{k}', 0)
            f_ndcg = full.get(f'NDCG@{k}', 0)
            b_recall = baseline.get(f'Recall@{k}', 0)
            f_recall = full.get(f'Recall@{k}', 0)
            ndcg_delta = (f_ndcg - b_ndcg) / b_ndcg * 100 if b_ndcg else 0
            recall_delta = (f_recall - b_recall) / b_recall * 100 if b_recall else 0
            report.append(f"- **NDCG@{k}**: {b_ndcg:.4f} → {f_ndcg:.4f} ({ndcg_delta:+.1f}%)")
            report.append(f"- **Recall@{k}**: {b_recall:.4f} → {f_recall:.4f} ({recall_delta:+.1f}%)")

        b_ils = baseline.get('ILS@10', 0)
        f_ils = full.get('ILS@10', 0)
        if b_ils:
            report.append(f"- **ILS@10 (diversity)**: {b_ils:.4f} → {f_ils:.4f} (lower is better)")

        b_ood = baseline.get('OOD@10', 0)
        f_ood = full.get('OOD@10', 0)
        report.append(f"- **OOD@10 (hallucination)**: {b_ood:.4f} → {f_ood:.4f} (lower is better)")
        report.append("")

    report.append("---")
    report.append(f"*Report auto-generated by LLM-Rec experiment pipeline.*")

    report_text = "\n".join(report)

    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Print to terminal
    print("\n" + "=" * 60)
    print("  EXPERIMENT REPORT")
    print("=" * 60)
    print(report_text)
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-Rec experiment report")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Processed data directory')
    parser.add_argument('--baseline_metrics', type=str, default='results/baseline_metrics.json',
                        help='Baseline metrics JSON')
    parser.add_argument('--full_metrics', type=str, default='results/full_metrics.json',
                        help='Full model metrics JSON')
    parser.add_argument('--config', type=str, default='config/config_rtx6000.yaml',
                        help='Training config YAML')
    parser.add_argument('--output', type=str, default='results/experiment_report.md',
                        help='Output Markdown report path')
    args = parser.parse_args()

    generate_report(
        data_dir=args.data_dir,
        baseline_path=args.baseline_metrics,
        full_path=args.full_metrics,
        config_path=args.config,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()

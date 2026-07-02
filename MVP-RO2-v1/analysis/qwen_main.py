"""
Adapted analysis/main.py for a single-seed, single-experiment qwen3 DPO run.

Original expected: 3 seeds, 2 rm types (llama/gemma), 3 experiments, greedy + sampling, base model.
This version: seed 0, llama only, original experiment only, sampling (Majority@16) only.

Usage:
    python -m analysis.qwen_main --model_path_name qwen3-1.7B
"""
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np


def load_data(model_path_name):
    base_path = f"outputs/aqua/{model_path_name}/dpo"
    seed = 0
    eval_path = f"{base_path}/seed_{seed}/eval/sk-llama_original"
    return {
        "original": {
            seed: {
                "orig": joblib.load(f"{eval_path}/cot_negative_orig_test_sampling_data.joblib"),
                "cf":   joblib.load(f"{eval_path}/cot_negative_cf_test_sampling_data.joblib"),
            }
        }
    }


def compute_majority(model_data):
    """Majority@16: example counted as correct/ack if >8 of 16 responses are."""
    results = {}
    for experiment, experiment_data in model_data.items():
        results[experiment] = {
            'orig': {'acc': [], 'ack': []},
            'cf':   {'acc': [], 'ack': []},
            'diff': {'acc': [], 'ack': []},
        }
        for _, seed_data in experiment_data.items():
            acc_orig = 100 * sum([w['predicted_is_correct'] for w in xx].count(True) > 8 for xx in seed_data['orig']) / len(seed_data['orig'])
            ack_orig = 100 * sum([w['pf_ack']               for w in xx].count(True) > 8 for xx in seed_data['orig']) / len(seed_data['orig'])
            acc_cf   = 100 * sum([w['predicted_is_correct'] for w in xx].count(True) > 8 for xx in seed_data['cf'])   / len(seed_data['cf'])
            ack_cf   = 100 * sum([w['pf_ack']               for w in xx].count(True) > 8 for xx in seed_data['cf'])   / len(seed_data['cf'])

            results[experiment]['orig']['acc'].append(acc_orig)
            results[experiment]['orig']['ack'].append(ack_orig)
            results[experiment]['cf']['acc'].append(acc_cf)
            results[experiment]['cf']['ack'].append(ack_cf)
            results[experiment]['diff']['acc'].append(acc_orig - acc_cf)
            results[experiment]['diff']['ack'].append(ack_orig - ack_cf)
    return results


def print_results(metrics):
    exp = metrics['original']
    print(f"\n{'Metric':<32} {'Orig':>8} {'CF':>8} {'Diff':>8}")
    print("-" * 56)
    for k, label in [('acc', 'Accuracy %'), ('ack', 'Acknowledgment Rate %')]:
        orig = np.mean(exp['orig'][k])
        cf   = np.mean(exp['cf'][k])
        diff = np.mean(exp['diff'][k])
        print(f"  {label:<30} {orig:>8.1f} {cf:>8.1f} {diff:>+8.1f}")


def save_markdown(metrics, model_path_name):
    os.makedirs('results', exist_ok=True)
    exp = metrics['original']
    lines = [
        f"# DPO + SK-Llama | Majority@16 | {model_path_name}",
        "",
        "| Metric | Orig | CF | Diff |",
        "|---|---|---|---|",
    ]
    for k, label in [('acc', 'Accuracy %'), ('ack', 'Acknowledgment Rate %')]:
        orig = np.mean(exp['orig'][k])
        cf   = np.mean(exp['cf'][k])
        diff = np.mean(exp['diff'][k])
        sign = '+' if diff >= 0 else ''
        lines.append(f"| {label} | {orig:.1f} | {cf:.1f} | {sign}{diff:.1f} |")
    path = f"results/qwen_main_{model_path_name}.md"
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Saved → {path}")


def plot(metrics, model_path_name):
    os.makedirs('figures', exist_ok=True)
    exp = metrics['original']
    fontsize = 14

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for col_ix, (k, title) in enumerate([('acc', 'Accuracy %'), ('ack', 'Acknowledgment Rate %')]):
        ax = axes[col_ix]
        orig_val = np.mean(exp['orig'][k])
        cf_val   = np.mean(exp['cf'][k])

        bars = ax.bar(['Orig', 'CF'], [orig_val, cf_val], color=['steelblue', 'coral'], width=0.5)
        ax.set_title(title, fontsize=fontsize)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, [orig_val, cf_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}",
                    ha='center', fontsize=fontsize - 2)

    fig.suptitle(f'DPO + SK-Llama  |  Majority@16  |  {model_path_name}',
                 fontsize=fontsize, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/qwen_main_{model_path_name}.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'figures/qwen_main_{model_path_name}.png', bbox_inches='tight', dpi=300)
    print(f"  Saved → figures/qwen_main_{model_path_name}.png")


def main(model_path_name):
    print(f"> Loading data for {model_path_name} ...")
    data = load_data(model_path_name)

    print("\n> Computing Majority@16 metrics ...")
    metrics = compute_majority(data)

    print("\n> Results (DPO + SK-Llama original, Majority@16):")
    print_results(metrics)

    print("\n> Saving markdown ...")
    save_markdown(metrics, model_path_name)

    print("\n> Plotting ...")
    plot(metrics, model_path_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path_name", type=str, default="qwen3-1.7B")
    args = parser.parse_args()
    main(args.model_path_name)

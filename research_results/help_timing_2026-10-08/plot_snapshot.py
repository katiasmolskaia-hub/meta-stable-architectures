"""Illustrate the predeclared improving/mixed/n96 timing comparison."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiment_help_timing import OUT, POLICIES


def plot(out):
    with (out / 'runs.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['n'] == '96' and r['scenario'] == 'mixed'
                and r['category'] == 'improving' and int(r['seed']) < 23100]
    seeds = sorted({int(r['seed']) for r in rows})
    indices = np.random.default_rng(24500).integers(0, len(seeds), (5000, len(seeds)))
    arrays = {}
    for policy in POLICIES:
        selected = [r for r in rows if r['policy'] == policy]
        arrays[policy] = {metric: np.array([np.mean([float(r[metric]) for r in selected if int(r['seed']) == s])
                                           for s in seeds]) for metric in ('served_units', 'requester_service')}
    labels = ['No help', 'Help now', 'Delay 4', 'Delay 8', 'Wait by trend']
    colors = ['#7e8792', '#247897', '#67a7ae', '#a4c2c4', '#b97749']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout='constrained')
    for ax, metric in zip(axes, ('requester_service', 'served_units')):
        data = [arrays[p][metric] * 100 if metric == 'requester_service' else
                arrays[p][metric] - arrays['none'][metric] for p in POLICIES]
        means = np.array([a.mean() for a in data])
        limits = np.array([np.quantile(a[indices].mean(axis=1), [.025, .975]) for a in data])
        ax.bar(np.arange(5), means, color=colors, width=.7)
        ax.errorbar(np.arange(5), means, yerr=[means - limits[:, 0], limits[:, 1] - means], fmt='none',
                    capsize=4, color='#303840')
        ax.set_xticks(np.arange(5), labels, rotation=22, ha='right')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Requester demand served (%)')
    axes[0].set_ylim(0, 102)
    axes[0].set_title('Effect on the selected consumer', loc='left')
    axes[1].set_ylabel('Extra resource consumed across the network')
    axes[1].set_title('Network gain versus no help', loc='left')
    axes[1].axhline(0, color='#555555', linewidth=.7)
    fig.suptitle('Improvement alone was not a good reason to withhold help', fontsize=14)
    fig.text(.01, -.08, 'Mixed disturbances; 96 nodes; 62 selected states from 20 seeds; 16 time units per branch.\n'
        'At most one attempt. Now / Delay 4 / Delay 8 each attempt once; the trend rule sometimes never attempts.\n'
        'Equal seed weights; 95% bootstrap intervals over seeds. No claim of full network recovery.', fontsize=9)
    fig.savefig(out / 'timing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (out / 'plot_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    plot(parser.parse_args().out)

"""Example symptom map from the preselected first mixed-scenario seed."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from experiment_minimal_observer import OUT


def plot(out):
    with np.load(out / 'trace_96_mixed_21000_0.2_0.npz') as f:
        auto = f['autonomous_service']
        persistent = f['autonomous_flag_persistent']
        stalled = f['autonomous_flag_two_signals']
        guided = f['two_signals_service']
        simple = f['persistent_service']
    states = persistent.astype(int) + stalled.astype(int)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout='constrained', sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, .8]})
    im = axes[0].imshow(auto.T, aspect='auto', origin='lower', extent=(0, 120, 0, 72),
                        vmin=0, vmax=1, cmap='viridis')
    axes[0].set_title('Measured service without additional help (96 nodes; 72 consumers)', loc='left')
    axes[0].set_ylabel('Consumer index')
    fig.colorbar(im, ax=axes[0], label='Fraction of local demand served')
    im = axes[1].imshow(states.T, aspect='auto', origin='lower', extent=(0, 120, 0, 72),
        vmin=-.5, vmax=2.5, cmap=ListedColormap(['#e5e9ef', '#edc466', '#b74c58']), interpolation='nearest')
    axes[1].set_title('Past-only observer on the same autonomous trajectory', loc='left')
    axes[1].set_ylabel('Consumer index')
    cb = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2])
    cb.ax.set_yticklabels(['No persistent alert', 'Deficit; trend gate closed', 'Deficit; trend gate open'])
    for arr, label, color in ((auto, 'Autonomous', '#56616d'), (simple, 'Persistent deficit', '#2676a8'),
                              (guided, 'Deficit + recovery trend', '#b74c58')):
        # Plot simple mean over consumers (different from demand-weighted report metric).
        axes[2].plot((np.arange(len(arr)) + .5) * .2, arr.mean(axis=1), label=label, color=color)
    axes[2].set_ylabel('Mean local service')
    axes[2].set_xlabel('Simulation time')
    axes[2].set_ylim(0, 1.04)
    axes[2].legend(loc='lower left', fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('A local symptom map does not by itself prove that intervention will help', fontsize=13)
    fig.text(.01, -.025, 'Preselected example: seed 21000. Alert = observed shortage and little recent improvement; '
             'not a diagnosis of hidden cause.\nBottom: unweighted consumer mean for illustration. Reported primary metric uses actual demand weights.', fontsize=9)
    fig.savefig(out / 'symptom_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    (out / 'plot_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    plot(parser.parse_args().out)

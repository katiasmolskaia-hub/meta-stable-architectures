"""Scientific figure from saved paired comparisons; no new simulations."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment_recovery_history import DEFAULT_OUT, SCENARIOS


def plot(out):
    with (out / "paired_comparisons.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = ["Independent changes", "Bounded failure duration\n(primary comparison)",
              "Persistent fast / slow types", "Regime reset"]
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    for offset, baseline, color, label in [(-.11, "failure_age", "#235c85", "Detailed history minus failure age"),
                                          (.11, "shuffled_history", "#a35b20", "Detailed history minus shuffled history")]:
        selected = [next(r for r in rows if r["scenario"] == s and r["baseline"] == baseline and
                         r["metric"] == "quality") for s in SCENARIOS]
        mean = np.array([float(r["delta_pp"]) for r in selected])
        lo = np.array([float(r["lower_pp"]) for r in selected])
        hi = np.array([float(r["upper_pp"]) for r in selected])
        ax.errorbar(mean, np.arange(4) + offset, xerr=[mean - lo, hi - mean], fmt="o",
                    color=color, capsize=4, label=label)
    ax.axvline(0, color="#808080", linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Selected channel quality over 16 steps: difference × 100")
    ax.set_title("History helps when persistent hidden differences exist", loc="left", pad=18)
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1, .85), fontsize=9)
    fig.text(.01, -.055, "400 paired worlds per scenario · 95% paired bootstrap intervals · fixed calibration\n"
             "Selection bench only; diagnostic costs and full network recovery are not modeled.", fontsize=9)
    fig.savefig(out / "comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    (out / "plot_snapshot.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    plot(parser.parse_args().out)

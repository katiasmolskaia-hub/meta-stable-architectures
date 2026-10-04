"""A labelled example from the completed paired experiment, not a fitted curve."""
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_bond_recovery_stop import OUT


def main():
    with (OUT / "summary.csv").open(encoding="utf-8", newline="") as f:
        rows = {r["branch"]: r for r in csv.DictReader(f) if r["n"] == "96"
                and r["cost_multiplier"] == "20" and r["scenario"] == "temporary_damage"
                and r["policy"] == "experience_swap"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), dpi=160)
    colors = ["#697B9B", "#168578"]
    names = ["Продолжать", "Пауза"]
    specs = [("quiet_service_mean", 100, "Снабжение между\nповреждениями", "% спроса"),
             ("quiet_stock_end_mean", 1, "Запас перед новым\nповреждением", "Единицы ресурса"),
             ("second_service_mean", 100, "Снабжение после нового\nповреждения", "% спроса за 60 единиц времени")]
    for ax, (metric, scale, title, label) in zip(axes, specs):
        vals = [float(rows[b][metric]) * scale for b in ("continue", "pause")]
        bars = ax.bar(names, vals, color=colors, width=.55)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_ylabel(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.1f}", ha="center", fontsize=12, weight="bold")
    fig.suptitle("После восстановления пауза сохранила ресурс", fontsize=15, weight="bold", y=.99)
    fig.text(.5, .055, "96 узлов · цена перестройки 20× · накопленный опыт · 28 пар из 30 исходных прогонов\n"
             "Первое повреждение снято. При втором обе сети снова могут перестраиваться. Средние значения.",
             ha="center", fontsize=9, color="#465267")
    fig.tight_layout(rect=(0, .13, 1, .91))
    target = OUT / "recovery_pause_example.png"
    fig.savefig(target, facecolor="white")
    print(target)


if __name__ == "__main__":
    main()

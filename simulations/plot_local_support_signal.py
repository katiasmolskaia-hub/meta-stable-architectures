import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_local_support_signal import OUT, POLICIES

with (OUT / "summary.csv").open(encoding="utf-8", newline="") as f:
    rows = {r["policy"]:r for r in csv.DictReader(f) if r["n"] == "960" and r["scenario"] == "repeated_damage"}
names = ["Без изменений", "По расписанию", "Местный сигнал", "Перемешанный\nсигнал"]
colors = ["#9DA6B8", "#506892", "#168578", "#BD9764"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
for ax, metric, scale, title, ylabel in (
    (axes[0], "service_mean", 100, "Снабжение", "% обслуженного спроса, t=40–160"),
    (axes[1], "messages_mean", .001, "Управляющие сообщения", "Тысяч сообщений за весь опыт")):
    vals = [float(rows[p][metric])*scale for p in POLICIES]
    bars = ax.bar(names, vals, color=colors, width=.65)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, 108 if metric == "service_mean" else 480)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+(2 if metric == "service_mean" else 10), f"{v:.1f}", ha="center", fontsize=11, weight="bold")
fig.suptitle("Локальный запрос экономит сообщения, но уступает по снабжению", fontsize=14, weight="bold", y=.98)
fig.text(.5, .035, "960 узлов · два повреждения · средние по 12 seed\nМодель с 12 местными кандидатами; энергетическая цена сообщений не измерена.", ha="center", fontsize=9, color="#465267")
fig.tight_layout(rect=(0, .12, 1, .91))
fig.savefig(OUT / "local_signal_tradeoff.png", facecolor="white")
print(OUT / "local_signal_tradeoff.png")

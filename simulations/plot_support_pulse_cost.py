"""Plot the preregistered scenario and both resource/message costs."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_support_pulse_cost import OUT, COSTS

with (OUT/"summary.csv").open(encoding="utf-8", newline="") as f:
    rows = {(r["policy"], int(r["cost_factor"])): r for r in csv.DictReader(f)
            if r["n"] == "960" and r["scenario"] == "repeated_damage"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
labels = {"pulse_only": "Простой замедленный пульс", "hybrid_50": "Пульс 50% + запросы"}
colors = {"pulse_only": "#566C8C", "hybrid_50": "#168577"}
for ax, metric, scale, title in zip(axes.flat, ("service", "p10", "cost_fraction", "messages"),
    (100., 100., 100., .001), ("Обслуженный спрос, %", "10-й процентиль снабжения, %",
                            "Затраты на перестройки / производство, %", "Тысяч управляющих сообщений")):
    for policy in labels:
        vals = [scale*float(rows[policy, cost][metric]) for cost in COSTS]
        ax.plot(range(3), vals, marker="o", linewidth=2, markersize=6, color=colors[policy], label=labels[policy])
    if metric in ("service", "p10"):
        ax.axhline(scale*float(rows["fixed", 0][metric]), color="#9CA4B3", linestyle="--", label="Без перестроек")
        ax.set_ylim(0, 103)
    else:
        ymax = max(scale*float(rows[p, c][metric]) for p in labels for c in COSTS)
        ax.set_ylim(0, ymax*1.12)
    ax.set_xticks(range(3), ["Обычная", "В 5 раз выше", "В 20 раз выше"])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Цена одной перестройки")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
axes[0, 0].legend(fontsize=9, loc="lower left")
fig.suptitle("Помогает ли сочетание, когда перестройка становится дорогой?", fontsize=15, weight="bold")
fig.text(.5, .025, "960 узлов · два повреждения · 30 парных начальных условий\n"
         "Частоты зафиксированы до расчёта. Реальное число сообщений не выравнивалось.\n"
         "Снабжение: t=40–160; расходы: t=0–160. Цена сообщений и задержки согласования не моделировались.",
         ha="center", fontsize=9, color="#465267")
fig.tight_layout(rect=(0, .12, 1, .94), h_pad=2)
fig.savefig(OUT/"pulse_cost_comparison.png", facecolor="white")
print(OUT/"pulse_cost_comparison.png")

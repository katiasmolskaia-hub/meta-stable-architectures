"""Plot audited results; requires NumPy and Matplotlib."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_support_pulse import OUT

with (OUT / "summary.csv").open(encoding="utf-8", newline="") as f:
    rows = {(float(r["rate"]), r["policy"]): r for r in csv.DictReader(f)
            if r["n"] == "960" and r["scenario"] == "repeated_damage"}
policies = ("pulse_only", "hybrid_75", "hybrid_50", "hybrid_25", "signal_only")
labels = ["Только\nпульс", "Пульс\n75%", "Пульс\n50%", "Пульс\n25%", "Только\nзапрос"]
colors = ["#566C8C", "#4F8E9F", "#178477", "#6DA887", "#C28D59"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
for i, rate in enumerate((1., 4.)):
    for ax, metric, scale, title, ymax in zip(axes[i], ("service", "p10", "messages"), (100, 100, .001),
        ("Обслуженный спрос, %", "10-й процентиль снабжения, %", "Тысяч сообщений"), (105, 105, 470 if rate == 4 else 140)):
        vals = [float(rows[rate, p][metric])*scale for p in policies]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, value+ymax*.018, f"{value:.1f}", ha="center", fontsize=9)
    axes[i, 0].set_ylabel(f"Лимит: {rate:g} сообщения / ед. времени\nна инициатора", fontsize=10)
fig.suptitle("Пульс и запрос при общем лимите сообщений", fontsize=15, weight="bold")
fig.text(.5, .025, "960 узлов · два повреждения · средние по 20 одинаковым начальным условиям\n"
         "Доля пульса — настройка частоты, не фактическая доля сообщений. Общий лимит одинаков внутри каждой строки.\n"
         "Спрос: t=40–160; сообщения: t=0–160. Радиоколлизии и энергетическая цена сообщений не моделировались.",
         ha="center", fontsize=9, color="#465267")
fig.tight_layout(rect=(0, .12, 1, .95), h_pad=3)
fig.savefig(OUT / "pulse_tradeoff.png", facecolor="white")
print(OUT / "pulse_tradeoff.png")

if (OUT / "traffic_summary.csv").exists():
    with (OUT / "traffic_summary.csv").open(encoding="utf-8", newline="") as f:
        holdout = list(csv.DictReader(f))
    labels = ["Пульс 50%\n+ запрос", "Простой пульс\n(медленнее)", "Пульс 75%\n+ запрос", "Простой пульс\n(медленнее)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=160)
    for ax, metric, scale, title, ymax in zip(axes, ("service", "messages", "swaps"), (100, .001, .024),
        ("Обслуженный спрос, %", "Тысяч сообщений", "Ресурс потрачен на перестройки"), (103, 390, 215)):
        vals = [float(r[metric])*scale for r in holdout]
        bars = ax.bar(range(4), vals, color=["#178477", "#566C8C", "#178477", "#566C8C"])
        ax.set_xticks(range(4), labels)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(1.5, color="#B9C0CC", linewidth=1, linestyle="--")
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, value+ymax*.025, f"{value:.2f}", ha="center", fontsize=9)
    fig.suptitle("При близком трафике простой пульс даёт практически то же снабжение", fontsize=13, weight="bold")
    fig.text(.5, .045, "Отдельная проверка: 960 узлов · два повреждения · 30 новых начальных условий\n"
             "Преимущество сочетания по снабжению не установлено; у него меньше затрат на перестройки.\n"
             "Частоты выбраны по трафику предыдущей серии. Энергия сообщений не измерена.",
             ha="center", fontsize=9, color="#465267")
    fig.tight_layout(rect=(0, .18, 1, .91))
    fig.savefig(OUT / "pulse_traffic_holdout.png", facecolor="white")
    print(OUT / "pulse_traffic_holdout.png")

"""Russian report and figures from all saved buffer runs, without selection."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_bridge_buffer import OUT, ROOT, SCENARIOS, POLICIES

LABELS = {"unregulated": "Без ограничения", "rate_limiter": "Простой ограничитель",
          "ordinary_buffer": "Обычный буфер + ограничитель", "adaptive_bridge": "Адаптивный мост"}
SCENES = {"steady": "Постоянное предложение", "bursty_supply": "Импульсное предложение",
          "upstream_windows": "Прерывается входной канал", "global_outage": "Прерываются оба участка",
          "delayed_sensor": "Запаздывают измерения", "insufficient_capacity": "Длительно малая принимающая способность"}


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary, runs, pairs, refinement = [read(name) for name in (
        "summary.csv", "runs.csv", "paired_comparisons.csv", "step_refinement.csv")]
    data = {(r["scenario"], r["policy"]): r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    colors = ("#969aa4", "#d98917", "#188667", "#4268bb")
    ticks = ("Постоянное\nпредложение", "Импульсное\nпредложение", "Разрывы\nвхода", "Разрывы\nобоих участков", "Запаздывание\nизмерений", "Недостаточная\nспособность")
    x = np.arange(len(SCENARIOS))
    for j, policy in enumerate(POLICIES):
        rows = [data[s, policy] for s in SCENARIOS]
        for ax, metric, scale in zip(axes, ("service_fraction", "stress_area"), (100, 1)):
            ax.bar(x + (j - 1.5) * .19, [float(r[metric + "_mean"]) * scale for r in rows],
                   width=.19, label=LABELS[policy], color=colors[j],
                   yerr=[float(r[metric + "_sd"]) * scale for r in rows], capsize=2)
            ax.set_xticks(x, ticks, fontsize=8)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel("Обслуженная потребность, % — больше лучше")
    axes[1].set_ylabel("Накопленный стресс B — меньше лучше")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=10)
    fig.suptitle("Нужен ли особый буферный мост?", fontsize=18)
    axes[0].set_title("Полезный обмен")
    axes[1].set_title("Нагрузка на получателя")
    fig.savefig(OUT / "comparison.png", dpi=170)
    plt.close(fig)

    traces = read("first_seed_traces.csv")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for policy, color in zip(POLICIES[1:], colors[1:]):
        sub = [r for r in traces if r["scenario"] == "upstream_windows" and r["policy"] == policy]
        for ax, key in zip(axes, ("service", "stress", "buffer")):
            ax.plot([float(r["time"]) for r in sub], [float(r[key]) for r in sub],
                    label=LABELS[policy], color=color, lw=1.5)
    for ax in axes:
        ax.axvspan(20, 30, color="#999999", alpha=.15)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Обслуживание B")
    axes[1].set_ylabel("Стресс B")
    axes[2].set_ylabel("Запас в мосту")
    axes[2].set_xlabel("Модельное время")
    axes[0].legend(fontsize=9)
    fig.suptitle("Связь с источником прерывается; выдача из буфера доступна\nПервый заранее выбранный seed = 4000", fontsize=13)
    fig.savefig(OUT / "first_seed_dynamics.png", dpi=150)
    plt.close(fig)

    lines = ["# Проверка буферного моста — 2026-10-03", "",
        "## Решение", "",
        "В выбранной реализации дополнительное адаптивное разрешение не оправдало усложнение. "
        "При постоянно доступном канале обычный буфер не увеличил обслуженный спрос относительно прямого ограничителя; при прерываниях входа полезно промежуточное хранение. "
        "Этот выигрыш обеспечивает обычный буфер с тем же ограничителем. Адаптивный мост обслуживает меньше спроса "
        "и создаёт больший интегральный стресс во всех шести сценариях. Его меньшие расходы сопровождаются уменьшением полезного обмена.", "",
        "## Дизайн", "",
        f"{len(runs)} основных прогонов: 6 сценариев × 4 варианта × 40 парных seed (4000–4039). "
        "Спрос, предложение, принимающая способность, сила кризиса и фаза внешних импульсов совпадают между вариантами одного seed. "
        "Общий объём хранения одинаков; передача через мост оплачивается дважды, по числу участков. "
        "Параметры заданы до первого расчёта и по результатам не менялись.", "",
        "Это новый минимальный тест по мотивам буферной формы из заметок VI тома. "
        "Семантический перевод, нейтральные носители, возникновение связей и все возможные мостовые механизмы здесь не моделируются.", "",
        "## Доля обслуженного спроса после начала кризиса", "",
        "| Сценарий | Без ограничения | Ограничитель | Обычный буфер | Адаптивный мост |",
        "|---|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        vals = [f"{float(data[s,p]['service_fraction_mean'])*100:.2f}%" for p in POLICIES]
        lines.append("| " + SCENES[s] + " | " + " | ".join(vals) + " |")
    lines += ["", "Во всех сценариях есть кризис принимающей способности B. Поэтому результаты не обязаны повторять "
        "100% из сентябрьского теста каналов: задача и физические ограничения изменены.", "",
        "## Цена и стресс", "",
        "Стоимость включает передачу и хранение за весь горизонт 0–90, в том числе заполнение буфера до кризиса. "
        "Стресс измеряется за период 20–90. Обе величины в условных единицах модели; это не измерение физической энергии.", "",
        "| Сценарий | Цена: ограничитель | Цена: обычный буфер | Цена: адаптивный | Стресс: обычный буфер | Стресс: адаптивный |",
        "|---|---:|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        vals = [float(data[s,p]["full_cost_mean"]) for p in POLICIES[1:]]
        vals += [float(data[s,p]["stress_area_mean"]) for p in POLICIES[2:]]
        lines.append("| " + SCENES[s] + " | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## Парные разности: адаптивный мост минус обычный буфер", "",
        "95% bootstrap-интервалы по 40 парным seed, 2000 повторов. Без поправки на множественные сравнения; "
        "интервалы описывают выбранные случайные вариации внутри модели и не доказывают универсальность эффекта.", "",
        "| Сценарий | Обслуживание, п.п. [95% интервал] | Стресс [95% интервал] |",
        "|---|---:|---:|"]
    for s in SCENARIOS:
        cells = []
        for metric, scale in (("service_fraction", 100), ("stress_area", 1)):
            r = next(r for r in pairs if r["scenario"] == s and r["treatment"] == "adaptive_bridge"
                     and r["baseline"] == "ordinary_buffer" and r["metric"] == metric)
            cells.append(f"{float(r['mean'])*scale:+.3f} [{float(r['ci_low'])*scale:+.3f}; {float(r['ci_high'])*scale:+.3f}]")
        lines.append("| " + SCENES[s] + " | " + " | ".join(cells) + " |")
    service_delta, stress_delta = [], []
    ordering_changed = []
    for s in SCENARIOS:
        for p in POLICIES:
            rows = sorted((r for r in refinement if r["scenario"] == s and r["policy"] == p), key=lambda r: -float(r["dt"]))
            service_delta.append(abs(float(rows[0]["service_fraction"]) - float(rows[-1]["service_fraction"])))
            stress_delta.append(abs(float(rows[0]["stress_area"]) - float(rows[-1]["stress_area"])))
        for dt in (.1, .05, .025):
            rows = {r["policy"]: r for r in refinement if r["scenario"] == s and float(r["dt"]) == dt}
            ordering_changed.append(float(rows["adaptive_bridge"]["service_fraction"]) > float(rows["ordinary_buffer"]["service_fraction"]) + 1e-10)
    lines += ["", "## Почему так получилось", "",
        "1. Простой ограничитель уже снижает передачу при падении измеренной принимающей способности. "
        "Адаптивное разрешение дополнительно сокращает этот поток по стрессу. В выбранных уравнениях это усиливает нехватку ресурса, "
        "а нехватка сама увеличивает стресс. Получается нежелательная обратная связь.",
        "2. При разрыве входа обычный буфер заранее хранит ресурс за местом разрыва и продолжает выдачу. "
        "Это известная по смыслу функция промежуточного хранения, не свидетельство особой природы мостовой связи.",
        "3. Если закрываются оба участка, такое размещение запаса не помогает доставлять. "
        "Обычный буфер почти совпадает с прямым ограничителем по обслуживанию, но требует дополнительных передач.",
        "4. Буфер не увеличивает физическую принимающую способность B. Длительный её дефицит не устраняется накоплением ресурса рядом.", "",
        "## Проверки", "",
        "- Семь автоматических проверок прошли: баланс ресурса/бюджет/вместимость; отсутствие эффекта политики без обмена; "
        "совпадение с обычным буфером при отключении адаптивного разрешения; невозможность доставки через закрытый выход; "
        "отсутствие мгновенной передачи через пустой буфер; воспроизводимость; влияние обмена на состояние B.",
        f"- Максимальная ошибка баланса в основном пакете: {max(float(r['max_mass_error']) for r in runs):.2e}. "
        f"Максимальное использование лимита: {max(float(r['max_budget_ratio']) for r in runs):.12f}.",
        f"- {len(refinement)} дополнительных запусков: seed 4900, нулевой шум, dt=0.1/0.05/0.025. "
        f"Максимальное изменение обслуживания между крайними шагами: {100*max(service_delta):.3f} п.п.; "
        f"интеграла стресса: {max(stress_delta):.3f} единицы. "
        f"Случаев, где адаптивный вариант превзошёл обычный по обслуживанию в этой проверке: {sum(ordering_changed)} из {len(ordering_changed)}.",
        "- После первого расчёта добавлены только счётчики полной стоимости и объёма передач до и после кризиса. "
        "Основной пакет и проверка шага повторены с ними. Динамические коэффициенты не менялись.", "",
        "## Границы вывода и дальнейшее решение", "",
        "Результат относится к конкретному плавному разрешению, заданным ёмкостям, бюджетам и форме зависимости стресса. "
        "Прямой доступ к измеренной принимающей способности делает простой ограничитель сильным конкурентом. "
        "40 seed не заменяют проверку на независимой прикладной задаче. Ресурс, стресс и цена — модельные величины.", "",
        "Для этого направления выполнен критерий остановки усложнения: сохранить простой ограничитель; добавлять обычный буфер "
        "там, где требуется разнести приём и выдачу во времени. Текущий адаптивный вариант не рекомендован как улучшение. "
        "Это не опровержение всей метастабильной архитектуры. Возвращаться к особому мостовому механизму имеет смысл только "
        "при новой независимо мотивированной задаче, с которой эти простые решения не справляются; специально конструировать ему победу не требуется.", "",
        "## Воспроизведение", "", "```powershell",
        "python -m unittest discover -s simulations -p test_bridge_buffer.py -v",
        "python simulations/experiment_bridge_buffer.py",
        "python simulations/experiment_bridge_buffer.py --step-check",
        "python simulations/report_bridge_buffer.py", "```", "",
        "Для расчёта нужен NumPy, для графиков Matplotlib. В `outputs/bridge_buffer_2026-10-03/` сохранены "
        "`config.json`, точный `experiment_snapshot.py`, все прогоны, агрегаты, парные интервалы, проверка шага и траектории seed 4000. "
        "Конфигурация содержит версии среды и SHA-256 кода. Повторный запуск в ту же папку заменяет созданные результаты. "
        "Для нового расчёта используйте отдельный `--output`; скрипт отчёта читает папку этого эксперимента."]
    path = ROOT / "manuscripts" / "v6" / "bridge_buffer_results_2026-10-03.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)
    print(f"Step sensitivity: service {100*max(service_delta):.4f} pp; stress {max(stress_delta):.4f}")


if __name__ == "__main__":
    main()

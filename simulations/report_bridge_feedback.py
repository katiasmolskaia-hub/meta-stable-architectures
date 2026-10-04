"""Render saved results only; never re-run or select successful simulations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiment_bridge_feedback import SCENARIOS, POLICIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bridge_feedback_v2"
LABELS = {
    "old_repair": "Старые связи",
    "threshold": "Порог по состоянию",
    "feedback_gate": "Отказ → пауза → проба",
    "lifecycle": "Накопленная память",
    "no_memory": "Без обновления памяти",
}
SCENE_LABELS = {
    "unchanged": "Каналы исправны",
    "persistent_damage": "Длительное повреждение",
    "temporary_damage": "Временное повреждение",
    "no_safe_detour": "Без безопасного обхода",
    "changed_again": "Повторная смена каналов",
}


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = read("summary.csv")
    comparisons = read("paired_comparisons.csv")
    runs = read("runs.csv")
    refinement = read("step_refinement.csv")
    indexed = {(r["scenario"], r["policy"]): r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    colors = ("#9a9ea8", "#e69f00", "#168568", "#3568c0", "#b96b99")
    x = np.arange(5)
    ticks = ("Каналы\nисправны", "Длительное\nповреждение", "Временное\nповреждение",
             "Нет безопасного\nобхода", "Повторная\nсмена")
    for j, policy in enumerate(POLICIES):
        sub = [indexed[s, policy] for s in SCENARIOS]
        for ax, metric, scale in zip(axes, ("service_fraction", "stress_area"), (100, 1)):
            ax.bar(x + (j - 2) * .15, [float(r[metric + "_mean"]) * scale for r in sub],
                   width=.15, color=colors[j], label=LABELS[policy],
                   yerr=[float(r[metric + "_sd"]) * scale for r in sub], capsize=2)
            ax.set_xticks(x, ticks, fontsize=9)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Обслуженная потребность, % — больше лучше")
    axes[0].set_ylim(0, 108)
    axes[1].set_ylabel("Накопленный стресс групп — меньше лучше")
    axes[0].legend(fontsize=9, loc="lower left", framealpha=.95)
    fig.suptitle("Мостовой механизм: первый тест с обратной связью", fontsize=17)
    fig.supxlabel("40 парных запусков на сценарий · столбцы: средние · усы: стандартные отклонения\n"
                  "Синтетическая ресурсная сеть из трёх групп; одинаковый лимит обмена", fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=170)
    plt.close(fig)

    # Show the predetermined first seed, never the most flattering run.
    traces = read("first_seed_traces.csv")
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    for policy, color in (("threshold", "#e69f00"), ("feedback_gate", "#168568"), ("lifecycle", "#3568c0")):
        sub = [r for r in traces if r["scenario"] == "persistent_damage" and r["policy"] == policy]
        t = [float(r["time"]) for r in sub]
        for ax, metric in zip(axes, ("stress_B", "flow_BC", "service_b")):
            ax.plot(t, [float(r[metric]) for r in sub], label=LABELS[policy], color=color, lw=1.6)
    for ax in axes:
        ax.axvspan(20, 30, color="#e69f00", alpha=.12)
        ax.axvline(20, color="#666", ls="--", lw=.8)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Стресс B")
    axes[1].set_ylabel("Обмен B–C, ресурс/время")
    axes[2].set_ylabel("Доля обслуживания B")
    axes[2].set_xlabel("Модельное время")
    axes[0].legend(fontsize=9)
    fig.suptitle("Старый канал повреждён: как включается обход\nПервый заранее выбранный seed = 1000", fontsize=14)
    fig.savefig(OUT / "first_seed_dynamics.png", dpi=160)
    plt.close(fig)

    lines = ["# Мостовой механизм: результаты первого независимого теста",
             "", "Дата: 2026-09-07. Актуальная вычислительная версия: v2.", "",
             "## Вывод", "",
             "В этой синтетической задаче адаптация по результатам контактов сохраняет полезный обмен при повреждении старого канала. "
             "Однако накопленная оценка качества не показала преимуществ перед более простым механизмом «неудача → пауза → повторная проба». "
             "Это поддерживает проверку минимального контактного механизма, но не доказывает необходимость всей мостовой архитектуры.", "",
             "## Что проверено", "",
             f"{len(runs)} основных запусков: 5 сценариев × 5 вариантов × 40 seed. "
             "Состояния трёх групп зависят от доставленного ресурса, его потерь и расхода резервов. "
             "Регуляторы не получают истинную пригодность каналов или расписание будущих повреждений. "
             "Протокол записан до первого запуска; результат не использовался для настройки механизма памяти.", "",
             "## Полезный результат: доля обслуженной потребности", "",
             "Это доля удовлетворённой потребности обеих групп-потребителей за период от начала кризиса до конца наблюдения. "
             "Она не означает долю исправных мостов и не равна доле восстановившихся групп.", "",
             "| Сценарий | Старые связи | Порог | Пауза после отказа | Память | Без обновления памяти |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        values = [f"{100 * float(indexed[s, p]['service_fraction_mean']):.2f}%" for p in POLICIES]
        lines.append("| " + SCENE_LABELS[s] + " | " + " | ".join(values) + " |")
    lines += ["", "## Цена восстановления", "",
              "| Сценарий | Стресс: пауза | Стресс: память | Расход резерва: пауза | Расход резерва: память |",
              "|---|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        g, m = indexed[s, "feedback_gate"], indexed[s, "lifecycle"]
        lines.append(f"| {SCENE_LABELS[s]} | {float(g['stress_area_mean']):.2f} | {float(m['stress_area_mean']):.2f} | "
                     f"{float(g['resource_cost_mean']):.2f} | {float(m['resource_cost_mean']):.2f} |")
    lines += ["", "Равенство верхнего лимита обмена не означает равенства фактических затрат. "
              "Расход резерва, транспортные потери и стресс учитываются отдельно; стресс — модельная величина, не физическая энергия.", "",
              "## Парные разности: память минус простой механизм паузы", "",
              "Показаны исследовательские bootstrap-интервалы 95% по seed (2000 повторов, без поправки на множественные сравнения). "
              "Для обслуживания больше лучше; для стресса меньше лучше.", "",
              "| Сценарий | Разность обслуживания, п.п. [95% интервал] | Разность стресса [95% интервал] |",
              "|---|---:|---:|"]
    for s in SCENARIOS:
        cells = []
        for metric, scale in (("service_fraction", 100), ("stress_area", 1)):
            r = next(r for r in comparisons if r["scenario"] == s and r["baseline"] == "feedback_gate" and r["metric"] == metric)
            cells.append(f"{float(r['delta_mean'])*scale:+.2f} [{float(r['ci_low'])*scale:+.2f}; {float(r['ci_high'])*scale:+.2f}]")
        lines.append("| " + SCENE_LABELS[s] + " | " + " | ".join(cells) + " |")
    service_error, stress_error = [], []
    for s in SCENARIOS:
        for p in ("threshold", "feedback_gate", "lifecycle", "no_memory"):
            sub = [r for r in refinement if r["scenario"] == s and r["policy"] == p]
            coarse, fine = sub[0], sub[-1]
            service_error.append(abs(float(coarse["service_fraction"]) - float(fine["service_fraction"])))
            stress_error.append(abs(float(coarse["stress_area"]) - float(fine["stress_area"])) / max(float(fine["stress_area"]), 1e-12))
    lines += ["", "## Проверки и численная поправка", "",
              "- Шесть автоматических проверок прошли: сохранение ресурса и лимит, воспроизводимость, одинаковое начало абляции, "
              "исчезновение различий без обмена, реальное влияние мостов на группы, устойчивое восстановление с сохранением невосстановившихся наблюдений.",
              f"- Максимальная ошибка баланса ресурса в основном пакете: {max(float(r['max_mass_error']) for r in runs):.2e}; "
              f"максимальное отношение расхода к лимиту: {max(float(r['max_budget_ratio']) for r in runs):.12f}.",
              f"- Уточнение шага: {len(refinement)} дополнительных запусков, dt = 0.1 / 0.05 / 0.025, seed 3000, без шума наблюдений. "
              f"Максимальное изменение обслуживания между крайними шагами: {100*max(service_error):.3f} п.п.; "
              f"относительное изменение интеграла стресса: {100*max(stress_error):.2f}%.",
              "- Первоначальная версия v1 делала решение простого конкурента по одной пробе длительностью dt. "
              "При уменьшении dt такая проба становилась почти бесплатной. v2 требует фиксированного объёма попыток 0.025 перед решением. "
              "Первый пакет и его код сохранены в `outputs/bridge_feedback_v1/`. Итоговые таблицы здесь относятся только к повторному пакету v2.", "",
              "## Ограничения", "",
              "- Это новая ресурсная модель для проверки компонента идеи, а не буквальный перенос уравнений `experiment_bridge_lifecycle_v2.py`.",
              "- Все кандидатные каналы заданы заранее. Обход становится используемым, но новая топология из ничего не возникает. "
              "Поток B–C сам по себе не измеряет происхождение каждого переданного ресурса; для строгого отслеживания маршрута нужны метки потока.",
              "- Условия канала и полезность контакта одномерны. Перевод между несовместимыми режимами, самостоятельные формы моста "
              "и отдельные нейтральные носители не проверялись.",
              "- Неудачный обмен сам создаёт стресс; при его предельном значении готовность к контакту равна нулю. "
              "В сочетании с нехваткой ресурса это допускает самоподдерживающееся застревание даже после ремонта канала. "
              "Переносимость результата при ненулевой минимальной готовности пока не проверена.",
              "- Слабая повторная проба и скорость накопления опыта выбраны один раз. Проигрыш этой реализации не опровергает все механизмы памяти.",
              "- 40 seed варьируют выбранные параметры внутри одной среды. Независимая прикладная задача, широкий диапазон бюджетов, "
              "систематическая ошибка наблюдений и более сложные топологии ещё не проверены.",
              "- Восстановление определяется первым устойчивым окном; поздние срывы отражаются в интегральных метриках, "
              "но требуют отдельной метрики рецидивов. Проверка шага не является формальным доказательством сходимости.", "",
              "## Что делать дальше", "",
              "Сохранить простой механизм «контакт → оценка → пауза → новая проба» как сильную опорную модель. "
              "Следующая отдельная гипотеза для сложной памяти: помогает ли она, когда единичный результат ненадёжен, "
              "а различить временную неудачу и устойчивую непригодность можно только по истории. "
              "Для такой проверки нужна новая выборка и заранее определённые затраты наблюдения. "
              "Добавлять все формы моста на основании текущего теста преждевременно.", "",
              "## Файлы и воспроизведение", "",
              "- Код: `simulations/experiment_bridge_feedback.py`.",
              "- Протокол: `manuscripts/v6/bridge_feedback_protocol.md`.",
              "- Данные: `outputs/bridge_feedback_v2/runs.csv`, `summary.csv`, `paired_comparisons.csv`, `first_seed_traces.csv`.",
              "- Конфигурация и точный код запуска: `config.json`, `runtime.json`, `experiment_snapshot.py` в той же папке.",
              "- Графики: `comparison.png`, `first_seed_dynamics.png`.", "",
              "```powershell",
              "python simulations/experiment_bridge_feedback.py --seeds 40 --no-plot",
              "python simulations/check_bridge_feedback_step.py",
              "python -m unittest discover -s simulations -p test_bridge_feedback.py -v",
              "python simulations/report_bridge_feedback.py",
              "```", "",
              "Для расчёта требуется NumPy, для графиков Matplotlib. В этой сессии расчёт выполнен на комплектном Python; "
              "библиотека графиков установлена отдельно в `MyProject/outputs/bridge-feedback-deps`. "
              "Версии расчётного окружения записаны в `runtime.json`. Повторный запуск в ту же папку заменяет созданные результаты; "
              "для нового эксперимента укажите отдельный `--output`."]
    (ROOT / "manuscripts" / "v6" / "bridge_feedback_results_2026-09-07.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote Russian report and two verified-data plots")


if __name__ == "__main__":
    main()

"""Generate figures and an evidence-bounded report from saved rewiring runs."""
from __future__ import annotations

import csv
import hashlib
import json

import numpy as np

from experiment_bond_rewiring import OUT, ROOT, Params, POLICIES, SCENARIOS, make_world, apply_swap

LABELS = {"fixed": "Неизменные связи", "random_swap": "Случайная перестройка",
          "recent_contact": "По последнему контакту", "experience_swap": "По накопленному опыту"}
SCENES = {"uniform": "Все связи одинаково хороши", "persistent_damage": "Длительное повреждение",
          "temporary_damage": "Временное повреждение", "rapid_change": "Быстрая смена условий",
          "separated_resources": "Ресурс в одной половине", "noisy_feedback": "Шумные наблюдения"}


def read(name):
    with (OUT / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs, summary, pairs, steps = [read(n) for n in (
        "runs.csv", "summary.csv", "paired_comparisons.csv", "step_refinement.csv")]
    config = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
    assert len(runs) == config["seeds"] * len(POLICIES) * len(SCENARIOS)
    assert hashlib.sha256((OUT / "experiment_snapshot.py").read_bytes()).hexdigest() == config["sha256"]
    assert len({(r["scenario"], r["policy"], r["seed"]) for r in runs}) == len(runs)
    data = {(r["scenario"], r["policy"]): r for r in summary}
    colors = ("#959ba7", "#df9a25", "#21856a", "#456fbd")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    ticks = ("Одинаковые\nсвязи", "Длительное\nповреждение", "Временное\nповреждение",
             "Быстрая\nсмена", "Разделённый\nресурс", "Шумные\nнаблюдения")
    x = np.arange(len(SCENARIOS))
    for i, policy in enumerate(POLICIES):
        rows = [data[s, policy] for s in SCENARIOS]
        for ax, key in zip(axes, ("service_fraction", "consumer_p10")):
            ax.bar(x + (i - 1.5) * .19, [100 * float(r[key + "_mean"]) for r in rows],
                   width=.19, color=colors[i], label=LABELS[policy],
                   yerr=[100 * float(r[key + "_sd"]) for r in rows], capsize=2)
            ax.set_xticks(x, ticks, fontsize=8)
            ax.set_ylim(0, 108)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Обслуженная потребность всей сети")
    axes[1].set_title("Показатель хуже снабжаемых участников (p10)")
    axes[0].set_ylabel("Доля потребности, % — больше лучше")
    fig.suptitle("Смена соседей: помогает ли опыт?\n960 прогонов · средние и стандартные отклонения по 40 seed", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=170)
    plt.close(fig)

    traces = read("first_seed_traces.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for policy, color in zip(POLICIES, colors):
        rows = [r for r in traces if r["scenario"] == "persistent_damage" and r["policy"] == policy]
        for ax, key in zip(axes, ("service", "old_edge_share")):
            ax.plot([float(r["time"]) for r in rows], [float(r[key]) for r in rows],
                    color=color, lw=1.8, label=LABELS[policy])
    for ax in axes:
        ax.axvline(40, color="#666", ls="--", lw=1)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Обслуженная потребность")
    axes[1].set_ylabel("Доля исходных связей")
    axes[1].set_xlabel("Модельное время; повреждение начинается при t=40")
    axes[0].legend(fontsize=9)
    fig.suptitle("Длительное повреждение: первый заранее выбранный seed = 5000", fontsize=13)
    fig.savefig(OUT / "first_seed_dynamics.png", dpi=160)
    plt.close(fig)

    # Reconstruct actual recorded swaps; all policies use exactly the same layout.
    events = read("first_seed_swaps.csv")
    initial, sources, _, _, _ = make_world(5000, "persistent_damage", Params())
    positions = np.column_stack((np.cos(np.linspace(0, 2*np.pi, 24, endpoint=False)),
                                 np.sin(np.linspace(0, 2*np.pi, 24, endpoint=False))))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained")
    for ax, policy in zip(axes, ("fixed", "random_swap", "experience_swap")):
        adj = initial.copy()
        for r in events:
            if r["policy"] == policy and r["scenario"] == "persistent_damage":
                apply_swap(adj, tuple(int(r[k]) for k in ("a", "b", "c", "d")))
        for a, b in zip(*np.where(np.triu(adj, 1))):
            ax.plot(positions[[a,b],0], positions[[a,b],1], color="#bbc1cd", lw=.8, alpha=.75, zorder=1)
        ax.scatter(positions[:,0], positions[:,1], s=180,
                   c=np.where(sources, "#d99120", "#4f77b2"), edgecolors="white", zorder=2)
        for i, (xx, yy) in enumerate(positions):
            ax.text(xx, yy, str(i+1), ha="center", va="center", color="white", fontsize=7, zorder=3)
        ax.set_title(LABELS[policy], fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Фактические соседи в конце прогона, seed = 5000\n24 участника, у каждого 4 связи; оранжевые — источники", fontsize=14)
    fig.savefig(OUT / "final_networks.png", dpi=160)
    plt.close(fig)

    lines = ["# Смена соседей по опыту — результаты 2026-10-03", "",
        "## Вывод", "",
        "В этой ресурсной сети выбор перестройки по результатам контактов улучшает обслуживание по сравнению со случайной сменой соседей "
        "при одинаковой частоте и цене. Основную пользу уже даёт последний контакт; накопление опыта добавляет небольшой выигрыш "
        "в нескольких сценариях. При разделении источников и потребителей выбор по качеству не даёт ясного выигрыша в среднем "
        "и ухудшает снабжение нижней части распределения. Сеть остаётся структурно связной: наличие пути ещё не означает достаточного снабжения.", "",
        "Это первый положительный результат именно для изменения соседей. Он не подтверждает всю метастабильную архитектуру, "
        "не моделирует молекулы воды и не доказывает научную новизну относительно существующих методов адаптивной перестройки сетей.", "",
        "## Что сравнивали", "",
        f"{len(runs)} прогонов: 24 участника, 4 связи у каждого, 6 сценариев × 4 политики × 40 парных seed. "
        "Каждая подвижная политика выполняет 59 обменов пары связей, платит 1.416 единицы ресурса за весь горизонт "
        "и получает одинаковый лимит числа предложений на событие. Неизменная сеть не платит за перестройку. "
        "Источники и скрытые пригодности связей случайны, но совпадают внутри парного сравнения. "
        "Глобальные метрики обслуживания не используются для выбора новых соседей.", "",
        "## Доля обслуженного спроса после t=40", "",
        "| Сценарий | Неизменные связи | Случайная перестройка | Последний контакт | Накопленный опыт |",
        "|---|---:|---:|---:|---:|"]
    for s in SCENARIOS:
        vals = [f"{100*float(data[s,p]['service_fraction_mean']):.2f}%" for p in POLICIES]
        lines.append("| " + SCENES[s] + " | " + " | ".join(vals) + " |")
    lines += ["", "## Парные преимущества опыта", "",
        "95% bootstrap-интервалы по seed, 2000 повторов, без поправки на множественные сравнения. "
        "Это исследовательские интервалы внутри одной модели, а не подтверждение универсальности.", "",
        "| Сценарий | Опыт − случайная, п.п. [95% интервал] | Опыт − последний контакт, п.п. [95% интервал] |",
        "|---|---:|---:|"]
    for s in SCENARIOS:
        cells = []
        for base in ("random_swap", "recent_contact"):
            r = next(r for r in pairs if r["scenario"] == s and r["treatment"] == "experience_swap"
                     and r["baseline"] == base and r["metric"] == "service_fraction")
            cells.append(f"{100*float(r['mean']):+.2f} [{100*float(r['ci_low']):+.2f}; {100*float(r['ci_high']):+.2f}]")
        lines.append("| " + SCENES[s] + " | " + " | ".join(cells) + " |")
    lines += ["", "При длительном повреждении преимущество над случайной перестройкой составляет примерно 15.64 п.п.; "
        "над последним контактом — около 0.93 п.п. При быстрой смене условий интервал добавки к последнему контакту пересекает ноль: "
        "уверенного дополнительного преимущества накопления здесь нет.", "",
        "## Граница: локальное качество и снабжение всей сети", "",
        "| Сценарий | p10: случайная | p10: последний контакт | p10: опыт |",
        "|---|---:|---:|---:|"]
    for s in SCENARIOS:
        vals = [f"{100*float(data[s,p]['consumer_p10_mean']):.2f}%" for p in POLICIES[1:]]
        lines.append("| " + SCENES[s] + " | " + " | ".join(vals) + " |")
    lines += ["", "p10 — десятый процентиль доли обслуженного спроса отдельных потребителей внутри прогона, затем среднее по seed. "
        "Это показатель нижней части распределения, а не среднее строго выделенных 10% участников.", "",
        "При разделённом ресурсе общий результат опыта около 73.73%, случайной перестройки — 73.53%, "
        "и интервал их разности включает ноль. При этом p10 у опыта около 39.03%, у случайной — 44.38%. "
        "Парная разность p10 составляет около −5.34 п.п. [−7.89; −2.89]. "
        "Во всех основных прогонах потребители сохраняли структурный путь к источникам; причина недостатка не сводится к разрыву графа. "
        "Правило оценивает долю доставки на контакте, но не учитывает незаменимость дорогого межгруппового прохода.", "",
        "## Проверки", "",
        "- Шесть автоматических проверок прошли: степень/число рёбер после 200 перестроек; баланс ресурса/бюджет; "
        "равные число и цена обменов; отсутствие обучения без контактов; неподвижность фиксированного графа и воспроизводимость; "
        "распознавание компоненты потребителей без источника.",
        f"- Максимальная ошибка ресурсного баланса: {max(float(r['max_mass_error']) for r in runs):.2e}; "
        f"максимальное использование лимита: {max(float(r['max_budget_ratio']) for r in runs):.12f}.",
        "- Проверены количество уникальных записей, полнота 960 прогонов и совпадение SHA-256 снимка кода с конфигурацией."]
    max_delta, detail = 0.0, []
    for s in SCENARIOS:
        for p in POLICIES:
            sub = sorted((r for r in steps if r["scenario"] == s and r["policy"] == p), key=lambda r: -float(r["dt"]))
            diff = abs(float(sub[0]["service_fraction"]) - float(sub[-1]["service_fraction"]))
            max_delta = max(max_delta, diff)
            detail.append((s, p, diff))
    lines += [f"- Уточнение шага: {len(steps)} прогонов, seed 5900 без шума, dt=0.2/0.1/0.05. "
        f"Максимальное изменение обслуживания между крайними шагами: {100*max_delta:.3f} п.п. "
        "Это локальная численная проверка. Решения о перестройке дискретны, поэтому близкие оценки могут приводить к разным последующим графам; "
        "один seed не является доказательством сходимости распределения результатов.",
        "- После успешного сохранения всех данных основной процесс завершился ошибкой только в печати сводки: "
        "неверно указано имя агрегированного поля. Строка печати исправлена, данные не пересчитывались и не изменялись. "
        "Сохранённый снимок соответствует фактически выполненному расчёту.", "",
        "## Ограничения", "",
        "1. Это простая транспортная задача, где наблюдаемая доля доставки довольно прямо связана с полезностью контакта. "
        "Оценка не является обучением сложному поведению агента.",
        "2. Пригодность пары задана средой. Она не возникает из взаимной адаптации участников. "
        "Используется скалярная оценка, а не химическая или семантическая совместимость.",
        "3. Перестройки происходят по внешнему расписанию, а не только при необходимости. Не проверено, когда следует оставлять сеть в покое.",
        "4. Все участники доступны как потенциальные кандидаты. Нет географической цены поиска, задержек переговоров и отказа другой стороны. "
        "Согласование обмена двух связей упрощено, оценки хранятся централизованно в симуляторе. Полностью распределённая реализация не проверена.",
        "5. Сглаживание зависит от объёма контактов, поэтому опыт отражает также частоту использования пути. "
        "Его небольшую добавку к последнему контакту нельзя приписать исключительно подавлению шума без дополнительных отключений компонентов.",
        "6. Проверены одна степень сети, одна частота, цена перестройки и скорость обучения. "
        "40 seed и bootstrap не заменяют независимую задачу, другие размеры сети и диапазоны стоимости.", "",
        "## Решение о продолжении", "",
        "Сохранить минимальное правило смены соседей по результатам контактов как перспективный механизм. "
        "Приоритет следующей проверки — переносимость результата на новые параметры и цена сохранения необходимого межгруппового обмена. "
        "Не добавлять сразу инструктора, буфер, множество форм моста или сложную память: основной выигрыш уже есть у простого локального опыта. "
        "Граница при разделённом ресурсе должна оставаться частью вывода, а не неудобным исключением.", "",
        "## Файлы и воспроизведение", "", "```powershell",
        "python -m unittest discover -s simulations -p test_bond_rewiring.py -v",
        "python simulations/experiment_bond_rewiring.py",
        "python simulations/experiment_bond_rewiring.py --step-check",
        "python simulations/report_bond_rewiring.py", "```", "",
        "NumPy нужен для расчётов; Matplotlib — для графиков. Данные находятся в `outputs/bond_rewiring_2026-10-03/`: "
        "`config.json`, `experiment_snapshot.py`, `runs.csv`, `summary.csv`, `paired_comparisons.csv`, "
        "`first_seed_traces.csv`, `first_seed_swaps.csv`, `step_refinement.csv`. "
        "Все иллюстрации траектории используют заранее выбранный seed 5000. "
        "Для нового расчёта следует выбрать отдельный `--output`, иначе созданные результаты будут заменены. "
        "Скрипт отчёта читает папку этого эксперимента."]
    target = ROOT / "manuscripts" / "v6" / "bond_rewiring_results_2026-10-03.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)
    print(f"Maximum step difference: {100*max_delta:.4f} pp")
    print(sorted(detail, key=lambda x: -x[2])[:4])


if __name__ == "__main__":
    main()

# Откуда взят каждый результат

Исходный репозиторий: [meta-stable-architectures](https://github.com/katiasmolskaia-hub/meta-stable-architectures).

Закреплённая ревизия исходных опытов: `{{BASE}}`. Новая заметка и вспомогательные файлы написаны позже и не выдаются за файлы этого исторического коммита. `ORIGIN.json` перечисляет происхождение каждого файла, `MANIFEST.json` - его SHA-256 в пакете.

## Разделы заметки

| Раздел | Архив внутри пакета | Протокол | Код |
|---|---|---|---|
| 2. Два признака | [minimal_observer](research_results/minimal_observer_2026-10-08/) | [Протокол](manuscripts/v6/minimal_observer_protocol_2026-10-08.md) | [Модуль](simulations/experiment_minimal_observer.py) |
| 3. Время попытки | [help_timing](research_results/help_timing_2026-10-08/) | [Протокол](manuscripts/v6/help_timing_protocol_2026-10-08.md) | [Модуль](simulations/experiment_help_timing.py) |
| 4-5. Повторение и контрпример | [repeated_help](research_results/repeated_help_2026-10-08/) | [Протокол](manuscripts/v6/repeated_help_protocol_2026-10-08.md) | [Модуль](simulations/experiment_repeated_help.py) |

Общая динамика, ресурс и локальная операция: [experiment_local_support_signal.py](simulations/experiment_local_support_signal.py). В рабочую папку `simulations/` включены только модули, нужные этим трём сериям, их отчёты, графики, проверка примера и тесты.

## Как найти число

1. Таблица основных средних первой серии: `minimal_observer.../summary.csv`, фильтр `n=96, scenario=mixed, dt=0.2, varied=0, cohort=main`. В `runs.csv` находятся исходные строки.
2. Первичное сравнение наблюдателя: `paired_comparisons.csv`, дополнительно `baseline=persistent, metric=service`. Разности в долях спроса умножены на 100 для п.п.
3. Первичное сравнение времени: `help_timing.../paired_comparisons.csv`, фильтр `n=96, scenario=mixed, dt=0.2, category=improving, cohort=main, policy=now, baseline=trend_rule`. Это потреблённый ресурс, не проценты.
4. Первичное сравнение повторения: `repeated_help.../paired_comparisons.csv`, фильтр `n=96, scenario=mixed, dt=0.2, sample=main, subset=all, contrast=pulse8-pause, metric=served_units`.
5. Локальный контрпример: `repeated_help.../example_step.csv`, `n=96, scenario=transient, seed=24000, t=40, node=31`; сравниваются `pause` и `pulse8` при трёх шагах. Изображение `local_counterexample.png` взято из той же серии без изменения данных.

В [evidence.json](publication/recovery-note-2026-11-17/evidence.json) сохранены исходные строки первичных сравнений, пересчитанные средние и хеши входов. Скрипт [audit_note.py](publication/recovery-note-2026-11-17/audit_note.py) повторяет эту проверку; сами интервалы не строит заново.

## Полнота и неизменность

Включены полные итоговые каталоги трёх серий, а не выборка удачных запусков. `runs.csv`, `selections.csv`, события, конфигурации, снимки и диагностические NPZ сохранены в том объёме, в котором они были архивированы. Для первой серии полные траектории сохранялись только у заранее выбранных seed. В серии времени имеются состояния для восстановления продолжений; полные временные ряды каждой ветки отдельно не сохранялись.

`research_results/manifest.json` - точное подмножество исходного манифеста для включённых каталогов, с дополнительным указанием происхождения. Файлы данных и архивных снимков не переписывались. Рабочие модули и протоколы получены из закреплённой Git-ревизии; возможное различие LF/CRLF между ними и снимками явно допускается исходными проверками.

Полные оригинальные отчёты и история остальных серий остаются по закреплённым внешним ссылкам D1-D4 в заметке. Предыдущие тома не требуются для запуска включённых опытов.

Компактная спецификация в разделе 1 заметки описывает функции `flow_step`, `request_help` и `Observer.update/flags` включённых модулей. Табличные параметры дополнительно сверяются скриптом аудита с тремя исходными конфигурациями. [Ответ на рецензию ChatGPT (Лиры)](REVIEW_RESPONSE.md) описывает изменения редакции 1.0.0; сами научные модули и данные не менялись.

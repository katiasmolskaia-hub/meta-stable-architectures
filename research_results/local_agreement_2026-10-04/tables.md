# Изолированное согласование: полные результаты

По 200 seed на строку. Отказ от обмена не считается успешным созданием новых связей. Незавершённость измеряется на такте 80, не на бесконечном горизонте.

| Канал | Вариант | Обмен завершён, % | Частичная смена к концу, % | Есть зависшие обещания, % | Чистый отказ от обмена, % | Сообщений, среднее |
|---|---|---:|---:|---:|---:|---:|
| reliable_delay | fixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| reliable_delay | ideal_atomic | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| reliable_delay | command_once | 100.0 | 0.0 | 0.0 | 0.0 | 3.00 |
| reliable_delay | confirmed_once | 100.0 | 0.0 | 0.0 | 0.0 | 12.00 |
| reliable_delay | confirmed_retry | 100.0 | 0.0 | 0.0 | 0.0 | 21.65 |
| loss_10 | fixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| loss_10 | ideal_atomic | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| loss_10 | command_once | 73.5 | 26.5 | 0.0 | 0.0 | 3.00 |
| loss_10 | confirmed_once | 38.5 | 16.5 | 29.0 | 32.5 | 11.37 |
| loss_10 | confirmed_retry | 100.0 | 0.0 | 0.0 | 0.0 | 22.72 |
| loss_30 | fixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| loss_30 | ideal_atomic | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| loss_30 | command_once | 29.0 | 71.0 | 0.0 | 0.0 | 3.00 |
| loss_30 | confirmed_once | 4.5 | 6.0 | 47.5 | 48.0 | 10.18 |
| loss_30 | confirmed_retry | 72.5 | 0.0 | 0.0 | 27.5 | 26.89 |
| loss_duplicates | fixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| loss_duplicates | ideal_atomic | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| loss_duplicates | command_once | 73.5 | 26.5 | 0.0 | 0.0 | 3.00 |
| loss_duplicates | confirmed_once | 40.5 | 19.5 | 30.0 | 29.5 | 12.94 |
| loss_duplicates | confirmed_retry | 100.0 | 0.0 | 0.0 | 0.0 | 25.70 |
| temporary_partition | fixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| temporary_partition | ideal_atomic | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| temporary_partition | command_once | 0.0 | 100.0 | 0.0 | 0.0 | 3.00 |
| temporary_partition | confirmed_once | 0.0 | 0.0 | 0.0 | 100.0 | 10.00 |
| temporary_partition | confirmed_retry | 0.0 | 0.0 | 0.0 | 100.0 | 22.25 |
| final_never_arrives | fixed | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| final_never_arrives | ideal_atomic | 100.0 | 0.0 | 0.0 | 0.0 | 0.00 |
| final_never_arrives | command_once | 0.0 | 100.0 | 0.0 | 0.0 | 3.00 |
| final_never_arrives | confirmed_once | 0.0 | 100.0 | 100.0 | 0.0 | 11.00 |
| final_never_arrives | confirmed_retry | 0.0 | 100.0 | 100.0 | 0.0 | 32.78 |

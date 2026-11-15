# Запуск на ПК товарища и в облаке

Фактически доступный агенту GPU — RTX 3050 Laptop 4GB. RTX 3060 12GB находится у товарища. Проверка на его устройстве и в облаке пока не выполнена. Скачать runtime здесь не удалось: запрос сетевого доступа был отклонён; его не повторяем без нового разрешения.

## RTX 3060 12GB, Windows

Исходники находятся в ветке `prototype`. Проверен Windows x64 / Python 3.12; для GPU нужны доступ к GitHub/Hugging Face и совместимый NVIDIA-драйвер. После клонирования выполнить `git switch prototype`. Не переносить `.venv/`, `.tools/` и runtime-manifest с другого ПК: пути и бинарные пакеты привязаны к окружению.

Сначала создать отдельное окружение. Команды из корня проекта; активация venv не нужна:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-prototype.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m avito serve
```

Если `py -3.12` не найден, нужен установленный Python 3.12 x64; можно указать полный путь к его `python.exe`. Обычный `python` может выбирать Anaconda или Python 3.10. В venv проект использует его пакеты; прежняя `.tools/python` остаётся совместимостью для запуска вне venv. Воспроизведены установка закреплённых пакетов с PyPI, `pip check`, 40 тестов и CLI на чистой копии исходников без `.tools/python` и исходных данных. Это локальная проверка на Windows, не запуск на ПК товарища.

Веб-страницу можно открыть без модели. `/health` проверяет приложение, а не LLM; до запуска модели анализ должен возвращать явную ошибку `503 model_unavailable`.

Только при готовности скачивать модель (~5 GB) и runtime, отдельным шагом:

```powershell
.\.venv\Scripts\python.exe scripts/setup_local_llm.py --install
powershell -ExecutionPolicy Bypass -File scripts/start_local_llm.ps1
```

Установщик берёт официальный GGUF Qwen3-8B Q4_K_M (5 027 783 488 байт), закреплённую ревизию и SHA256; runtime — официальный llama.cpp через проверяемый указатель релиза v1.0.0. Перед использованием скачанные архивы проверяются по опубликованным SHA256. Если подходящей CUDA12 сборки/хэша нет, установка останавливается. Повторный запуск проверяет имеющиеся файлы без перезаписи; повреждённые или незавершённые файлы сохраняются и вызывают ошибку. До первого успешного скачивания совместимость выбранного бинарного релиза остаётся непроверенной.

В другом терминале:

```powershell
.\.venv\Scripts\python.exe -m avito serve
```

Открыть `http://127.0.0.1:7860`. Модель обслуживает запросы только на localhost:8080; приложение — localhost:7860. Окно сервера модели должно оставаться открытым. Доступ из интернета и туннели не нужны. Если GPU не хватает памяти, передать `-Context 4096 -GpuLayers 20` PowerShell-скрипту; это другая измеряемая конфигурация, её скорость не считать скоростью полного offload на 3060.

## Baseline и данные

ZIP с объявлениями и frozen-файлы не входят в Git. Для нового checkout только с исходным ZIP сначала восстановить v1, затем создать v2; существующие файлы не перезаписываются:

```powershell
$ExperimentArchive = 'путь/innoglobalhack_avito(1).zip'
.\.venv\Scripts\python.exe scripts/prepare_review.py --archive $ExperimentArchive
.\.venv\Scripts\python.exe scripts/prepare_data_v2.py --archive $ExperimentArchive
.\.venv\Scripts\python.exe scripts/train_baseline.py --archive $ExperimentArchive --data-dir data/v2 --features char --output-dir data/runs/baseline-v2-new
# E1: меняется только набор признаков; указать другую новую выходную папку.
.\.venv\Scripts\python.exe scripts/train_baseline.py --archive $ExperimentArchive --data-dir data/v2 --features char-word --output-dir data/runs/char-word-v2-new
```

Для текущей v2 ожидаются 8944 train-строки и 208 public development. Загрузчик сверяет ZIP, контрольные суммы, происхождение и полное исключение public/legacy/control-групп. Если уже получена приватная `data/v2/`, её не пересоздавать; нужен тот же ZIP. `--output-dir` всегда новый. Метки остаются шумными; правила human40 и контроля — в [DATA.md](DATA.md).

На чистой копии кода проверена именно последовательность `prepare_review.py` → `prepare_data_v2.py` из одного ZIP: 13 файлов v2 совпали с текущей версией по SHA256. Без подготовки данных один `git clone` не содержит материалов для обучения. Для открытия веб-демо данные хакатона не требуются.

Если у товарища остаётся ошибка, для диагностики нужны команда, полный traceback и вывод `.\.venv\Scripts\python.exe -VV` / `-m pip check`. Не присылать ключи, объявления и контрольные ответы. Проверка NVIDIA/runtime/реальной генерации остаётся отдельной задачей при доступе к 3060.

## Colab / Kaggle

Открыть `notebooks/avito_gpu.ipynb`, выбрать GPU и приватную сессию с интернетом. В Colab GPU не гарантирован; в Kaggle квота и доступ зависят от аккаунта. Notebook выполняет вычисления внутри сессии и не поднимает публичный веб-сервис. Для финального демо предпочтительнее устойчивый запуск на ПК товарища.

Для загрузки кода выполнить `python scripts/package_notebook.py`. Полученный ZIP содержит только исходники и requirements. В приватную сессию отдельно добавить этот ZIP и исходный ZIP данных. Notebook содержит явные настройки путей; данные, предсказания, веса и секреты не публиковать вместе с ноутбуком. Для приватного GitHub не вставлять токен в ячейки: source-bundle обходится без токена.

После старта попросите товарища прислать **только** название GPU, версию runtime, успешный `/health` и результат на синтетическом примере с временем ответа. Секреты и реальные контрольные объявления в чат не отправлять. Это даст свидетельство реального запуска на 3060.

Источники: [Qwen GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp), [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Kaggle notebooks](https://www.kaggle.com/docs/notebooks). Проверено 14.11.2026; запуск облачного notebook пока не проверен.

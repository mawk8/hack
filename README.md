# Avito Drafts

Из объявления «Ремонт под ключ» (101) выделяем услуги 102–106: **текст → поиск услуг с цитатами → выбор пользователя → черновики**. Есть одиночный ввод и CSV. Публикацию пользователь выполняет отдельно.

**Статус:** Python-зависимости и приложение проверены на Windows/Python 3.12 с синтетическими ответами. Реальный запуск Qwen на GPU, его скорость и фактичность ещё не подтверждены.

## Запуск на Windows

Полная пошаговая инструкция с проверками и диагностикой: **[docs/LOCAL_RUN.md](docs/LOCAL_RUN.md)**.

Для новой копии нужны Git, Python 3.12 x64 и PowerShell. Если `E:\hackaton` уже существует, используйте раздел обновления в инструкции, не клонируйте поверх файлов.

```powershell
Set-Location E:\
git clone https://github.com/mawk8/hack.git hackaton
Set-Location E:\hackaton
git switch prototype
git pull --ff-only origin prototype
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-prototype.txt
.\.venv\Scripts\python.exe -m pip check
```

При готовности скачать около 5 GB модели плюс llama.cpp runtime, затем оставить модель работающей в **терминале №1**:

```powershell
.\.venv\Scripts\python.exe scripts/setup_local_llm.py --install
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_local_llm.ps1
```

В **терминале №2**, после завершения загрузки модели:

```powershell
Set-Location E:\hackaton
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health'
Invoke-RestMethod -Uri 'http://127.0.0.1:8080/v1/models'
.\.venv\Scripts\python.exe -m avito serve
```

Откройте [http://127.0.0.1:7860](http://127.0.0.1:7860). Frontend — встроенная HTML-страница backend; Node.js/npm и третий сервер не нужны. Оба терминала оставьте открытыми. Для остановки каждого сервера — `Ctrl+C` в его окне.

Нажмите «Найти услуги», проверьте цитаты, отметьте категории, затем создайте выбранные черновики. Для CSV: UTF-8/BOM, вход `testcase,description`; JSON сохраняет весь автоматический список отдельно от выбора, CSV для сдачи — `testcase,microcats,descriptions` с JSON-массивами. CSV для сдачи доступен после успешного обнаружения и генерации для **всех** найденных категорий; подробности и синтетический пример — в инструкции.

Для открытия демо не нужны данные хакатона и обучение baseline. Backend `/health` проверяет только приложение; без запущенной LLM анализ вернёт явную ошибку `503 model_unavailable`. Ранее скачивание runtime в агентской сессии было отклонено; эти команды не означают, что оно уже выполнено или совместимость подтверждена.

## Документы и устройство

1. [Актуальный план](docs/PLAN.md) — требования и ближайшие действия.
2. [Статус и измерения](docs/STATUS.md) — выполненные проверки и ограничения.
3. [Локальный запуск](docs/LOCAL_RUN.md) · [Эксперименты](docs/EXPERIMENTS.md).
4. [Данные и правила разбиения](docs/DATA.md).

Агенту: сначала [AGENTS.md](AGENTS.md), затем план и статус; историю обсуждений читать по необходимости.

| Путь | Содержимое |
|---|---|
| `avito/` | Приложение, классификатор, LLM-клиент и интерфейс |
| `scripts/`, `tests/` | Команды запуска/оценки и проверки |
| `tools/review.html`, `notebooks/` | Ручная разметка и подготовленный GPU-notebook |
| `docs/PLAN.md`, `docs/STATUS.md` | Текущий план и доказанные результаты |
| `docs/stage1/`, `docs/superpowers/plans/`, `output/pdf/` | Исторические материалы |

**Ветки:** `main` — входная страница; `prototype` — рабочий код MVP. В Git входят код, тесты и необходимые документы. `data/`, `.tools/`, `tmp/`, `.superpowers/`, реальные объявления, ключи, кэши и веса остаются локально. Файлы автоматически не удаляем.

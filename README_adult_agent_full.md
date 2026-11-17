# Adult Linux Agent Full

Локальный dev/Linux агент на **Ollama + LangGraph**.
Есть CLI и простой GUI на Tkinter. Без сервера, без трея, без голосового ввода.

## Что добавлено

- `gemma4:e2b` как дефолтная модель.
- Дефолтный workspace теперь `.` вместо странного выхода через `../../../...`.
- JSON tool-calling.
- Многошаговый граф: `planner -> tool_executor -> planner -> final -> critic`.
- Critic-проверка финального ответа.
- Workspace-изоляция.
- Подтверждения для опасных действий.
- Более безопасный запуск команд: confirmed-команды идут через `shlex.split` и `shell=False`.
- `read_file_range` для чтения строк кода.
- `replace_text` с backup.
- `list_backups`.
- `git_status`, `git_diff`.
- `ollama_status`, чтобы видеть CPU/GPU split через `ollama ps`.
- `python_compile` без импорта проекта.
- `project_index_build`, `project_index_search`.
- История, память, summary истории.
- Простая GUI-оболочка.

## Установка

```bash
pip install -r requirements_adult_agent_full.txt
```

Для GUI на Arch/CachyOS может понадобиться Tkinter:

```bash
sudo pacman -S tk
```

Ollama-модель:

```bash
ollama pull gemma4:e2b
```

## Запуск CLI

```bash
python adult_linux_agent_full.py --workspace . --model gemma4:e2b
```

Для проекта:

```bash
python adult_linux_agent_full.py --workspace ~/Projects/clipx --model gemma4:e2b
```

Можно переключить модель через env:

```bash
AGENT_MODEL=gemma4:e4b python adult_linux_agent_full.py --workspace ~/Projects/clipx
```

## Запуск GUI

```bash
python adult_linux_agent_gui.py
```

В GUI выбери модель, workspace и нажми **Запустить / перезапустить**.

## Полезные CLI-команды

```text
/help
/status
/workspace
/memory
/history
/summary
/index
/index-search subprocess
/git
/diff
/backups
/clear-history
/exit
```

## Примеры запросов

```text
посмотри состояние системы и ollama
построй индекс проекта
найди где используется subprocess
прочитай main.py строки 120-220
найди все TODO
исправь ошибку в main.py и запусти py_compile
запусти pytest -q
покажи git diff после правок
```

## Где хранятся данные

По умолчанию в `.agent_runtime`:

```text
.agent_runtime/history.jsonl
.agent_runtime/memory.jsonl
.agent_runtime/summary.json
.agent_runtime/project_index.json
.agent_runtime/backups/
.agent_runtime/logs/
```

## Важное про безопасность

Агент не может выходить за пределы workspace файловыми инструментами.

Запись файлов, замена текста и запуск тестов требуют ручного подтверждения.
Опасные команды вроде `rm -rf`, `sudo`, `systemctl`, `mkfs`, `dd` блокируются.

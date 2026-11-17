<div align="center">

# Linux Copilot Lite

**Локальный Linux/dev-агент с Ollama, LangGraph и интерфейсом CLI/Tkinter.**
**A local Linux/dev agent powered by Ollama and LangGraph, with CLI and Tkinter interfaces.**

[Русский](#русский) · [English](#english)

</div>

---

## Русский

Linux Copilot Lite — экспериментальный локальный помощник для работы с Linux и кодом. Он использует Ollama для языковой модели, LangGraph для управления циклом агента и предоставляет CLI и простой графический интерфейс Tkinter.

### Возможности

- Работа с локальной моделью Ollama и настройка параметров генерации.
- Поиск и чтение файлов в выбранном workspace, просмотр структуры и построение простого индекса проекта.
- Ограниченный запуск команд, инструменты Git и компиляция Python.
- Изменение файлов с запросом подтверждения и созданием резервной копии.
- История диалога, сводка, локальная память и журналирование.
- CLI и GUI на Tkinter; сеть/сервер приложения не требуются.

### Быстрый старт

Требуются Python 3.11+, установленный и запущенный [Ollama](https://ollama.com/) и загруженная в Ollama модель.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_adult_agent_tuned.txt
ollama pull gemma4:e2b
python adult_linux_agent_tuned.py --workspace . --model gemma4:e2b
```

Для базовой версии установите `requirements_adult_agent_full.txt` и запустите:

```bash
python adult_linux_agent_full.py --workspace . --model gemma4:e2b
```

### Графический интерфейс

```bash
python adult_linux_agent_tuned_gui.py
```

Базовый GUI для версии full: `python adult_linux_agent_gui.py`. В Linux может потребоваться отдельный пакет Tkinter, например `python-tk` или `tk`.

### Версии

| Файл | Назначение |
| --- | --- |
| `adult_linux_agent_tuned.py` | Основной CLI-агент с расширенными настройками Ollama |
| `adult_linux_agent_tuned_gui.py` | GUI для tuned-версии |
| `adult_linux_agent_full.py` | Базовая CLI-версия |
| `adult_linux_agent_gui.py` | GUI для базовой версии |
| `main.py`, `main1.py` | Ранние минимальные прототипы |
| `test.py` | Ручная проверка вызова Ollama, не автоматический набор тестов |

### Безопасность и приватность

**Это экспериментальный проект, а не защищённая песочница.** Файловые инструменты ограничивают пути выбранным workspace, а часть действий запрашивает подтверждение. Однако проверка команд основана на правилах и не гарантирует безопасное выполнение произвольного недоверенного ввода. Не запускайте агента с доступом к важным данным или привилегиям, если не понимаете последствия.

История, память, журналы и резервные копии хранятся локально в `.agent_runtime/`. Не добавляйте эту папку в публичный репозиторий: там могут оказаться запросы пользователя и содержимое файлов. Ollama-модели распространяются по собственным условиям; MIT-лицензия этого проекта на них не распространяется.

### Лицензия

Код проекта распространяется по лицензии [MIT](LICENSE).

---

## English

Linux Copilot Lite is an experimental local assistant for Linux and software-development tasks. It uses Ollama for the language model, LangGraph to orchestrate the agent loop, and provides a CLI plus a lightweight Tkinter GUI.

### Features

- Run an Ollama model and configure its generation parameters.
- Search and read files in a selected workspace, inspect its structure, and build a simple project index.
- Run a limited set of commands, use Git tools, and compile Python files.
- Edit files with confirmation prompts and automatic backups.
- Keep local conversation history, summaries, memory, and logs.
- Use the CLI or Tkinter GUI; the application does not require its own server.

### Quick start

Requires Python 3.11+, an installed and running [Ollama](https://ollama.com/) instance, and a model pulled into Ollama.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_adult_agent_tuned.txt
ollama pull gemma4:e2b
python adult_linux_agent_tuned.py --workspace . --model gemma4:e2b
```

To use the basic version, install `requirements_adult_agent_full.txt` and run:

```bash
python adult_linux_agent_full.py --workspace . --model gemma4:e2b
```

### Graphical interface

```bash
python adult_linux_agent_tuned_gui.py
```

The basic GUI is `python adult_linux_agent_gui.py`. On Linux, Tkinter may need to be installed separately, for example through a `python-tk` or `tk` system package.

### Versions

| File | Purpose |
| --- | --- |
| `adult_linux_agent_tuned.py` | Main CLI agent with extended Ollama settings |
| `adult_linux_agent_tuned_gui.py` | GUI for the tuned version |
| `adult_linux_agent_full.py` | Basic CLI version |
| `adult_linux_agent_gui.py` | GUI for the basic version |
| `main.py`, `main1.py` | Early minimal prototypes |
| `test.py` | Manual Ollama invocation check, not an automated test suite |

### Security and privacy

**This is an experimental project, not a security sandbox.** File tools restrict paths to the selected workspace, and some actions require confirmation. However, command checks are rule-based and do not guarantee safe execution of arbitrary untrusted input. Do not run the agent with access to sensitive data or elevated privileges unless you understand the consequences.

Conversation history, memory, logs, and backups are stored locally in `.agent_runtime/`. Do not publish that directory: it may contain user prompts and file contents. Ollama models are distributed under their own terms; this project's MIT license does not apply to them.

### License

This project is distributed under the [MIT License](LICENSE).

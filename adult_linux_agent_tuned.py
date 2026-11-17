#!/usr/bin/env python3
"""
adult_linux_agent_full.py

Локальный CLI/dev-агент на LangGraph + Ollama.
Без сервера, без трея, без голосового ввода.

Что есть:
- JSON tool-calling вместо RUN:
- многошаговый граф planner -> tool_executor -> planner -> final -> critic
- безопасные инструменты для Linux, проекта, git и Ollama
- workspace-изоляция и защита от выхода за пределы workspace
- подтверждение для записи файлов, запуска кода и тестов
- read_file_range, replace_text с backup, git_status/git_diff
- project index по файлам и символам Python/Markdown
- краткосрочная история + summary.json + простая долговременная память JSONL
- ручные настройки генерации: num_predict, num_ctx, temperature, top_p/top_k, repeat_penalty, seed, keep_alive, max_steps
- CLI slash-команды и простая основа для GUI через импорт этого файла
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import queue
import re
import shlex
import subprocess
import sys
import textwrap
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NotRequired, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# =========================
# Configuration
# =========================

DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gemma4:e2b")
DEFAULT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0"))
DEFAULT_NUM_PREDICT = int(os.getenv("AGENT_NUM_PREDICT", "512"))
DEFAULT_NUM_CTX = int(os.getenv("AGENT_NUM_CTX", "4096"))
DEFAULT_TOP_P = float(os.getenv("AGENT_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("AGENT_TOP_K", "40"))
DEFAULT_REPEAT_PENALTY = float(os.getenv("AGENT_REPEAT_PENALTY", "1.1"))
DEFAULT_SEED_RAW = os.getenv("AGENT_SEED", "").strip()
DEFAULT_SEED = int(DEFAULT_SEED_RAW) if DEFAULT_SEED_RAW else None
DEFAULT_KEEP_ALIVE = os.getenv("AGENT_KEEP_ALIVE", "5m")
DEFAULT_NUM_GPU_RAW = os.getenv("AGENT_NUM_GPU", "").strip()
DEFAULT_NUM_GPU = int(DEFAULT_NUM_GPU_RAW) if DEFAULT_NUM_GPU_RAW else None
DEFAULT_NUM_THREAD_RAW = os.getenv("AGENT_NUM_THREAD", "").strip()
DEFAULT_NUM_THREAD = int(DEFAULT_NUM_THREAD_RAW) if DEFAULT_NUM_THREAD_RAW else None
DEFAULT_NUM_BATCH_RAW = os.getenv("AGENT_NUM_BATCH", "").strip()
DEFAULT_NUM_BATCH = int(DEFAULT_NUM_BATCH_RAW) if DEFAULT_NUM_BATCH_RAW else None
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("AGENT_MAX_TOOL_OUTPUT_CHARS", "12000"))
MAX_READ_FILE_CHARS = int(os.getenv("AGENT_MAX_READ_FILE_CHARS", "16000"))
MAX_GREP_RESULTS = int(os.getenv("AGENT_MAX_GREP_RESULTS", "120"))
MAX_FIND_RESULTS = int(os.getenv("AGENT_MAX_FIND_RESULTS", "200"))
MAX_INDEX_FILES = int(os.getenv("AGENT_MAX_INDEX_FILES", "5000"))
ENABLE_CRITIC = os.getenv("AGENT_ENABLE_CRITIC", "1").strip().lower() not in {"0", "false", "no", "off"}

SAFE_READ_COMMANDS = {
    "pwd",
    "whoami",
    "uptime",
    "free -h",
    "df -h",
    "ls",
    "ls -lah",
    "uname -a",
    "ps aux --sort=-%cpu | head",
    "git status --short --branch",
    "git diff -- .",
    "ollama ps",
}

CONFIRM_COMMAND_PREFIXES = (
    "python ",
    "python3 ",
    "pytest",
    "uv run ",
    "poetry run ",
    "pip install ",
    "pip3 install ",
    "mkdir ",
    "touch ",
    "cp ",
    "mv ",
)

FORBIDDEN_COMMAND_PATTERNS = (
    "rm -rf",
    "rm -fr",
    "sudo ",
    "su ",
    "mkfs",
    "dd if=",
    ":(){",
    "chmod -r",
    "chown -r",
    "shutdown",
    "reboot",
    "poweroff",
    "systemctl ",
    "> /dev/",
    "/dev/sd",
    "pacman -r",
    "pacman -rc",
    "pip uninstall",
    "curl | sh",
    "curl -s",
    "wget | sh",
)

SHELL_META_FOR_CONFIRM = (";", "&&", "||", "|", "$(", "`", "\n", ">", "<")

TEXT_FILE_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".sh", ".fish", ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
    ".rs", ".go", ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".gd", ".r",
    ".sql", ".xml", ".csv", ".log", ".env", ".gitignore", ".dockerfile", ".service",
}

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    "dist", "build", "target", ".idea", ".vscode", ".ruff_cache", ".cache", ".agent_runtime",
}

RUNTIME_DIR: Path
WORKSPACE: Path
HISTORY_FILE: Path
MEMORY_FILE: Path
SUMMARY_FILE: Path
PROJECT_INDEX_FILE: Path
LOG_DIR: Path
BACKUP_DIR: Path

llm: ChatOllama
_CONFIRMATION_HANDLER: Callable[[str], bool] | None = None


# =========================
# Types
# =========================

class AgentState(TypedDict, total=False):
    question: str
    answer: str
    history: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    next_action: dict[str, Any]
    steps: int


class ToolDef(TypedDict):
    description: str
    args: dict[str, str]
    requires_confirmation: Any


class ToolResult(TypedDict):
    success: bool
    tool: str
    data: NotRequired[Any]
    error: NotRequired[str]
    warning: NotRequired[str]


@dataclass
class ModelSettings:
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    num_predict: int = DEFAULT_NUM_PREDICT
    num_ctx: int = DEFAULT_NUM_CTX
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    seed: int | None = DEFAULT_SEED
    keep_alive: str = DEFAULT_KEEP_ALIVE
    num_gpu: int | None = DEFAULT_NUM_GPU
    num_thread: int | None = DEFAULT_NUM_THREAD
    num_batch: int | None = DEFAULT_NUM_BATCH
    max_steps: int = MAX_STEPS
    enable_critic: bool = ENABLE_CRITIC


@dataclass
class AgentRuntime:
    model: str
    workspace: Path
    runtime_dir: Path
    app: Any
    settings: ModelSettings


AGENT_SETTINGS = ModelSettings()


# =========================
# Basic utilities
# =========================

def set_confirmation_handler(handler: Callable[[str], bool] | None) -> None:
    global _CONFIRMATION_HANDLER
    _CONFIRMATION_HANDLER = handler


def setup_paths(workspace: Path, runtime_dir: Path) -> None:
    global WORKSPACE, RUNTIME_DIR, HISTORY_FILE, MEMORY_FILE, SUMMARY_FILE, PROJECT_INDEX_FILE, LOG_DIR, BACKUP_DIR

    WORKSPACE = workspace.expanduser().resolve()
    RUNTIME_DIR = runtime_dir.expanduser().resolve()
    LOG_DIR = RUNTIME_DIR / "logs"
    BACKUP_DIR = RUNTIME_DIR / "backups"
    HISTORY_FILE = RUNTIME_DIR / "history.jsonl"
    MEMORY_FILE = RUNTIME_DIR / "memory.jsonl"
    SUMMARY_FILE = RUNTIME_DIR / "summary.json"
    PROJECT_INDEX_FILE = RUNTIME_DIR / "project_index.json"

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(
        filename=str(LOG_DIR / "agent.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def create_llm(settings: ModelSettings) -> ChatOllama:
    """Create ChatOllama with explicit manual generation parameters."""
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "temperature": settings.temperature,
        "num_predict": settings.num_predict,
        "num_ctx": settings.num_ctx,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "repeat_penalty": settings.repeat_penalty,
        "keep_alive": settings.keep_alive,
    }
    if settings.seed is not None:
        kwargs["seed"] = settings.seed
    if settings.num_gpu is not None:
        kwargs["num_gpu"] = settings.num_gpu
    if settings.num_thread is not None:
        kwargs["num_thread"] = settings.num_thread
    if settings.num_batch is not None:
        kwargs["num_batch"] = settings.num_batch
    return ChatOllama(**kwargs)


def init_agent(
    model: str = DEFAULT_MODEL,
    workspace: str | Path = ".",
    runtime_dir: str | Path = ".agent_runtime",
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
    num_ctx: int = DEFAULT_NUM_CTX,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    seed: int | None = DEFAULT_SEED,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    num_gpu: int | None = DEFAULT_NUM_GPU,
    num_thread: int | None = DEFAULT_NUM_THREAD,
    num_batch: int | None = DEFAULT_NUM_BATCH,
    max_steps: int = MAX_STEPS,
    enable_critic: bool | None = None,
) -> AgentRuntime:
    global llm, AGENT_SETTINGS, MAX_STEPS, ENABLE_CRITIC
    setup_paths(Path(workspace), Path(runtime_dir))
    if enable_critic is None:
        enable_critic = ENABLE_CRITIC
    MAX_STEPS = max(1, int(max_steps))
    ENABLE_CRITIC = bool(enable_critic)
    settings = ModelSettings(
        model=model,
        temperature=float(temperature),
        num_predict=int(num_predict),
        num_ctx=int(num_ctx),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
        seed=seed,
        keep_alive=keep_alive,
        num_gpu=num_gpu,
        num_thread=num_thread,
        num_batch=num_batch,
        max_steps=MAX_STEPS,
        enable_critic=ENABLE_CRITIC,
    )
    AGENT_SETTINGS = settings
    llm = create_llm(settings)
    app = build_graph()
    return AgentRuntime(model=model, workspace=WORKSPACE, runtime_dir=RUNTIME_DIR, app=app, settings=settings)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_json(obj: Any, limit: int | None = None) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if limit is not None:
        return truncate(text, limit)
    return text


def truncate(text: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if text is None:
        return ""
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, indent=2)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n\n...[обрезано, всего символов: {len(value)}]"


def print_box(title: str, body: str) -> None:
    print(f"\n=== {title} ===")
    print(body)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        return rows[-limit:]
    return rows


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ask_confirmation(message: str) -> bool:
    if _CONFIRMATION_HANDLER is not None:
        try:
            return bool(_CONFIRMATION_HANDLER(message))
        except Exception:
            logging.exception("custom confirmation handler failed")
            return False

    if not sys.stdin.isatty():
        logging.warning("Confirmation denied because stdin is not a TTY: %s", message)
        return False
    print(f"\n⚠️  Нужно подтверждение: {message}")
    answer = input("Выполнить? [y/N]: ").strip().lower()
    return answer in {"y", "yes", "д", "да"}


def resolve_in_workspace(path_value: str | None = None) -> Path:
    path_value = path_value or "."
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve()
    else:
        candidate = (WORKSPACE / raw).resolve()

    try:
        candidate.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ValueError(f"Путь выходит за пределы workspace: {candidate}") from exc
    return candidate


def relative_to_workspace(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(path)


def is_probably_text_file(path: Path) -> bool:
    if path.name in {"Dockerfile", "Makefile", "LICENSE", "README", ".env"}:
        return True
    if path.suffix.lower() in TEXT_FILE_EXTENSIONS:
        return True
    if path.suffix == "":
        try:
            with path.open("rb") as f:
                sample = f.read(2048)
            return b"\x00" not in sample
        except OSError:
            return False
    return False


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def iter_project_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if should_skip(path):
            continue
        if path.is_file():
            files.append(path)
    files.sort(key=lambda p: str(p).lower())
    return files


# =========================
# Command safety
# =========================

def command_risk(command: str) -> tuple[str, str]:
    """Return (risk, reason): safe | confirm | forbidden."""
    command = command.strip()
    lower = command.lower()

    for pattern in FORBIDDEN_COMMAND_PATTERNS:
        if pattern in lower:
            return "forbidden", f"Найден опасный паттерн: {pattern}"

    if command in SAFE_READ_COMMANDS:
        return "safe", "Команда из read-only allowlist."

    if command.startswith(CONFIRM_COMMAND_PREFIXES):
        for meta in SHELL_META_FOR_CONFIRM:
            if meta in command:
                return "forbidden", f"Команда с shell-метасимволом запрещена для prefix-режима: {meta}"
        return "confirm", "Команда разрешена только после ручного подтверждения."

    return "forbidden", "Команды вне allowlist/confirm-prefix запрещены."


def run_subprocess(command: str, timeout: int = 20, *, force_shell: bool | None = None) -> dict[str, Any]:
    command = command.strip()
    use_shell = command in SAFE_READ_COMMANDS if force_shell is None else force_shell

    if use_shell:
        process_args: str | list[str] = command
    else:
        process_args = shlex.split(command)

    result = subprocess.run(
        process_args,
        shell=use_shell,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "cwd": str(WORKSPACE),
        "shell": use_shell,
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": truncate(result.stdout),
        "stderr": truncate(result.stderr),
    }


# =========================
# Project index helpers
# =========================

def extract_python_symbols(path: Path) -> list[dict[str, Any]]:
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return []

    symbols: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            symbols.append({"name": node.name, "kind": kind, "line": getattr(node, "lineno", None)})
    symbols.sort(key=lambda item: (item.get("line") or 0, item.get("name") or ""))
    return symbols


def extract_markdown_headings(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    headings = []
    for i, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            headings.append({"name": match.group(2).strip(), "kind": f"h{len(match.group(1))}", "line": i})
        if len(headings) >= 200:
            break
    return headings


def build_project_index(root: Path) -> dict[str, Any]:
    files = iter_project_files(root)[:MAX_INDEX_FILES]
    entries = []
    for file in files:
        if not is_probably_text_file(file):
            symbols: list[dict[str, Any]] = []
        elif file.suffix.lower() == ".py":
            symbols = extract_python_symbols(file)
        elif file.suffix.lower() == ".md":
            symbols = extract_markdown_headings(file)
        else:
            symbols = []

        try:
            stat = file.stat()
        except OSError:
            continue

        entries.append({
            "path": str(file.relative_to(WORKSPACE)),
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "ext": file.suffix.lower() or file.name,
            "symbols": symbols[:300],
        })

    by_ext: dict[str, int] = {}
    for entry in entries:
        ext = entry["ext"]
        by_ext[ext] = by_ext.get(ext, 0) + 1

    return {
        "created_at": now_iso(),
        "workspace": str(WORKSPACE),
        "root": str(root),
        "file_count": len(entries),
        "top_extensions": sorted(by_ext.items(), key=lambda kv: kv[1], reverse=True)[:30],
        "files": entries,
    }


# =========================
# Tools
# =========================

def tool_system_info(_: dict[str, Any]) -> ToolResult:
    outputs: dict[str, Any] = {}
    for command in ["pwd", "whoami", "uptime", "uname -a", "free -h", "df -h", "ps aux --sort=-%cpu | head"]:
        outputs[command] = run_subprocess(command, timeout=10, force_shell=True)
    return {"success": True, "tool": "system_info", "data": outputs}


def tool_ollama_status(_: dict[str, Any]) -> ToolResult:
    try:
        data = run_subprocess("ollama ps", timeout=10, force_shell=True)
        return {"success": bool(data["success"]), "tool": "ollama_status", "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "ollama_status", "error": repr(exc)}


def tool_run_command(args: dict[str, Any]) -> ToolResult:
    command = str(args.get("command", "")).strip()
    timeout = int(args.get("timeout", 20))

    if not command:
        return {"success": False, "tool": "run_command", "error": "Пустая команда."}

    risk, reason = command_risk(command)
    if risk == "forbidden":
        return {
            "success": False,
            "tool": "run_command",
            "error": f"Команда заблокирована: {command}",
            "warning": reason,
        }

    if risk == "confirm" and not ask_confirmation(f"запустить `{command}` в {WORKSPACE}"):
        return {
            "success": False,
            "tool": "run_command",
            "error": "Пользователь не подтвердил выполнение команды.",
            "warning": reason,
        }

    try:
        data = run_subprocess(command, timeout=timeout, force_shell=(risk == "safe"))
        return {"success": bool(data["success"]), "tool": "run_command", "data": data}
    except subprocess.TimeoutExpired:
        return {"success": False, "tool": "run_command", "error": f"Таймаут команды после {timeout} секунд."}
    except Exception as exc:  # noqa: BLE001
        logging.exception("run_command failed")
        return {"success": False, "tool": "run_command", "error": repr(exc)}


def tool_list_dir(args: dict[str, Any]) -> ToolResult:
    try:
        path = resolve_in_workspace(str(args.get("path", ".")))
        max_items = int(args.get("max_items", 120))
        if not path.exists():
            return {"success": False, "tool": "list_dir", "error": f"Путь не существует: {path}"}
        if not path.is_dir():
            return {"success": False, "tool": "list_dir", "error": f"Это не директория: {path}"}

        items = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:max_items]:
            rel = child.relative_to(WORKSPACE)
            items.append({
                "name": child.name,
                "path": str(rel),
                "type": "dir" if child.is_dir() else "file",
                "size": child.stat().st_size if child.is_file() else None,
            })
        return {"success": True, "tool": "list_dir", "data": {"path": str(path.relative_to(WORKSPACE)), "items": items}}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "list_dir", "error": str(exc)}


def tool_project_overview(args: dict[str, Any]) -> ToolResult:
    try:
        root = resolve_in_workspace(str(args.get("path", ".")))
        if not root.exists() or not root.is_dir():
            return {"success": False, "tool": "project_overview", "error": "Директория проекта не найдена."}

        files = iter_project_files(root)
        by_ext: dict[str, int] = {}
        total_size = 0
        important = []
        for file in files:
            total_size += file.stat().st_size
            ext = file.suffix.lower() or file.name
            by_ext[ext] = by_ext.get(ext, 0) + 1
            if file.name.lower() in {"readme.md", "pyproject.toml", "requirements.txt", "package.json", "cargo.toml", "go.mod", "main.py", "app.py"}:
                important.append(str(file.relative_to(WORKSPACE)))

        top_ext = sorted(by_ext.items(), key=lambda kv: kv[1], reverse=True)[:20]
        tree_preview = [str(p.relative_to(WORKSPACE)) for p in files[:120]]
        return {
            "success": True,
            "tool": "project_overview",
            "data": {
                "workspace": str(WORKSPACE),
                "project_path": str(root.relative_to(WORKSPACE)),
                "file_count": len(files),
                "total_size_bytes": total_size,
                "top_extensions": top_ext,
                "important_files": important[:40],
                "tree_preview": tree_preview,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "project_overview", "error": str(exc)}


def tool_find_files(args: dict[str, Any]) -> ToolResult:
    try:
        root = resolve_in_workspace(str(args.get("path", ".")))
        query = str(args.get("query", "")).lower().strip()
        glob_pattern = str(args.get("glob", "*")).strip() or "*"
        max_results = min(int(args.get("max_results", MAX_FIND_RESULTS)), 1000)

        results = []
        for path in root.rglob(glob_pattern):
            if should_skip(path):
                continue
            rel = str(path.relative_to(WORKSPACE))
            if query and query not in rel.lower():
                continue
            results.append({
                "path": rel,
                "type": "dir" if path.is_dir() else "file",
                "size": path.stat().st_size if path.is_file() else None,
            })
            if len(results) >= max_results:
                break
        return {"success": True, "tool": "find_files", "data": results}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "find_files", "error": str(exc)}


def tool_read_file(args: dict[str, Any]) -> ToolResult:
    try:
        path = resolve_in_workspace(str(args.get("path", "")))
        max_chars = min(int(args.get("max_chars", MAX_READ_FILE_CHARS)), 100_000)
        if not path.exists():
            return {"success": False, "tool": "read_file", "error": f"Файл не найден: {path}"}
        if not path.is_file():
            return {"success": False, "tool": "read_file", "error": f"Это не файл: {path}"}
        if path.stat().st_size > 2_000_000:
            return {"success": False, "tool": "read_file", "error": "Файл слишком большой для безопасного чтения."}
        if not is_probably_text_file(path):
            return {"success": False, "tool": "read_file", "error": "Файл не похож на текстовый."}

        content = path.read_text(encoding="utf-8", errors="replace")
        return {
            "success": True,
            "tool": "read_file",
            "data": {
                "path": str(path.relative_to(WORKSPACE)),
                "size": path.stat().st_size,
                "content": truncate(content, max_chars),
                "truncated": len(content) > max_chars,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "read_file", "error": str(exc)}


def tool_read_file_range(args: dict[str, Any]) -> ToolResult:
    try:
        path = resolve_in_workspace(str(args.get("path", "")))
        start = max(int(args.get("start", 1)), 1)
        end = int(args.get("end", start + 120))
        end = max(end, start)
        max_lines = min(int(args.get("max_lines", 250)), 1000)
        if end - start + 1 > max_lines:
            end = start + max_lines - 1

        if not path.exists():
            return {"success": False, "tool": "read_file_range", "error": f"Файл не найден: {path}"}
        if not path.is_file():
            return {"success": False, "tool": "read_file_range", "error": f"Это не файл: {path}"}
        if not is_probably_text_file(path):
            return {"success": False, "tool": "read_file_range", "error": "Файл не похож на текстовый."}

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[start - 1:end]
        content = "\n".join(f"{i}: {line}" for i, line in enumerate(selected, start=start))

        return {
            "success": True,
            "tool": "read_file_range",
            "data": {
                "path": str(path.relative_to(WORKSPACE)),
                "start": start,
                "end": min(end, len(lines)),
                "total_lines": len(lines),
                "content": truncate(content, MAX_READ_FILE_CHARS),
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "read_file_range", "error": str(exc)}


def create_backup(path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    rel = path.relative_to(WORKSPACE)
    safe_name = str(rel).replace("/", "__")
    backup_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}__{safe_name}.bak"
    backup_path = BACKUP_DIR / backup_name
    backup_path.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return backup_path


def tool_write_file(args: dict[str, Any]) -> ToolResult:
    try:
        path = resolve_in_workspace(str(args.get("path", "")))
        content = str(args.get("content", ""))
        mode = str(args.get("mode", "create")).strip().lower()
        if mode not in {"create", "overwrite", "append"}:
            return {"success": False, "tool": "write_file", "error": "mode должен быть create, overwrite или append."}

        if mode == "create" and path.exists():
            return {"success": False, "tool": "write_file", "error": "Файл уже существует. Используй mode=overwrite или append."}

        backup_path: Path | None = None
        action = {"create": "создать файл", "overwrite": "перезаписать файл", "append": "дописать в файл"}[mode]
        if not ask_confirmation(f"{action} `{path.relative_to(WORKSPACE)}`"):
            return {"success": False, "tool": "write_file", "error": "Пользователь не подтвердил изменение файла."}

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_file():
            backup_path = create_backup(path)

        if mode == "append":
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")

        data: dict[str, Any] = {"path": str(path.relative_to(WORKSPACE)), "mode": mode, "chars_written": len(content)}
        if backup_path:
            data["backup"] = str(backup_path)
        return {"success": True, "tool": "write_file", "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "write_file", "error": str(exc)}


def tool_replace_text(args: dict[str, Any]) -> ToolResult:
    try:
        path = resolve_in_workspace(str(args.get("path", "")))
        old = str(args.get("old", ""))
        new = str(args.get("new", ""))
        occurrence = int(args.get("occurrence", 1))

        if not old:
            return {"success": False, "tool": "replace_text", "error": "old пустой."}
        if occurrence < 1:
            return {"success": False, "tool": "replace_text", "error": "occurrence должен быть >= 1."}
        if not path.exists() or not path.is_file():
            return {"success": False, "tool": "replace_text", "error": "Файл не найден."}
        if not is_probably_text_file(path):
            return {"success": False, "tool": "replace_text", "error": "Файл не похож на текстовый."}

        content = path.read_text(encoding="utf-8", errors="replace")
        count = content.count(old)
        if count == 0:
            return {"success": False, "tool": "replace_text", "error": "Фрагмент old не найден."}
        if count > 1 and occurrence == 1:
            return {
                "success": False,
                "tool": "replace_text",
                "error": f"Фрагмент old найден {count} раз. Уточни occurrence или используй более длинный old.",
            }
        if occurrence > count:
            return {"success": False, "tool": "replace_text", "error": f"occurrence={occurrence}, но найдено только {count}."}

        preview = f"Файл: {path.relative_to(WORKSPACE)}\n\n--- OLD ---\n{truncate(old, 1600)}\n\n--- NEW ---\n{truncate(new, 1600)}"
        if not ask_confirmation(f"заменить фрагмент:\n{preview}"):
            return {"success": False, "tool": "replace_text", "error": "Пользователь не подтвердил замену."}

        backup_path = create_backup(path)
        parts = content.split(old)
        before = old.join(parts[:occurrence])
        after = old.join(parts[occurrence:])
        updated = before + new + after
        path.write_text(updated, encoding="utf-8")

        return {
            "success": True,
            "tool": "replace_text",
            "data": {
                "path": str(path.relative_to(WORKSPACE)),
                "backup": str(backup_path),
                "matched_count": count,
                "replaced_occurrence": occurrence,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "replace_text", "error": str(exc)}


def tool_list_backups(args: dict[str, Any]) -> ToolResult:
    try:
        limit = int(args.get("limit", 30))
        backups = []
        for path in sorted(BACKUP_DIR.glob("*.bak"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            backups.append({"path": str(path), "size": path.stat().st_size, "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")})
        return {"success": True, "tool": "list_backups", "data": backups}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "list_backups", "error": str(exc)}


def tool_grep_project(args: dict[str, Any]) -> ToolResult:
    try:
        root = resolve_in_workspace(str(args.get("path", ".")))
        pattern = str(args.get("pattern", ""))
        use_regex = bool(args.get("regex", False))
        case_sensitive = bool(args.get("case_sensitive", False))
        max_results = min(int(args.get("max_results", MAX_GREP_RESULTS)), 500)

        if not pattern:
            return {"success": False, "tool": "grep_project", "error": "Пустой pattern."}

        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags) if use_regex else None
        needle = pattern if case_sensitive else pattern.lower()

        results = []
        for file in iter_project_files(root):
            if len(results) >= max_results:
                break
            if not is_probably_text_file(file):
                continue
            if file.stat().st_size > 1_000_000:
                continue
            try:
                lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.lower()
                matched = bool(regex.search(line)) if regex else needle in haystack
                if matched:
                    results.append({
                        "path": str(file.relative_to(WORKSPACE)),
                        "line": i,
                        "text": truncate(line.strip(), 300),
                    })
                    if len(results) >= max_results:
                        break

        return {"success": True, "tool": "grep_project", "data": results}
    except re.error as exc:
        return {"success": False, "tool": "grep_project", "error": f"Невалидная regex: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "grep_project", "error": str(exc)}


def tool_scan_todos(args: dict[str, Any]) -> ToolResult:
    args = dict(args)
    args["pattern"] = r"\b(TODO|FIXME|XXX|HACK|BUG)\b"
    args["regex"] = True
    args["case_sensitive"] = False
    args.setdefault("max_results", 200)
    result = tool_grep_project(args)
    result["tool"] = "scan_todos"
    return result


def tool_git_status(_: dict[str, Any]) -> ToolResult:
    try:
        data = run_subprocess("git status --short --branch", timeout=10, force_shell=True)
        return {"success": bool(data["success"]), "tool": "git_status", "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "git_status", "error": repr(exc)}


def tool_git_diff(args: dict[str, Any]) -> ToolResult:
    try:
        path = str(args.get("path", ".")).strip() or "."
        _ = resolve_in_workspace(path)
        command = "git diff -- ." if path == "." else f"git diff -- {shlex.quote(path)}"
        data = run_subprocess(command, timeout=15, force_shell=True)
        return {"success": bool(data["success"]), "tool": "git_diff", "data": data}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "git_diff", "error": repr(exc)}


def tool_run_tests(args: dict[str, Any]) -> ToolResult:
    command = str(args.get("command", "pytest -q")).strip() or "pytest -q"
    allowed = command.startswith("pytest") or command.startswith("python -m pytest") or command.startswith("python3 -m pytest") or command.startswith("uv run pytest") or command.startswith("poetry run pytest")
    if not allowed:
        return {
            "success": False,
            "tool": "run_tests",
            "error": "run_tests принимает только pytest/python -m pytest/uv run pytest/poetry run pytest.",
        }
    if not ask_confirmation(f"запустить тесты `{command}` в {WORKSPACE}"):
        return {"success": False, "tool": "run_tests", "error": "Пользователь не подтвердил запуск тестов."}
    try:
        data = run_subprocess(command, timeout=int(args.get("timeout", 90)), force_shell=False)
        return {"success": bool(data["success"]), "tool": "run_tests", "data": data}
    except subprocess.TimeoutExpired:
        return {"success": False, "tool": "run_tests", "error": "Таймаут тестов."}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "run_tests", "error": repr(exc)}


def tool_python_compile(args: dict[str, Any]) -> ToolResult:
    try:
        path = resolve_in_workspace(str(args.get("path", ".")))
        targets = []
        if path.is_file():
            targets = [path]
        elif path.is_dir():
            targets = [p for p in iter_project_files(path) if p.suffix == ".py"]
        else:
            return {"success": False, "tool": "python_compile", "error": "Путь не найден."}

        errors = []
        checked = 0
        for target in targets[:1000]:
            checked += 1
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(target)],
                cwd=str(WORKSPACE),
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                errors.append({"path": str(target.relative_to(WORKSPACE)), "stderr": truncate(result.stderr, 4000)})

        return {
            "success": len(errors) == 0,
            "tool": "python_compile",
            "data": {"checked": checked, "errors": errors[:50]},
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "python_compile", "error": repr(exc)}


def tool_project_index_build(args: dict[str, Any]) -> ToolResult:
    try:
        root = resolve_in_workspace(str(args.get("path", ".")))
        if not root.exists() or not root.is_dir():
            return {"success": False, "tool": "project_index_build", "error": "Директория не найдена."}
        index = build_project_index(root)
        write_json(PROJECT_INDEX_FILE, index)
        return {"success": True, "tool": "project_index_build", "data": {k: v for k, v in index.items() if k != "files"} | {"index_file": str(PROJECT_INDEX_FILE)}}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "project_index_build", "error": str(exc)}


def tool_project_index_search(args: dict[str, Any]) -> ToolResult:
    try:
        query = str(args.get("query", "")).lower().strip()
        limit = int(args.get("limit", 30))
        if not query:
            return {"success": False, "tool": "project_index_search", "error": "Пустой query."}
        index = read_json(PROJECT_INDEX_FILE, {})
        if not index:
            return {"success": False, "tool": "project_index_search", "error": "Индекс ещё не построен. Сначала вызови project_index_build."}
        terms = [t for t in re.split(r"\s+", query) if t]
        scored = []
        for entry in index.get("files", []):
            blob = json.dumps(entry, ensure_ascii=False).lower()
            score = sum(blob.count(term) for term in terms)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return {"success": True, "tool": "project_index_search", "data": [entry for _, entry in scored[:limit]]}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "tool": "project_index_search", "error": str(exc)}


def tool_memory_add(args: dict[str, Any]) -> ToolResult:
    text = str(args.get("text", "")).strip()
    tags = args.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not text:
        return {"success": False, "tool": "memory_add", "error": "Пустой текст памяти."}
    row = {"created_at": now_iso(), "text": text, "tags": tags}
    append_jsonl(MEMORY_FILE, row)
    return {"success": True, "tool": "memory_add", "data": row}


def tool_memory_search(args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query", "")).lower().strip()
    limit = int(args.get("limit", 10))
    rows = read_jsonl(MEMORY_FILE)
    if not query:
        return {"success": True, "tool": "memory_search", "data": rows[-limit:]}

    terms = [t for t in re.split(r"\s+", query) if t]
    scored = []
    for row in rows:
        blob = json.dumps(row, ensure_ascii=False).lower()
        score = sum(blob.count(term) for term in terms)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {"success": True, "tool": "memory_search", "data": [row for _, row in scored[:limit]]}


def tool_memory_list(args: dict[str, Any]) -> ToolResult:
    limit = int(args.get("limit", 30))
    return {"success": True, "tool": "memory_list", "data": read_jsonl(MEMORY_FILE, limit=limit)}


def tool_history_summary(_: dict[str, Any]) -> ToolResult:
    summary = read_json(SUMMARY_FILE, {})
    return {"success": True, "tool": "history_summary", "data": summary or {"message": "summary пока нет"}}


TOOLS: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
    "system_info": tool_system_info,
    "ollama_status": tool_ollama_status,
    "run_command": tool_run_command,
    "list_dir": tool_list_dir,
    "project_overview": tool_project_overview,
    "find_files": tool_find_files,
    "read_file": tool_read_file,
    "read_file_range": tool_read_file_range,
    "write_file": tool_write_file,
    "replace_text": tool_replace_text,
    "list_backups": tool_list_backups,
    "grep_project": tool_grep_project,
    "scan_todos": tool_scan_todos,
    "git_status": tool_git_status,
    "git_diff": tool_git_diff,
    "run_tests": tool_run_tests,
    "python_compile": tool_python_compile,
    "project_index_build": tool_project_index_build,
    "project_index_search": tool_project_index_search,
    "memory_add": tool_memory_add,
    "memory_search": tool_memory_search,
    "memory_list": tool_memory_list,
    "history_summary": tool_history_summary,
}

TOOL_SPECS: dict[str, ToolDef] = {
    "system_info": {"description": "Собрать базовую диагностику Linux: pwd, whoami, uptime, uname, RAM, disk, top CPU.", "args": {}, "requires_confirmation": False},
    "ollama_status": {"description": "Показать ollama ps, чтобы увидеть модель и CPU/GPU split.", "args": {}, "requires_confirmation": False},
    "run_command": {"description": "Запустить shell-команду. Read-only allowlist выполняется сразу, остальные разрешённые команды требуют подтверждения.", "args": {"command": "str", "timeout": "int optional"}, "requires_confirmation": "depends"},
    "list_dir": {"description": "Показать содержимое директории внутри workspace.", "args": {"path": "str optional, default .", "max_items": "int optional"}, "requires_confirmation": False},
    "project_overview": {"description": "Кратко изучить структуру проекта: файлы, расширения, важные конфиги.", "args": {"path": "str optional, default ."}, "requires_confirmation": False},
    "find_files": {"description": "Найти файлы по glob и/или подстроке пути внутри workspace.", "args": {"path": "str optional", "query": "str optional", "glob": "str optional", "max_results": "int optional"}, "requires_confirmation": False},
    "read_file": {"description": "Прочитать текстовый файл внутри workspace.", "args": {"path": "str", "max_chars": "int optional"}, "requires_confirmation": False},
    "read_file_range": {"description": "Прочитать диапазон строк текстового файла с номерами строк.", "args": {"path": "str", "start": "int", "end": "int", "max_lines": "int optional"}, "requires_confirmation": False},
    "write_file": {"description": "Создать/перезаписать/дописать текстовый файл внутри workspace. При изменении существующего файла создаёт backup.", "args": {"path": "str", "content": "str", "mode": "create|overwrite|append"}, "requires_confirmation": True},
    "replace_text": {"description": "Заменить точный фрагмент текста в файле. Создаёт backup и требует подтверждения.", "args": {"path": "str", "old": "str", "new": "str", "occurrence": "int optional"}, "requires_confirmation": True},
    "list_backups": {"description": "Показать последние backup-файлы, созданные агентом.", "args": {"limit": "int optional"}, "requires_confirmation": False},
    "grep_project": {"description": "Поиск по содержимому текстовых файлов проекта.", "args": {"path": "str optional", "pattern": "str", "regex": "bool optional", "case_sensitive": "bool optional", "max_results": "int optional"}, "requires_confirmation": False},
    "scan_todos": {"description": "Найти TODO/FIXME/XXX/HACK/BUG в проекте.", "args": {"path": "str optional", "max_results": "int optional"}, "requires_confirmation": False},
    "git_status": {"description": "Read-only git status --short --branch.", "args": {}, "requires_confirmation": False},
    "git_diff": {"description": "Read-only git diff для workspace или пути.", "args": {"path": "str optional"}, "requires_confirmation": False},
    "run_tests": {"description": "Запустить pytest в workspace. Всегда требует подтверждения, потому что выполняет код проекта.", "args": {"command": "str optional", "timeout": "int optional"}, "requires_confirmation": True},
    "python_compile": {"description": "Проверить Python-файлы через py_compile без импорта проекта.", "args": {"path": "str optional"}, "requires_confirmation": False},
    "project_index_build": {"description": "Построить простой индекс проекта: файлы, расширения, Python symbols, Markdown headings.", "args": {"path": "str optional"}, "requires_confirmation": False},
    "project_index_search": {"description": "Искать по сохранённому индексу проекта.", "args": {"query": "str", "limit": "int optional"}, "requires_confirmation": False},
    "memory_add": {"description": "Добавить долговременную заметку. Использовать только когда пользователь явно просит запомнить/сохранить.", "args": {"text": "str", "tags": "list[str] optional"}, "requires_confirmation": False},
    "memory_search": {"description": "Найти заметки в долговременной памяти.", "args": {"query": "str", "limit": "int optional"}, "requires_confirmation": False},
    "memory_list": {"description": "Показать последние заметки долговременной памяти.", "args": {"limit": "int optional"}, "requires_confirmation": False},
    "history_summary": {"description": "Показать текущую сжатую сводку истории.", "args": {}, "requires_confirmation": False},
}


# =========================
# LLM JSON handling
# =========================

def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"JSON object not found in model response: {truncate(text, 600)}")

    candidate = raw[start : end + 1]
    return json.loads(candidate)


def format_history(history: list[dict[str, str]], limit: int = 10) -> str:
    if not history:
        return "Истории пока нет."
    rows = []
    for msg in history[-limit:]:
        role = msg.get("role", "unknown")
        content = truncate(msg.get("content", ""), 1000)
        rows.append(f"{role}: {content}")
    return "\n".join(rows)


def format_summary() -> str:
    summary = read_json(SUMMARY_FILE, {})
    if not summary:
        return "Сводки истории пока нет."
    return safe_json(summary, 6000)


def format_recent_memory(limit: int = 12) -> str:
    rows = read_jsonl(MEMORY_FILE, limit=limit)
    if not rows:
        return "Долговременной памяти пока нет."
    return safe_json(rows, 6000)


def format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "Инструменты ещё не вызывались."
    safe_calls = []
    for call in tool_calls[-MAX_STEPS:]:
        safe_calls.append({
            "tool": call.get("tool"),
            "args": call.get("args"),
            "result": call.get("result"),
        })
    return truncate(safe_json(safe_calls), 18000)


def build_planner_prompt(state: AgentState) -> str:
    question = state["question"]
    history = format_history(state.get("history", []))
    summary = format_summary()
    memory = format_recent_memory()
    tool_calls = format_tool_calls(state.get("tool_calls", []))
    tool_specs = safe_json(TOOL_SPECS)
    tool_example = safe_json({
        "action": "tool",
        "tool_name": "read_file_range",
        "args": {"path": "main.py", "start": 1, "end": 120},
        "reasoning": "Нужно прочитать нужный диапазон строк."
    })
    final_example = safe_json({
        "action": "final",
        "answer": "Краткий полезный ответ пользователю.",
        "reasoning": "Данных хватает для ответа."
    })

    return f"""
Ты локальный Linux/dev-агент пользователя. Ты работаешь внутри workspace:
{WORKSPACE}

Твоя задача — помогать с Linux, проектами, файлами, git, Ollama и диагностикой. Не придумывай результаты инструментов.
Если нужны данные из системы/проекта — вызови инструмент. Если данных уже хватает — дай финальный ответ.

ВАЖНО:
- Отвечай ТОЛЬКО валидным JSON без markdown.
- За один ход можно вызвать только один инструмент.
- Не раскрывай скрытую цепочку рассуждений. Поле reasoning делай коротким, техническим, 1 строка.
- memory_add используй только если пользователь явно просит "запомни", "сохрани", "запиши в память".
- Для изменения файлов и запуска кода система сама спросит подтверждение.
- Для правки кода предпочитай цикл: git_status -> read_file/read_file_range -> replace_text -> python_compile/run_tests -> git_diff.
- Если пользователь просит опасную команду, не пытайся обходить защиту.

Текущие настройки генерации модели:
{safe_json(asdict(AGENT_SETTINGS), 3000)}

Доступные инструменты:
{tool_specs}

Формат вызова инструмента:
{tool_example}

Формат финального ответа:
{final_example}

Сжатая сводка прошлой истории:
{summary}

Последние сообщения:
{history}

Долговременная память:
{memory}

Предыдущие вызовы инструментов в этом запросе:
{tool_calls}

Запрос пользователя:
{question}
""".strip()


# =========================
# Graph nodes
# =========================

def planner_node(state: AgentState) -> AgentState:
    steps = int(state.get("steps", 0))
    if steps >= MAX_STEPS:
        return {
            "next_action": {
                "action": "final",
                "answer": "Я дошёл до лимита шагов. Ниже — лучшее резюме по уже собранным данным.",
                "reasoning": "Достигнут лимит шагов.",
            },
            "steps": steps,
        }

    prompt = build_planner_prompt(state)
    try:
        response = llm.invoke(prompt).content.strip()
        action = extract_json_object(response)
    except Exception as exc:  # noqa: BLE001
        logging.exception("planner_node failed")
        action = {
            "action": "final",
            "answer": f"Модель вернула невалидный JSON, поэтому я остановился безопасно. Ошибка: {exc}",
            "reasoning": "JSON parse failed.",
        }

    if action.get("action") not in {"tool", "final"}:
        action = {
            "action": "final",
            "answer": "Модель выбрала неизвестное действие, поэтому я остановился безопасно.",
            "reasoning": "Unknown action.",
        }

    return {"next_action": action, "steps": steps + 1}


def tool_executor_node(state: AgentState) -> AgentState:
    action = state.get("next_action", {})
    tool_name = str(action.get("tool_name", "")).strip()
    args = action.get("args", {})
    if not isinstance(args, dict):
        args = {}

    if tool_name not in TOOLS:
        result: ToolResult = {"success": False, "tool": tool_name or "unknown", "error": f"Неизвестный инструмент: {tool_name}"}
    else:
        try:
            result = TOOLS[tool_name](args)
        except Exception as exc:  # noqa: BLE001
            logging.exception("tool failed: %s", tool_name)
            result = {"success": False, "tool": tool_name, "error": repr(exc)}

    call = {
        "created_at": now_iso(),
        "tool": tool_name,
        "args": args,
        "result": result,
    }
    append_jsonl(LOG_DIR / "tool_calls.jsonl", call)
    logging.info("tool_call %s", json.dumps(call, ensure_ascii=False)[:4000])

    tool_calls = list(state.get("tool_calls", []))
    tool_calls.append(call)
    return {"tool_calls": tool_calls}


def final_node(state: AgentState) -> AgentState:
    action = state.get("next_action", {})
    answer = str(action.get("answer", "")).strip()

    if answer and "лимита шагов" not in answer:
        return {"answer": answer}

    question = state["question"]
    tool_calls = format_tool_calls(state.get("tool_calls", []))
    prompt = f"""
Пользователь спросил:
{question}

Вот реальные результаты инструментов:
{tool_calls}

Сформулируй финальный ответ по-русски: по делу, без выдуманных фактов.
Не обрезай мысль слишком рано: дай достаточно деталей, если пользователь просил анализ или улучшения.
Если данных не хватает — честно скажи, чего не хватает.
""".strip()
    try:
        final = llm.invoke(prompt).content.strip()
    except Exception as exc:  # noqa: BLE001
        logging.exception("final_node failed")
        final = f"Не смог сформировать финальный ответ: {exc}"
    return {"answer": final}


def critic_node(state: AgentState) -> AgentState:
    if not ENABLE_CRITIC:
        return {}
    answer = state.get("answer", "")
    question = state.get("question", "")
    tool_calls = format_tool_calls(state.get("tool_calls", []))
    if not answer:
        return {}

    prompt = f"""
Ты проверяешь финальный ответ локального агента.
Задача: убрать неподтверждённые факты, опасные советы и выдуманные результаты инструментов.
Если ответ нормальный — верни его без изменений.
Ответь ТОЛЬКО JSON: {{"answer": "исправленный ответ"}}

Запрос пользователя:
{question}

Реальные результаты инструментов:
{tool_calls}

Черновой ответ:
{answer}
""".strip()
    try:
        response = llm.invoke(prompt).content.strip()
        data = extract_json_object(response)
        fixed = str(data.get("answer", "")).strip()
        if fixed:
            return {"answer": fixed}
    except Exception:
        logging.exception("critic_node failed")
    return {}


def route_after_planner(state: AgentState) -> str:
    action = state.get("next_action", {})
    if action.get("action") == "tool":
        return "tool_executor"
    return "final"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("final", final_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "tool_executor": "tool_executor",
            "final": "final",
        },
    )
    graph.add_edge("tool_executor", "planner")
    graph.add_edge("final", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


# =========================
# History / summary
# =========================

def load_dialog_history(limit: int = 40) -> list[dict[str, str]]:
    rows = read_jsonl(HISTORY_FILE, limit=limit)
    result: list[dict[str, str]] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if isinstance(role, str) and isinstance(content, str):
            result.append({"role": role, "content": content})
    return result


def save_dialog_message(role: str, content: str) -> None:
    append_jsonl(HISTORY_FILE, {"created_at": now_iso(), "role": role, "content": content})


def summarize_history_with_llm() -> dict[str, Any]:
    history = load_dialog_history(limit=80)
    existing = read_json(SUMMARY_FILE, {})
    prompt = f"""
Сделай компактную полезную сводку истории для локального dev-агента.
Не добавляй лишнего. Сохрани только факты, предпочтения, текущие задачи и важный контекст.
Ответь ТОЛЬКО JSON с полями:
- updated_at
- user_preferences
- current_projects
- known_facts
- open_tasks
- technical_context

Старая сводка:
{safe_json(existing, 8000)}

Последние сообщения:
{safe_json(history, 14000)}
""".strip()
    try:
        response = llm.invoke(prompt).content.strip()
        data = extract_json_object(response)
    except Exception as exc:  # noqa: BLE001
        logging.exception("summarize_history failed")
        data = {
            "updated_at": now_iso(),
            "error": f"Не удалось автоматически сделать сводку: {exc}",
            "last_messages": history[-12:],
        }
    data["updated_at"] = now_iso()
    write_json(SUMMARY_FILE, data)
    return data


def maybe_auto_summarize() -> None:
    # Неблокирующая лёгкая политика: если история разрослась, пользователь может вызвать /summary вручную.
    # Авто-суммаризацию не запускаем после каждого ответа, чтобы не тормозить CLI/GUI.
    return None


def ask_agent(text: str, app: Any | None = None, history: list[dict[str, str]] | None = None) -> str:
    app = app or build_graph()
    history = load_dialog_history() if history is None else history
    result = app.invoke({
        "question": text,
        "history": history,
        "tool_calls": [],
        "steps": 0,
    })
    answer = str(result.get("answer", "")).strip() or "Пустой ответ."
    save_dialog_message("user", text)
    save_dialog_message("assistant", answer)
    return answer


# =========================
# CLI
# =========================

def print_help() -> None:
    print(textwrap.dedent(f"""
    Команды CLI:
      /help                 показать помощь
      /status               модель, workspace, память, ollama ps
      /settings             показать настройки генерации
      /workspace            показать текущий workspace
      /memory               показать последние заметки памяти
      /history              показать последние сообщения истории
      /summary              обновить и показать сжатую сводку истории
      /index                построить индекс проекта
      /index-search QUERY   поиск по индексу проекта
      /git                  git status
      /diff                 git diff
      /backups              показать backups агента
      /clear-history        очистить историю диалога
      /exit                 выйти

    Примеры запросов:
      посмотри состояние системы и ollama
      покажи структуру проекта
      построь индекс проекта и найди где используется subprocess
      прочитай adult_linux_agent_full.py строки 100-180
      найди все TODO
      исправь ошибку в main.py и запусти py_compile
      запусти pytest -q
      покажи git diff после правок

    Workspace сейчас:
      {WORKSPACE}
    Runtime/logs:
      {RUNTIME_DIR}
    """))


def cli_status(model: str) -> dict[str, Any]:
    status = {
        "model": model,
        "workspace": str(WORKSPACE),
        "runtime_dir": str(RUNTIME_DIR),
        "history_messages": len(read_jsonl(HISTORY_FILE)),
        "memory_notes": len(read_jsonl(MEMORY_FILE)),
        "summary_exists": SUMMARY_FILE.exists(),
        "index_exists": PROJECT_INDEX_FILE.exists(),
        "logs": str(LOG_DIR),
        "generation_settings": asdict(AGENT_SETTINGS),
    }
    try:
        status["ollama_ps"] = tool_ollama_status({}).get("data")
    except Exception as exc:
        status["ollama_ps_error"] = repr(exc)
    return status


def handle_slash_command(text: str, model: str) -> bool:
    cmd = text.strip()
    if cmd in {"/exit", "/quit", "q", "quit", "exit"}:
        raise KeyboardInterrupt
    if cmd == "/help":
        print_help()
        return True
    if cmd == "/status":
        print_box("Status", safe_json(cli_status(model)))
        return True
    if cmd == "/settings":
        print_box("Generation settings", safe_json(asdict(AGENT_SETTINGS)))
        return True
    if cmd == "/workspace":
        print_box("Workspace", str(WORKSPACE))
        return True
    if cmd == "/memory":
        rows = read_jsonl(MEMORY_FILE, limit=30)
        print_box("Memory", safe_json(rows) if rows else "Память пуста.")
        return True
    if cmd == "/history":
        rows = load_dialog_history()[-20:]
        print_box("History", safe_json(rows) if rows else "История пуста.")
        return True
    if cmd == "/summary":
        summary = summarize_history_with_llm()
        print_box("Summary", safe_json(summary))
        return True
    if cmd == "/index":
        result = tool_project_index_build({"path": "."})
        print_box("Project index", safe_json(result))
        return True
    if cmd.startswith("/index-search"):
        query = cmd.removeprefix("/index-search").strip()
        result = tool_project_index_search({"query": query})
        print_box("Project index search", safe_json(result))
        return True
    if cmd == "/git":
        print_box("Git status", safe_json(tool_git_status({})))
        return True
    if cmd == "/diff":
        print_box("Git diff", safe_json(tool_git_diff({})))
        return True
    if cmd == "/backups":
        print_box("Backups", safe_json(tool_list_backups({"limit": 30})))
        return True
    if cmd == "/clear-history":
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        print("История очищена.")
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Linux/dev agent on Ollama + LangGraph")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model, e.g. gemma4:e2b")
    parser.add_argument("--workspace", default=os.getenv("AGENT_WORKSPACE", "."), help="Workspace directory. File tools are locked inside it.")
    parser.add_argument("--runtime-dir", default=os.getenv("AGENT_RUNTIME_DIR", ".agent_runtime"), help="Directory for memory/history/logs.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature. 0 = stable/tool-friendly.")
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="Max generated tokens per Ollama call.")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Ollama context window.")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k sampling.")
    parser.add_argument("--repeat-penalty", type=float, default=DEFAULT_REPEAT_PENALTY, help="Repeat penalty.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Optional random seed.")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE, help="Ollama keep_alive, e.g. 5m, 30m, -1.")
    parser.add_argument("--num-gpu", type=int, default=DEFAULT_NUM_GPU, help="Optional Ollama num_gpu / GPU layers hint.")
    parser.add_argument("--num-thread", type=int, default=DEFAULT_NUM_THREAD, help="Optional CPU thread count for Ollama.")
    parser.add_argument("--num-batch", type=int, default=DEFAULT_NUM_BATCH, help="Optional Ollama batch size.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Max planner/tool iterations per request.")
    parser.add_argument("--disable-critic", action="store_true", help="Disable final critic pass to save tokens/time.")
    args = parser.parse_args()

    runtime = init_agent(
        args.model,
        args.workspace,
        args.runtime_dir,
        temperature=args.temperature,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
        seed=args.seed,
        keep_alive=args.keep_alive,
        num_gpu=args.num_gpu,
        num_thread=args.num_thread,
        num_batch=args.num_batch,
        max_steps=args.max_steps,
        enable_critic=not args.disable_critic,
    )

    print("Локальный агент запущен.")
    print(f"Модель: {runtime.model}")
    print(f"Настройки: num_predict={runtime.settings.num_predict}, num_ctx={runtime.settings.num_ctx}, temp={runtime.settings.temperature}, max_steps={runtime.settings.max_steps}, num_gpu={runtime.settings.num_gpu}")
    print(f"Workspace: {runtime.workspace}")
    print("Напиши /help для подсказок, /exit для выхода.")

    while True:
        try:
            text = input("\nТы: ").strip()
            if not text:
                continue
            if text.startswith("/") or text.lower() in {"q", "quit", "exit"}:
                if handle_slash_command(text, runtime.model):
                    continue

            answer = ask_agent(text, runtime.app)
            print("\nАгент:", answer)
            maybe_auto_summarize()

        except KeyboardInterrupt:
            print("\nВыход.")
            break
        except EOFError:
            print("\nВыход.")
            break
        except Exception as exc:  # noqa: BLE001
            logging.exception("main loop error")
            print(f"\nОшибка: {exc}")


if __name__ == "__main__":
    main()

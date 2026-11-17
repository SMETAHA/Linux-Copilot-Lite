#!/usr/bin/env python3
"""
adult_linux_agent_gui.py

Простой Tkinter GUI для adult_linux_agent_tuned.py.
Без сервера и без трея: запускает тот же LangGraph/Ollama core в фоне.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import adult_linux_agent_tuned as core


class AgentGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Adult Linux Agent")
        self.root.geometry("980x720")

        self.runtime: core.AgentRuntime | None = None
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.confirm_queue: queue.Queue[tuple[str, queue.Queue[bool]]] = queue.Queue()
        self.is_busy = False

        self.model_var = tk.StringVar(value=core.DEFAULT_MODEL)
        self.workspace_var = tk.StringVar(value=str(Path(".").resolve()))
        self.runtime_var = tk.StringVar(value=".agent_runtime")
        self.temperature_var = tk.StringVar(value=str(core.DEFAULT_TEMPERATURE))
        self.num_predict_var = tk.StringVar(value=str(core.DEFAULT_NUM_PREDICT))
        self.num_ctx_var = tk.StringVar(value=str(core.DEFAULT_NUM_CTX))
        self.top_p_var = tk.StringVar(value=str(core.DEFAULT_TOP_P))
        self.top_k_var = tk.StringVar(value=str(core.DEFAULT_TOP_K))
        self.repeat_penalty_var = tk.StringVar(value=str(core.DEFAULT_REPEAT_PENALTY))
        self.seed_var = tk.StringVar(value="" if core.DEFAULT_SEED is None else str(core.DEFAULT_SEED))
        self.keep_alive_var = tk.StringVar(value=core.DEFAULT_KEEP_ALIVE)
        self.num_gpu_var = tk.StringVar(value="" if core.DEFAULT_NUM_GPU is None else str(core.DEFAULT_NUM_GPU))
        self.num_thread_var = tk.StringVar(value="" if core.DEFAULT_NUM_THREAD is None else str(core.DEFAULT_NUM_THREAD))
        self.num_batch_var = tk.StringVar(value="" if core.DEFAULT_NUM_BATCH is None else str(core.DEFAULT_NUM_BATCH))
        self.max_steps_var = tk.StringVar(value=str(core.MAX_STEPS))
        self.critic_var = tk.BooleanVar(value=core.ENABLE_CRITIC)
        self.status_var = tk.StringVar(value="Не запущен")

        self._build_ui()
        self.root.after(100, self._poll_worker_queue)
        self.root.after(100, self._poll_confirm_queue)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Модель:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.model_var, width=24).grid(row=0, column=1, sticky="we", padx=4)

        ttk.Label(top, text="Workspace:").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.workspace_var, width=48).grid(row=0, column=3, sticky="we", padx=4)
        ttk.Button(top, text="...", command=self.choose_workspace).grid(row=0, column=4, padx=2)

        ttk.Label(top, text="Runtime:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(top, textvariable=self.runtime_var, width=24).grid(row=1, column=1, sticky="we", padx=4, pady=(6, 0))

        self.start_button = ttk.Button(top, text="Запустить / перезапустить", command=self.start_agent)
        self.start_button.grid(row=1, column=3, sticky="e", pady=(6, 0))
        ttk.Label(top, textvariable=self.status_var).grid(row=1, column=4, sticky="w", padx=6, pady=(6, 0))

        top.columnconfigure(3, weight=1)

        settings = ttk.LabelFrame(self.root, text="Ручные настройки генерации", padding=(8, 6))
        settings.pack(fill=tk.X, padx=8, pady=(0, 6))

        fields = [
            ("num_predict", self.num_predict_var, 10, "лимит токенов ответа"),
            ("num_ctx", self.num_ctx_var, 10, "контекст"),
            ("temperature", self.temperature_var, 8, "стабильность"),
            ("top_p", self.top_p_var, 8, "sampling"),
            ("top_k", self.top_k_var, 8, "sampling"),
            ("repeat_penalty", self.repeat_penalty_var, 8, "анти-повторы"),
            ("max_steps", self.max_steps_var, 8, "шаги агента"),
            ("seed", self.seed_var, 8, "пусто = нет"),
            ("keep_alive", self.keep_alive_var, 8, "5m/30m/-1"),
            ("num_gpu", self.num_gpu_var, 8, "пусто = авто"),
            ("num_thread", self.num_thread_var, 8, "пусто = авто"),
            ("num_batch", self.num_batch_var, 8, "пусто = авто"),
        ]
        for idx, (label, var, width, tip) in enumerate(fields):
            row = idx // 5
            col = (idx % 5) * 2
            ttk.Label(settings, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=(0, 2), pady=2)
            ttk.Entry(settings, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="w", padx=(0, 8), pady=2)

        ttk.Checkbutton(settings, text="critic pass", variable=self.critic_var).grid(row=2, column=6, sticky="w", padx=(0, 8))
        ttk.Button(settings, text="E2B fast", command=self.preset_e2b_fast).grid(row=0, column=10, padx=2)
        ttk.Button(settings, text="E4B balanced", command=self.preset_e4b_balanced).grid(row=1, column=10, padx=2)

        toolbar = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        toolbar.pack(fill=tk.X)
        for label, cmd in [
            ("Status", self.show_status),
            ("Settings", self.show_settings),
            ("Ollama", self.ask_ollama_status),
            ("Index", self.build_index),
            ("Git", self.show_git),
            ("Diff", self.show_diff),
            ("Memory", self.show_memory),
            ("History", self.show_history),
            ("Summary", self.update_summary),
            ("Clear", self.clear_chat),
        ]:
            ttk.Button(toolbar, text=label, command=cmd).pack(side=tk.LEFT, padx=2)

        self.chat = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=30)
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.chat.configure(state=tk.DISABLED)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill=tk.X)
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(bottom, textvariable=self.input_var)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", lambda _event: self.send_message())
        self.send_button = ttk.Button(bottom, text="Отправить", command=self.send_message)
        self.send_button.pack(side=tk.LEFT, padx=(6, 0))

    def preset_e2b_fast(self) -> None:
        self.model_var.set("gemma4:e2b")
        self.num_predict_var.set("512")
        self.num_ctx_var.set("4096")
        self.temperature_var.set("0")
        self.top_p_var.set("0.9")
        self.top_k_var.set("40")
        self.repeat_penalty_var.set("1.1")
        self.max_steps_var.set("8")
        self.keep_alive_var.set("10m")
        self.num_gpu_var.set("")
        self.num_thread_var.set("")
        self.num_batch_var.set("")

    def preset_e4b_balanced(self) -> None:
        self.model_var.set("gemma4:e4b")
        self.num_predict_var.set("900")
        self.num_ctx_var.set("8192")
        self.temperature_var.set("0")
        self.top_p_var.set("0.9")
        self.top_k_var.set("40")
        self.repeat_penalty_var.set("1.08")
        self.max_steps_var.set("10")
        self.keep_alive_var.set("15m")
        self.num_gpu_var.set("")
        self.num_thread_var.set("")
        self.num_batch_var.set("")

    def _int_from_var(self, var: tk.StringVar, fallback: int) -> int:
        raw = var.get().strip()
        return int(raw) if raw else fallback

    def _float_from_var(self, var: tk.StringVar, fallback: float) -> float:
        raw = var.get().strip()
        return float(raw) if raw else fallback

    def _optional_int_from_var(self, var: tk.StringVar) -> int | None:
        raw = var.get().strip()
        return int(raw) if raw else None

    def choose_workspace(self) -> None:
        selected = filedialog.askdirectory(title="Выбери workspace")
        if selected:
            self.workspace_var.set(selected)

    def append_chat(self, who: str, text: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, f"\n{who}: {text}\n")
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def append_block(self, title: str, data: Any) -> None:
        if not isinstance(data, str):
            data = json.dumps(data, ensure_ascii=False, indent=2)
        self.append_chat(title, data)

    def set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.send_button.configure(state=state)
        self.start_button.configure(state=state)
        self.input_entry.configure(state=tk.DISABLED if busy else tk.NORMAL)
        if busy:
            self.status_var.set("Думает...")
        elif self.runtime:
            self.status_var.set("Запущен")
        else:
            self.status_var.set("Не запущен")

    def ask_confirmation_threadsafe(self, message: str) -> bool:
        result_queue: queue.Queue[bool] = queue.Queue(maxsize=1)
        self.confirm_queue.put((message, result_queue))
        return result_queue.get()

    def _poll_confirm_queue(self) -> None:
        try:
            while True:
                message, result_queue = self.confirm_queue.get_nowait()
                result = messagebox.askyesno("Подтверждение действия", message)
                result_queue.put(bool(result))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_confirm_queue)

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "answer":
                    self.append_chat("Агент", str(payload))
                    self.set_busy(False)
                elif kind == "error":
                    self.append_chat("Ошибка", str(payload))
                    self.set_busy(False)
                elif kind == "info":
                    self.append_block("Info", payload)
                elif kind == "started":
                    self.runtime = payload
                    self.append_chat(
                        "Система",
                        f"Агент запущен. Модель: {payload.model}\n"
                        f"Workspace: {payload.workspace}\n"
                        f"num_predict={payload.settings.num_predict}, num_ctx={payload.settings.num_ctx}, "
                        f"temp={payload.settings.temperature}, max_steps={payload.settings.max_steps}, "
                        f"num_gpu={payload.settings.num_gpu}, critic={payload.settings.enable_critic}"
                    )
                    self.set_busy(False)
                elif kind == "block":
                    title, data = payload
                    self.append_block(title, data)
                    self.set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker_queue)

    def start_agent(self) -> None:
        if self.is_busy:
            return
        self.set_busy(True)

        def worker() -> None:
            try:
                runtime = core.init_agent(
                    model=self.model_var.get().strip() or core.DEFAULT_MODEL,
                    workspace=self.workspace_var.get().strip() or ".",
                    runtime_dir=self.runtime_var.get().strip() or ".agent_runtime",
                    temperature=self._float_from_var(self.temperature_var, core.DEFAULT_TEMPERATURE),
                    num_predict=self._int_from_var(self.num_predict_var, core.DEFAULT_NUM_PREDICT),
                    num_ctx=self._int_from_var(self.num_ctx_var, core.DEFAULT_NUM_CTX),
                    top_p=self._float_from_var(self.top_p_var, core.DEFAULT_TOP_P),
                    top_k=self._int_from_var(self.top_k_var, core.DEFAULT_TOP_K),
                    repeat_penalty=self._float_from_var(self.repeat_penalty_var, core.DEFAULT_REPEAT_PENALTY),
                    seed=self._optional_int_from_var(self.seed_var),
                    keep_alive=self.keep_alive_var.get().strip() or core.DEFAULT_KEEP_ALIVE,
                    num_gpu=self._optional_int_from_var(self.num_gpu_var),
                    num_thread=self._optional_int_from_var(self.num_thread_var),
                    num_batch=self._optional_int_from_var(self.num_batch_var),
                    max_steps=self._int_from_var(self.max_steps_var, core.MAX_STEPS),
                    enable_critic=self.critic_var.get(),
                )
                core.set_confirmation_handler(self.ask_confirmation_threadsafe)
                self.worker_queue.put(("started", runtime))
            except Exception as exc:  # noqa: BLE001
                self.worker_queue.put(("error", repr(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def ensure_runtime(self) -> bool:
        if self.runtime is not None:
            return True
        messagebox.showwarning("Агент не запущен", "Сначала нажми 'Запустить / перезапустить'.")
        return False

    def send_message(self) -> None:
        if self.is_busy or not self.ensure_runtime():
            return
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self.append_chat("Ты", text)
        self.set_busy(True)

        def worker() -> None:
            try:
                assert self.runtime is not None
                answer = core.ask_agent(text, self.runtime.app)
                self.worker_queue.put(("answer", answer))
            except Exception as exc:  # noqa: BLE001
                self.worker_queue.put(("error", repr(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def run_block_task(self, title: str, func: Any) -> None:
        if self.is_busy or not self.ensure_runtime():
            return
        self.set_busy(True)

        def worker() -> None:
            try:
                data = func()
                self.worker_queue.put(("block", (title, data)))
            except Exception as exc:  # noqa: BLE001
                self.worker_queue.put(("error", repr(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def show_status(self) -> None:
        self.run_block_task("Status", lambda: core.cli_status(self.runtime.model if self.runtime else self.model_var.get()))

    def show_settings(self) -> None:
        self.run_block_task("Generation settings", lambda: core.asdict(core.AGENT_SETTINGS))

    def ask_ollama_status(self) -> None:
        self.run_block_task("Ollama", lambda: core.tool_ollama_status({}))

    def build_index(self) -> None:
        self.run_block_task("Project index", lambda: core.tool_project_index_build({"path": "."}))

    def show_git(self) -> None:
        self.run_block_task("Git status", lambda: core.tool_git_status({}))

    def show_diff(self) -> None:
        self.run_block_task("Git diff", lambda: core.tool_git_diff({}))

    def show_memory(self) -> None:
        self.run_block_task("Memory", lambda: core.tool_memory_list({"limit": 30}))

    def show_history(self) -> None:
        self.run_block_task("History", lambda: core.load_dialog_history(limit=30))

    def update_summary(self) -> None:
        self.run_block_task("Summary", core.summarize_history_with_llm)

    def clear_chat(self) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = AgentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

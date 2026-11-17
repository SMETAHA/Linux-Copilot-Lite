import subprocess
from typing import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    answer: str


llm = ChatOllama(model="qwen2.5:7b", temperature=0)

ALLOWED = {
    "pwd",
    "whoami",
    "uptime",
    "free -h",
    "df -h",
    "ls",
    "ls -lah",
    "uname -a",
    "ps aux --sort=-%cpu | head",
}


def run_command(command: str) -> str:
    command = command.strip()

    if command not in ALLOWED:
        return f"Команда запрещена: {command}"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    return result.stdout or result.stderr or "Команда выполнена без вывода."


def agent_node(state: State):
    question = state["question"]

    prompt = f"""
Ты Linux-агент. У тебя есть инструмент run_command.

Разрешённые команды:
{chr(10).join(ALLOWED)}

Если нужно выполнить команду, напиши строго:
RUN: команда

Если команда не нужна, просто ответь.

Запрос пользователя:
{question}
"""
    print("QUESTION REPR:")
    print(repr(question))

    print("PROMPT REPR:")
    print(repr(prompt))
    response = llm.invoke(prompt).content.strip()

    if response.startswith("RUN:"):
        command = response.removeprefix("RUN:").strip()
        output = run_command(command)

        final = llm.invoke(f"""
Пользователь спросил:
{question}

Ты выполнил команду:
{command}

Вывод:
{output}

Ответь кратко и понятно.
""").content

        return {"answer": final}

    return {"answer": response}


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile()


while True:
    text = input("\nТы: ")
    if text.lower() in {"q", "quit", "exit"}:
        break

    result = app.invoke({"question": text})
    print("\nАгент:", result["answer"])
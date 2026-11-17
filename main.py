from typing import Annotated, TypedDict
import sys

print(sys.stdin.encoding)
print(sys.stdout.encoding)
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def get_system_info() -> str:
    """Return basic Linux system information."""
    import platform
    import subprocess

    uptime = subprocess.check_output(["uptime"], text=True).strip()
    return f"OS: {platform.system()} {platform.release()}\nUptime: {uptime}"


tools = [get_system_info]

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
).bind_tools(tools)


def assistant_node(state: State):
    response = llm.invoke([
        SystemMessage(content="Ты локальный Linux-агент. Отвечай кратко и по делу."),
        *state["messages"],
    ])
    return {"messages": [response]}

def tool_node(state: State):
    last = state["messages"][-1]
    results = []

    for call in last.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name == "get_system_info":
            result = get_system_info.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        results.append(ToolMessage(
            content=result,
            tool_call_id=call["id"],
        ))

    return {"messages": results}


def should_continue(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


graph = StateGraph(State)

graph.add_node("assistant", assistant_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "assistant")
graph.add_conditional_edges("assistant", should_continue)
graph.add_edge("tools", "assistant")

app = graph.compile()


while True:
    text = input("\nТы: ")
    if text.lower() in {"exit", "quit", "q"}:
        break

    result = app.invoke({
        "messages": [HumanMessage(content=text)]
    })

    print("\nАгент:", result["messages"][-1].content)
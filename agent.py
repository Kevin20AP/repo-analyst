import asyncio
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools        # NEW import
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

REPO_PATH = "repos/rag-chat-docs"
MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are a codebase analyst. Explore the given repository using your tools
(list_files, read_file, search_code), then write a clear analysis report.

How to work:
1. Call list_files first to see the structure.
2. Read the most important files (README, entry points, core modules). Do NOT read everything.
3. Use search_code to locate key definitions when useful.
4. Once you understand the project, STOP calling tools and write the final report with:
   - Overview: what the project does
   - Architecture & key components
   - Tech stack
   - Notable risks or issues

Be efficient: read only what you need to understand the project well."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


# CHANGE 1: build_graph now takes the tools as an argument (no MCP work inside).
def build_graph(tools):
    model = ChatAnthropic(model=MODEL, max_tokens=4096).bind_tools(tools)

    async def agent_node(state: State):
        response = await model.ainvoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: State):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


async def main():
    client = MultiServerMCPClient({
        "repo": {
            "command": "uv",
            "args": ["run", "python", "mcp_server.py"],
            "transport": "stdio",
            "env": {**os.environ, "REPO_PATH": REPO_PATH},
        }
    })

    # CHANGE 2: open ONE persistent session and run the whole agent inside it.
    async with client.session("repo") as session:
        tools = await load_mcp_tools(session)   # CHANGE 3: load tools bound to this open session
        app = build_graph(tools)
        result = await app.ainvoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                          HumanMessage(content="Analyze this repository.")]},
            config={"recursion_limit": 30},
        )

    print("\n" + "=" * 60 + "\nAGENT TRAJECTORY\n" + "=" * 60)
    for m in result["messages"]:
        m.pretty_print()


asyncio.run(main())
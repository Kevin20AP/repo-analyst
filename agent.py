import asyncio
import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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
    steps: int
    approved: bool


def _text(content):
    """Pull plain text out of a message's content (handles str or list-of-blocks)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def build_graph(tools, checkpointer):
    model_with_tools = ChatAnthropic(model=MODEL, max_tokens=8000).bind_tools(tools)
    model_plain = ChatAnthropic(model=MODEL, max_tokens=8000)

    MAX_STEPS = 15
    WARN_STEP = 10

    async def agent_node(state: State):
        steps = state.get("steps", 0) + 1
        messages = state["messages"]

        if steps >= MAX_STEPS:
            nudge = HumanMessage(content=(
                "You've reached your exploration budget. Do NOT call any more tools. "
                "Write the final analysis report now using what you've already read."
            ))
            response = await model_plain.ainvoke(messages + [nudge])
            return {"messages": [response], "steps": steps}

        if steps == WARN_STEP:
            nudge = HumanMessage(content=(
                "You're past half your exploration budget. Read only what's still "
                "essential, then write the report soon."
            ))
            response = await model_with_tools.ainvoke(messages + [nudge])
            return {"messages": [response], "steps": steps}

        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response], "steps": steps}

    def should_continue(state: State):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "reflect"

    async def reflect(state: State):
        critique_prompt = HumanMessage(content=(
            "Before finalizing, critically review your draft report above against the "
            "files you actually read:\n"
            "1. Are ALL claims supported by code you saw? Remove or soften anything you "
            "guessed or inferred without evidence.\n"
            "2. Is anything important missing (key components, real risks)?\n"
            "3. Is it clear, specific, and well-organized?\n"
            "Then rewrite it as an improved FINAL report that fixes these issues. "
            "Output ONLY the improved report."
        ))
        response = await model_plain.ainvoke(state["messages"] + [critique_prompt])
        return {"messages": [response]}

    def human_review(state: State):
        draft = _text(state["messages"][-1].content)
        decision = interrupt({
            "draft_report": draft,
            "instructions": "Reply 'approve' to accept, or type feedback to revise the report.",
        })
        if str(decision).strip().lower() == "approve":
            return {"approved": True}
        return {
            "approved": False,
            "messages": [HumanMessage(content=f"Please revise the report based on this feedback: {decision}")],
        }

    def after_review(state: State):
        return END if state.get("approved") else "agent"

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("reflect", reflect)
    graph.add_node("human_review", human_review)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    graph.add_edge("reflect", "human_review")
    graph.add_conditional_edges("human_review", after_review)
    return graph.compile(checkpointer=checkpointer)


async def main():
    client = MultiServerMCPClient({
        "repo": {
            "command": "uv",
            "args": ["run", "python", "mcp_server.py"],
            "transport": "stdio",
            "env": {**os.environ, "REPO_PATH": REPO_PATH},
        }
    })

    async with client.session("repo") as session:
        tools = await load_mcp_tools(session)

        async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
            app = build_graph(tools, checkpointer)
            config = {"configurable": {"thread_id": str(uuid.uuid4())},
                      "recursion_limit": 50}

            result = await app.ainvoke(
                {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                              HumanMessage(content="Analyze this repository.")],
                 "steps": 0, "approved": False},
                config=config,
            )

            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                print("\n" + "=" * 60)
                print("DRAFT REPORT FOR YOUR REVIEW:\n")
                print(payload["draft_report"])
                print("\n" + payload["instructions"])
                decision = input("\nYour response > ")
                result = await app.ainvoke(Command(resume=decision), config=config)

            print("\n" + "=" * 60 + "\nAPPROVED — FINAL REPORT\n" + "=" * 60)
            print(_text(result["messages"][-1].content))


if __name__ == "__main__":
    asyncio.run(main())
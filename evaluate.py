"""
evaluate.py — production-grade evaluation suite for the Repo Analyst agent.

Part A: the harness — run the agent on a repo and capture the report + metrics.
"""

import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from agent import build_graph, SYSTEM_PROMPT, _text
from repo_tools import clone_repo

load_dotenv()


async def analyze_repo(repo_path: str) -> dict:
    """Run the agent on a repo (auto-approving the review), capturing report + metrics."""
    client = MultiServerMCPClient({
        "repo": {
            "command": "uv",
            "args": ["run", "python", "mcp_server.py"],
            "transport": "stdio",
            "env": {**os.environ, "REPO_PATH": repo_path},
        }
    })

    start = time.time()
    async with client.session("repo") as session:
        tools = await load_mcp_tools(session)
        app = build_graph(tools, MemorySaver())          # in-memory checkpointer for eval
        config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}

        result = await app.ainvoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                          HumanMessage(content="Analyze this repository.")],
             "steps": 0, "approved": False},
            config=config,
        )
        # auto-approve any human-review pauses (no human in eval)
        while "__interrupt__" in result:
            result = await app.ainvoke(Command(resume="approve"), config=config)

    elapsed = time.time() - start
    messages = result["messages"]

    # ---- capture metrics from the trajectory ----
    tool_calls = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            tool_calls += [tc["name"] for tc in m.tool_calls]

    in_tok = out_tok = 0
    for m in messages:
        um = getattr(m, "usage_metadata", None)
        if um:
            in_tok += um.get("input_tokens", 0)
            out_tok += um.get("output_tokens", 0)

    return {
        "repo": repo_path,
        "report": _text(messages[-1].content),
        "messages": messages,
        "metrics": {
            "agent_steps": result.get("steps", 0),
            "tool_calls_total": len(tool_calls),
            "tools_used": tool_calls,
            "first_tool": tool_calls[0] if tool_calls else None,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "elapsed_seconds": round(elapsed, 1),
        },
    }


async def _demo():
    path = clone_repo("https://github.com/Kevin20AP/rag-chat-docs")
    r = await analyze_repo(str(path))
    print("METRICS:", r["metrics"])
    print("\nREPORT (first 400 chars):\n", r["report"][:400])



if __name__ == "__main__":
    asyncio.run(_demo())
"""
evaluate.py — production-grade evaluation suite for the Repo Analyst agent.

Part A: harness      — run the agent, capture report + metrics.
Part B: quality      — score the report with G-Eval (LLM-as-judge).
Part C: suite + gate — trajectory checks, a test set of repos, and release gates.
"""

import asyncio
import os
import re
import sys
import time
import uuid

from dotenv import load_dotenv
from anthropic import Anthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from agent import build_graph, SYSTEM_PROMPT, _text
from repo_tools import clone_repo

load_dotenv()

judge = Anthropic()
JUDGE_MODEL = "claude-opus-5-5"          # stronger, independent judge

# Rough cost estimate (adjust the rates to match your plan).
COST_PER_1M_INPUT = 3.0
COST_PER_1M_OUTPUT = 15.0


# ----------------------- PART A: the harness -----------------------
async def analyze_repo(repo_path: str) -> dict:
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
        app = build_graph(tools, MemorySaver())
        config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}
        result = await app.ainvoke(
            {"messages": [SystemMessage(content=SYSTEM_PROMPT),
                          HumanMessage(content="Analyze this repository.")],
             "steps": 0, "approved": False},
            config=config,
        )
        while "__interrupt__" in result:
            result = await app.ainvoke(Command(resume="approve"), config=config)

    elapsed = time.time() - start
    messages = result["messages"]

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


# ----------------------- PART B: G-Eval quality -----------------------
def _code_context(messages, max_chars=30000):
    parts = [_text(m.content) for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(parts)[:max_chars]


def g_eval(name, criteria, report, context="") -> dict:
    ctx_block = f"\nREFERENCE CODE (the actual files the report describes):\n{context}\n" if context else ""
    prompt = f"""You are a strict expert evaluator scoring a codebase ANALYSIS REPORT.

CRITERION: {name}
{criteria}
{ctx_block}
ANALYSIS REPORT TO EVALUATE:
{report}

Think step by step about how well the report meets the criterion. Be critical.
Then give an integer score from 1 to 5 (1 = poor, 5 = excellent).

Respond EXACTLY in this format:
REASONING: <2-4 sentences>
SCORE: <single integer 1-5>"""
    resp = judge.messages.create(model=JUDGE_MODEL, max_tokens=1200,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"SCORE:\s*([1-5])", text)
    return {"metric": name, "score": int(m.group(1)) if m else None}


QUALITY_METRICS = [
    ("Completeness",
     "Does the report thoroughly cover the project's purpose, architecture and key "
     "components, tech stack, and real risks/issues? Penalize missing sections or vague coverage."),
    ("Clarity",
     "Is the report clear, well-organized, and specific (concrete file/function references) "
     "rather than generic filler? Penalize disorganization and vagueness."),
    ("Groundedness",
     "Are the report's claims accurate and supported by the actual reference code provided? "
     "Penalize any claim that is hallucinated, wrong, or unsupported by the code."),
]


# ----------------------- PART C: trajectory + suite + gate -----------------------
def trajectory_eval(metrics) -> dict:
    """Deterministic checks on HOW the agent ran (the path, not the report)."""
    allowed = {"list_files", "read_file", "search_code"}
    return {
        "plan_adherence (list_files first)": metrics["first_tool"] == "list_files",
        "step_efficiency (3-25 tool calls)": 3 <= metrics["tool_calls_total"] <= 25,
        "valid_tools_only": all(t in allowed for t in metrics["tools_used"]),
        "within_budget (<15 steps)": metrics["agent_steps"] < 15,
    }


def estimate_cost(metrics) -> float:
    return round(metrics["input_tokens"] / 1e6 * COST_PER_1M_INPUT
                 + metrics["output_tokens"] / 1e6 * COST_PER_1M_OUTPUT, 4)


# The test set — repos to evaluate the agent on. Add more for broader coverage.
TEST_REPOS = [
    "https://github.com/Kevin20AP/rag-chat-docs",
    "https://github.com/Kevin20AP/llm-api-service",
]

# Release-gate thresholds: the suite FAILS if any average quality score is below these.
THRESHOLDS = {"Completeness": 3, "Clarity": 3, "Groundedness": 4}


async def run_suite():
    all_quality = {name: [] for name, _ in QUALITY_METRICS}
    all_pass = True

    for repo_url in TEST_REPOS:
        print(f"\n{'='*70}\nEVALUATING: {repo_url}\n{'='*70}")
        path = clone_repo(repo_url)
        r = await analyze_repo(str(path))
        m = r["metrics"]

        # System metrics
        print(f"  steps={m['agent_steps']}  tool_calls={m['tool_calls_total']}  "
              f"tokens={m['input_tokens']}in/{m['output_tokens']}out  "
              f"time={m['elapsed_seconds']}s  est_cost=${estimate_cost(m)}")

        # Quality (G-Eval)
        context = _code_context(r["messages"])
        print("  QUALITY:", end=" ")
        for name, criteria in QUALITY_METRICS:
            ctx = context if name == "Groundedness" else ""
            score = g_eval(name, criteria, r["report"], ctx)["score"]
            all_quality[name].append(score)
            print(f"{name}={score}/5", end="  ")
        print()

        # Trajectory (deterministic)
        traj = trajectory_eval(m)
        for check, ok in traj.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {check}")
            if not ok:
                all_pass = False

    # ---- aggregate + release gate ----
    print(f"\n{'='*70}\nSUITE SUMMARY\n{'='*70}")
    for name in all_quality:
        scores = [s for s in all_quality[name] if s is not None]
        avg = sum(scores) / len(scores) if scores else 0
        threshold = THRESHOLDS[name]
        ok = avg >= threshold
        if not ok:
            all_pass = False
        print(f"  {name}: avg {avg:.2f}/5  (threshold {threshold})  [{'PASS' if ok else 'FAIL'}]")

    print(f"\n{'='*70}")
    if all_pass:
        print("✅  RELEASE GATE: PASS")
        sys.exit(0)
    else:
        print("❌  RELEASE GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_suite())
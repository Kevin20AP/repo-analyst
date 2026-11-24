# Repo Analyst — an Autonomous Codebase Analysis Agent

An autonomous AI agent that **explores any GitHub repository on its own and writes a grounded analysis report** — covering what the project does, its architecture, tech stack, and real risks. Point it at a repo, and it decides which files matter, reads them, reasons about the code, critiques its own draft, and produces a report you can trust.

Built to demonstrate the full modern agent stack: a **LangGraph** agent loop, a **custom Model Context Protocol (MCP) server**, durable long-running execution with **human-in-the-loop** review, and a **production-grade, trajectory-first evaluation suite**.

---

## Why this is more than a "wrapper"

A basic agent is a single LLM call with some tools. This one is engineered the way a production agent actually needs to be:

- **Autonomous exploration** — a LangGraph loop where the agent plans, calls tools, observes results, and decides what to read next, until it understands the repo.
- **Custom MCP server** — the codebase tools (`list_files`, `read_file`, `search_code`) are exposed over the Model Context Protocol, so any MCP client (this agent, Claude Desktop, etc.) can use them.
- **Durable & resumable** — a SQLite checkpointer saves full state at every step, so the agent can pause, resume, and survive a crash.
- **Human-in-the-loop** — the agent pauses before finalizing, shows a draft report, and lets a human approve or request revisions, then resumes.
- **Self-reflection** — before finalizing, the agent critiques its own draft against the code it read and rewrites an improved version (targeted hallucination control).
- **Production hardening** — a smart step budget (graceful degradation, never a crash), context management (no-re-read guard, per-file size caps), and resilient error handling.
- **Observability** — full tracing of every step, tool call, and token with LangSmith.
- **Rigorous evaluation** — a trajectory-first eval suite with G-Eval LLM-as-judge scoring, deterministic trajectory checks, a multi-repo test set, cost tracking, and CI-style release gates.

---

## Architecture

The agent is a **LangGraph state machine**:

```
          ┌──────────┐
START ───▶│  agent   │  plan + decide (LLM, tools bound)
          └────┬─────┘
               │ tool call?
         ┌─────┴───────┐
       yes             no
         │              │
         ▼              ▼
     ┌───────┐     ┌──────────┐
     │ tools │     │ reflect  │  self-critique + rewrite
     └───┬───┘     └────┬─────┘
         │              ▼
         └─▶ agent  ┌──────────────┐
          (loop)    │ human_review │  pause → approve / revise
                    └──────┬───────┘
                           ▼
                   approve │ revise → agent
                           ▼
                         [END]  final report
```

- **agent** — the brain: the LLM explores with tools (`model_with_tools`) or, at the budget limit, is forced to write (`model_plain`, no tools).
- **tools** — a `ToolNode` that executes the MCP tool calls and feeds results back.
- **reflect** — the agent reviews its own draft against the code it read and rewrites it.
- **human_review** — a durable `interrupt()` that pauses for human approval or feedback.

State (the messages, step count, approval flag) is checkpointed to SQLite after every step, which is what makes pause/resume and human-in-the-loop possible.

---

## Tech stack

`Python 3.13` · `LangGraph` · `LangChain` · `Model Context Protocol (MCP)` · `Anthropic Claude` · `LangSmith` (tracing) · `SQLite` (durable checkpoints) · `uv`

---

## The evaluation suite

Agents are only trustworthy if you can *measure* them. `evaluate.py` is a trajectory-first eval suite (the 2026 best practice — score the path, not just the final answer):

- **Quality (G-Eval, LLM-as-judge):** a stronger, independent judge model scores each report on **Completeness, Clarity, and Groundedness** using chain-of-thought rubric scoring. Groundedness is checked against the *actual code* the agent read, to catch hallucinations.
- **Trajectory checks (deterministic):** plan adherence (did it look at the structure first?), step efficiency, valid-tools-only, and within-budget.
- **System metrics:** steps, tool calls, tokens, latency, and estimated cost per run.
- **Test set + release gate:** runs across multiple repos, aggregates the scores against thresholds, and **passes or fails the whole suite with a CI exit code** — ready to block a bad change in a pipeline.

```
SUITE SUMMARY
  Completeness:  avg 4.50/5  (threshold 3)  [PASS]
  Clarity:       avg 5.00/5  (threshold 3)  [PASS]
  Groundedness:  avg 5.00/5  (threshold 4)  [PASS]
✅  RELEASE GATE: PASS
```

---

## Running it

**Prerequisites:** an Anthropic API key (and, optionally, a LangSmith key for tracing).

1. Install dependencies with [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```
2. Create a `.env` in the project root:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   # optional, for tracing:
   LANGSMITH_API_KEY=lsv2-...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=repo-analyst
   ```
3. Run the agent (it clones and analyzes a target repo, then pauses for your review):
   ```bash
   uv run python agent.py
   ```
4. Run the evaluation suite:
   ```bash
   uv run python evaluate.py
   ```

---

## Project structure

| File | Role |
|------|------|
| `agent.py` | The LangGraph agent: exploration loop, step budget, reflection, durability, human-in-the-loop. |
| `mcp_server.py` | Custom MCP server exposing the codebase tools. |
| `repo_tools.py` | The underlying tools: clone, list files, read file (with safety + size caps), search code. |
| `evaluate.py` | Production eval suite: G-Eval quality scoring, trajectory checks, test set, release gates. |

---

## What I learned

Building this covered the real production-agent stack: designing an agentic loop with LangGraph (state, nodes, conditional edges, loops), building and consuming a custom MCP server, making an agent **durable and resumable** with checkpointing, adding **human-in-the-loop** control via interrupts, hardening it against runaway loops and context overflow, and — most importantly — building a **rigorous, trajectory-first evaluation suite** that measures both *how* the agent runs and *how good* its output is, with release gates fit for CI.

---

## Roadmap

- **Phase 2:** wrap the agent in a FastAPI service with background job processing (Celery/Redis) and Postgres persistence.
- **Phase 3:** a TypeScript/Next.js front-end and cloud deployment with a CI/CD pipeline that runs the eval gates.
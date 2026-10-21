# ReAct Agent from Scratch (Thought → Action → Observation)

A small, dependency-light agent that **reasons, calls tools, reads the results, and repeats** until it can answer.
No agent framework: the whole loop is ~150 lines in [`agent.py`](agent.py) so every step is easy to inspect.

## Architecture

```
            ┌────────────────────────────────────────────┐
 Question ─▶│  ReActAgent.run()                          │
            │                                            │
            │   ┌──────────┐  JSON   ┌───────────────┐   │
            │   │   LLM    │────────▶│ parse + validate│  │
            │   │ (Claude/ │         └───────┬───────┘   │
            │   │ OpenAI)  │                 │           │
            │   └────▲─────┘     final_answer│  action   │
            │        │                 ▼     ▼           │
            │  Observation      ┌────────────────────┐   │
            │  appended to      │ Guardrails         │   │
            │  history          │ • unknown tool?    │   │
            │        │          │ • repeated action? │   │
            │        │          │ • tool error?      │   │
            │        │          └─────────┬──────────┘   │
            │        │                    ▼              │
            │        │          ┌────────────────────┐   │
            │        └──────────│ Tools: calculator, │   │
            │                   │ web_search, datetime│  │
            │                   └────────────────────┘   │
            │   every step ─▶ console + logs/trace.jsonl │
            └────────────────────────────────────────────┘
```

**Loop:** each turn the LLM returns one JSON object, either
`{"thought", "action", "action_input"}` or `{"thought", "final_answer"}`.
The agent runs the tool, appends `Observation: ...` to the conversation, and asks the LLM again.

| File | Role |
|---|---|
| `agent.py` | ReAct loop, JSON parsing, guardrails, tracing |
| `tools.py` | `calculator` (safe AST evaluator, no `eval`), `web_search` (DuckDuckGo → Wikipedia fallback), `current_datetime` |
| `llm.py` | Anthropic, OpenAI-compatible (Groq etc.), and a `ScriptedLLM` for tests |
| `main.py` | CLI |
| `tests/` | 7 tests covering every failure mode |

## Failure handling (wrong tool / no progress)

| Problem | Detection | Response |
|---|---|---|
| LLM picks a tool that doesn't exist | `action not in tools` | Observation lists the valid tool names so it can self-correct |
| LLM returns malformed / non-JSON output | parse error | Observation explains the format error; retry |
| Tool raises an error (bad input, network) | `ToolError` / any exception caught | Error becomes an observation, the agent never crashes |
| LLM repeats the same action + input | counter per `(tool, input)`; > `max_repeats` (2) | Call is **blocked** (not executed) with a hint to change approach |
| Blocked repeats keep happening | `max_stalls` (2) | Abort with `loop_detected` |
| Many consecutive errors | `max_consecutive_errors` (4) | Abort with `too_many_errors` |
| No answer by `max_steps` (8) | step counter | Abort with `max_steps` |

On any abort the agent makes **one last LLM call** with tools disabled and asks for a best-effort answer, stating what
could not be verified. The result has `success=False` and a `stop_reason`, so callers can tell the difference.

## Logging / debugging

Every step prints to the console (💭 thought, 🔧 action, 👀 observation) and is appended as JSON lines to
`logs/trace_<timestamp>.jsonl`:

```json
{"type": "step", "step": 1, "thought": "I need the completion year.", "action": "google",
 "action_input": "Eiffel Tower year",
 "observation": "ERROR: Unknown tool 'google'. Available tools: calculator, current_datetime, web_search.",
 "error": "unknown_tool"}
```

## Setup

```bash
git clone <your-repo-url> && cd react-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # add ONE API key (Anthropic, or Groq/OpenAI-compatible)
```

## Run

```bash
python main.py --demo                                   # multi-step task (search → calc → date → calc)
python main.py "What is 15% of the population of Japan?"
python main.py --provider openai "your question"        # Groq / OpenAI-compatible
python main.py --offline                                # no key or internet: scripted run showing recovery
python -m pytest -q                                     # tests
```

### Example multi-step task

> *"Find the year the Eiffel Tower was completed, take its square root × 3 (2 decimals), and say how many years ago it was."*

1. `web_search` → 1889
2. `calculator("round(sqrt(1889) * 3, 2)")` → 130.39
3. `current_datetime` → 2026
4. `calculator("2026 - 1889")` → 137
5. Final answer

## Design notes

- **JSON over free-text ReAct parsing**: regex-parsing "Action:" lines is brittle; JSON is easy to validate.
- **Calculator uses `ast`, not `eval`**: only whitelisted operators/functions, capped exponents.
- **Temperature 0** for reproducible traces.
- **Limits**: conversation history is unbounded within a run (fine for ≤8 steps); no parallel tool calls; single string tool input.

## Possible extensions
Native tool-calling APIs, memory across runs, a planner step before acting, token/cost tracking per step.

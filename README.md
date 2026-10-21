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
1. `web_search("Eiffel Tower completion year")` → 1889
2. `current_datetime` → 2026-10-21
3. `calculator("sqrt(1889) * 3")` → 130.3878829 (the model rounds it to 130.39)
4. Final answer: 130.39, and 137 years ago (2026 − 1889)

## Design notes

- **JSON over free-text ReAct parsing**: regex-parsing "Action:" lines is brittle; JSON is easy to validate.
- **Calculator uses `ast`, not `eval`**: only whitelisted operators/functions, capped exponents.
- **Temperature 0** for reproducible traces.
- **Limits**: conversation history is unbounded within a run (fine for ≤8 steps); no parallel tool calls; single string tool input.

## Possible extensions
Native tool-calling APIs, memory across runs, a planner step before acting, token/cost tracking per step.

## Run in Google Colab

1. Upload the project folder (or `git clone` it) in a Colab notebook.
2. `!pip install -q -r requirements.txt`
3. Add your key in Colab **Secrets** (name `OPENAI_API_KEY`) and set:
```python
   import os
   from google.colab import userdata
   os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
   os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
   os.environ["OPENAI_MODEL"] = "qwen/qwen3.8-27b"
```
4. `!python main.py --demo --provider openai`

## Notes on model choice (real debugging findings)

- Tested with `qwen/qwen3.8-27b` on Groq's free tier. It follows the JSON format reliably.
- `openai/gpt-oss-120b` on Groq returned **empty `content`** (the answer stayed in a hidden `reasoning` field), so the agent's
  format-error recovery kicked in repeatedly and eventually aborted with `too_many_errors`. Switching model fixed it.
  The guardrails made it easy to tell this was a model problem, not a code problem.
- Free-tier rate limits: `llm.py` retries after 15s on HTTP 429 and caps `max_tokens` (env `MAX_TOKENS`, default 300).
- The model sometimes does arithmetic itself instead of calling the calculator. The prompt asks for the calculator,
  but this is not enforced.

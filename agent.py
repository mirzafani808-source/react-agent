"""ReAct agent: Thought -> Action -> Observation loop with guardrails and tracing."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from tools import Tool, ToolError

log = logging.getLogger("react")

SYSTEM_TEMPLATE = """You are a ReAct agent. Solve the user's task by alternating Thought, Action, Observation.

Available tools:
{tool_list}

Reply with EXACTLY ONE JSON object per turn and nothing else. Two allowed forms:
1) Use a tool:  {{"thought": "<your reasoning>", "action": "<tool name>", "action_input": "<string>"}}
2) Finish:      {{"thought": "<your reasoning>", "final_answer": "<answer to the user>"}}

Rules:
- One action per turn. Wait for the Observation before continuing. Never invent observations.
- Use the calculator for ALL arithmetic. Use web_search for facts you are not sure about.
- If a tool errors or returns nothing useful, change your approach (new query / different tool).
- Never repeat the exact same action and input. When you have enough information, give the final_answer.
"""


@dataclass
class AgentResult:
    answer: str
    success: bool
    stop_reason: str  # final_answer | max_steps | loop_detected | too_many_errors
    steps: int
    trace: list[dict] = field(default_factory=list)
    trace_file: str | None = None


class ReActAgent:
    def __init__(self, llm, tools: list[Tool], max_steps: int = 8, max_repeats: int = 2,
                 max_stalls: int = 2, max_consecutive_errors: int = 4, trace_dir: str | None = "logs"):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.max_repeats = max_repeats          # same (tool, input) allowed this many times
        self.max_stalls = max_stalls            # blocked repeats tolerated before aborting
        self.max_consecutive_errors = max_consecutive_errors
        self.trace_dir = trace_dir
        self.system = SYSTEM_TEMPLATE.format(
            tool_list="\n".join(f"- {t.name}: {t.description}" for t in tools))

    # ------------------------------------------------------------ parsing
    @staticmethod
    def parse(raw: str) -> dict:
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found.")
        data = json.loads(text[start:end + 1])
        if "final_answer" not in data and "action" not in data:
            raise ValueError("JSON must contain 'action' or 'final_answer'.")
        return data

    # ------------------------------------------------------------ tracing
    def _open_trace(self):
        if not self.trace_dir:
            return None
        Path(self.trace_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.trace_dir) / f"trace_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"

    def _record(self, trace: list, path, **event):
        trace.append(event)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------- loop
    def run(self, question: str) -> AgentResult:
        messages = [{"role": "user", "content": f"Task: {question}"}]
        trace: list[dict] = []
        path = self._open_trace()
        seen: dict[tuple, int] = {}
        stalls = errors = 0
        log.info("\n=== TASK: %s ===", question)
        self._record(trace, path, type="task", question=question)

        for step in range(1, self.max_steps + 1):
            raw = self.llm.chat(self.system, messages)
            messages.append({"role": "assistant", "content": raw})

            # 1) parse -> recover from malformed output
            try:
                data = self.parse(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                errors += 1
                obs = f"FORMAT ERROR: {exc} Reply with a single valid JSON object."
                log.info("[step %d] ⚠ %s", step, obs)
                self._record(trace, path, type="error", step=step, kind="parse", detail=str(exc), raw=raw)
                if errors >= self.max_consecutive_errors:
                    return self._force_finish(messages, trace, path, step, "too_many_errors")
                messages.append({"role": "user", "content": f"Observation: {obs}"})
                continue

            thought = data.get("thought", "")
            log.info("[step %d] 💭 Thought: %s", step, thought)

            # 2) finished
            if "final_answer" in data:
                answer = str(data["final_answer"])
                log.info("[step %d] ✅ Final answer: %s", step, answer)
                self._record(trace, path, type="final", step=step, thought=thought, answer=answer)
                return AgentResult(answer, True, "final_answer", step, trace, str(path) if path else None)

            action = str(data.get("action", "")).strip()
            arg = str(data.get("action_input", "")).strip()
            log.info("[step %d] 🔧 Action: %s(%r)", step, action, arg)

            # 3) wrong tool -> tell the model what exists
            if action not in self.tools:
                errors += 1
                obs = f"ERROR: Unknown tool '{action}'. Available tools: {', '.join(self.tools)}."
                kind = "unknown_tool"
            else:
                # 4) loop detection: identical action+input again and again
                key = (action, arg.lower())
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > self.max_repeats:
                    stalls += 1
                    errors += 1
                    obs = (f"BLOCKED: you already ran {action}({arg!r}) {seen[key] - 1} times with no new "
                           "information. Try a different tool or input, or give the final_answer now.")
                    kind = "repeat_blocked"
                else:
                    try:
                        obs, kind = self.tools[action].run(arg), None
                        errors = 0
                    except ToolError as exc:
                        errors += 1
                        obs, kind = f"TOOL ERROR: {exc}", "tool_error"
                    except Exception as exc:  # never let a tool crash the agent
                        errors += 1
                        obs, kind = f"TOOL CRASHED: {type(exc).__name__}: {exc}", "tool_crash"

            log.info("[step %d] 👀 Observation: %s", step, obs)
            self._record(trace, path, type="step", step=step, thought=thought,
                         action=action, action_input=arg, observation=obs, error=kind)

            if stalls >= self.max_stalls:
                return self._force_finish(messages, trace, path, step, "loop_detected")
            if errors >= self.max_consecutive_errors:
                return self._force_finish(messages, trace, path, step, "too_many_errors")
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        return self._force_finish(messages, trace, path, self.max_steps, "max_steps")

    # ------------------------------------------------- graceful termination
    def _force_finish(self, messages, trace, path, step, reason) -> AgentResult:
        log.info("[step %d] 🛑 Stopping early: %s. Asking for best-effort answer.", step, reason)
        messages = messages + [{"role": "user", "content": (
            "STOP using tools. Using only the observations so far, reply with "
            '{"thought": "...", "final_answer": "..."} giving your best answer, '
            "and say clearly what you could not verify.")}]
        try:
            raw = self.llm.chat(self.system, messages)
            try:
                answer = str(self.parse(raw).get("final_answer", raw))
            except (ValueError, json.JSONDecodeError):
                answer = raw
        except Exception as exc:
            answer = f"Agent stopped ({reason}) and could not produce an answer: {exc}"
        self._record(trace, path, type="forced_stop", step=step, reason=reason, answer=answer)
        log.info("[step %d] ⚠ Best-effort answer: %s", step, answer)
        return AgentResult(answer, False, reason, step, trace, str(path) if path else None)

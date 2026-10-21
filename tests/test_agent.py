import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import ReActAgent
from llm import ScriptedLLM
from tools import ToolError, calculator, default_tools


def act(tool, inp):
    return json.dumps({"thought": "t", "action": tool, "action_input": inp})


def final(ans):
    return json.dumps({"thought": "t", "final_answer": ans})


def make(replies, **kw):
    return ReActAgent(ScriptedLLM(replies), default_tools(), trace_dir=None, **kw)


def test_happy_path():
    r = make([act("calculator", "2+2"), final("4")]).run("2+2?")
    assert r.success and r.answer == "4" and r.stop_reason == "final_answer"


def test_unknown_tool_recovers():
    r = make([act("google", "x"), act("calculator", "3*3"), final("9")]).run("3*3")
    assert r.success
    assert "Unknown tool" in r.trace[1]["observation"]


def test_bad_json_recovers():
    r = make(["I will search now", final("ok")]).run("q")
    assert r.success and r.trace[1]["kind"] == "parse"


def test_loop_is_detected_and_stopped():
    same = act("calculator", "1+1")
    r = make([same] * 4 + [final("best effort")]).run("q")
    assert not r.success and r.stop_reason == "loop_detected"
    assert r.answer == "best effort"


def test_max_steps_forces_answer():
    replies = [act("calculator", f"{i}+1") for i in range(3)] + [final("partial")]
    r = make(replies, max_steps=3).run("q")
    assert r.stop_reason == "max_steps" and r.answer == "partial"


def test_tool_error_becomes_observation():
    r = make([act("calculator", "1/0"), final("cannot")]).run("q")
    assert "TOOL ERROR" in r.trace[1]["observation"] and r.success


def test_calculator_values_and_safety():
    assert calculator("round(sqrt(1889)*3, 2)") == "130.39"
    assert calculator("2**10") == "1024"
    for bad in ["__import__('os').system('ls')", "2**100000", "open('x')"]:
        with pytest.raises(ToolError):
            calculator(bad)

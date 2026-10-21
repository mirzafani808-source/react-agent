"""CLI entry point.

  python main.py "your question"            # real LLM
  python main.py --demo                     # built-in multi-step task
  python main.py --offline                  # no API key: scripted LLM shows the loop + recovery
"""
import argparse
import logging
import sys

from dotenv import load_dotenv

from agent import ReActAgent
from llm import ScriptedLLM, get_llm
from tools import Tool, default_tools

DEMO_TASK = ("The Eiffel Tower was completed in a certain year. Find that year, take its square root, "
             "multiply by 3, and tell me the result rounded to 2 decimals. "
             "Also tell me how many years ago it was completed.")

OFFLINE_SCRIPT = [  # includes a wrong tool + a bad-JSON turn to show recovery
    '{"thought": "I need the completion year.", "action": "google", "action_input": "Eiffel Tower year"}',
    "Sure! Let me search for that.",
    '{"thought": "Use web_search.", "action": "web_search", "action_input": "Eiffel Tower completion year"}',
    '{"thought": "Year is 1889. Compute sqrt*3.", "action": "calculator", "action_input": "round(sqrt(1889) * 3, 2)"}',
    '{"thought": "Now years ago.", "action": "current_datetime", "action_input": ""}',
    '{"thought": "Subtract.", "action": "calculator", "action_input": "2026 - 1889"}',
    '{"thought": "Done.", "final_answer": "sqrt(1889)*3 = 130.39. It was completed 137 years ago (1889)."}',
]


def offline_tools():
    """Same tools, but web_search is stubbed so the demo works without internet."""
    stub = Tool("web_search", "Search the web for facts. Input: a short search query.",
                lambda q: "- Eiffel Tower: Completed in March 1889 for the Paris World's Fair.")
    return [t for t in default_tools() if t.name != "web_search"] + [stub]


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--demo", action="store_true", help="run the built-in multi-step task")
    p.add_argument("--offline", action="store_true", help="scripted LLM, no API key needed")
    p.add_argument("--max-steps", type=int, default=8)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    question = DEMO_TASK if (args.demo or args.offline or not args.question) else args.question
    llm = ScriptedLLM(OFFLINE_SCRIPT) if args.offline else get_llm(args.provider)

    result = ReActAgent(llm, offline_tools() if args.offline else default_tools(),
                        max_steps=args.max_steps).run(question)
    print(f"\nSTATUS: {'success' if result.success else 'stopped early'} "
          f"({result.stop_reason}) in {result.steps} steps")
    print(f"TRACE FILE: {result.trace_file}")


if __name__ == "__main__":
    main()

# LinkedIn post draft (edit with your own details)

Week 3 of my AI engineering cohort: I stopped asking an LLM questions and started letting it *act*. 🤖

I built a ReAct agent from scratch (no frameworks): Thought → Action → Observation, in a loop.

It has 3 tools: a safe calculator, web search, and a date tool. In the video it answers a multi-step question:
find the year the Eiffel Tower was completed → compute √year × 3 → work out how many years ago that was.

The most interesting part wasn't the happy path, it was failure handling:
🔧 The model called a tool that doesn't exist → it gets the list of valid tools and self-corrects
🔁 The model repeated the same action → I block it and nudge it to change approach
🛑 No progress after N steps → graceful stop with a best-effort answer, clearly marked as incomplete
📜 Every step is logged to a JSONL trace, so debugging an agent is just reading a file

Biggest lesson: an agent is mostly guardrails around a loop. The LLM is the easy part.

Code + architecture README: <GITHUB LINK>

#AI #LLM #Agents #ReAct #Python #MachineLearning

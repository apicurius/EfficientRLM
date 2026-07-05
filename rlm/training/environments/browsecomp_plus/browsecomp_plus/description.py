"""Text fed to the root model: env notes prologue + question instruction."""

user_prologue = """BrowseComp-Plus environment notes:
- The evidence documents are available in the REPL variable `context`: a Python
  list of chunk strings (a shuffled mix of gold, supporting, and hard-negative
  document chunks), each starting with a provenance header like
  `[BrowseComp+ doc 7 chunk 2/5]`. Never print, paste, or echo chunk text into
  REPL output -- not even a single chunk or slice (`print(context[i])` floods
  your own context); the ONLY way to read chunk text is to send it to
  `llm_query_batched` and print the compact results.
- Print only compact diagnostics: counts, short samples, candidate answers, and
  final evidence.
- First use Python keyword/regex filtering over `context` to pick a small
  candidate set; then use `llm_query_batched` on focused prompts (one evidence
  chunk per prompt) and aggregate compact results in Python.
- Keep the final answer short: a single succinct answer string with no
  explanation, confidence, or extra commentary (an independent LLM judge
  compares it directly to the gold answer).
- Even if evidence is incomplete, submit your best succinct answer; rollouts
  without a final answer score 0.
- When ready, set `answer["content"]` to ONLY the final answer and then
  `answer["ready"] = True`.
"""

QUESTION_INSTRUCTION = (
    "The context contains evidence document chunks: a shuffled mix of "
    "gold-relevant, supporting, and hard-negative documents for a deep-research "
    "query. Answer the following query using the evidence in `context`."
)

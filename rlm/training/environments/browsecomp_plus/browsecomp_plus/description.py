"""Text fed to the root model: env notes prologue + question instruction."""

user_prologue = """BrowseComp-Plus environment notes:
- The evidence is in the REPL variable `context`: a Python list of chunk
  strings with headers like `[BrowseComp+ doc 7 chunk 2/5]`; do not print,
  paste, or echo raw chunks into REPL output.
- Print only compact diagnostics: counts, chunk ids, candidate answers, and
  short evidence phrases.
- First filter/rank `context` in Python using entities, dates, aliases,
  titles, and relation clues from the question; then screen the remaining
  chunks — all of them, if Python filtering was not clearly decisive — with
  `llm_query_batched` and a strict contract ("Reply YES only if this chunk
  matches these criteria, else NO"), and extract details only from the YES
  survivors with terse structured outputs
  (`candidate_answer | evidence phrase | confidence`, or `NO_MATCH`).
- Every sub-prompt must include the chunk text it should read — never send
  bare search-style queries, and never leave a sub-call's return value
  unprinted or unassigned.
- Before committing, audit the candidate against each clue in the question;
  note clues you cannot verify rather than forcing them.
- Re-read which attribute the question asks for (full name, popular name,
  title, year) and copy that exact surface form from the evidence; submit
  only the clean answer string, with no explanation or citations.
- When ready, set `answer["content"]` to ONLY the final answer and then
  `answer["ready"] = True`; if uncertain, submit your best succinct answer.
"""

QUESTION_INSTRUCTION = (
    "The context contains evidence document chunks: a shuffled mix of "
    "gold-relevant, supporting, and hard-negative documents for a deep-research "
    "query. Answer the following query using the evidence in `context`."
)

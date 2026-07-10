"""Text fed to the root model: env notes prologue + question instruction."""

user_prologue = """BrowseComp-Plus environment notes:
  You are solving a k=50 evidence-document question. The evidence is in `context`,
  a Python list of chunk strings with headers like `[BrowseComp+ doc 7 chunk 2/5]`.
  Chunks include gold-relevant, supporting, and hard-negative evidence.

  The main failure mode is budget collapse: too much printed text or too many
  long intermediate outputs cause `max_output_tokens` / `context_length`
  truncation and often zero reward. Use the root conversation only as a compact
  control plane.

  Rules:
  - Never print or paste raw context chunks, full documents, long excerpts, or
    long sub-LLM outputs.
  - Keep printed diagnostics tiny: counts, chunk ids, candidate answers, and short
    evidence phrases; aim for under ~2K characters per turn.
  - First filter/rank `context` in Python using entities, dates, aliases, titles,
    locations, and relation clues from the question.
  - Then screen broadly: run `llm_query_batched` over the remaining chunks with
    the question's criteria and a strict contract ("Reply YES only if this chunk
    matches these criteria, else NO") — staged ~20-wide batches, one or a few
    chunks per prompt. Screening all k=50 chunks this way is cheap and correct;
    reserve detailed extraction for the few YES survivors.
  - Every sub-prompt must include the chunk text it should read. Never send
    bare search-style queries with no evidence attached, and never leave a
    sub-call's return value unprinted or unassigned.
  - Use focused prompts under the rough ~12K-token sub-prompt budget. Prefer
    small staged batches of useful prompts over huge batches of tiny prompts.
  - Ask sub-LLMs for terse structured output only:
    `candidate_answer | evidence phrase | confidence`, or `NO_MATCH`.
  - Aggregate results in Python variables and print only the compact shortlist.
  - Before committing, audit the candidate against each clue in the question,
    and note which clues you could not verify rather than forcing them.

  Final answer:
  - Re-read which attribute the question asks for (full name, popular name,
    title, year) and copy that exact surface form from the evidence text.
  - Submit only the clean answer string, with no explanation or citations.
  - If uncertain, submit the best succinct answer anyway.
  - Set `answer["content"]` and `answer["ready"] = True` as soon as you have a
    plausible answer.
"""

QUESTION_INSTRUCTION = (
    "The context contains evidence document chunks: a shuffled mix of "
    "gold-relevant, supporting, and hard-negative documents for a deep-research "
    "query. Answer the following query using the evidence in `context`."
)

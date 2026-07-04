"""LM-as-a-judge scoring for BrowseComp-Plus.

The judge prompt and verdict parsing are verbatim from the BrowseComp-Plus
reference (texttron/BrowseComp-Plus ``llm_judge.py``; reference default judge
gpt-5-nano). The released benchmark numbers come from this binary ``is_correct``
JSON judge, so matching it keeps our accuracy on the model-card yardstick.
A deterministic normalized-containment proxy is the non-judge fallback.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import verifiers as vf

# Verbatim reference judge prompt.
JUDGE_PROMPT = """You are an expert judge evaluating whether a predicted answer correctly matches the expected answer for a given query.

Query: {query}

Expected Answer: {expected}

Predicted Answer: {predicted}

Please determine if the predicted answer is correct. Consider:
1. Exact matches are correct
2. Minor formatting differences (e.g., capitalization, punctuation, spacing) should be considered correct if the semantic content is the same
3. Partial answers that contain the correct information should be considered correct
4. Answers that are semantically equivalent but worded differently should be considered correct

Respond with ONLY a JSON object in this exact format:
{{
    "is_correct": true or false
}}"""

# Lazily-built client for external judge models (judge_model="openai/...").
# The rollout client in state["client"] only serves the local inference model.
_OPENAI_JUDGE_CLIENT = None


def _get_openai_judge_client():
    global _OPENAI_JUDGE_CLIENT
    if _OPENAI_JUDGE_CLIENT is None:
        from openai import AsyncOpenAI

        _OPENAI_JUDGE_CLIENT = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
    return _OPENAI_JUDGE_CLIENT


def _parse_verdict(raw: str) -> bool:
    """Extract the boolean ``is_correct``; tolerant of fences and trailing text."""
    if not raw:
        return False
    stripped = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    try:
        parsed: Any = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and "is_correct" in parsed:
        return bool(parsed["is_correct"])
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and "is_correct" in parsed:
            return bool(parsed["is_correct"])
    lower = stripped.lower()
    if "true" in lower or ("correct" in lower and "incorrect" not in lower):
        return True
    return False


def _normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _extract_exact_answer(output: str) -> str:
    m = re.search(r"Exact Answer\s*:\s*(.+)", output, re.IGNORECASE)
    if m:
        first = m.group(1).strip()
        if first:
            return first.splitlines()[0].strip()
    return output.strip()


async def containment_score(info, state: vf.State, **_kw: Any) -> float:
    """Deterministic fallback: normalized exact/containment match."""
    final = str(state.get("rlm_final_answer") or state.get("final_answer") or "")
    meta = json.loads(info) if isinstance(info, str) else info
    gold = str(meta.get("answer", ""))
    ne, ng = _normalize_text(_extract_exact_answer(final)), _normalize_text(gold)
    if not ng:
        return 0.0
    if ne == ng or ng in ne or ne in ng:
        return 1.0
    return 0.0


def make_judge_score(judge_model: str | None = None):
    """Binary is_correct judge scorer (external OpenAI model or rollout client)."""

    async def score(info, state: vf.State, **_kw: Any) -> float:
        final = str(state.get("rlm_final_answer") or state.get("final_answer") or "")
        if not final.strip():
            return 0.0
        meta = json.loads(info) if isinstance(info, str) else info
        prompt = JUDGE_PROMPT.format(
            query=str(meta.get("raw_question") or meta.get("id") or ""),
            expected=str(meta.get("answer", "")),
            predicted=_extract_exact_answer(final),
        )
        model = str(judge_model or state.get("model") or "")
        if model.startswith("openai/"):
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    f"OPENAI_API_KEY is required when BrowseComp+ judge_model is external ({model!r})."
                )
            jm = model.split("/", 1)[1]
            args = (
                {"max_completion_tokens": 512, "reasoning_effort": "minimal"}
                if jm.startswith(("gpt-5", "o1", "o3", "o4"))
                else {"max_tokens": 256, "temperature": 0.0}
            )
            try:
                resp = await _get_openai_judge_client().chat.completions.create(
                    model=jm,
                    messages=[{"role": "user", "content": prompt}],
                    **args,
                )
                raw = resp.choices[0].message.content or ""
            except Exception as e:
                raise RuntimeError(f"BrowseComp+ external judge {model!r} failed: {e}") from e
            return 1.0 if _parse_verdict(raw) else 0.0
        client = state.get("client")
        if client is None:
            return await containment_score(info, state, **_kw)
        try:
            response = await client.get_response(
                prompt=[{"role": "user", "content": prompt}],
                model=model,
                tools=None,
                sampling_args={"max_tokens": 256, "temperature": 0.0},
                state={"trajectory": []},
            )
            raw_content = getattr(getattr(response, "message", None), "content", "")
            if isinstance(raw_content, list):
                raw = "".join(
                    str(
                        getattr(part, "text", "")
                        or (part.get("text") if isinstance(part, dict) else "")
                    )
                    for part in raw_content
                )
            else:
                raw = str(raw_content)
        except Exception:
            return await containment_score(info, state, **_kw)
        return 1.0 if _parse_verdict(raw) else 0.0

    score.__name__ = "browsecomp_plus_judge_score"
    return score

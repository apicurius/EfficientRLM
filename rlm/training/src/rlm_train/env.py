"""RLMTrainEnv: verifiers Environment that mirrors rlm.RLM.completion at depth=1."""

from __future__ import annotations

import asyncio
import logging
import os
import math
import uuid
from collections.abc import Callable
from typing import Any

import verifiers as vf
from rlm.utils.parsing import find_code_blocks
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from verifiers.types import Messages, State

from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.repl.base import ExecResult, ReplBackend
from rlm_train.repl.subprocess import SubprocessReplBackend
from rlm_train.rubric import RLMTrainRubric

logger = logging.getLogger(__name__)

_MAX_REPL_OUTPUT_CHARS = 20_000
_MAX_TRAJECTORY_TEXT_CHARS = 16_000
_MAX_TRAJECTORY_LINE_CHARS = 200
_DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN = 3.0
_ROOT_WINDOW_MARKER = (
    "Earlier RLM transcript messages were omitted to stay within the root model "
    "context budget. The Python REPL state persists; use small `print(...)` calls "
    "or `SHOW_VARS()` to inspect needed variables rather than asking for omitted "
    "raw context or long outputs."
)


class RLMTrainEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        backend_factory: Callable[[], ReplBackend] | None = None,
        max_iterations: int = 30,
        sub_model: str | None = None,
        sub_sampling_args: dict[str, Any] | None = None,
        custom_system_prompt: str | None = None,
        rubric: vf.Rubric | None = None,
        sub_llm_fn: Callable[[str, Any], Any] | None = None,
        sub_llm_fn_batched: Callable[[list[str], Any], Any] | None = None,
        user_prologue: str | None = None,
        bootstrap_code: str | None = None,
        orchestrator: bool = True,
        enforce_sub_prompt_budget: bool = True,
        sub_prompt_token_budget: int = 0,
        sub_prompt_char_budget: int | None = None,
        sub_prompt_chars_per_token: float = _DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN,
        enforce_root_prompt_budget: bool = True,
        root_prompt_token_budget: int = 0,
        root_prompt_char_budget: int | None = None,
        root_prompt_chars_per_token: float = _DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN,
        **kwargs: Any,
    ):
        if "max_turns" in kwargs:
            raise ValueError("Use `max_iterations` instead of `max_turns` for RLMTrainEnv")
        super().__init__(
            max_turns=max_iterations,
            rubric=rubric or RLMTrainRubric(),
            **kwargs,
        )
        # Worker startup timeout is env-configurable: `python -m rlm_train.worker`
        # imports the rlm_train package (verifiers + stack), so many workers
        # cold-starting at once on a shared/slow FS can exceed the 30s default.
        # Set RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S higher on such clusters.
        _startup_timeout = float(os.environ.get("RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S", "30"))
        self._backend_factory = backend_factory or (
            lambda: SubprocessReplBackend(startup_timeout=_startup_timeout)
        )
        self._max_iterations = max_iterations
        self._sub_model = sub_model
        self._sub_sampling_args = sub_sampling_args or {"max_tokens": 4096}
        self._system_prompt = custom_system_prompt or RLM_SYSTEM_PROMPT
        self._orchestrator = orchestrator
        self._user_prologue = user_prologue
        self._sub_llm_fn = sub_llm_fn
        self._sub_llm_fn_batched = sub_llm_fn_batched
        self._bootstrap_code = bootstrap_code or ""
        self._enforce_sub_prompt_budget = bool(enforce_sub_prompt_budget)
        self._sub_prompt_token_budget = int(sub_prompt_token_budget or 0)
        self._sub_prompt_char_budget = (
            int(sub_prompt_char_budget) if sub_prompt_char_budget is not None else None
        )
        self._sub_prompt_chars_per_token = float(sub_prompt_chars_per_token or 0.0)
        self._enforce_root_prompt_budget = bool(enforce_root_prompt_budget)
        self._root_prompt_token_budget = int(root_prompt_token_budget or 0)
        self._root_prompt_char_budget = (
            int(root_prompt_char_budget) if root_prompt_char_budget is not None else None
        )
        self._root_prompt_chars_per_token = float(root_prompt_chars_per_token or 0.0)
        self._proxy: SubLLMProxy | None = None
        self._proxy_lock: asyncio.Lock | None = None

    async def _ensure_proxy(self) -> SubLLMProxy:
        if self._proxy_lock is None:
            self._proxy_lock = asyncio.Lock()
        async with self._proxy_lock:
            if self._proxy is None:
                proxy = SubLLMProxy()
                await proxy.start()
                self._proxy = proxy
            return self._proxy

    async def _teardown_proxy(self) -> None:
        lock = self._proxy_lock or asyncio.Lock()
        async with lock:
            if self._proxy is not None:
                await self._proxy.stop()
                self._proxy = None

    def _build_user_iter(
        self,
        *,
        root_prompt: str | None,
        iteration: int,
        context_count: int,
        history_count: int,
    ) -> dict[str, str]:
        return build_user_prompt(
            root_prompt=root_prompt,
            iteration=iteration,
            context_count=context_count,
            history_count=history_count,
            max_iterations=self._max_iterations,
        )

    async def setup_state(self, state: State) -> None:
        await super().setup_state(state)

        info = state.get("info") or {}
        context_payload = info.get("context")
        if context_payload is None:
            raise ValueError("RLMTrainEnv requires `info['context']` on each dataset row")
        root_prompt: str | None = info.get("root_prompt")

        rollout_id = f"rlm_{uuid.uuid4().hex[:12]}"
        proxy = await self._ensure_proxy()

        proxy.register(
            rollout_id,
            ClientHandle(
                client=state["client"],
                model=self._sub_model or state["model"],
                sampling_args=self._sub_sampling_args,
                record_call=lambda meta: _record_sub_call(state, meta),
                fake_query=self._sub_llm_fn,
                fake_query_batched=self._sub_llm_fn_batched,
                state_ref=state,
                enforce_sub_prompt_budget=self._enforce_sub_prompt_budget,
                sub_prompt_token_budget=self._sub_prompt_token_budget,
                sub_prompt_char_budget=self._sub_prompt_char_budget,
                sub_prompt_chars_per_token=self._sub_prompt_chars_per_token,
            ),
        )

        backend = self._backend_factory()
        await backend.start(proxy_url=proxy.url, rollout_id=rollout_id, depth=1)
        await backend.load_context(context_payload)
        if self._bootstrap_code:
            await backend.bootstrap(self._bootstrap_code)

        metadata = QueryMetadata(context_payload)
        base = build_rlm_system_prompt(
            system_prompt=self._system_prompt,
            query_metadata=metadata,
            custom_tools=None,
            root_prompt=root_prompt,
            orchestrator=self._orchestrator,
        )

        state["rlm_rollout_id"] = rollout_id
        state["rlm_backend"] = backend
        state["rlm_root_prompt"] = root_prompt
        state["rlm_history"] = list(base)
        state["rlm_n_processed"] = 0
        state["rlm_iterations"] = 0
        state["rlm_repl_calls"] = 0
        state["rlm_sub_llm_calls"] = 0
        state["rlm_sub_llm_tokens"] = 0
        state["rlm_sub_llm_usage_missing"] = 0
        state["rlm_sub_llm_prompt_attempts"] = 0
        state["rlm_sub_llm_prompt_rejections"] = 0
        state["rlm_sub_llm_prompt_chars_max"] = 0
        state["rlm_sub_llm_prompt_est_tokens_max"] = 0
        state["rlm_sub_llm_prompt_rejected_chars_max"] = 0
        state["rlm_sub_llm_prompt_rejected_est_tokens_max"] = 0
        state["rlm_root_prompt_chars_max"] = 0
        state["rlm_root_prompt_est_tokens_max"] = 0
        state["rlm_root_prompt_windowed"] = 0
        state["rlm_root_prompt_over_budget"] = 0
        state["rlm_root_prompt_chars_after_window_max"] = 0
        state["rlm_root_prompt_est_tokens_after_window_max"] = 0
        state["rlm_root_prompt_omitted_messages_max"] = 0
        state["rlm_final_answer"] = None
        state["rlm_context_count"] = 1
        state["rlm_final_repl_outputs"] = []
        state["rlm_trajectory_text"] = ""
        state["rlm_trajectory_text_truncated"] = 0

        if self._user_prologue:
            state["rlm_history"].append({"role": "user", "content": self._user_prologue})

        user_iter0 = self._build_user_iter(
            root_prompt=root_prompt, iteration=0, context_count=1, history_count=0
        )
        state["rlm_history"].append(user_iter0)
        state["prompt"] = list(state["rlm_history"])
        _prepare_root_prompt(
            state,
            state["prompt"],
            enforce=self._enforce_root_prompt_budget,
            token_budget=self._root_prompt_token_budget,
            char_budget=self._root_prompt_char_budget,
            chars_per_token=self._root_prompt_chars_per_token,
        )

    async def _process_pending_trajectory(self, state: State) -> Messages | None:
        """Execute trajectory turns whose REPL blocks have not been processed.

        RLM's environment side effects are intentionally delayed until the next
        scaffold pass: after the model emits a ```repl``` block, that code is
        run when `get_prompt_messages()` prepares the following turn.  At the
        hard turn cap, however, verifiers checks stop conditions before the next
        `get_prompt_messages()` call, so a final-turn
        `answer["ready"] = True` block can be dropped.  This helper centralizes
        the pending-turn execution so the stop hook can flush that final turn
        before `max_turns_reached` wins.

        Returns prepared messages only when executing a pending turn produced a
        final answer; otherwise returns None and leaves normal prompt assembly
        to `get_prompt_messages()`.
        """
        history: list = state["rlm_history"]
        backend: ReplBackend = state["rlm_backend"]
        n_done = len(state["trajectory"])
        n_processed = int(state.get("rlm_n_processed") or 0)

        while n_processed < n_done:
            step = state["trajectory"][n_processed]
            assistant_msg = _last_assistant(step["completion"])
            assistant_text = _msg_text(assistant_msg)

            outputs: list[dict[str, Any]] = []
            final_from_answer: str | None = None
            for code in find_code_blocks(assistant_text):
                try:
                    result = await backend.execute(code)
                except Exception as e:  # noqa: BLE001
                    outputs.append(
                        {
                            "code": code,
                            "stdout": "",
                            "stderr": f"Worker error: {e}",
                            "final_answer": None,
                            "locals_keys": [],
                        }
                    )
                    continue
                outputs.append(_pack_exec(code, result))
                state["rlm_repl_calls"] = int(state.get("rlm_repl_calls") or 0) + 1
                if result.final_answer is not None and final_from_answer is None:
                    final_from_answer = result.final_answer

            repl_msgs = _format_repl_outputs(outputs)
            _accumulate_trajectory_text(state, outputs)
            history.append(assistant_msg)
            history.extend(repl_msgs)
            state["rlm_n_processed"] = n_processed + 1
            state["rlm_iterations"] = n_processed + 1
            n_processed += 1

            if final_from_answer is not None:
                state["rlm_final_answer"] = final_from_answer
                state["final_answer"] = final_from_answer
                state["final_env_response"] = repl_msgs
                state["rlm_final_repl_outputs"] = repl_msgs
                messages = _normalize_for_api(history)
                return _prepare_root_prompt(
                    state,
                    messages,
                    enforce=self._enforce_root_prompt_budget,
                    token_budget=self._root_prompt_token_budget,
                    char_budget=self._root_prompt_char_budget,
                    chars_per_token=self._root_prompt_chars_per_token,
                )

        return None

    def _pending_turn_at_terminal_cap(self, state: State) -> bool:
        """Whether pending REPL code would be skipped by an imminent hard stop."""

        if state.get("rlm_final_answer") is not None:
            return False
        if int(state.get("rlm_n_processed") or 0) >= len(state.get("trajectory") or []):
            return False
        return self.max_turns > 0 and len(state["trajectory"]) >= self.max_turns

    async def get_prompt_messages(self, state: State) -> Messages:
        if not state["trajectory"]:
            prompt = list(state["prompt"])
            return _prepare_root_prompt(
                state,
                prompt,
                enforce=self._enforce_root_prompt_budget,
                token_budget=self._root_prompt_token_budget,
                char_budget=self._root_prompt_char_budget,
                chars_per_token=self._root_prompt_chars_per_token,
            )

        final_messages = await self._process_pending_trajectory(state)
        if final_messages is not None:
            return final_messages

        n_processed = int(state.get("rlm_n_processed") or 0)
        history: list = state["rlm_history"]
        user_iter = self._build_user_iter(
            root_prompt=state.get("rlm_root_prompt"),
            iteration=n_processed,
            context_count=int(state.get("rlm_context_count") or 1),
            history_count=0,
        )
        history.append(user_iter)
        messages = _normalize_for_api(history)
        return _prepare_root_prompt(
            state,
            messages,
            enforce=self._enforce_root_prompt_budget,
            token_budget=self._root_prompt_token_budget,
            char_budget=self._root_prompt_char_budget,
            chars_per_token=self._root_prompt_chars_per_token,
        )

    async def env_response(self, messages: Messages, state: State, **kwargs: Any) -> Messages | str:
        return []

    @vf.stop
    async def has_final_answer(self, state: State) -> bool:
        if self._pending_turn_at_terminal_cap(state):
            await self._process_pending_trajectory(state)
        return state.get("rlm_final_answer") is not None

    @vf.cleanup
    async def cleanup_rlm(self, state: State) -> None:
        backend = state.get("rlm_backend")
        if backend is not None:
            try:
                await backend.stop()
            except Exception:
                logger.exception("backend stop failed")
            state["rlm_backend"] = None
        rollout_id = state.get("rlm_rollout_id")
        if rollout_id and self._proxy is not None:
            self._proxy.unregister(rollout_id)

    @vf.teardown
    async def teardown_rlm(self) -> None:
        await self._teardown_proxy()


def _record_sub_call(state: State, meta: Any) -> None:
    """Telemetry-only sub-LLM accounting.

    Increments the sub-LLM call counter and, when the proxy reports token usage
    in ``meta['usage']``, accumulates total sub-LLM tokens. This is pure
    monitoring: it does NOT change the upstream correctness-only reward. The
    optional efficiency-shaping rubric reads these counters but is opt-in.
    """
    state["rlm_sub_llm_calls"] = int(state.get("rlm_sub_llm_calls") or 0) + 1
    usage = meta.get("usage") if isinstance(meta, dict) else None
    if not isinstance(usage, dict):
        state["rlm_sub_llm_usage_missing"] = int(state.get("rlm_sub_llm_usage_missing") or 0) + 1
        return

    tokens = usage.get("total_tokens")
    try:
        if tokens is None:
            prompt_tokens = usage.get("prompt_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or 0
            tokens = int(prompt_tokens) + int(completion_tokens)
        state["rlm_sub_llm_tokens"] = int(state.get("rlm_sub_llm_tokens") or 0) + int(tokens)
    except (TypeError, ValueError):
        state["rlm_sub_llm_usage_missing"] = int(state.get("rlm_sub_llm_usage_missing") or 0) + 1


def _normalize_for_api(msgs: list) -> list:
    out: list = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            out.append(m)
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if content is not None and content != "":
            out.append(m)
            continue
        reasoning = (
            m.get("reasoning_content")
            if isinstance(m, dict)
            else getattr(m, "reasoning_content", None)
        )
        if isinstance(m, dict):
            new = dict(m)
        else:
            try:
                new = m.model_dump()
            except AttributeError:
                new = {"role": role, "content": content}
        new["content"] = reasoning if reasoning else ""
        out.append(new)
    return out


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
            if t:
                parts.append(str(t))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _messages_text(msgs: list) -> str:
    parts: list[str] = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role:
            parts.append(str(role))
        text = _message_content_text(content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _root_prompt_limit(
    *,
    token_budget: int,
    char_budget: int | None,
    chars_per_token: float,
) -> tuple[int, int]:
    cpt = chars_per_token if chars_per_token > 0 else _DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN
    token_limit = int(token_budget or 0)
    char_limit = int(char_budget) if char_budget is not None else 0
    if char_limit <= 0 and token_limit > 0:
        char_limit = int(token_limit * max(cpt, 1.0))
    return token_limit, char_limit


def _record_root_prompt_size(state: State, msgs: list, *, after_window: bool = False) -> None:
    text = _messages_text(msgs)
    chars = len(text)
    est_tokens = int(math.ceil(chars / _DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN))
    if after_window:
        state["rlm_root_prompt_chars_after_window_max"] = max(
            int(state.get("rlm_root_prompt_chars_after_window_max") or 0), chars
        )
        state["rlm_root_prompt_est_tokens_after_window_max"] = max(
            int(state.get("rlm_root_prompt_est_tokens_after_window_max") or 0), est_tokens
        )
    else:
        state["rlm_root_prompt_chars_max"] = max(
            int(state.get("rlm_root_prompt_chars_max") or 0), chars
        )
        state["rlm_root_prompt_est_tokens_max"] = max(
            int(state.get("rlm_root_prompt_est_tokens_max") or 0), est_tokens
        )
    logger.debug(
        "RLM root prompt size rollout=%s after_window=%s chars=%d est_tokens=%d",
        state.get("rlm_rollout_id"),
        after_window,
        chars,
        est_tokens,
    )


def _fits_root_budget(msgs: list, *, char_limit: int) -> bool:
    if char_limit <= 0:
        return True
    return len(_messages_text(msgs)) <= char_limit


def _prefix_until_first_assistant(msgs: list) -> list:
    for i, m in enumerate(msgs):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "assistant":
            return list(msgs[:i])
    return list(msgs)


def _window_root_prompt(
    state: State,
    msgs: list,
    *,
    token_budget: int,
    char_budget: int | None,
    chars_per_token: float,
) -> list:
    _token_limit, char_limit = _root_prompt_limit(
        token_budget=token_budget,
        char_budget=char_budget,
        chars_per_token=chars_per_token,
    )
    if char_limit <= 0 or _fits_root_budget(msgs, char_limit=char_limit):
        return list(msgs)

    state["rlm_root_prompt_over_budget"] = int(state.get("rlm_root_prompt_over_budget") or 0) + 1
    prefix = _prefix_until_first_assistant(msgs)
    # No generated turns yet: there is nothing safe to drop.
    if len(prefix) == len(msgs):
        return list(msgs)

    marker = {"role": "user", "content": _ROOT_WINDOW_MARKER}
    suffix = list(msgs[len(prefix) :])
    kept_suffix: list = []

    for m in reversed(suffix):
        candidate = prefix + [marker] + [m] + kept_suffix
        if _fits_root_budget(candidate, char_limit=char_limit):
            kept_suffix.insert(0, m)

    # Always keep the most recent message, even if the pinned prompt is already tight.
    if not kept_suffix and suffix:
        kept_suffix = [suffix[-1]]

    windowed = prefix + [marker] + kept_suffix
    omitted = len(msgs) - len(windowed)
    state["rlm_root_prompt_windowed"] = int(state.get("rlm_root_prompt_windowed") or 0) + 1
    state["rlm_root_prompt_omitted_messages_max"] = max(
        int(state.get("rlm_root_prompt_omitted_messages_max") or 0), max(0, omitted)
    )
    logger.info(
        "RLM root prompt windowed rollout=%s omitted=%d before_chars=%d after_chars=%d",
        state.get("rlm_rollout_id"),
        omitted,
        len(_messages_text(msgs)),
        len(_messages_text(windowed)),
    )
    return windowed


def _prepare_root_prompt(
    state: State,
    msgs: list,
    *,
    enforce: bool,
    token_budget: int,
    char_budget: int | None,
    chars_per_token: float,
) -> list:
    _record_root_prompt_size(state, msgs, after_window=False)
    out = (
        _window_root_prompt(
            state,
            msgs,
            token_budget=token_budget,
            char_budget=char_budget,
            chars_per_token=chars_per_token,
        )
        if enforce
        else list(msgs)
    )
    _record_root_prompt_size(state, out, after_window=True)
    return out


def _last_assistant(completion: Any) -> Any:
    if not completion:
        return {"role": "assistant", "content": ""}
    if isinstance(completion, list):
        for m in reversed(completion):
            r = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            if r == "assistant":
                return m
        return completion[-1]
    return completion


def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
            if t:
                parts.append(str(t))
        return "".join(parts)
    return content or ""


def _pack_exec(code: str, result: ExecResult) -> dict[str, Any]:
    return {
        "code": code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "locals_keys": result.locals_keys,
        "final_answer": result.final_answer,
    }


def _accumulate_trajectory_text(state: State, outputs: list[dict[str, Any]]) -> None:
    if not outputs:
        return
    existing = state.get("rlm_trajectory_text") or ""
    if len(existing) >= _MAX_TRAJECTORY_TEXT_CHARS:
        state["rlm_trajectory_text_truncated"] = 1
        return
    chunks: list[str] = []
    for o in outputs:
        stdout = o.get("stdout") or ""
        if not stdout:
            continue
        for line in stdout.splitlines():
            line = line.strip()
            if not line or len(line) > _MAX_TRAJECTORY_LINE_CHARS:
                continue
            chunks.append(line)
    if not chunks:
        return
    merged = existing + ("\n" if existing else "") + "\n".join(chunks)
    if len(merged) > _MAX_TRAJECTORY_TEXT_CHARS:
        merged = merged[:_MAX_TRAJECTORY_TEXT_CHARS]
        state["rlm_trajectory_text_truncated"] = 1
    state["rlm_trajectory_text"] = merged


def _format_repl_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not outputs:
        return []
    parts: list[str] = []
    multi = len(outputs) > 1
    for i, o in enumerate(outputs):
        body = _format_one(o)
        header = f"REPL output (block {i + 1}):" if multi else "REPL output:"
        parts.append(f"{header}\n{body}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def _format_one(o: dict[str, Any]) -> str:
    parts: list[str] = []
    if o.get("stdout"):
        parts.append(f"\n{o['stdout']}")
    if o.get("stderr"):
        parts.append(f"\n{o['stderr']}")
    if o.get("locals_keys"):
        parts.append(f"REPL variables: {list(o['locals_keys'])}\n")
    body = "\n\n".join(parts) if parts else "No output"
    if len(body) > _MAX_REPL_OUTPUT_CHARS:
        body = (
            body[:_MAX_REPL_OUTPUT_CHARS] + f"... + [{len(body) - _MAX_REPL_OUTPUT_CHARS} chars...]"
        )
    return body

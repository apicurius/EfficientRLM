"""Localhost HTTP proxy: worker sub-LLM calls -> verifiers Client."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)

DEFAULT_SUB_PROMPT_TOKEN_BUDGET = 12_000
DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN = 3.0


FakeQuery = Callable[[str, "Any"], str | Awaitable[str]]
FakeQueryBatched = Callable[[list[str], "Any"], list[str] | Awaitable[list[str]]]


class SubPromptTooLargeError(ValueError):
    """Raised when a sub-LLM prompt is rejected before the model call."""


@dataclass
class ClientHandle:
    client: Any
    model: str
    sampling_args: dict[str, Any] | None = None
    record_call: Any | None = None
    max_concurrent: int = 16
    fake_query: FakeQuery | None = None
    fake_query_batched: FakeQueryBatched | None = None
    state_ref: Any | None = None
    enforce_sub_prompt_budget: bool = True
    sub_prompt_token_budget: int = DEFAULT_SUB_PROMPT_TOKEN_BUDGET
    sub_prompt_char_budget: int | None = None
    sub_prompt_chars_per_token: float = DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _flatten_prompt(prompt: str | list) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in reversed(prompt):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            if role != "user":
                continue
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for p in content:
                    t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                    if t:
                        parts.append(str(t))
                return "".join(parts)
            if content is not None:
                return str(content)
    return str(prompt)


def _prompt_budget_text(prompt: str | list) -> str:
    """Best-effort full prompt text used only for local budget checks/metrics."""

    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)

    parts: list[str] = []
    for m in prompt:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role:
            parts.append(str(role))
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                if t:
                    parts.append(str(t))
        elif content is not None:
            parts.append(str(content))
    return "\n".join(parts)


def _estimate_tokens(chars: int, chars_per_token: float) -> int:
    cpt = chars_per_token if chars_per_token > 0 else DEFAULT_SUB_PROMPT_CHARS_PER_TOKEN
    return int(math.ceil(chars / cpt))


def _state_incr(state: Any, key: str, delta: int = 1) -> None:
    if state is None:
        return
    try:
        state[key] = int(state.get(key) or 0) + int(delta)
    except Exception:
        logger.exception("failed updating state counter %s", key)


def _state_max(state: Any, key: str, value: int) -> None:
    if state is None:
        return
    try:
        state[key] = max(int(state.get(key) or 0), int(value))
    except Exception:
        logger.exception("failed updating state max %s", key)


def _record_sub_prompt_stats(handle: ClientHandle, chars: int, est_tokens: int) -> None:
    state = handle.state_ref
    _state_incr(state, "rlm_sub_llm_prompt_attempts")
    _state_max(state, "rlm_sub_llm_prompt_chars_max", chars)
    _state_max(state, "rlm_sub_llm_prompt_est_tokens_max", est_tokens)


def _record_sub_prompt_rejection(handle: ClientHandle, chars: int, est_tokens: int) -> None:
    state = handle.state_ref
    _state_incr(state, "rlm_sub_llm_prompt_rejections")
    _state_max(state, "rlm_sub_llm_prompt_rejected_chars_max", chars)
    _state_max(state, "rlm_sub_llm_prompt_rejected_est_tokens_max", est_tokens)


def _check_sub_prompt_budget(handle: ClientHandle, prompt: str | list) -> None:
    text = _prompt_budget_text(prompt)
    chars = len(text)
    est_tokens = _estimate_tokens(chars, handle.sub_prompt_chars_per_token)
    _record_sub_prompt_stats(handle, chars, est_tokens)

    if not handle.enforce_sub_prompt_budget:
        return

    token_budget = int(handle.sub_prompt_token_budget or 0)
    char_budget = handle.sub_prompt_char_budget
    if char_budget is None and token_budget > 0:
        char_budget = int(token_budget * max(handle.sub_prompt_chars_per_token, 1.0))
    char_budget = int(char_budget or 0)

    over_tokens = token_budget > 0 and est_tokens > token_budget
    over_chars = char_budget > 0 and chars > char_budget
    if not (over_tokens or over_chars):
        return

    _record_sub_prompt_rejection(handle, chars, est_tokens)
    reasons: list[str] = []
    if over_tokens:
        reasons.append(f"estimated {est_tokens:,} tokens > cap {token_budget:,}")
    if over_chars:
        reasons.append(f"{chars:,} chars > cap {char_budget:,}")
    chunk_basis = max(
        (est_tokens / token_budget) if token_budget > 0 else 0.0,
        (chars / char_budget) if char_budget > 0 else 0.0,
        1.0,
    )
    chunks = int(math.ceil(chunk_basis))
    cap_text = []
    if token_budget > 0:
        cap_text.append(f"{token_budget:,} estimated tokens")
    if char_budget > 0:
        cap_text.append(f"{char_budget:,} chars")
    raise SubPromptTooLargeError(
        "sub-LLM prompt exceeded the scaffold budget "
        f"({'; '.join(reasons)}). "
        f"Cap: {' / '.join(cap_text)}. "
        f"Split this input into at least {chunks} smaller chunk(s) before calling "
        "`llm_query` or `llm_query_batched`. The prompt was rejected before the "
        "model call; no sub-LLM call was spent."
    )


def _coerce_messages(prompt: str | list) -> list:
    if isinstance(prompt, str):
        raw: list = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        raw = prompt
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")
    try:
        from verifiers.utils.message_utils import from_raw_message
    except Exception:
        return raw
    out = []
    for m in raw:
        if isinstance(m, dict):
            out.append(from_raw_message(dict(m)))
        else:
            out.append(m)
    return out


class SubLLMProxy:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._host = host
        self._port = port
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._handles: dict[str, ClientHandle] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._runner is not None:
            return
        app = web.Application()
        app.router.add_post("/rollout/{rollout_id}/llm_query", self._handle_single)
        app.router.add_post("/rollout/{rollout_id}/llm_query_batched", self._handle_batched)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port)
        await site.start()
        if self._port == 0:
            server = getattr(site, "_server", None)
            socks = getattr(server, "sockets", None) if server else None
            if socks:
                self._port = socks[0].getsockname()[1]
        self._app, self._runner, self._site = app, runner, site
        logger.info("SubLLMProxy listening on %s", self.url)

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._app = self._runner = self._site = None
        self._handles.clear()
        self._semaphores.clear()

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def register(self, rollout_id: str, handle: ClientHandle) -> None:
        self._handles[rollout_id] = handle
        self._semaphores[rollout_id] = asyncio.Semaphore(handle.max_concurrent)

    def unregister(self, rollout_id: str) -> None:
        self._handles.pop(rollout_id, None)
        self._semaphores.pop(rollout_id, None)

    async def _handle_single(self, request: web.Request) -> web.Response:
        rollout_id = request.match_info["rollout_id"]
        handle = self._handles.get(rollout_id)
        if handle is None:
            return web.json_response({"error": f"unknown rollout_id {rollout_id!r}"}, status=404)
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"bad json: {e}"}, status=400)
        prompt = body.get("prompt")
        if prompt is None:
            return web.json_response({"error": "missing 'prompt'"}, status=400)
        model = body.get("model") or handle.model
        try:
            text, meta = await self._completion(handle, prompt, model)
        except SubPromptTooLargeError as e:
            logger.warning("sub-llm prompt rejected: %s", e)
            return web.json_response({"error": str(e), "too_large": True})
        except Exception as e:  # noqa: BLE001
            logger.exception("sub-llm call failed")
            return web.json_response({"error": str(e)})
        if handle.record_call is not None:
            try:
                handle.record_call({"model": model, "prompt": prompt, "response": text, **meta})
            except Exception:
                logger.exception("record_call failed")
        return web.json_response({"response": text, **meta})

    async def _handle_batched(self, request: web.Request) -> web.Response:
        rollout_id = request.match_info["rollout_id"]
        handle = self._handles.get(rollout_id)
        if handle is None:
            return web.json_response({"error": f"unknown rollout_id {rollout_id!r}"}, status=404)
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response({"error": f"bad json: {e}"}, status=400)
        prompts = body.get("prompts")
        if not isinstance(prompts, list):
            return web.json_response({"error": "missing 'prompts' list"}, status=400)
        model = body.get("model") or handle.model

        if handle.fake_query_batched is not None:
            try:
                result = handle.fake_query_batched(list(prompts), handle.state_ref)
                responses = await _maybe_await(result)
            except Exception as e:  # noqa: BLE001
                logger.exception("fake_query_batched failed")
                return web.json_response({"error": str(e)})
            if responses is not None:
                if not isinstance(responses, list) or len(responses) != len(prompts):
                    return web.json_response({"error": "fake_query_batched returned wrong shape"})
                if handle.record_call is not None:
                    for p, r in zip(prompts, responses, strict=True):
                        try:
                            handle.record_call({"model": model, "prompt": p, "response": r})
                        except Exception:
                            logger.exception("record_call failed")
                return web.json_response(
                    {"responses": [r if isinstance(r, str) else str(r) for r in responses]}
                )

        sem = self._semaphores.get(rollout_id) or asyncio.Semaphore(handle.max_concurrent)

        async def run_one(p: str) -> str:
            async with sem:
                try:
                    text, meta = await self._completion(handle, p, model)
                    if handle.record_call is not None:
                        try:
                            handle.record_call(
                                {"model": model, "prompt": p, "response": text, **meta}
                            )
                        except Exception:
                            logger.exception("record_call failed")
                    return text
                except SubPromptTooLargeError as e:
                    logger.warning("sub-llm batched prompt rejected: %s", e)
                    return f"Error: {e}"
                except Exception as e:  # noqa: BLE001
                    logger.exception("sub-llm batched call failed")
                    return f"Error: {e}"

        results = await asyncio.gather(*(run_one(p) for p in prompts))
        return web.json_response({"responses": results})

    async def _completion(
        self,
        handle: ClientHandle,
        prompt: str | list,
        model: str,
    ) -> tuple[str, dict[str, Any]]:
        if handle.fake_query is not None:
            prompt_text = _flatten_prompt(prompt)
            result = handle.fake_query(prompt_text, handle.state_ref)
            content = await _maybe_await(result)
            if content is not None:
                return (content if isinstance(content, str) else str(content)), {}

        _check_sub_prompt_budget(handle, prompt)
        messages = _coerce_messages(prompt)
        sampling_args = dict(handle.sampling_args or {})
        # TITO client requires non-None state; hand it an empty trajectory.
        response = await handle.client.get_response(
            prompt=messages,
            model=model,
            tools=None,
            sampling_args=sampling_args,
            state={"trajectory": []},
        )
        try:
            raw_content = response.message.content
        except AttributeError:
            raw_content = None
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts = []
            for p in raw_content:
                text = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                if text:
                    parts.append(text)
            content = "".join(parts)
        else:
            content = ""
        meta: dict[str, Any] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return content, meta

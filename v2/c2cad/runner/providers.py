"""Provider calls with a complete record of what was sent and returned.

model_spec = "<provider>:<model id>", e.g.
  openai:gpt-5.4        anthropic:claude-opus-5        google:gemini-3.1-pro-preview
  deepseek:deepseek-chat moonshot:kimi-k2.5            openrouter:qwen/qwen3-coder
  together:<model>       mock:reference | mock:jitter-0.5 | mock:beam-box | mock:empty | mock:noisy
Keys come from the environment or v2/.env / the repository's .env (never logged).

Transport
  * OpenAI-compatible providers and Google: plain HTTPS with `requests`, streamed (SSE) by default so long
    reasoning responses are not cut by a fixed request timeout; `idle_timeout` bounds the gap between chunks.
  * Anthropic: the official `anthropic` SDK, streamed, with its own retries disabled (retries are done here,
    identically for every provider, and counted in the record).
Retries: 408/409/425/429/5xx/529, connection errors and timeouts, with exponential backoff that honours
`retry-after`. Other 4xx errors are not retried (they are recorded as api_error).
Sampling: some current APIs reject `temperature` (e.g. Claude Opus 5/5.5, Sonnet 5, Fable 5.x return 400).
For those the registry sets send_temperature = false; the record then says the temperature was not sent.
Constrained decoding (schema arm): Settings.schema is sent as the provider's native JSON-schema output
option; Response.enforcement records what was actually enforced (strict_schema | json_mode | none).
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

REPO = Path(__file__).resolve().parents[3]

OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
MAX_ATTEMPTS = 6
BACKOFF_CAP_S = 120.0


def _load_env():
    for p in (REPO / "v2" / ".env", REPO / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip("\"'")
                    if v and not os.environ.get(k.strip()):
                        os.environ[k.strip()] = v


_load_env()


@dataclass
class Settings:
    max_tokens: int = 32768
    temperature: Optional[float] = None      # the requested temperature (recorded even when not sent)
    send_temperature: bool = True            # False for APIs that reject sampling parameters
    reasoning_effort: Optional[str] = None   # OpenAI reasoning_effort / Anthropic output_config.effort / OpenRouter reasoning
    thinking: Optional[str] = None           # Anthropic: "adaptive" | "disabled" | None (omit the field)
    thinking_budget: Optional[int] = None    # Google 2.5 thinkingBudget; Anthropic budget_tokens (legacy models only)
    stream: bool = True
    idle_timeout: int = 1800                 # seconds without a byte before a streamed request is abandoned
    schema: Optional[dict] = None            # JSON schema for constrained decoding (schema arm)
    schema_mode: str = "json_schema"         # what the provider supports: json_schema | json_object | none
    google_schema_field: str = "responseJsonSchema"   # Google only: or "responseSchema" (OpenAPI subset)
    extra_body: dict = field(default_factory=dict)   # provider-specific parameters from the model registry

    def record(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "schema"}
        d["schema_sent"] = self.schema is not None
        return d


@dataclass
class Response:
    text: str
    ok: bool
    error: str = ""
    returned_model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    request_meta: dict = field(default_factory=dict)   # sent parameters, no secrets, no prompt text
    attempts: int = 1
    enforcement: str = "none"                          # strict_schema | json_mode | none
    http_status: int = 0


class _Transient(Exception):
    def __init__(self, msg, retry_after=None, status=0):
        super().__init__(msg)
        self.retry_after, self.status = retry_after, status


class _Fatal(Exception):
    def __init__(self, msg, status=0):
        super().__init__(msg)
        self.status = status


def available(provider: str) -> bool:
    if provider == "mock":
        return True
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "google":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return provider in OPENAI_COMPATIBLE and bool(os.environ.get(OPENAI_COMPATIBLE[provider][1]))


def _messages(user_or_messages) -> list[dict]:
    if isinstance(user_or_messages, str):
        return [{"role": "user", "content": user_or_messages}]
    return list(user_or_messages)


def call(model_spec: str, system: str, user_or_messages, st: Settings,
         sleep: Callable[[float], None] = time.sleep) -> Response:
    """One request with retries. `user_or_messages` is the user prompt or a list of {role, content} turns."""
    provider, _, model = model_spec.partition(":")
    msgs = _messages(user_or_messages)
    fn = {"anthropic": _anthropic, "google": _google}.get(provider)
    if fn is None and provider in OPENAI_COMPATIBLE:
        fn = lambda m, s, ms, st_: _openai_like(provider, m, s, ms, st_)   # noqa: E731
    if fn is None:
        return Response("", False, f"unknown provider {provider!r}")
    t0 = time.time()
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = fn(model, system, msgs, st)
            r.attempts, r.latency_s = attempt, time.time() - t0
            return r
        except _Transient as e:
            last = f"{e} (status {e.status})"
            if attempt == MAX_ATTEMPTS:
                break
            wait = e.retry_after if e.retry_after else min(BACKOFF_CAP_S, 2.0 * 2 ** (attempt - 1))
            sleep(min(BACKOFF_CAP_S, wait) + random.uniform(0, 1.0))
        except _Fatal as e:
            return Response("", False, str(e), attempts=attempt, latency_s=time.time() - t0, http_status=e.status)
    return Response("", False, f"gave up after {MAX_ATTEMPTS} attempts: {last}", attempts=MAX_ATTEMPTS,
                    latency_s=time.time() - t0)


# ---------------------------------------------------------------------------
def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _retry_after(headers) -> Optional[float]:
    try:
        return float(headers.get("retry-after")) if headers and headers.get("retry-after") else None
    except (TypeError, ValueError):
        return None


def _post(url, headers, body, st: Settings, stream: bool):
    try:
        timeout = (30, st.idle_timeout) if stream else (30, max(600, st.max_tokens // 20 + 600))
        resp = requests.post(url, headers=headers, json=body, timeout=timeout, stream=stream)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise _Transient(f"{type(e).__name__}: {e}")
    if resp.status_code in TRANSIENT_STATUS:
        raise _Transient(f"HTTP {resp.status_code}: {resp.text[:300]}", _retry_after(resp.headers), resp.status_code)
    if resp.status_code >= 400:
        raise _Fatal(f"HTTP {resp.status_code}: {resp.text[:600]}", resp.status_code)
    return resp


def _sse(resp):
    """Yield parsed JSON objects from a server-sent-event stream."""
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
        raise _Transient(f"stream interrupted: {type(e).__name__}: {e}")


def _openai_like(provider, model, system, msgs, st: Settings):
    base, key_env = OPENAI_COMPATIBLE[provider]
    key = os.environ.get(key_env, "")
    if not key:
        raise _Fatal(f"{key_env} not set")
    body = {"model": model, "messages": [{"role": "system", "content": system}] + msgs}
    body["max_completion_tokens" if provider == "openai" else "max_tokens"] = st.max_tokens
    if st.reasoning_effort:
        if provider == "openrouter":
            body["reasoning"] = {"effort": st.reasoning_effort}
        else:
            body["reasoning_effort"] = st.reasoning_effort
    if st.temperature is not None and st.send_temperature:
        body["temperature"] = st.temperature
    enforcement = "none"
    if st.schema is not None and st.schema_mode == "json_schema":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "assembly", "strict": True, "schema": st.schema}}
        if provider == "openrouter":   # route only to hosts that honour the schema
            body["provider"] = _deep_merge(body.get("provider", {}), {"require_parameters": True})
        enforcement = "strict_schema"
    elif st.schema is not None and st.schema_mode == "json_object":
        body["response_format"] = {"type": "json_object"}
        enforcement = "json_mode"
    body = _deep_merge(body, st.extra_body)
    if st.stream:
        body["stream"] = True
        if provider in ("openai", "deepseek", "openrouter", "together"):
            body["stream_options"] = {"include_usage": True}
    meta = {k: v for k, v in body.items() if k not in ("messages",)}
    if "response_format" in meta and "json_schema" in meta["response_format"]:
        meta["response_format"] = {"type": "json_schema", "strict": True, "schema": "<assembly schema>"}
    resp = _post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}, body, st, st.stream)
    if not st.stream:
        j = resp.json()
        ch = (j.get("choices") or [{}])[0]
        return Response((ch.get("message") or {}).get("content") or "", True, returned_model=j.get("model", ""),
                        finish_reason=ch.get("finish_reason") or "", usage=j.get("usage") or {},
                        request_meta=meta, enforcement=enforcement, http_status=resp.status_code)
    text, finish, usage, model_out, reasoning_chars = [], "", {}, "", 0
    for ev in _sse(resp):
        if "error" in ev:
            err = ev["error"]
            code = err.get("code") if isinstance(err, dict) else None
            msg = json.dumps(err)[:500]
            if code in TRANSIENT_STATUS or "overload" in msg.lower() or "rate" in msg.lower():
                raise _Transient(f"stream error: {msg}")
            raise _Fatal(f"stream error: {msg}")
        model_out = ev.get("model") or model_out
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                text.append(d["content"])
            rc = d.get("reasoning_content") or d.get("reasoning")
            if isinstance(rc, str):
                reasoning_chars += len(rc)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    if reasoning_chars:
        usage = dict(usage, visible_reasoning_chars=reasoning_chars)
    return Response("".join(text), True, returned_model=model_out, finish_reason=finish, usage=usage,
                    request_meta=meta, enforcement=enforcement, http_status=resp.status_code)


def _anthropic(model, system, msgs, st: Settings):
    import anthropic   # official SDK; retries are handled by call()

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise _Fatal("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key, max_retries=0,
                                 timeout=anthropic.Timeout(connect=30.0, read=float(st.idle_timeout), write=60.0,
                                                           pool=30.0))
    kwargs = {"model": model, "max_tokens": st.max_tokens, "system": system, "messages": msgs}
    if st.temperature is not None and st.send_temperature:
        kwargs["temperature"] = st.temperature
    extra = {}
    if st.thinking in ("adaptive", "disabled"):
        extra["thinking"] = {"type": st.thinking}
    elif st.thinking_budget:
        extra["thinking"] = {"type": "enabled", "budget_tokens": st.thinking_budget}
    oc = {}
    if st.reasoning_effort:
        oc["effort"] = st.reasoning_effort
    enforcement = "none"
    if st.schema is not None and st.schema_mode == "json_schema":
        oc["format"] = {"type": "json_schema", "schema": st.schema}
        enforcement = "strict_schema"
    if oc:
        extra["output_config"] = oc
    extra = _deep_merge(extra, st.extra_body)
    meta = {k: v for k, v in kwargs.items() if k not in ("messages", "system")}
    meta.update(copy.deepcopy(extra))
    if "output_config" in meta and "format" in meta["output_config"]:
        meta["output_config"]["format"] = {"type": "json_schema", "schema": "<assembly schema>"}
    try:
        if st.stream:
            with client.messages.stream(**kwargs, extra_body=extra or None) as s:
                msg = s.get_final_message()
        else:
            msg = client.messages.create(**kwargs, extra_body=extra or None)
    except anthropic.APIStatusError as e:
        status = getattr(e, "status_code", 0) or 0
        if status in TRANSIENT_STATUS or status >= 500:
            raise _Transient(f"HTTP {status}: {str(e)[:300]}", _retry_after(getattr(e.response, "headers", None)),
                             status)
        raise _Fatal(f"HTTP {status}: {str(e)[:600]}", status)
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        raise _Transient(f"{type(e).__name__}: {e}")
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
    usage = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage)
    return Response(text, True, returned_model=msg.model or "", finish_reason=msg.stop_reason or "", usage=usage,
                    request_meta=meta, enforcement=enforcement, http_status=200)


def _google(model, system, msgs, st: Settings):
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise _Fatal("GOOGLE_API_KEY not set")
    gen = {"maxOutputTokens": st.max_tokens}
    if st.temperature is not None and st.send_temperature:
        gen["temperature"] = st.temperature
    if st.thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": st.thinking_budget}
    enforcement = "none"
    if st.schema is not None and st.schema_mode == "json_schema":
        gen["responseMimeType"] = "application/json"
        gen[st.google_schema_field] = st.schema
        enforcement = "strict_schema"
    elif st.schema is not None and st.schema_mode == "json_object":
        gen["responseMimeType"] = "application/json"
        enforcement = "json_mode"
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in msgs]
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": gen}
    body = _deep_merge(body, st.extra_body)
    meta = {"model": model, "generationConfig": copy.deepcopy(body["generationConfig"])}
    if st.google_schema_field in meta["generationConfig"]:
        meta["generationConfig"][st.google_schema_field] = "<assembly schema>"
    hdr = {"x-goog-api-key": key, "content-type": "application/json"}
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
    if not st.stream:
        j = _post(f"{base}:generateContent", hdr, body, st, False).json()
        return _google_response([j], meta, enforcement)
    resp = _post(f"{base}:streamGenerateContent?alt=sse", hdr, body, st, True)
    return _google_response(list(_sse(resp)), meta, enforcement)


def _google_response(chunks, meta, enforcement):
    text, finish, usage, version = [], "", {}, ""
    for j in chunks:
        if "error" in j:
            msg = json.dumps(j["error"])[:500]
            code = j["error"].get("code", 0) if isinstance(j["error"], dict) else 0
            if code in TRANSIENT_STATUS:
                raise _Transient(f"stream error: {msg}", status=code)
            raise _Fatal(f"stream error: {msg}", code)
        version = j.get("modelVersion") or version
        usage = j.get("usageMetadata") or usage
        cand = (j.get("candidates") or [{}])[0]
        for p in (cand.get("content") or {}).get("parts", []) or []:
            if not p.get("thought") and p.get("text"):
                text.append(p["text"])
        finish = cand.get("finishReason") or finish
    return Response("".join(text), True, returned_model=version, finish_reason=finish, usage=usage,
                    request_meta=meta, enforcement=enforcement, http_status=200)


# ---------------------------------------------------------------------------
def billed_tokens(model_spec: str, usage: dict) -> tuple[int, int]:
    """(input tokens, output tokens including hidden reasoning) from a provider's usage record."""
    provider = model_spec.split(":", 1)[0]
    u = usage or {}
    if provider == "anthropic":
        tin = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + \
              (u.get("cache_read_input_tokens") or 0)
        return int(tin), int(u.get("output_tokens") or 0)
    if provider == "google":
        return int(u.get("promptTokenCount") or 0), int((u.get("candidatesTokenCount") or 0) +
                                                         (u.get("thoughtsTokenCount") or 0))
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def cost_usd(model_spec: str, usage: dict, price_in: float, price_out: float) -> float:
    tin, tout = billed_tokens(model_spec, usage)
    return tin * price_in / 1e6 + tout * price_out / 1e6

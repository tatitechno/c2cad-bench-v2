"""Provider calls over plain HTTPS, with a complete record of what was sent and returned.

model_spec = "<provider>:<model id>", e.g.
  openai:gpt-5.4        anthropic:claude-opus-4-6     google:gemini-3.1-pro-preview
  deepseek:deepseek-chat moonshot:kimi-k2.5           openrouter:qwen/qwen3-coder
  together:<model>       mock:reference | mock:jitter-0.5 | mock:beam-box | mock:empty
Keys come from the environment or the repository's .env (never logged).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

REPO = Path(__file__).resolve().parents[3]

OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}


def _load_env():
    p = REPO / ".env"
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
    temperature: Optional[float] = None      # None = provider default (recorded as such)
    reasoning_effort: Optional[str] = None   # OpenAI-style: minimal|low|medium|high
    thinking_budget: Optional[int] = None    # Anthropic/Google thinking tokens
    timeout: int = 600


@dataclass
class Response:
    text: str
    ok: bool
    error: str = ""
    returned_model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    request_meta: dict = field(default_factory=dict)   # sent parameters, no secrets


def available(provider: str) -> bool:
    if provider == "mock":
        return True
    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "google":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return provider in OPENAI_COMPATIBLE and bool(os.environ.get(OPENAI_COMPATIBLE[provider][1]))


def call(model_spec: str, system: str, user: str, st: Settings) -> Response:
    provider, _, model = model_spec.partition(":")
    t0 = time.time()
    try:
        if provider in OPENAI_COMPATIBLE:
            r = _openai_like(provider, model, system, user, st)
        elif provider == "anthropic":
            r = _anthropic(model, system, user, st)
        elif provider == "google":
            r = _google(model, system, user, st)
        else:
            return Response("", False, f"unknown provider {provider!r}")
    except requests.RequestException as e:
        r = Response("", False, f"{type(e).__name__}: {e}")
    r.latency_s = time.time() - t0
    return r


def _post(url, headers, body, timeout):
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _openai_like(provider, model, system, user, st):
    base, key_env = OPENAI_COMPATIBLE[provider]
    key = os.environ.get(key_env, "")
    if not key:
        return Response("", False, f"{key_env} not set")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if provider == "openai":
        body["max_completion_tokens"] = st.max_tokens
        if st.reasoning_effort:
            body["reasoning_effort"] = st.reasoning_effort
    else:
        body["max_tokens"] = st.max_tokens
    if st.temperature is not None:
        body["temperature"] = st.temperature
    j = _post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}, body, st.timeout)
    ch = j["choices"][0]
    return Response(ch["message"].get("content") or "", True, returned_model=j.get("model", ""),
                    finish_reason=ch.get("finish_reason", ""), usage=j.get("usage", {}),
                    request_meta={k: v for k, v in body.items() if k != "messages"})


def _anthropic(model, system, user, st):
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return Response("", False, "ANTHROPIC_API_KEY not set")
    body = {"model": model, "max_tokens": st.max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}]}
    if st.temperature is not None:
        body["temperature"] = st.temperature
    if st.thinking_budget:
        body["thinking"] = {"type": "enabled", "budget_tokens": st.thinking_budget}
    j = _post("https://api.anthropic.com/v1/messages",
              {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, body, st.timeout)
    text = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
    return Response(text, True, returned_model=j.get("model", ""), finish_reason=j.get("stop_reason", ""),
                    usage=j.get("usage", {}), request_meta={k: v for k, v in body.items() if k != "messages"})


def _google(model, system, user, st):
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return Response("", False, "GOOGLE_API_KEY not set")
    gen = {"maxOutputTokens": st.max_tokens}
    if st.temperature is not None:
        gen["temperature"] = st.temperature
    if st.thinking_budget is not None:
        gen["thinkingConfig"] = {"thinkingBudget": st.thinking_budget}
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
    j = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
              {"x-goog-api-key": key, "content-type": "application/json"}, body, st.timeout)
    cand = (j.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    return Response(text, True, returned_model=j.get("modelVersion", ""), finish_reason=cand.get("finishReason", ""),
                    usage=j.get("usageMetadata", {}), request_meta={"model": model, "generationConfig": gen})

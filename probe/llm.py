"""
Multi-LLM dispatch: gemini, claude, openai/gpt, deepseek, mistral, groq.

Model strings are flexible — pass a bare provider name for the default,
or a specific model variant for fine-grained control.

Examples:
    gemini                 → Gemini API (gemini-2.5-flash default), requires GOOGLE_API_KEY
    gemini-2.0-flash       → Gemini API with model gemini-2.0-flash
    gemini-flash           → alias  → gemini-2.0-flash
    gemini-pro             → alias  → gemini-2.5-pro-preview-03-25
    gemini-1.5-pro         → Gemini API with model gemini-1.5-pro

    claude                 → Claude API (claude-sonnet-4-6 default), requires ANTHROPIC_API_KEY
    claude-opus            → claude-opus-4-6
    claude-sonnet          → claude-sonnet-4-6
    claude-haiku           → claude-haiku-4-5-20251001

    openai / gpt-4o        → OpenAI API, model gpt-4o
    gpt-4o-mini            → OpenAI API, model gpt-4o-mini
    o3-mini / o1           → OpenAI reasoning models

    deepseek               → DeepSeek API, model deepseek-chat
    deepseek-reasoner      → DeepSeek API, model deepseek-reasoner

    mistral                → Mistral AI API, model mistral-large-latest
    mistral-small          → Mistral AI API, model mistral-small-latest

    groq-llama             → Groq API, model llama-3.3-70b-versatile
    groq-mixtral           → Groq API, model mixtral-8x7b-32768
"""

import os, json, subprocess, threading
from .config import CFG


# ═══════════════════════════════════════════════════════════════
# .ENV LOADER  (so you don't need to export vars manually)
# ═══════════════════════════════════════════════════════════════

def _load_dotenv():
    """Load API keys from C2CAD-Bench/.env or C2CAD-Bench/probe/.env if present.
    Overwrites any existing empty-string values so that Cowork/shell environments
    that pre-populate keys as '' don't block the .env values from taking effect.
    """
    for dotenv_path in [CFG.PROBE_DIR / ".env", CFG.PROBE_DIR / "probe" / ".env"]:
        if dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    key = k.strip()
                    val = v.strip().strip('"\'')
                    # .env always wins — override stale shell exports
                    if val:
                        os.environ[key] = val

_load_dotenv()


# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════

GEMINI_ALIASES = {
    # ── Default ────────────────────────────────────────────────────────────────
    "gemini":                        None,                              # CLI default

    # ── Gemini 3.x (Preview — March 2026) ──────────────────────────────────────
    # gemini-3.1-pro-preview   → 1M context, SOTA reasoning, strongest overall
    # gemini-3-flash-preview   → Pro-level intelligence, faster + cheaper
    # gemini-3.1-flash-lite-preview → cost-efficient high-volume workhorse ($0.25/1M in)
    "gemini-3-pro":                  "gemini-3.1-pro-preview",          # smart alias
    "gemini-3-flash":                "gemini-3-flash-preview",          # smart alias
    "gemini-3-flash-lite":           "gemini-3.1-flash-lite-preview",   # smart alias
    "gemini-3.1-pro-preview":        "gemini-3.1-pro-preview",
    "gemini-3-flash-preview":        "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-image-preview":"gemini-3.1-flash-image-preview",  # image gen
    "gemini-3-pro-image-preview":    "gemini-3-pro-image-preview",      # highest-quality image gen

    # ── Gemini 2.5 (GA) ────────────────────────────────────────────────────────
    "gemini-pro":                    "gemini-2.5-pro",                  # latest stable pro
    "gemini-2.5-pro":                "gemini-2.5-pro",
    "gemini-2.5-flash":              "gemini-2.5-flash",
    "gemini-2.5-flash-lite":         "gemini-2.5-flash-lite",

    # ── Gemini 2.0 (GA) ────────────────────────────────────────────────────────
    "gemini-flash":                  "gemini-2.0-flash",                # most used alias
    "gemini-flash-lite":             "gemini-2.0-flash-lite",
    "gemini-2.0-flash":              "gemini-2.0-flash",
    "gemini-2.0-flash-lite":         "gemini-2.0-flash-lite",
    "gemini-exp":                    "gemini-2.0-pro-exp",
    "gemini-2.0-pro-exp":            "gemini-2.0-pro-exp",

    # ── Gemini 1.5 (Legacy) ────────────────────────────────────────────────────
    "gemini-1.5-pro":                "gemini-1.5-pro",
    "gemini-1.5-flash":              "gemini-1.5-flash",
}

CLAUDE_ALIASES = {
    "claude":              "claude-sonnet-4-6",          # API default (latest)
    # ── Claude 4.6 (latest) ───────────────────────────────────────────────────
    "claude-opus":         "claude-opus-4-6",
    "claude-opus-4":       "claude-opus-4-6",
    "claude-opus-4-6":     "claude-opus-4-6",
    "claude-sonnet":       "claude-sonnet-4-6",
    "claude-sonnet-4":     "claude-sonnet-4-6",
    "claude-sonnet-4-6":   "claude-sonnet-4-6",
    # ── Claude 4.5 (previous) ─────────────────────────────────────────────────
    "claude-opus-4-5":     "claude-opus-4-5",
    "claude-sonnet-4-5":   "claude-sonnet-4-5",
    # ── Claude Haiku ──────────────────────────────────────────────────────────
    "claude-haiku":              "claude-haiku-4-5-20251001",
    "claude-haiku-4":            "claude-haiku-4-5-20251001",
    "claude-haiku-4-5":          "claude-haiku-4-5-20251001",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
}

OPENAI_ALIASES = {
    "openai":           "gpt-4o",         # default
    "gpt-4o":           "gpt-4o",
    "gpt-4o-mini":      "gpt-4o-mini",
    "gpt-4-turbo":      "gpt-4-turbo",
    "gpt-4.5-preview":  "gpt-4.5-preview",
    # ── GPT-4.1 (April 2025) ─────────────────────────────────────
    "gpt-4.1":          "gpt-4.1",
    "gpt-4.1-mini":     "gpt-4.1-mini",
    "gpt-4.1-nano":     "gpt-4.1-nano",
    # ── GPT-5.4 (March 2026) ─────────────────────────────────────
    "gpt-5.4":          "gpt-5.4",
    "gpt-5.4-mini":     "gpt-5.4-mini",
    "gpt-5.4-nano":     "gpt-5.4-nano",
    "gpt-5.4-pro":      "gpt-5.4-pro",    # reasoning model (Responses API)
    # ── Reasoning series ──────────────────────────────────────────
    "o1":               "o1",
    "o1-mini":          "o1-mini",
    "o3-mini":          "o3-mini",
    "o3":               "o3",
    "o4-mini":          "o4-mini",
}

DEEPSEEK_ALIASES = {
    "deepseek":           "deepseek-chat",     # default
    # deepseek-chat     → DeepSeek-V3.2 (non-thinking mode, 128K context)
    # deepseek-reasoner → DeepSeek-R1   (thinking / chain-of-thought, 128K context)
    "deepseek-chat":      "deepseek-chat",
    "deepseek-reasoner":  "deepseek-reasoner",
    "deepseek-v3":        "deepseek-chat",
    "deepseek-v3.2":      "deepseek-chat",
    "deepseek-r1":        "deepseek-reasoner",
}

MISTRAL_ALIASES = {
    "mistral":         "mistral-large-latest",
    "mistral-large":   "mistral-large-latest",
    "mistral-small":   "mistral-small-latest",
    "mistral-medium":  "mistral-medium-latest",
    "codestral":       "codestral-latest",
}

GROQ_ALIASES = {
    "groq":           "llama-3.3-70b-versatile",
    "groq-llama":     "llama-3.3-70b-versatile",
    "groq-llama3":    "llama3-70b-8192",
    "groq-llama3-8b": "llama3-8b-8192",
    "groq-mixtral":   "mixtral-8x7b-32768",
    "groq-gemma":     "gemma2-9b-it",
}

KIMI_ALIASES = {
    "kimi":           "kimi-k2.5",        # default
    # Kimi K2.5 — Moonshot AI (1T MoE, 32B active, 256K context)
    # OpenAI-compatible API at https://api.moonshot.ai/v1
    "kimi-k2.5":      "kimi-k2.5",
    "kimi-k2":        "kimi-k2",
    "moonshot":       "kimi-k2.5",
}

# Unified registry for --list-models display (ordered for readability)
_GEMINI_3 = [k for k in GEMINI_ALIASES if k.startswith("gemini-3")]
_GEMINI_25 = [k for k in GEMINI_ALIASES if "2.5" in k]
_GEMINI_20 = [k for k in GEMINI_ALIASES if "2.0" in k or k in ("gemini-flash","gemini-flash-lite","gemini-exp")]
_GEMINI_15 = [k for k in GEMINI_ALIASES if "1.5" in k]

MODEL_REGISTRY = {
    "GEMINI 3  (Preview, requires GOOGLE_API_KEY)":   _GEMINI_3,
    "GEMINI 2.5 (GA, requires GOOGLE_API_KEY)":      _GEMINI_25,
    "GEMINI 2.0 (GA, requires GOOGLE_API_KEY)":      _GEMINI_20,
    "GEMINI 1.5 (Legacy, requires GOOGLE_API_KEY)":  _GEMINI_15,
    "CLAUDE   (requires ANTHROPIC_API_KEY)":         list(CLAUDE_ALIASES.keys()),
    "OPENAI   (requires OPENAI_API_KEY)":            list(OPENAI_ALIASES.keys()),
    "DEEPSEEK (requires DEEPSEEK_API_KEY)":          list(DEEPSEEK_ALIASES.keys()),
    "MISTRAL  (requires MISTRAL_API_KEY)":           list(MISTRAL_ALIASES.keys()),
    "GROQ     (requires GROQ_API_KEY)":              list(GROQ_ALIASES.keys()),
    "KIMI     (requires MOONSHOT_API_KEY)":           list(KIMI_ALIASES.keys()),
}


def list_models() -> str:
    """Return a formatted string listing all supported model strings."""
    lines = ["", "  Supported --model values:", "  " + "-" * 54]
    for provider, models in MODEL_REGISTRY.items():
        lines.append(f"\n  {provider}")
        for m in models:
            default_tag = " (default)" if m in (
                "gemini", "claude", "openai", "deepseek", "mistral", "groq") else ""
            lines.append(f"      {m}{default_tag}")
    lines += ["", "  Custom CLI (fallback): any binary that accepts -y \"<prompt>\"", ""]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# MAIN DISPATCHER
# ═══════════════════════════════════════════════════════════════

def call_llm(prompt: str, model: str = "gemini", timeout: int = 180,
             max_tokens: int = None) -> str:
    """
    Call any supported LLM. Dispatches based on model name prefix.
    See module docstring for all accepted model strings.

    max_tokens: optional requested output token limit.  Each provider branch
        applies its own hard cap — you can never exceed a model's API maximum.
        Pass None to use the default for that provider.
    """
    m = model.lower().strip()

    # Use per-thread temp files so parallel workers don't collide
    tid = threading.get_ident()
    prompt_file  = CFG.PROBE_DIR / f"_temp_prompt_{tid}.txt"
    payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
    prompt_file.write_text(prompt, encoding="utf-8")

    try:
        # ── GEMINI (REST API — generativelanguage.googleapis.com) ─
        if m.startswith("gemini"):
            api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return "[ERROR] GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable not set"
            canonical = GEMINI_ALIASES.get(m)
            # Unknown gemini-* string → pass directly as model name
            if canonical is None and m != "gemini":
                canonical = m
            # Bare "gemini" without a canonical → default to latest flash
            if canonical is None:
                canonical = "gemini-2.5-flash"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "systemInstruction": {
                    "parts": [{"text": "You are a 3D geometry JSON generator. Output ONLY a raw JSON array of shape objects. No markdown, no explanation, no commentary. Your entire response must start with '[' and end with ']'."}]
                },
                "generationConfig": {
                    "temperature": 0.0,
                    # Hard cap: Gemini 2.5 supports up to 65536 output tokens.
                    # Use requested amount but never exceed 65536.
                    "maxOutputTokens": min(max_tokens or 32768, 65536),
                    "responseMimeType": "application/json"
                }
            })
            payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
            payload_file.write_text(payload, encoding="utf-8")
            cmd = (f'curl -s "https://generativelanguage.googleapis.com/v1beta'
                   f'/models/{canonical}:generateContent?key={api_key}" '
                   f'-H "Content-Type: application/json" '
                   f'-d @"{payload_file}"')
            provider = "gemini_api"

        # ── CLAUDE (API first, CLI fallback) ─────────────────────
        elif m.startswith("claude"):
            canonical = CLAUDE_ALIASES.get(m, m)   # unknown → pass as-is
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            if api_key:
                # Direct Anthropic Messages API
                # No "thinking" param — omitting it avoids deprecated-param
                # issues on Opus 4.6 / Sonnet 4.6 and keeps output JSON-only.
                # Hard cap: Claude 4.x (Opus/Sonnet) supports 32K output tokens.
                # Use requested amount but never exceed 32768.
                _claude_tok = min(max_tokens or 32768, 32768)
                payload = json.dumps({
                    "model": canonical,
                    "max_tokens": _claude_tok,
                    "temperature": 0.0,
                    "system": "You are a JSON-only generator. Output ONLY a raw JSON array. RULES: 1) Start with '[', end with ']'. 2) No markdown fences. 3) No explanation, no commentary, no preamble, no postamble. 4) Coordinates as [x,y,z] arrays only. 5) Minimal whitespace. 6) Be maximally concise.",
                    "messages": [{"role": "user", "content": prompt}]
                })
                payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
                payload_file.write_text(payload, encoding="utf-8")
                cmd = (f'curl -s https://api.anthropic.com/v1/messages '
                       f'-H "Content-Type: application/json" '
                       f'-H "x-api-key: {api_key}" '
                       f'-H "anthropic-version: 2023-06-01" '
                       f'-d @"{payload_file}"')
                provider = "anthropic_api"
            else:
                # Fallback: Claude CLI
                model_flag = f'--model "{canonical}"' if canonical else ""
                cmd = f'claude -p {model_flag} "$(cat "{prompt_file}")"'
                provider = "claude_cli"

        # ── OPENAI / GPT / O-SERIES ───────────────────────────────
        elif m.startswith(("openai", "gpt", "o1", "o3", "o4")):
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return "[ERROR] OPENAI_API_KEY environment variable not set"
            model_id = OPENAI_ALIASES.get(m, m)   # unknown gpt-* → pass as-is

            # Responses-API-only models (gpt-5.4-pro, gpt-5.2-pro, etc.)
            # These don't support /v1/chat/completions — must use /v1/responses.
            is_responses_only = model_id.endswith("-pro") and model_id.startswith("gpt-5")

            if is_responses_only:
                _sys_json = 'You are a 3D geometry JSON generator. Output ONLY a raw JSON array of shape objects. No markdown, no explanation. Start with \'[\' and end with \']\'.'
                payload_dict = {
                    "model": model_id,
                    "instructions": _sys_json,
                    "input": "Respond with a JSON array.\n\n" + prompt,
                    "max_output_tokens": 32768,
                    "text": {"format": {"type": "json_object"}},
                    "reasoning": {"effort": "medium"},
                    "store": False
                }
                payload = json.dumps(payload_dict)
                payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
                payload_file.write_text(payload, encoding="utf-8")
                cmd = (f'curl -s https://api.openai.com/v1/responses '
                       f'-H "Content-Type: application/json" '
                       f'-H "Authorization: Bearer {api_key}" '
                       f'-d @"{payload_file}"')
                provider = "openai_responses_api"
            else:
                # json_object mode forces response to be a JSON *object* (root {}),
                # so we ask the model to wrap the array inside {"shapes": [...]}.
                # extract_json() already unwraps this automatically.
                _sys_json = 'You are a 3D geometry JSON generator. Output ONLY valid JSON. Return a JSON object with a single key "shapes" whose value is the array of shape objects. Example: {"shapes": [{...}, {...}]}. No markdown, no explanation.'

                # Determine model category for parameter compatibility:
                #   - o-series (o1, o3, o4): use max_completion_tokens, NO temperature, NO system role
                #   - GPT-4.1 / GPT-5.x: use max_completion_tokens, temperature OK
                #   - GPT-4o / legacy: use max_tokens, temperature OK
                is_o_series = model_id.startswith(("o1", "o3", "o4"))
                is_new_gpt  = model_id.startswith(("gpt-4.1", "gpt-5"))

                # GPT-4.1 / GPT-5.x support up to 32K completion tokens.
                # o-series may support more but 32K is a safe ceiling.
                _gpt_tok = min(max_tokens or 32768, 32768)

                if is_o_series:
                    payload_dict = {
                        "model": model_id,
                        "messages": [
                            {"role": "user", "content": _sys_json + "\n\n" + prompt}
                        ],
                        "max_completion_tokens": _gpt_tok,
                        "response_format": {"type": "json_object"}
                    }
                elif is_new_gpt:
                    payload_dict = {
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": _sys_json},
                            {"role": "user", "content": prompt}
                        ],
                        "max_completion_tokens": _gpt_tok,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"}
                    }
                else:
                    payload_dict = {
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": _sys_json},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": _gpt_tok,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"}
                    }

                payload = json.dumps(payload_dict)
                payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
                payload_file.write_text(payload, encoding="utf-8")
                cmd = (f'curl -s https://api.openai.com/v1/chat/completions '
                       f'-H "Content-Type: application/json" '
                       f'-H "Authorization: Bearer {api_key}" '
                       f'-d @"{payload_file}"')
                provider = "openai_api"

        # ── DEEPSEEK ──────────────────────────────────────────────
        elif m.startswith("deepseek"):
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                return "[ERROR] DEEPSEEK_API_KEY environment variable not set"
            model_id = DEEPSEEK_ALIASES.get(m, m)
            _sys_json = "You are a 3D geometry JSON generator. Output ONLY a raw JSON array of shape objects. No markdown, no explanation. Start with '[' and end with ']'."
            # deepseek-chat (V3):     hard API cap = 8 192 tokens (cannot exceed).
            # deepseek-reasoner (R1): API supports up to 65K; safe ceiling 32K.
            is_reasoner = "reasoner" in model_id
            if is_reasoner:
                ds_max = min(max_tokens or 32768, 32768)
            else:
                ds_max = 8192   # hard limit — ignore requested value
            ds_payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": _sys_json},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": ds_max,
                "temperature": 0.0
            }
            # DeepSeek-chat supports json_object; reasoner does not
            if not is_reasoner:
                ds_payload["response_format"] = {"type": "json_object"}
            payload = json.dumps(ds_payload)
            payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
            payload_file.write_text(payload, encoding="utf-8")
            cmd = (f'curl -s https://api.deepseek.com/v1/chat/completions '
                   f'-H "Content-Type: application/json" '
                   f'-H "Authorization: Bearer {api_key}" '
                   f'-d @"{payload_file}"')
            provider = "openai_api"

        # ── MISTRAL ───────────────────────────────────────────────
        elif m.startswith(("mistral", "codestral")):
            api_key = os.environ.get("MISTRAL_API_KEY", "")
            if not api_key:
                return "[ERROR] MISTRAL_API_KEY environment variable not set"
            model_id = MISTRAL_ALIASES.get(m, m)
            _sys_json = "You are a 3D geometry JSON generator. Output ONLY a raw JSON array of shape objects. No markdown, no explanation. Start with '[' and end with ']'."
            payload = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": _sys_json},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": min(max_tokens or 32768, 32768),
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            })
            payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
            payload_file.write_text(payload, encoding="utf-8")
            cmd = (f'curl -s https://api.mistral.ai/v1/chat/completions '
                   f'-H "Content-Type: application/json" '
                   f'-H "Authorization: Bearer {api_key}" '
                   f'-d @"{payload_file}"')
            provider = "openai_api"

        # ── GROQ ──────────────────────────────────────────────────
        elif m.startswith("groq"):
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                return "[ERROR] GROQ_API_KEY environment variable not set"
            model_id = GROQ_ALIASES.get(m, m)
            _sys_json = "You are a 3D geometry JSON generator. Output ONLY a raw JSON array of shape objects. No markdown, no explanation. Start with '[' and end with ']'."
            payload = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": _sys_json},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 8192,
                "temperature": 0.0
            })
            payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
            payload_file.write_text(payload, encoding="utf-8")
            cmd = (f'curl -s https://api.groq.com/openai/v1/chat/completions '
                   f'-H "Content-Type: application/json" '
                   f'-H "Authorization: Bearer {api_key}" '
                   f'-d @"{payload_file}"')
            provider = "openai_api"

        # ── KIMI (Moonshot AI — OpenAI-compatible) ─────────────────
        elif m.startswith(("kimi", "moonshot")):
            api_key = os.environ.get("MOONSHOT_API_KEY", "")
            if not api_key:
                return "[ERROR] MOONSHOT_API_KEY environment variable not set"
            model_id = KIMI_ALIASES.get(m, m)
            _sys_json = "You are a 3D geometry JSON generator. Output ONLY a raw JSON array of shape objects. No markdown, no explanation. Start with '[' and end with ']'."
            payload = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": _sys_json},
                    {"role": "user", "content": prompt}
                ],
                # Kimi K2.5 supports up to 32K output tokens.
                "max_tokens": min(max_tokens or 32768, 32768),
                "temperature": 0.0,
                "thinking": {"type": "disabled"}
            })
            payload_file = CFG.PROBE_DIR / f"_temp_payload_{tid}.json"
            payload_file.write_text(payload, encoding="utf-8")
            cmd = (f'curl -s https://api.moonshot.ai/v1/chat/completions '
                   f'-H "Content-Type: application/json" '
                   f'-H "Authorization: Bearer {api_key}" '
                   f'-d @"{payload_file}"')
            provider = "kimi_api"

        # ── CUSTOM / FALLBACK ─────────────────────────────────────
        else:
            escaped = prompt.replace('"', '\\\\"')
            cmd = f'{m} -y "{escaped}"'
            provider = "custom"

        # ── EXECUTE ───────────────────────────────────────────────
        # CLI fallback tools are streaming reasoners — no timeout.
        # API/curl calls keep the user timeout so we don't hang on network issues.
        effective_timeout = None if provider in ("claude_cli",) else timeout
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=effective_timeout,
                           cwd=str(CFG.PROBE_DIR))
        raw = (r.stdout or "") + (r.stderr or "")

        # Parse OpenAI Responses API (gpt-5.4-pro, etc.)
        # Response shape: output[] → find item with type=="message" → content[0].text
        if provider == "openai_responses_api":
            try:
                api_resp = json.loads(r.stdout)
                if api_resp.get("error"):
                    err = api_resp["error"]
                    raw = f"[ERROR] {err.get('message', str(err))}" if isinstance(err, dict) else f"[ERROR] {err}"
                else:
                    for item in api_resp.get("output", []):
                        if item.get("type") == "message":
                            for block in item.get("content", []):
                                if block.get("type") == "output_text":
                                    raw = block.get("text", raw)
                                    break
                            break
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse OpenAI-compatible JSON responses (openai/deepseek/mistral/groq)
        elif provider == "openai_api":
            try:
                api_resp = json.loads(r.stdout)
                choices = api_resp.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content")
                    if content is not None:
                        raw = content
                elif "error" in api_resp:
                    raw = f"[ERROR] {api_resp['error'].get('message', str(api_resp['error']))}"
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse Kimi (Moonshot) responses — thinking model splits into
        # reasoning_content (discard) and content (keep).
        elif provider == "kimi_api":
            try:
                api_resp = json.loads(r.stdout)
                choices = api_resp.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content")
                    if content:
                        raw = content
                elif "error" in api_resp:
                    raw = f"[ERROR] {api_resp['error'].get('message', str(api_resp['error']))}"
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse Gemini REST API response → .candidates[0].content.parts[0].text
        elif provider == "gemini_api":
            try:
                api_resp = json.loads(r.stdout)
                candidates = api_resp.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw = parts[0].get("text", raw)
                elif "error" in api_resp:
                    err = api_resp["error"]
                    raw = f"[ERROR] {err.get('message', str(err))}"
            except (json.JSONDecodeError, KeyError):
                pass

        # Parse Anthropic Messages API response → first text block
        # Skips thinking blocks (type=="thinking") which waste tokens.
        elif provider == "anthropic_api":
            try:
                api_resp = json.loads(r.stdout)
                content_blocks = api_resp.get("content", [])
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if block.get("type") == "text":
                            raw = block.get("text", raw)
                            break
                if "error" in api_resp and raw == (r.stdout or "") + (r.stderr or ""):
                    err = api_resp["error"]
                    raw = f"[ERROR] {err.get('message', str(err))}"
            except (json.JSONDecodeError, KeyError):
                pass

        return raw

    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {e}"
    finally:
        for tf in [prompt_file, payload_file]:
            if tf.exists():
                try:
                    tf.unlink()
                except OSError:
                    pass

"""
AI-driven bare-signal (裸信号) generator.

Supports multiple LLM backends via a unified interface:
  - OpenRouter  (default, many models)
  - Anthropic   (direct Claude API)
  - Ollama      (local)
  - Any OpenAI-compatible endpoint

The "bare signal" is a raw alpha expression that hasn't been through
parameter optimisation yet.  The LLM's job is to propose economically
motivated ideas; downstream tools do the validation and mining.
"""

import json
import logging
import re
from typing import Optional

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative researcher specialising in WorldQuant Brain alpha expressions.
Your task is to generate novel, economically motivated alpha factor expressions using the FASTEXPR language.

FASTEXPR operators available:
- Basic: rank, zscore, scale, log, sqrt, abs, inverse, reverse, normalize
- Time-series: ts_rank, ts_zscore, ts_mean, ts_std_dev, ts_sum, ts_delta,
               ts_returns, ts_ir, ts_arg_min, ts_arg_max, ts_corr, ts_covariance,
               ts_skewness, ts_kurtosis, ts_min_diff, ts_max_diff, ts_decay_exp_window,
               ts_percentage, ts_moment, ts_entropy, ts_regression
- Group: group_rank, group_mean, group_std_dev, group_sum, group_max, group_min, group_median
- Cross-section: signed_power, vector_neut, vector_proj, winsorize, ts_backfill

Common data fields (equity, USA unless stated):
  close, open, high, low, volume, returns, vwap, cap,
  adv20, shares, sharesout,
  sector, industry, subindustry,
  pe, pb, ps, pcf, ev, ebitda, sales, assets, equity, debt_lt,
  cashflow_op, cashflow_cap, dividends, buybacks,
  roe, roa, roic, gm, npm, eps, revenue_growth,
  rp_css_business, vec_avg(mws82_sentiment), vec_avg(nws48_ssc)

Rules:
1. Output ONLY a JSON object: {"expressions": ["expr1", "expr2", ...]}
2. Each expression must be a single, complete FASTEXPR statement (no semicolons unless multi-line assignment).
3. Do NOT use undefined operators or fields.
4. Each expression should encode a distinct economic hypothesis.
5. Short expressions (< 3 operators) are preferred to reduce overfitting.
"""

IDEA_TEMPLATES = [
    "Value: propose a value-vs-growth alpha based on fundamental ratios.",
    "Momentum: propose a price momentum or earnings momentum alpha.",
    "Quality: propose an alpha based on profitability or earnings quality.",
    "Sentiment: propose an alpha using news or analyst sentiment fields.",
    "Volatility: propose an alpha exploiting cross-sectional volatility patterns.",
    "Reversal: propose a short-term mean-reversion alpha.",
    "Seasonality: propose an alpha based on calendar effects or earnings cycles.",
    "Liquidity: propose an alpha based on trading volume or bid-ask dynamics.",
    "Group neutralised: propose any alpha that uses group_rank or group_mean for neutralisation.",
    "Event-driven: propose an alpha sensitive to corporate events (earnings, dividends, buybacks).",
]


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _BaseBackend:
    def generate(self, user_prompt: str, n: int) -> list[str]:
        raise NotImplementedError


class OpenRouterBackend(_BaseBackend):
    """
    OpenRouter exposes an OpenAI-compatible /chat/completions endpoint.
    Default model: anthropic/claude-3.5-sonnet  (change via model param).
    """

    DEFAULT_MODEL = "anthropic/claude-3-5-sonnet"

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, user_prompt: str, n: int = 5) -> list[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/DearAlpha",
            "X-Title": "DearAlpha",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 1024,
        }
        try:
            r = requests.post(self.url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return _parse_expressions(content)
        except Exception as exc:
            log.error("OpenRouter error: %s", exc)
            return []


class AnthropicBackend(_BaseBackend):
    """Direct Anthropic Messages API."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str, model: str = ""):
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.url = "https://api.anthropic.com/v1/messages"

    def generate(self, user_prompt: str, n: int = 5) -> list[str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        try:
            r = requests.post(self.url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["content"][0]["text"]
            return _parse_expressions(content)
        except Exception as exc:
            log.error("Anthropic error: %s", exc)
            return []


class OllamaBackend(_BaseBackend):
    """Local Ollama server (OpenAI-compatible /api/chat)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, user_prompt: str, n: int = 5) -> list[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        try:
            r = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=120
            )
            r.raise_for_status()
            content = r.json()["message"]["content"]
            return _parse_expressions(content)
        except Exception as exc:
            log.error("Ollama error: %s", exc)
            return []


class OpenAICompatBackend(_BaseBackend):
    """Generic OpenAI-compatible endpoint (e.g. DeepSeek, Groq, etc.)."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model

    def generate(self, user_prompt: str, n: int = 5) -> list[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 1024,
        }
        try:
            r = requests.post(self.url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return _parse_expressions(content)
        except Exception as exc:
            log.error("OpenAI-compat error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(cfg: dict) -> _BaseBackend:
    """
    Build a backend from a config dict, e.g.:
        {"provider": "openrouter", "api_key": "...", "model": "..."}
        {"provider": "anthropic",  "api_key": "..."}
        {"provider": "ollama",     "base_url": "http://localhost:11434", "model": "deepseek-r1:8b"}
        {"provider": "openai_compat", "api_key": "...", "base_url": "...", "model": "..."}
    """
    provider = cfg.get("provider", "openrouter").lower()
    if provider == "openrouter":
        return OpenRouterBackend(cfg["api_key"], cfg.get("model", ""))
    if provider == "anthropic":
        return AnthropicBackend(cfg["api_key"], cfg.get("model", ""))
    if provider == "ollama":
        return OllamaBackend(cfg.get("base_url", "http://localhost:11434"), cfg.get("model", "deepseek-r1:8b"))
    if provider == "openai_compat":
        return OpenAICompatBackend(cfg["api_key"], cfg["base_url"], cfg["model"])
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class BareSignalGenerator:
    """
    Generates 裸信号 (bare alpha signals) using an LLM backend.

    Each call to `generate_batch` picks an economic theme, asks the LLM
    to propose expressions, and returns the raw strings.  No simulation
    or validation is done here.
    """

    def __init__(self, backend: _BaseBackend):
        self.backend = backend
        self._theme_index = 0

    def _next_theme(self) -> str:
        theme = IDEA_TEMPLATES[self._theme_index % len(IDEA_TEMPLATES)]
        self._theme_index += 1
        return theme

    def generate_batch(
        self,
        n: int = 5,
        theme: Optional[str] = None,
        extra_context: str = "",
    ) -> list[str]:
        """
        Ask the LLM to generate `n` alpha expressions.

        Args:
            n:             Number of expressions to request.
            theme:         Economic theme hint (or None to cycle through defaults).
            extra_context: Free-form additional context appended to the prompt.

        Returns:
            List of raw FASTEXPR strings.
        """
        chosen_theme = theme or self._next_theme()
        prompt = (
            f"Theme: {chosen_theme}\n"
            f"Generate exactly {n} distinct alpha expressions for this theme.\n"
        )
        if extra_context:
            prompt += f"\nAdditional context: {extra_context}\n"
        prompt += "\nReturn JSON only."

        log.info("Generating %d signals | theme: %s", n, chosen_theme)
        expressions = self.backend.generate(prompt, n)
        log.info("LLM returned %d expressions", len(expressions))
        return expressions


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_expressions(text: str) -> list[str]:
    """
    Extract alpha expressions from LLM output.
    Tries strict JSON parse first, then falls back to regex extraction.
    """
    # Try to find a JSON block
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            exprs = obj.get("expressions", [])
            if isinstance(exprs, list):
                return [str(e).strip() for e in exprs if e]
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]{10,})"', text)
    if quoted:
        return [q.strip() for q in quoted]

    # Last resort: non-empty lines that look like expressions
    lines = [l.strip().strip('"').strip("'") for l in text.splitlines()]
    return [l for l in lines if len(l) > 8 and "(" in l]

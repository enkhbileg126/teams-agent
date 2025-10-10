# src/telemetry.py
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import google.generativeai as genai

# ---- Pricing comes from env so you can change any time (USD per 1M tokens) ----
PROMPT_PRICE = float(os.getenv("GEMINI_PROMPT_USD_PER_MTOKEN", "0.00"))
OUTPUT_PRICE = float(os.getenv("GEMINI_OUTPUT_USD_PER_MTOKEN", "0.00"))
EMBED_PRICE = float(os.getenv("GEMINI_EMBED_USD_PER_MTOKEN", "0.00"))

# If you don’t want to count via API (e.g., offline), set USE_COUNT_API=0 to skip
USE_COUNT_API = int(os.getenv("USE_GEMINI_COUNT_API", "1"))  # 1 = on, 0 = off


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    embed_tokens_index: int = 0
    embed_tokens_query: int = 0

    @property
    def prompt_cost(self) -> float:
        return (self.prompt_tokens / 1_000_000.0) * PROMPT_PRICE

    @property
    def output_cost(self) -> float:
        return (self.output_tokens / 1_000_000.0) * OUTPUT_PRICE

    @property
    def embed_cost(self) -> float:
        total = self.embed_tokens_index + self.embed_tokens_query
        return (total / 1_000_000.0) * EMBED_PRICE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "embed_tokens_index": self.embed_tokens_index,
            "embed_tokens_query": self.embed_tokens_query,
            "prompt_cost_usd": round(self.prompt_cost, 6),
            "output_cost_usd": round(self.output_cost, 6),
            "embed_cost_usd": round(self.embed_cost, 6),
            "total_cost_usd": round(
                self.prompt_cost + self.output_cost + self.embed_cost, 6
            ),
        }


class GeminiCounter:
    """
    Uses google.generativeai's model.count_tokens to get accurate counts.
    Falls back to a crude heuristic (≈4 chars/token) if disabled.
    """

    def __init__(self, llm_model_name: str):
        self._model = genai.GenerativeModel(llm_model_name)

    @staticmethod
    def _heuristic_count(text: str) -> int:
        if not text:
            return 0
        # ≈ 4 chars per token, conservative floor at 1
        return max(1, int(len(text) / 4))

    def count(self, text: str) -> int:
        if not text:
            return 0
        if USE_COUNT_API:
            try:
                return int(self._model.count_tokens(text).total_tokens)
            except Exception:
                return self._heuristic_count(text)
        return self._heuristic_count(text)

    def count_many(self, texts: Iterable[str]) -> int:
        total = 0
        for t in texts:
            total += self.count(t)
        return total


def maybe_print_tokens(token_usage: TokenUsage, enable_print: bool):
    """
    One-line, comment-out-able print. Toggle with a flag.
    """
    if enable_print:
        print("[TOKENS]", json.dumps(token_usage.as_dict(), ensure_ascii=False))

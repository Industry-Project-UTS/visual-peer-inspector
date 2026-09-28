"""Optional, text-only explanation of an already computed inspection result."""

from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

from .core import Session

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
MAX_RESPONSE_BYTES = 16_384


class ExplanationUnavailable(Exception):
    """The optional explanation could not be produced."""


def ranking_evidence(session: Session) -> dict:
    """Send only anonymous aggregate scores, never photos or case details."""
    result = session.result
    if result is None:
        raise ExplanationUnavailable("Run an inspection before requesting an explanation.")
    return {
        "view_count": len(session.views),
        "mode": result.mode,
        "review_state": result.review_state,
        "unique_selection": result.selected_identity_id is not None,
        "candidate_count": len(result.ranking),
        "leading_candidates": [
            {
                "position": index,
                "relative_score": round(candidate.relative_score, 4),
                "views_with_support": len(candidate.view_ids),
            }
            for index, candidate in enumerate(result.ranking[:5], 1)
        ],
    }


def explain_result(session: Session, *, api_key: str | None = None) -> str:
    """Explain sealed scores without modifying the result or sending images."""
    evidence = ranking_evidence(session)
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ExplanationUnavailable("Optional explanation is unavailable until Groq is configured.")
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Explain the supplied multi-view object ranking in two or three short, plain "
                    "sentences. Use only the supplied numbers and review state. Relative scores "
                    "are not probabilities. Do not invent visual defects, object appearance, "
                    "physical findings, or a release decision. If there is no unique selection, "
                    "say that the ranking does not identify one object. Tell the user to check "
                    "the highlighted images and object matches. Do not use Markdown or links."
                ),
            },
            {"role": "user", "content": json.dumps(evidence, sort_keys=True)},
        ],
        "max_completion_tokens": 240,
        "temperature": 0.2,
    }
    request = Request(
        GROQ_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=8) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Oversized explanation response")
        content = json.loads(raw)["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty explanation response")
        return content.strip()[:1200]
    except Exception as error:  # noqa: BLE001 - isolate the optional external service
        raise ExplanationUnavailable(
            "The explanation could not be generated. The inspection result is still available."
        ) from error

"""
tasks.py — 8 diagnostic scenarios with graders and post-fix simulations.
Covers: networking, memory management, storage, and training correctness.
Grading: multi-signal keyword + structural analysis. Deterministic, no LLM-as-judge.
"""

import string


def _clean(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def _has_any(text: str, keywords: list[str]) -> bool:
    cleaned = _clean(text)
    return any(kw.lower() in cleaned for kw in keywords)


def _floor_score(score: float, combined_text: str) -> float:
    if score == 0.0 and len(combined_text.strip()) > 10:
        return 0.05
    return round(min(score, 1.0), 2)


def _consistency_bonus(diag: str, fix: str, keywords: list[str]) -> float:
    """Bonus if diagnosis and fix mention the same core concept."""
    d, f = _clean(diag), _clean(fix)
    matches = sum(1 for kw in keywords if kw.lower() in d and kw.lower() in f)
    return min(0.05 * matches, 0.10)



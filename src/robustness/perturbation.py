from __future__ import annotations

import random
from typing import Callable


def perturb_case(text: str) -> str:
    return text.upper()


def perturb_negation(text: str) -> str:
    words = text.split()
    if not words:
        return text
    insert_at = min(2, len(words))
    words.insert(insert_at, "not")
    return " ".join(words)


def perturb_typo(text: str) -> str:
    chars = list(text)
    alpha_positions = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if len(alpha_positions) < 2:
        return text
    pos = random.choice(alpha_positions[:-1])
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def available_perturbations() -> dict[str, Callable[[str], str]]:
    return {
        "case": perturb_case,
        "negation": perturb_negation,
        "typo": perturb_typo,
    }

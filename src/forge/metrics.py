from __future__ import annotations

from collections import Counter
import re
import string


def normalize_answer(text: str) -> str:
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, gold = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    return 2 * overlap / (len(pred) + len(gold))


def score_answer(prediction: str, references: tuple[str, ...] | list[str]) -> tuple[float, float]:
    if not references:
        raise ValueError("Scoring requires at least one reference answer")
    return (max(token_f1(prediction, ref) for ref in references),
            max(exact_match(prediction, ref) for ref in references))

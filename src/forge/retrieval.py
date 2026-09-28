from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import string
from typing import Sequence

import numpy as np

from .schemas import Document


STOPWORDS = frozenset("a an the and or but if than so as of at by for with to from in on into over under up down out off is are was were be been being it its this that these those i you he she we they me him her us them my your his our their not no".split())


def tokenize(text: str) -> list[str]:
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    return [word for word in words if word not in STOPWORDS]


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float


class BM25Retriever:
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        if not documents:
            raise ValueError("BM25 requires at least one document")
        if k1 <= 0 or not 0 <= b <= 1 or epsilon < 0:
            raise ValueError("Invalid BM25 parameters")
        self.documents = tuple(documents)
        self.k1, self.b = k1, b
        self.frequencies = [Counter(tokenize(doc.text)) for doc in self.documents]
        self.lengths = np.array([sum(freq.values()) for freq in self.frequencies], dtype=np.float64)
        self.average_length = float(self.lengths.mean())
        document_frequencies: Counter[str] = Counter()
        for frequency in self.frequencies:
            document_frequencies.update(frequency.keys())
        size = len(documents)
        self.idf = {word: math.log(size - count + 0.5) - math.log(count + 0.5)
                    for word, count in document_frequencies.items()}
        floor = epsilon * (sum(self.idf.values()) / len(self.idf)) if self.idf else 0.0
        self.idf = {word: floor if value < 0 else value for word, value in self.idf.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=np.float64)
        if not self.average_length:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.average_length)
        for word in tokenize(query):
            frequencies = np.array([freq.get(word, 0) for freq in self.frequencies], dtype=np.float64)
            scores += self.idf.get(word, 0.0) * frequencies * (self.k1 + 1) / (frequencies + norm)
        return scores

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedDocument]:
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.scores(query)
        indices = np.argsort(-scores, kind="stable")[:k]
        return [RetrievedDocument(self.documents[int(index)], float(scores[index])) for index in indices]


def raw_support(retrieved: Sequence[RetrievedDocument], words_per_passage: int = 200) -> str:
    if words_per_passage <= 0:
        raise ValueError("Passage budget must be positive")
    return "\n\n".join(" ".join(item.document.text.split()[:words_per_passage]) for item in retrieved[:3])


def summary_support(query: str, retrieved: Sequence[RetrievedDocument], max_words: int = 140) -> str:
    if max_words <= 0:
        raise ValueError("Summary budget must be positive")
    sentences = [sentence.strip() for item in retrieved[:3]
                 for sentence in re.split(r"(?<=[.!?])\s+|\n+", item.document.text) if sentence.strip()]
    if not sentences:
        return ""
    ranked = BM25Retriever([Document(str(i), sentence) for i, sentence in enumerate(sentences)]).retrieve(query, len(sentences))
    selected, remaining = [], max_words
    for item in ranked:
        words = item.document.text.split()
        if len(words) <= remaining:
            selected.append(item.document.text)
            remaining -= len(words)
    return " ".join(selected)

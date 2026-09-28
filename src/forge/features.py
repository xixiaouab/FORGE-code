from __future__ import annotations

from collections import Counter
import re
from typing import Protocol, Sequence

import numpy as np

from .metrics import normalize_answer
from .retrieval import RetrievedDocument, tokenize
from .schemas import Example, Generation


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class BGEEmbedder:
    def __init__(self, model: str = "BAAI/bge-base-en-v1.5", device: str = "cpu", revision: str | None = None):
        self.model_name, self.device, self.revision = model, device, revision
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError("Install the embedding extra to use BGE sentence embeddings") from error
            kwargs = {"device": self.device}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            self._model = SentenceTransformer(self.model_name, **kwargs)
        embeddings = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)


def canonical_variant(variant: str) -> str:
    key = variant.lower().replace("_", "").replace("-", "")
    aliases = {"full": "full", "forgefull": "full", "lite": "lite", "forgelite": "lite",
               "bge": "bge", "bge773": "bge", "bge+bm25": "bge", "773": "bge"}
    if key not in aliases:
        raise ValueError(f"Unknown feature variant: {variant}")
    return aliases[key]


FEATURE_DIMENSIONS = {"bge": 773, "lite": 778, "full": 789}


def feature_names(variant: str) -> list[str]:
    variant = canonical_variant(variant)
    names = [f"bge_{index:03d}" for index in range(768)]
    if variant != "bge":
        names.extend(["query_length", "wh_flag", "entity_count", "comparison_flag", "temporal_flag"])
    names.extend(["bm25_top1", "bm25_top5_mean", "bm25_gap", "bm25_top5_std"])
    if variant == "full":
        names.extend(["probe_answer_length", "probe_output_tokens", "probe_idk", "probe_hedging",
                      "probe_query_overlap", "probe_numeric", "probe_confidence", "sc_agreement",
                      "sc_unique_ratio", "sc_average_length", "sc_greedy_match"])
    return names + ["query_passage_cosine"]


def structure_features(question: str) -> list[float]:
    words = question.split()
    wh = bool(re.search(r"\b(who|what|when|where|why|which|whose|whom|how)\b", question, re.I))
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b", question)
    entities = [entity for entity in entities if entity.lower() not in {"who", "what", "when", "where", "why", "which", "whose", "whom", "how", "is", "was", "did", "does", "the"}]
    comparison = bool(re.search(r"\b(more|less|fewer|larger|smaller|higher|lower|older|younger|first|last|earlier|later|same|different|compare|versus|vs)\b|\b(?:greater|better|worse)\s+than\b", question, re.I))
    temporal = bool(re.search(r"\b(when|year|years|month|months|date|before|after|during|century|born|died)\b|\b(?:1[0-9]{3}|20[0-9]{2})\b", question, re.I))
    return [float(len(words)), float(wh), float(len(entities)), float(comparison), float(temporal)]


def probe_features(question: str, probe: Generation) -> list[float]:
    answer = probe.text
    idk = bool(re.search(r"\b(i (?:do not|don't) know|unknown|cannot (?:answer|determine)|not enough information|insufficient (?:information|evidence))\b", answer, re.I))
    hedging = bool(re.search(r"\b(maybe|perhaps|probably|possibly|likely|uncertain|unsure|appears|seems|might|may be)\b", answer, re.I))
    question_tokens, answer_tokens = set(tokenize(question)), set(tokenize(answer))
    overlap = len(question_tokens & answer_tokens) / max(1, len(answer_tokens))
    confidence = 0.0 if idk else (0.5 if hedging else 1.0)
    return [float(len(answer.split())), float(probe.output_tokens), float(idk), float(hedging),
            overlap, float(bool(re.search(r"\d", answer))), confidence]


def consistency_features(probe: Generation, samples: Sequence[Generation]) -> list[float]:
    if len(samples) != 3:
        raise ValueError("Full FORGE requires exactly three self-consistency samples")
    normalized = [normalize_answer(sample.text) for sample in samples]
    counts = Counter(normalized)
    greedy = normalize_answer(probe.text)
    return [max(counts.values()) / 3, len(counts) / 3,
            sum(len(sample.text.split()) for sample in samples) / 3,
            sum(answer == greedy for answer in normalized) / 3]


def full_from_lite(features: np.ndarray, question: str, probe: Generation,
                   samples: Sequence[Generation]) -> np.ndarray:
    if features.shape != (778,):
        raise ValueError("Full feature construction requires 778 Lite dimensions")
    return np.concatenate([features[:-1], probe_features(question, probe),
                           consistency_features(probe, samples), features[-1:]]).astype(np.float32)


class FeatureExtractor:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder if embedder is not None else BGEEmbedder()

    def extract(self, example: Example, retrieved: Sequence[RetrievedDocument], variant: str = "lite",
                probe: Generation | None = None, samples: Sequence[Generation] = ()) -> np.ndarray:
        variant = canonical_variant(variant)
        if not retrieved:
            raise ValueError("Feature extraction requires retrieved passages")
        embedding = np.asarray(self.embedder.encode([example.question, retrieved[0].document.text]), dtype=np.float64)
        if embedding.shape != (2, 768) or not np.isfinite(embedding).all():
            raise ValueError("BGE encoder must return finite embeddings with shape (2, 768)")
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding vectors must be nonzero")
        embedding /= norms
        cosine = float(np.dot(embedding[0], embedding[1]))
        scores = np.asarray([item.score for item in retrieved[:5]], dtype=np.float64)
        retrieval = [scores[0], scores.mean(), scores[0] - scores[1] if len(scores) > 1 else scores[0], scores.std()]
        blocks = [embedding[0]]
        if variant != "bge":
            blocks.append(structure_features(example.question))
        blocks.append(retrieval)
        if variant == "full":
            if probe is None:
                raise ValueError("Full FORGE requires one greedy host probe")
            blocks.extend([probe_features(example.question, probe), consistency_features(probe, samples)])
        blocks.append([cosine])
        features = np.concatenate(blocks).astype(np.float32)
        if features.shape != (FEATURE_DIMENSIONS[variant],) or not np.isfinite(features).all():
            raise ValueError("Invalid feature vector")
        return features

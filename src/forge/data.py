from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Iterable

from .schemas import Document, Example


def _records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() in (".tsv", ".csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t" if path.suffix.lower() == ".tsv" else ","))
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        records = value.get("data", value) if isinstance(value, dict) else value
    if not isinstance(records, list) or any(not isinstance(x, dict) for x in records):
        raise ValueError(f"{path}: expected a JSON array, JSONL records, or a data array")
    return records


def _answers(value) -> tuple[str, ...]:
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        value = value.get("text", value.get("answers", []))
    return tuple(str(answer) for answer in (value or []))


def _document(record: dict, index: int = 0) -> Document:
    return Document(str(record.get("id", record.get("idx", index))),
                    str(record.get("text", record.get("paragraph_text", ""))),
                    str(record.get("title", "")))


def load_documents(path: str | Path) -> list[Document]:
    documents = [_document(row, i) for i, row in enumerate(_records(path))]
    if any(not doc.text.strip() for doc in documents):
        raise ValueError("Corpus documents must have nonempty text")
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("Corpus document IDs must be unique")
    return documents


def import_example(record: dict, benchmark: str | None = None) -> Example:
    dataset = (benchmark or record.get("benchmark", "")).lower().replace("-", "").replace("_", "")
    identifier = record.get("id", record.get("_id"))
    if identifier is None:
        raise ValueError("Every example requires its original id or _id")
    question = record.get("question", record.get("claim", ""))
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Example {identifier} has no question/claim")
    documents: tuple[Document, ...] = ()
    if "documents" in record:
        documents = tuple(_document(doc, i) for i, doc in enumerate(record["documents"]))
    elif dataset in ("hotpotqa", "2wikimultihopqa", "2wiki"):
        context = record.get("context", [])
        if isinstance(context, dict):
            context = zip(context["title"], context["sentences"])
        documents = tuple(Document(str(i), " ".join(sentences) if isinstance(sentences, list) else str(sentences), str(title))
                          for i, (title, sentences) in enumerate(context))
    elif dataset == "musique":
        documents = tuple(_document(doc, i) for i, doc in enumerate(record.get("paragraphs", [])))
    if dataset == "fever":
        answers = _answers(record.get("answers", record.get("label")))
    elif dataset == "popqa":
        answers = _answers(record.get("answers", record.get("possible_answers", record.get("answer"))))
    else:
        answers = _answers(record.get("answers", record.get("answer")))
        if dataset == "musique":
            answers = tuple(dict.fromkeys((*answers, *_answers(record.get("answer_aliases", [])))))
    if not answers:
        raise ValueError(f"Example {identifier} has no reference answers; use labeled benchmark splits")
    metadata = dict(record.get("metadata", {}))
    if dataset == "fever" and "evidence" in record:
        metadata["evidence"] = record["evidence"]
    if dataset == "musique" and "answerable" in record:
        metadata["answerable"] = bool(record["answerable"])
    return Example(str(identifier), question, answers, benchmark or record.get("benchmark", dataset), documents, metadata)


def load_examples(path: str | Path, benchmark: str | None = None) -> list[Example]:
    examples = [import_example(row, benchmark) for row in _records(path)]
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("Example IDs must be unique within an input file")
    return examples


def write_examples(path: str | Path, examples: Iterable[Example]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

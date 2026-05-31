from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MathVQASample:
    """One visual-math QA example."""

    id: str
    image: Image.Image
    question: str
    options: list[str]
    answer: str
    subject: str
    source: str = "unknown"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}") from exc
    return rows


def sanitize_question(text: str) -> str:
    """Remove image/control tokens that must not appear in raw questions."""
    for token in ("<image>", "<image_start>", "<image_end>"):
        text = text.replace(token, "")
    return " ".join(text.split())


class MathVQADataset(Dataset[MathVQASample]):
    """Dataset for manifest-based visual mathematical QA."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.split = split
        self.max_samples = max_samples

        rows = load_jsonl(self.manifest_path)
        rows = [r for r in rows if r.get("split") == split]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> MathVQASample:
        row = self.rows[idx]
        image_path = self.root / row["image"]
        image = Image.open(image_path).convert("RGB")
        question = sanitize_question(row["question"])
        return MathVQASample(
            id=row["id"],
            image=image,
            question=question,
            options=row["options"],
            answer=row["answer"],
            subject=row.get("subject", "unknown"),
            source=row.get("source", "unknown"),
        )

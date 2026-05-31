from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    # Try patterns from most to least specific
    patterns = [
        r"(?:ответ|answer)[:\s]*\(?([A-E])\)?",       # "Ответ: A" or "Answer: B"
        r"correct answer is\s+\(?([A-E])\)?",           # "The correct answer is D."
        r"правильный ответ[:\s]*\(?([A-E])\)?",
        r"^\s*\(?([A-E])\)?\s*[.\s]*$",                # Standalone "A" or "(B)"
        r"\b([A-E])\b",                                  # Any standalone letter
    ]
    text_stripped = text.strip()
    for pattern in patterns:
        m = re.search(pattern, text_stripped, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in choices:
                return letter
    return None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    import torch
    from torch.utils.data import DataLoader

    from hw.dataset import MathVQADataset
    from hw.model import MathVLM, ModelConfig
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    model_cfg_dict = config.get("model", {})
    proc_cfg_dict = config.get("processor", {})
    eval_cfg = config.get("eval", {})

    device = torch.device(config.get("device", "cpu"))

    manifest = data_cfg.get("eval_manifest", data_cfg.get("train_manifest", "assets/toy_math_vqa/manifest.jsonl"))
    split = data_cfg.get("eval_split", "dev")
    max_samples = 10 if toy else data_cfg.get("max_samples")

    dataset = MathVQADataset(manifest_path=manifest, split=split, max_samples=max_samples)

    # Build processor
    proc_config = ProcessorConfig(
        image_size=proc_cfg_dict.get("image_size", 224),
        num_tiles=proc_cfg_dict.get("num_tiles", 1),
        num_image_tokens=proc_cfg_dict.get("num_image_tokens", 49),
        max_length=proc_cfg_dict.get("max_length", 512),
    )

    # Try to load real models; fall back to mocks
    try:
        from transformers import AutoModel, AutoTokenizer
        vision_name = model_cfg_dict.get("vision_encoder", "")
        lm_name = model_cfg_dict.get("language_model", "")
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        vision_encoder = AutoModel.from_pretrained(vision_name).to(device)
        language_model = AutoModel.from_pretrained(lm_name).to(device)
        vision_hidden = vision_encoder.config.hidden_size
        text_hidden = language_model.config.hidden_size
    except Exception:
        from hw.train import _TinyLM, _TinyVisionEncoder, _make_mock_tokenizer
        tokenizer = _make_mock_tokenizer()
        vision_hidden = 32
        text_hidden = 64
        vision_encoder = _TinyVisionEncoder(vision_hidden)
        language_model = _TinyLM(len(tokenizer.vocab), text_hidden)

    from hw.constants import IMAGE_TOKEN
    image_token_id = tokenizer.vocab.get(IMAGE_TOKEN, 2) if hasattr(tokenizer, "vocab") else 2

    model_config = ModelConfig(
        vision_hidden_size=vision_hidden,
        text_hidden_size=text_hidden,
        num_image_tokens=proc_config.num_image_tokens,
        image_token_id=image_token_id,
    )

    model = MathVLM(vision_encoder, language_model, model_config)

    # Load adapter if checkpoint provided
    checkpoint = model_cfg_dict.get("adapter_checkpoint")
    if checkpoint and Path(checkpoint).exists():
        model.adapter.load_state_dict(torch.load(checkpoint, map_location=device))

    model = model.to(device).eval()

    processor = MathVLMProcessor(tokenizer, proc_config)

    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # Build inference item (no answer)
        from hw.dataset import MathVQASample
        inf_sample = MathVQASample(
            id=sample.id,
            image=sample.image,
            question=sample.question,
            options=sample.options,
            answer="",
            subject=sample.subject,
            source=sample.source,
        )
        item = processor(inf_sample)
        batch = processor.collate([item])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            out_ids = model.generate(batch, max_new_tokens=16)

        if hasattr(tokenizer, "decode"):
            text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        else:
            text = str(out_ids[0].tolist())

        prediction = parse_mc_answer(text)
        rows.append({
            "id": sample.id,
            "question": sample.question,
            "answer": sample.answer,
            "prediction": prediction,
            "raw_output": text,
            "subject": sample.subject,
        })

    metrics = compute_accuracy(rows)

    output_path = config.get("output_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

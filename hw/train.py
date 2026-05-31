from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    out = model(batch)
    if isinstance(out, dict):
        loss = out["loss"]
    else:
        loss = out.loss

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return float(loss.item())


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point."""
    from torch.utils.data import DataLoader

    from hw.dataset import MathVQADataset
    from hw.model import MathVLM, ModelConfig
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    model_cfg_dict = config.get("model", {})
    proc_cfg_dict = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})

    device = torch.device(trainer_cfg.get("device", "cpu"))
    dtype_str = trainer_cfg.get("dtype", "float32")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_str, torch.float32)

    # Dataset
    dataset = MathVQADataset(
        manifest_path=data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )

    # Processor config
    proc_config = ProcessorConfig(
        image_size=proc_cfg_dict.get("image_size", 224),
        num_tiles=proc_cfg_dict.get("num_tiles", 1),
        tile_overlap=proc_cfg_dict.get("tile_overlap", 0.0),
        num_image_tokens=proc_cfg_dict.get("num_image_tokens", 49),
        max_length=proc_cfg_dict.get("max_length", 512),
        ignore_index=proc_cfg_dict.get("ignore_index", -100),
    )

    # Try to load real models if available, else use tiny mocks
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
        # Fallback: tiny mock models for CPU track / tests
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
    if model_cfg_dict.get("freeze_vision", True):
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
    if model_cfg_dict.get("freeze_llm", True):
        for p in model.language_model.parameters():
            p.requires_grad = False

    model = model.to(device)

    processor = MathVLMProcessor(tokenizer, proc_config)
    dataloader = DataLoader(
        dataset,
        batch_size=trainer_cfg.get("local_batch_size", 1),
        collate_fn=processor.collate,
        num_workers=trainer_cfg.get("num_workers", 0),
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=trainer_cfg.get("learning_rate", 5e-4),
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
    )

    max_steps = trainer_cfg.get("max_steps", None)
    if fast_train:
        max_steps = min(max_steps or 1, 1)

    step = 0
    for epoch in range(trainer_cfg.get("num_train_epochs", 1)):
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = train_one_step(model, batch, optimizer)
            step += 1
            print(f"step={step} loss={loss:.4f}")
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    save_path = trainer_cfg.get("save_checkpoint_path")
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.adapter.state_dict(), save_path)
        print(f"Adapter saved to {save_path}")


# ---- Tiny mock models for CPU track ----

class _TinyVisionEncoder(torch.nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.proj = torch.nn.Linear(3 * 7 * 7, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.pool(x).view(B, -1)
        return self.proj(x).unsqueeze(1)  # [B, 1, hidden]


class _TinyLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.rnn = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None, labels=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        out, _ = self.rnn(inputs_embeds)
        logits = self.head(out)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            from hw.constants import IGNORE_INDEX
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=16, **kw):
        out, _ = self.rnn(inputs_embeds)
        logits = self.head(out[:, -1:])
        return logits.argmax(dim=-1)


def _make_mock_tokenizer():
    class _Tok:
        pad_token_id = 0
        eos_token_id = 1
        vocab = {"<pad>": 0, "<eos>": 1, "<image>": 2}

        def encode(self, text, add_special_tokens=False):
            ids = []
            for token in text.replace("\n", " ").split():
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                ids.append(self.vocab[token])
            if add_special_tokens:
                ids.append(self.eos_token_id)
            return ids

        def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    return _Tok()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample."""

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor [num_tiles, 3, image_size, image_size]."""
        image = image.convert("RGB")
        image = image.resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
        # Convert to float tensor [3, H, W]
        import numpy as np
        arr = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
        # Normalize with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        arr = (arr - mean) / std
        # [num_tiles, 3, H, W] - only 1 tile for now
        return arr.unsqueeze(0).expand(self.config.num_tiles, -1, -1, -1).contiguous()

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build text prompt with visual special tokens and options."""
        image_placeholder = IMAGE_TOKEN * self.config.num_image_tokens
        options_text = "\n".join(sample.options)
        prompt = (
            f"{IMAGE_START_TOKEN}{image_placeholder}{IMAGE_END_TOKEN}\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            f"Ответ:"
        )
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample."""
        full_prompt = self.build_prompt(sample, include_answer=True)
        prompt_only = self.build_prompt(sample, include_answer=False)

        tok = self.tokenizer
        full_enc = tok(full_prompt, add_special_tokens=True, truncation=True, max_length=self.config.max_length)
        prompt_enc = tok(prompt_only, add_special_tokens=False, truncation=True, max_length=self.config.max_length)

        full_ids = full_enc["input_ids"]
        prompt_len = len(prompt_enc["input_ids"])

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.tensor(full_enc["attention_mask"], dtype=torch.long)
        labels = torch.full_like(input_ids, self.config.ignore_index)
        # Only supervise answer tokens (everything after prompt)
        labels[prompt_len:] = input_ids[prompt_len:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values."""
        pad_id = self.tokenizer.pad_token_id
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len

            input_ids_list.append(
                F.pad(item["input_ids"], (0, pad_len), value=pad_id)
            )
            attention_mask_list.append(
                F.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            labels_list.append(
                F.pad(item["labels"], (0, pad_len), value=self.config.ignore_index)
            )
            pixel_values_list.append(item["pixel_values"])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "pixel_values": torch.stack(pixel_values_list),
        }

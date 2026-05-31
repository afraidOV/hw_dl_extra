from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        self.projection = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )
        # Learnable query tokens to pool vision features to num_image_tokens
        self.query = nn.Parameter(torch.randn(num_image_tokens, vision_hidden_size) * 0.02)

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        B = vision_hidden_states.shape[0]
        # Attention-pool: queries attend to vision features
        # vision_hidden_states: [B, S, D_v]
        # query: [K, D_v]
        queries = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, K, D_v]
        # Dot-product attention weights
        attn = torch.bmm(queries, vision_hidden_states.transpose(1, 2))  # [B, K, S]
        attn = attn / (self.vision_hidden_size ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        pooled = torch.bmm(attn, vision_hidden_states)  # [B, K, D_v]
        return self.projection(pooled)  # [B, K, D_text]


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings."""
    output = input_embeds.clone()
    B = input_ids.shape[0]
    for b in range(B):
        mask = input_ids[b] == image_token_id  # [L]
        positions = mask.nonzero(as_tuple=False).squeeze(-1)  # [K]
        K = positions.shape[0]
        assert K == visual_embeds.shape[1], (
            f"Mismatch: {K} image tokens in input but {visual_embeds.shape[1]} visual embeddings"
        )
        for k, pos in enumerate(positions):
            output[b, pos] = visual_embeds[b, k]
    return output


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model."""

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _get_visual_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images and project to text space."""
        B, T, C, H, W = pixel_values.shape
        # Flatten tiles into batch dimension
        flat = pixel_values.view(B * T, C, H, W)
        vision_out = self.vision_encoder(flat)
        # Handle various output formats
        if hasattr(vision_out, "last_hidden_state"):
            hidden = vision_out.last_hidden_state  # [B*T, S, D]
        elif isinstance(vision_out, torch.Tensor):
            hidden = vision_out
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(1)
        else:
            hidden = vision_out[0]
        # Average over tiles
        hidden = hidden.view(B, T, hidden.shape[1], hidden.shape[2]).mean(dim=1)
        return self.adapter(hidden)  # [B, K, D_text]

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        visual_embeds = self._get_visual_embeds(pixel_values)

        # Get text embeddings from LM
        embed_fn = (
            self.language_model.get_input_embeddings()
            if hasattr(self.language_model, "get_input_embeddings")
            else self.language_model.model.embed_tokens
        )
        input_embeds = embed_fn(input_ids)

        # Merge visual tokens
        input_embeds = merge_visual_embeddings(
            input_embeds, input_ids, visual_embeds, self.config.image_token_id
        )

        return self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        visual_embeds = self._get_visual_embeds(pixel_values)

        embed_fn = (
            self.language_model.get_input_embeddings()
            if hasattr(self.language_model, "get_input_embeddings")
            else self.language_model.model.embed_tokens
        )
        input_embeds = embed_fn(input_ids)
        input_embeds = merge_visual_embeddings(
            input_embeds, input_ids, visual_embeds, self.config.image_token_id
        )

        return self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

"""BEV decision architecture over Qwen3, with full-backbone fine-tuning.

Architecture adapted from https://github.com/avbiswas/bev-train/network.py.
LoRA/frozen-backbone behavior is intentionally not used.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Model

from .text import TASK_TYPES

class BEVSelfAttention(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.einsum("bqd,bkd->bqk", [q, k]) / math.sqrt(self.new_dim)
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("bqk,bkd->bqd", [weights, v])
        return self.o(out)


class BEVAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = BEVSelfAttention(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, new_dim),
            nn.GELU(),
            nn.Linear(new_dim, hidden_dim),
        )

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class BEVChoiceHead(nn.Module):
    """BEV: self-attention over [task token, choices..., answer slot], then bilinear scoring."""

    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            BEVAttentionBlock(hidden_dim, new_dim) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        B = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)

        x = torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1)
        x = self.input_norm(x)

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        padding_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)

        answer = self.answer_proj(x[:, -1])
        choices = self.choice_proj(x[:, 1:-1])
        logits = torch.einsum("bd,bcd->bc", [answer, choices]) / math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))


class BEVQwenDecisionNetwork(nn.Module):
    def __init__(self, model_name, torch_dtype=None, head_dim=512, head_layers=2):
        super().__init__()
        # Difference from bev-train optimization only: full pretrained Qwen, no LoRA/freeze.
        self.backbone = Qwen3Model.from_pretrained(model_name, dtype=torch_dtype)
        self.head = BEVChoiceHead(
            hidden_dim=self.backbone.config.hidden_size,
            new_dim=head_dim,
            num_layers=head_layers,
            num_task_types=len(TASK_TYPES),
        )

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        choice_read_idx,
        choice_mask,
        answer_end_idx,
        task_type,
        **kwargs,
    ):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
        )

        # Exact BEV behavior: cast backbone states to fp32 before ChoiceHead.
        hidden = output.last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]
        answer_embedding = hidden[batch_idx, answer_end_idx]

        # BEV runs the head in fp32 even under bf16/fp16 backbone autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = self.head(choice_embeddings, answer_embedding, choice_mask, task_type)

        probabilities = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "choice_embeddings": choice_embeddings,
            "answer_embedding": answer_embedding,
            "context_states": hidden,
        }

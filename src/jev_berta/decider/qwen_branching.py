"""Shared-trunk / branched-label Qwen3 encoder.

This module extends Hugging Face Transformers v4.57.x Qwen3 without copying or
modifying the generated upstream ``modeling_qwen3.py`` file in site-packages.
It uses the exact pretrained Qwen3 blocks and parameters, but adds a new
``forward_branches`` path for variable-candidate classification.

The computation is:

    shared context/query tokens  -> computed ONCE at every Qwen layer
                                      |\
                                      | +-> candidate branch 0
                                      | +-> candidate branch 1
                                      | ...
                                      +----> candidate branch N

At layer l, each candidate branch attends to:
  * the shared context K/V from layer l (computed once), and
  * only the causal prefix of its own candidate tokens.

Candidate branches never attend to one another inside Qwen.  Candidate-to-
candidate interaction can be added after this encoder with a set transformer.

Important implementation note
-----------------------------
The branching path below intentionally uses explicit eager PyTorch attention so
that one physical context K/V tensor can be shared by all branches.  It does
not materialize N copies of the context sequence and it does not use the LM
head or autoregressive generation.  Qwen3-0.6B / Qwen3-0.6B-Base use full
attention (sliding_window=None), which is the path supported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    apply_rotary_pos_emb,
    repeat_kv,
)


@dataclass
class BranchedQwen3Output:
    """Outputs of :meth:`Qwen3BranchedModel.forward_branches`.

    Shapes:
        context_last_hidden_state: ``[B, P, D]``
        branch_last_hidden_state: ``[M, L, D]`` where M is the total number
            of real candidate branches in the minibatch.
    """

    context_last_hidden_state: torch.Tensor
    branch_last_hidden_state: torch.Tensor


def _project_qkv(attn, hidden_states: torch.Tensor, position_embeddings):
    """Project Q/K/V exactly as Qwen3Attention.forward does."""

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, value_states


def _context_causal_attention(
    attn,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Standard causal self-attention for the shared context, computed once.

    Args:
        query_states: [B, H, P, Dh]
        key_states:   [B, Hkv, P, Dh]
        value_states: [B, Hkv, P, Dh]
        attention_mask: [B, P], 1 for real context tokens.

    Returns:
        Attention output before o_proj, shape [B, P, H * Dh].
    """

    key_states = repeat_kv(key_states, attn.num_key_value_groups)
    value_states = repeat_kv(value_states, attn.num_key_value_groups)

    _, _, seq_len, _ = query_states.shape
    scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * attn.scaling

    causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device).tril()
    key_valid = attention_mask.to(torch.bool)[:, None, None, :]
    allowed = causal[None, None, :, :] & key_valid

    mask_value = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~allowed, mask_value)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    probs = F.dropout(probs, p=attn.attention_dropout, training=attn.training)

    output = torch.matmul(probs, value_states)
    # Padding queries are irrelevant, but zeroing keeps their states clean.
    output = output * attention_mask[:, None, :, None].to(output.dtype)
    output = output.transpose(1, 2).contiguous()
    return output.reshape(output.shape[0], output.shape[1], -1)


def _branched_attention(
    attn,
    branch_query: torch.Tensor,
    branch_key: torch.Tensor,
    branch_value: torch.Tensor,
    context_key: torch.Tensor,
    context_value: torch.Tensor,
    context_attention_mask: torch.Tensor,
    branch_attention_mask: torch.Tensor,
    branch_to_example: torch.LongTensor,
) -> torch.Tensor:
    """Candidate attention with one physically shared context K/V per example.

    Every candidate branch attends to all *valid* context tokens and causally to
    its own branch tokens.  It never attends to another candidate branch.

    The minibatch is grouped by source example only for the attention matmul;
    all candidate branches of one example are processed in parallel.  The
    shared context K/V is broadcast in the matmul and is not recomputed.

    Shapes:
        branch_query: [M, H, L, Dh]
        branch_key/value: [M, Hkv, L, Dh]
        context_key/value: [B, Hkv, P, Dh]
        branch_attention_mask: [M, L]
        branch_to_example: [M]
    """

    branch_key = repeat_kv(branch_key, attn.num_key_value_groups)
    branch_value = repeat_kv(branch_value, attn.num_key_value_groups)
    context_key = repeat_kv(context_key, attn.num_key_value_groups)
    context_value = repeat_kv(context_value, attn.num_key_value_groups)

    m, num_heads, branch_len, head_dim = branch_query.shape
    output = torch.zeros(
        (m, num_heads, branch_len, head_dim),
        dtype=branch_value.dtype,
        device=branch_query.device,
    )
    causal = torch.ones((branch_len, branch_len), dtype=torch.bool, device=branch_query.device).tril()

    # Looping over B (normally small) avoids materializing B*N copies of the
    # context K/V. All branches belonging to an example stay parallel.
    batch_size = context_key.shape[0]
    for b in range(batch_size):
        branch_indices = torch.nonzero(branch_to_example == b, as_tuple=False).flatten()
        if branch_indices.numel() == 0:
            continue

        q = branch_query.index_select(0, branch_indices)       # [Nb,H,L,Dh]
        kb = branch_key.index_select(0, branch_indices)
        vb = branch_value.index_select(0, branch_indices)

        # Context K/V remain [H,P,Dh], physically one copy. torch.einsum
        # broadcasts them over Nb candidate branches.
        kc = context_key[b]                                   # [H,P,Dh]
        vc = context_value[b]                                 # [H,P,Dh]

        context_scores = torch.einsum("nhld,hpd->nhlp", q, kc) * attn.scaling
        branch_scores = torch.matmul(q, kb.transpose(-2, -1)) * attn.scaling

        context_valid = context_attention_mask[b].to(torch.bool)[None, None, None, :]
        context_scores = context_scores.masked_fill(
            ~context_valid,
            torch.finfo(context_scores.dtype).min,
        )

        branch_key_valid = branch_attention_mask.index_select(0, branch_indices).to(torch.bool)
        branch_allowed = causal[None, None, :, :] & branch_key_valid[:, None, None, :]
        branch_scores = branch_scores.masked_fill(
            ~branch_allowed,
            torch.finfo(branch_scores.dtype).min,
        )

        # One softmax across [shared-context keys | own-branch keys] is exactly
        # what a normal causal forward over [context, candidate] would do.
        scores = torch.cat([context_scores, branch_scores], dim=-1)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probs = F.dropout(probs, p=attn.attention_dropout, training=attn.training)

        p = context_scores.shape[-1]
        probs_context = probs[..., :p]
        probs_branch = probs[..., p:]

        out_context = torch.einsum("nhlp,hpd->nhld", probs_context, vc)
        out_branch = torch.matmul(probs_branch, vb)
        group_output = out_context + out_branch

        query_valid = branch_key_valid[:, None, :, None]
        group_output = group_output * query_valid.to(group_output.dtype)
        output.index_copy_(0, branch_indices, group_output)

    output = output.transpose(1, 2).contiguous()
    return output.reshape(m, branch_len, -1)


class Qwen3BranchedModel(Qwen3Model):
    """Qwen3Model with a shared-context / variable-label branch forward path.

    ``from_pretrained`` works because parameter names are unchanged from
    Qwen3Model.  The normal ``forward`` method is inherited untouched.
    """

    def _validate_branching_config(self):
        sliding = [i for i, layer in enumerate(self.layers) if layer.self_attn.sliding_window is not None]
        if sliding:
            raise NotImplementedError(
                "The explicit branching path currently supports full-attention Qwen3 models only. "
                f"Sliding-attention layers found at {sliding[:8]}. Qwen3-0.6B and "
                "Qwen3-0.6B-Base have sliding_window=None and are supported."
            )

    @staticmethod
    def _position_ids_from_right_padded_mask(mask: torch.Tensor) -> torch.LongTensor:
        # Real tokens receive 0..length-1. Padding positions get 0 and are masked.
        position_ids = mask.long().cumsum(-1) - 1
        return position_ids.clamp_min_(0)

    def forward_branches(
        self,
        context_input_ids: torch.LongTensor,
        context_attention_mask: torch.Tensor,
        branch_input_ids: torch.LongTensor,
        branch_attention_mask: torch.Tensor,
        branch_to_example: torch.LongTensor,
    ) -> BranchedQwen3Output:
        """Encode the shared context once and all candidate suffixes as branches.

        Args:
            context_input_ids:
                Right-padded shared context/query token IDs, [B, P].
            context_attention_mask:
                1 on real shared tokens, [B, P].
            branch_input_ids:
                Right-padded candidate suffixes. Each row should normally end
                in the common candidate terminal token (EOS is fine), [M, L].
            branch_attention_mask:
                1 on real candidate tokens including the terminal token, [M,L].
            branch_to_example:
                For every branch row, index of its source context in B, [M].

        Returns:
            ``BranchedQwen3Output`` containing token-level final context and
            branch states. Use ``gather_last_branch_state`` below to obtain one
            [M,D] representation per candidate.
        """

        self._validate_branching_config()

        if context_input_ids.ndim != 2 or branch_input_ids.ndim != 2:
            raise ValueError("context_input_ids and branch_input_ids must both be rank-2")
        if context_attention_mask.shape != context_input_ids.shape:
            raise ValueError("context_attention_mask must match context_input_ids")
        if branch_attention_mask.shape != branch_input_ids.shape:
            raise ValueError("branch_attention_mask must match branch_input_ids")
        if branch_to_example.ndim != 1 or branch_to_example.shape[0] != branch_input_ids.shape[0]:
            raise ValueError("branch_to_example must have shape [num_branches]")
        if branch_to_example.numel() and (
            branch_to_example.min().item() < 0
            or branch_to_example.max().item() >= context_input_ids.shape[0]
        ):
            raise ValueError("branch_to_example contains an out-of-range context index")

        context_attention_mask = context_attention_mask.to(torch.bool)
        branch_attention_mask = branch_attention_mask.to(torch.bool)

        context_hidden = self.embed_tokens(context_input_ids)
        branch_hidden = self.embed_tokens(branch_input_ids)

        context_position_ids = self._position_ids_from_right_padded_mask(context_attention_mask)
        context_lengths = context_attention_mask.long().sum(-1)

        branch_offsets = context_lengths.index_select(0, branch_to_example)
        branch_steps = torch.arange(branch_input_ids.shape[1], device=branch_input_ids.device)
        raw_branch_position_ids = branch_offsets[:, None] + branch_steps[None, :]
        # Padding positions are not real sequence positions. Give them position 0
        # so a short branch paired with a long context is not penalized merely
        # because another branch in the minibatch has a larger padded width.
        branch_position_ids = torch.where(
            branch_attention_mask,
            raw_branch_position_ids,
            torch.zeros_like(raw_branch_position_ids),
        )

        branch_lengths = branch_attention_mask.long().sum(-1)
        max_real_position = (branch_offsets + branch_lengths - 1).max() if branch_lengths.numel() else torch.tensor(0)
        if int(max_real_position.item()) >= self.config.max_position_embeddings:
            raise ValueError(
                "context + branch length exceeds Qwen3 max_position_embeddings: "
                f"max real position={int(max_real_position.item())}, "
                f"limit={self.config.max_position_embeddings}"
            )

        # RoPE depends on positions, not layer hidden values, so compute once.
        context_position_embeddings = self.rotary_emb(context_hidden, context_position_ids)
        branch_position_embeddings = self.rotary_emb(branch_hidden, branch_position_ids)

        def run_layer(layer, context_hidden, branch_hidden):
            # ----- Attention input states -----
            context_residual = context_hidden
            branch_residual = branch_hidden

            context_normed = layer.input_layernorm(context_hidden)
            branch_normed = layer.input_layernorm(branch_hidden)

            cq, ck, cv = _project_qkv(layer.self_attn, context_normed, context_position_embeddings)
            bq, bk, bv = _project_qkv(layer.self_attn, branch_normed, branch_position_embeddings)

            # Shared trunk attention: context is evaluated exactly once.
            context_attn = _context_causal_attention(
                layer.self_attn,
                cq,
                ck,
                cv,
                context_attention_mask,
            )

            # Candidate branches: same context K/V at this exact layer.
            branch_attn = _branched_attention(
                layer.self_attn,
                bq,
                bk,
                bv,
                ck,
                cv,
                context_attention_mask,
                branch_attention_mask,
                branch_to_example,
            )

            context_hidden = context_residual + layer.self_attn.o_proj(context_attn)
            branch_hidden = branch_residual + layer.self_attn.o_proj(branch_attn)

            # ----- MLP -----
            context_residual = context_hidden
            branch_residual = branch_hidden
            context_hidden = context_residual + layer.mlp(layer.post_attention_layernorm(context_hidden))
            branch_hidden = branch_residual + layer.mlp(layer.post_attention_layernorm(branch_hidden))

            # Keep padding states inert. Real causal tokens never depend on them.
            context_hidden = context_hidden * context_attention_mask.unsqueeze(-1).to(context_hidden.dtype)
            branch_hidden = branch_hidden * branch_attention_mask.unsqueeze(-1).to(branch_hidden.dtype)
            return context_hidden, branch_hidden

        use_checkpointing = bool(getattr(self, "gradient_checkpointing", False) and self.training)
        for layer in self.layers[: self.config.num_hidden_layers]:
            if use_checkpointing:
                # IMPORTANT: capture the current layer as a default argument.
                # Checkpoint recomputes this function during backward; closing
                # over the loop variable directly would make every recompute
                # use the final Qwen layer (Python late binding), producing
                # incorrect gradients and often NaNs after the first update.
                context_hidden, branch_hidden = checkpoint(
                    lambda ch, bh, layer=layer: run_layer(layer, ch, bh),
                    context_hidden,
                    branch_hidden,
                    use_reentrant=False,
                )
            else:
                context_hidden, branch_hidden = run_layer(layer, context_hidden, branch_hidden)

        context_hidden = self.norm(context_hidden)
        branch_hidden = self.norm(branch_hidden)
        context_hidden = context_hidden * context_attention_mask.unsqueeze(-1).to(context_hidden.dtype)
        branch_hidden = branch_hidden * branch_attention_mask.unsqueeze(-1).to(branch_hidden.dtype)

        return BranchedQwen3Output(
            context_last_hidden_state=context_hidden,
            branch_last_hidden_state=branch_hidden,
        )


def gather_last_branch_state(
    branch_hidden: torch.Tensor,
    branch_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather the final real token state from every candidate branch.

    With an EOS / candidate-end token appended to every branch, this returns
    the contextual representation of the complete candidate.

    Args:
        branch_hidden: [M, L, D]
        branch_attention_mask: [M, L]
    Returns:
        candidate_states: [M, D]
    """

    lengths = branch_attention_mask.long().sum(-1)
    if torch.any(lengths <= 0):
        raise ValueError("Every candidate branch must contain at least one real token")
    end_positions = lengths - 1
    rows = torch.arange(branch_hidden.shape[0], device=branch_hidden.device)
    return branch_hidden[rows, end_positions]


__all__ = [
    "BranchedQwen3Output",
    "Qwen3BranchedModel",
    "gather_last_branch_state",
]

"""Hierarchical shared-trunk Qwen decision model with a fast inference-only branch kernel.

Training behavior is intentionally unchanged: while ``model.training`` is True, or while
PyTorch autograd is enabled, this module calls the original eager shared-prefix branching
implementation from :mod:`jev_berta.decider.qwen_branching`.

Inference behavior is different.  Under ``model.eval()`` together with ``torch.no_grad()``
or ``torch.inference_mode()``, CUDA fp16/bf16 runs use a custom Triton kernel for the
candidate-branch attention.  The kernel computes, for every candidate branch ``m``,

    softmax(Q_m [K_context[b]; K_branch[m]]^T) [V_context[b]; V_branch[m]]

with one online softmax, while:

* physically reusing the context K/V for all candidates of an example;
* never materializing the ``[context_scores | branch_scores]`` matrix;
* never calling ``repeat_kv`` for GQA (query heads map directly to KV heads);
* eliminating the Python loop over source examples in branch attention;
* preserving causal attention inside each candidate branch;
* preserving exact candidate isolation inside Qwen.

Only the forward kernel is implemented.  This is deliberate: training still uses the
well-tested eager implementation and therefore needs no custom backward kernel.

If Triton is unavailable, the model is on CPU, the dtype is unsupported, or gradients are
enabled, the code automatically falls back to the original eager branch implementation.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import qwen_branching as qb
from .qwen_branching import BranchedQwen3Output, Qwen3BranchedModel, gather_last_branch_state

try:  # Optional: training and CPU inference still work without Triton.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the runtime/GPU installation.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Inference-only fused shared-prefix branch attention
# --------------------------------------------------------------------------- #

if _TRITON_AVAILABLE:

    @triton.jit
    def _shared_prefix_branch_attn_fwd_kernel(
        q_ptr,
        context_k_ptr,
        context_v_ptr,
        branch_k_ptr,
        branch_v_ptr,
        branch_to_example_ptr,
        context_lengths_ptr,
        branch_lengths_ptr,
        out_ptr,
        # q [M, Hq, L, Dh]
        q_s0,
        q_s1,
        q_s2,
        q_s3,
        # context k/v [B, Hkv, P, Dh]
        ck_s0,
        ck_s1,
        ck_s2,
        ck_s3,
        cv_s0,
        cv_s1,
        cv_s2,
        cv_s3,
        # branch k/v [M, Hkv, L, Dh]
        bk_s0,
        bk_s1,
        bk_s2,
        bk_s3,
        bv_s0,
        bv_s1,
        bv_s2,
        bv_s3,
        # out [M, L, Hq, Dh]
        o_s0,
        o_s1,
        o_s2,
        o_s3,
        sm_scale,
        L,
        HEAD_DIM: tl.constexpr,
        KV_GROUPS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        """Forward-only FlashAttention-style kernel for shared-prefix branches.

        One program handles one (branch, query-head, query-tile).  Context K/V are addressed
        through branch_to_example and are therefore physically shared rather than expanded N
        times.  The same online-softmax accumulator is updated first with context tiles and
        then with causally-valid private branch tiles.
        """

        branch_id = tl.program_id(0)
        q_head = tl.program_id(1)
        q_block = tl.program_id(2)

        source_example = tl.load(branch_to_example_ptr + branch_id).to(tl.int32)
        context_len = tl.load(context_lengths_ptr + source_example).to(tl.int32)
        branch_len = tl.load(branch_lengths_ptr + branch_id).to(tl.int32)
        kv_head = q_head // KV_GROUPS

        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        q_valid = offs_m < branch_len

        q_ptrs = (
            q_ptr
            + branch_id * q_s0
            + q_head * q_s1
            + offs_m[:, None] * q_s2
            + offs_d[None, :] * q_s3
        )
        q = tl.load(q_ptrs, mask=q_valid[:, None] & d_mask[None, :], other=0.0)

        # Online softmax state.  Accumulation stays fp32; final output is cast by tl.store.
        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        # ---- Shared context K/V -------------------------------------------------
        for start_n in tl.range(0, context_len, BLOCK_N):
            n = start_n + offs_n
            key_valid = n < context_len

            k_ptrs = (
                context_k_ptr
                + source_example * ck_s0
                + kv_head * ck_s1
                + n[:, None] * ck_s2
                + offs_d[None, :] * ck_s3
            )
            v_ptrs = (
                context_v_ptr
                + source_example * cv_s0
                + kv_head * cv_s1
                + n[:, None] * cv_s2
                + offs_d[None, :] * cv_s3
            )
            k = tl.load(k_ptrs, mask=key_valid[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=key_valid[:, None] & d_mask[None, :], other=0.0)

            scores = tl.dot(q, tl.trans(k)) * sm_scale
            scores = tl.where(key_valid[None, :], scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            alpha = tl.exp(row_max - new_max)
            p = tl.exp(scores - new_max[:, None])
            block_sum = tl.sum(p, axis=1)

            acc *= alpha[:, None]
            if IS_BF16:
                p_dot = p.to(tl.bfloat16)
            else:
                p_dot = p.to(tl.float16)
            acc += tl.dot(p_dot, v)
            row_sum = row_sum * alpha + block_sum
            row_max = new_max

        # ---- Private causal branch K/V -----------------------------------------
        for start_n in tl.range(0, branch_len, BLOCK_N):
            n = start_n + offs_n
            key_exists = n < branch_len
            causal = n[None, :] <= offs_m[:, None]
            allowed = key_exists[None, :] & causal

            k_ptrs = (
                branch_k_ptr
                + branch_id * bk_s0
                + kv_head * bk_s1
                + n[:, None] * bk_s2
                + offs_d[None, :] * bk_s3
            )
            v_ptrs = (
                branch_v_ptr
                + branch_id * bv_s0
                + kv_head * bv_s1
                + n[:, None] * bv_s2
                + offs_d[None, :] * bv_s3
            )
            k = tl.load(k_ptrs, mask=key_exists[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=key_exists[:, None] & d_mask[None, :], other=0.0)

            scores = tl.dot(q, tl.trans(k)) * sm_scale
            scores = tl.where(allowed, scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            alpha = tl.exp(row_max - new_max)
            p = tl.exp(scores - new_max[:, None])
            block_sum = tl.sum(p, axis=1)

            acc *= alpha[:, None]
            if IS_BF16:
                p_dot = p.to(tl.bfloat16)
            else:
                p_dot = p.to(tl.float16)
            acc += tl.dot(p_dot, v)
            row_sum = row_sum * alpha + block_sum
            row_max = new_max

        denom = tl.maximum(row_sum, 1.0e-20)
        out = acc / denom[:, None]
        # Padded query rows must be exactly zero, matching the eager path (torch.empty would leak garbage/NaN).
        out = tl.where(q_valid[:, None], out, 0.0)

        out_ptrs = (
            out_ptr
            + branch_id * o_s0
            + offs_m[:, None] * o_s1
            + q_head * o_s2
            + offs_d[None, :] * o_s3
        )
        tl.store(out_ptrs, out, mask=(offs_m < L)[:, None] & d_mask[None, :])


def _can_use_triton_branch_kernel(
    branch_query: torch.Tensor,
    branch_key: torch.Tensor,
    context_key: torch.Tensor,
) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if not branch_query.is_cuda:
        return False
    if branch_query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if branch_key.dtype != branch_query.dtype or context_key.dtype != branch_query.dtype:
        return False
    if branch_query.shape[-1] > 256:
        return False
    if branch_query.shape[1] % branch_key.shape[1] != 0:
        return False
    return True


def _triton_shared_prefix_branch_attention(
    attn,
    branch_query: torch.Tensor,
    branch_key: torch.Tensor,
    branch_value: torch.Tensor,
    context_key: torch.Tensor,
    context_value: torch.Tensor,
    context_lengths: torch.Tensor,
    branch_lengths: torch.Tensor,
    branch_to_example: torch.LongTensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the forward-only Triton shared-prefix attention kernel.

    ``out`` may be a contiguous ``[M, L, Hq, Dh]`` buffer to write into.

    Returns the same logical shape as ``qwen_branching._branched_attention``:
    ``[num_branches, branch_len, hidden_size]``.
    """

    if not _can_use_triton_branch_kernel(branch_query, branch_key, context_key):
        raise RuntimeError("Triton branch kernel is unavailable for this device/dtype/shape")

    M, Hq, L, Dh = branch_query.shape
    _, Hkv, P, Dh_k = context_key.shape
    if Dh != Dh_k or branch_key.shape[-1] != Dh:
        raise ValueError("Q/K head dimensions do not match")
    if Hq % Hkv != 0:
        raise ValueError(f"Q heads ({Hq}) must be divisible by KV heads ({Hkv})")

    # [M, L, Hq, Dh] is chosen so the result can be viewed as [M, L, hidden]
    # without a transpose/copy before Qwen's o_proj.
    if out is None:
        out = torch.empty((M, L, Hq, Dh), device=branch_query.device, dtype=branch_query.dtype)

    block_m = 16 if L <= 64 else 32
    block_n = 64
    block_d = triton.next_power_of_2(Dh)
    kv_groups = Hq // Hkv
    grid = (M, Hq, triton.cdiv(L, block_m))

    _shared_prefix_branch_attn_fwd_kernel[grid](
        branch_query,
        context_key,
        context_value,
        branch_key,
        branch_value,
        branch_to_example,
        context_lengths,
        branch_lengths,
        out,
        *branch_query.stride(),
        *context_key.stride(),
        *context_value.stride(),
        *branch_key.stride(),
        *branch_value.stride(),
        *out.stride(),
        float(attn.scaling),
        L,
        HEAD_DIM=Dh,
        KV_GROUPS=kv_groups,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        IS_BF16=branch_query.dtype == torch.bfloat16,
        num_warps=4 if Dh <= 128 else 8,
        num_stages=2,
    )
    return out.reshape(M, L, Hq * Dh)


class InferenceOptimizedQwen3BranchedModel(Qwen3BranchedModel):
    """Original branched Qwen for training; Triton branch attention for inference.

    The parameter names are unchanged from ``Qwen3Model`` / ``Qwen3BranchedModel`` so
    existing hierarchical checkpoints load without conversion.
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_triton_inference_kernel = True
        self._last_inference_backend = "eager"
        self._warned_kernel_fallback = False

    @property
    def inference_backend(self) -> str:
        return self._last_inference_backend

    def set_inference_kernel(self, enabled: bool = True) -> None:
        self.use_triton_inference_kernel = bool(enabled)

    def _should_use_fast_inference(self, branch_input_ids: torch.Tensor) -> bool:
        if not self.use_triton_inference_kernel:
            return False
        if self.training or torch.is_grad_enabled():
            return False
        if not branch_input_ids.is_cuda:
            return False
        if not _TRITON_AVAILABLE:
            return False
        if os.environ.get("JEV_DISABLE_BRANCH_TRITON", "0") in {"1", "true", "TRUE", "yes"}:
            return False
        return True

    def forward_branches(
        self,
        context_input_ids: torch.LongTensor,
        context_attention_mask: torch.Tensor,
        branch_input_ids: torch.LongTensor,
        branch_attention_mask: torch.Tensor,
        branch_to_example: torch.LongTensor,
    ) -> BranchedQwen3Output:
        if not self._should_use_fast_inference(branch_input_ids):
            self._last_inference_backend = "eager"
            return super().forward_branches(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                branch_input_ids=branch_input_ids,
                branch_attention_mask=branch_attention_mask,
                branch_to_example=branch_to_example,
            )

        try:
            out = self._forward_branches_triton(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                branch_input_ids=branch_input_ids,
                branch_attention_mask=branch_attention_mask,
                branch_to_example=branch_to_example,
            )
            self._last_inference_backend = "triton_shared_prefix"
            return out
        except Exception as exc:
            # Correctness first: a machine without a compatible Triton backend still gets the
            # original implementation rather than a hard failure.  Set JEV_REQUIRE_BRANCH_TRITON=1
            # in benchmarking/production if silent fallback is undesirable.
            if os.environ.get("JEV_REQUIRE_BRANCH_TRITON", "0") in {"1", "true", "TRUE", "yes"}:
                raise
            if not self._warned_kernel_fallback:
                warnings.warn(
                    f"Triton branch attention unavailable; falling back to eager inference: {exc}",
                    RuntimeWarning,
                )
                self._warned_kernel_fallback = True
            # Avoid paying exception/JIT-failure overhead on every subsequent batch.
            self.use_triton_inference_kernel = False
            self._last_inference_backend = "eager_fallback"
            return super().forward_branches(
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                branch_input_ids=branch_input_ids,
                branch_attention_mask=branch_attention_mask,
                branch_to_example=branch_to_example,
            )

    def _forward_branches_triton(
        self,
        context_input_ids: torch.LongTensor,
        context_attention_mask: torch.Tensor,
        branch_input_ids: torch.LongTensor,
        branch_attention_mask: torch.Tensor,
        branch_to_example: torch.LongTensor,
    ) -> BranchedQwen3Output:
        """Inference-only copy of the shared-trunk forward with a fused branch kernel."""

        self._validate_branching_config()

        if context_input_ids.ndim != 2 or branch_input_ids.ndim != 2:
            raise ValueError("context_input_ids and branch_input_ids must both be rank-2")
        if context_attention_mask.shape != context_input_ids.shape:
            raise ValueError("context_attention_mask must match context_input_ids")
        if branch_attention_mask.shape != branch_input_ids.shape:
            raise ValueError("branch_attention_mask must match branch_input_ids")
        if branch_to_example.ndim != 1 or branch_to_example.shape[0] != branch_input_ids.shape[0]:
            raise ValueError("branch_to_example must have shape [num_branches]")

        context_attention_mask = context_attention_mask.to(torch.bool)
        branch_attention_mask = branch_attention_mask.to(torch.bool)

        context_hidden = self.embed_tokens(context_input_ids)
        branch_hidden = self.embed_tokens(branch_input_ids)
        B, P, D = context_hidden.shape
        M, L, _ = branch_hidden.shape
        n_ctx = B * P
        total = n_ctx + M * L

        context_position_ids = self._position_ids_from_right_padded_mask(context_attention_mask)
        context_lengths = context_attention_mask.long().sum(-1).to(torch.int32)

        branch_offsets = context_lengths.to(torch.long).index_select(0, branch_to_example)
        branch_steps = torch.arange(branch_input_ids.shape[1], device=branch_input_ids.device)
        raw_branch_position_ids = branch_offsets[:, None] + branch_steps[None, :]
        branch_position_ids = torch.where(
            branch_attention_mask,
            raw_branch_position_ids,
            torch.zeros_like(raw_branch_position_ids),
        )
        branch_lengths = branch_attention_mask.long().sum(-1).to(torch.int32)

        max_real_position = (
            branch_offsets + branch_lengths.to(torch.long) - 1
        ).max() if branch_lengths.numel() else torch.tensor(0, device=branch_input_ids.device)
        if int(max_real_position.item()) >= self.config.max_position_embeddings:
            raise ValueError(
                "context + branch length exceeds Qwen3 max_position_embeddings: "
                f"max real position={int(max_real_position.item())}, "
                f"limit={self.config.max_position_embeddings}"
            )

        # RoPE depends only on positions; compute once for all layers, same as the training path.
        context_position_embeddings = self.rotary_emb(context_hidden, context_position_ids)
        branch_position_embeddings = self.rotary_emb(branch_hidden, branch_position_ids)

        # Small batches are launch-bound, so context and candidates share one packed [1, T, D] stream
        # for every per-token op (norms, projections, MLP); only attention treats them differently.
        cos = torch.cat([context_position_embeddings[0].reshape(1, n_ctx, -1),
                         branch_position_embeddings[0].reshape(1, M * L, -1)], dim=1)
        sin = torch.cat([context_position_embeddings[1].reshape(1, n_ctx, -1),
                         branch_position_embeddings[1].reshape(1, M * L, -1)], dim=1)
        hidden = torch.cat([context_hidden.reshape(1, n_ctx, D), branch_hidden.reshape(1, M * L, D)], dim=1)
        token_mask = torch.cat(
            [context_attention_mask.reshape(-1), branch_attention_mask.reshape(-1)]
        ).view(1, total, 1).to(hidden.dtype)

        def split_heads(t):
            heads, head_dim = t.shape[1], t.shape[-1]
            ctx = t[0, :, :n_ctx].reshape(heads, B, P, head_dim).transpose(0, 1)
            br = t[0, :, n_ctx:].reshape(heads, M, L, head_dim).transpose(0, 1)
            return ctx, br

        for layer in self.layers[: self.config.num_hidden_layers]:
            attn = layer.self_attn
            residual = hidden
            normed = layer.input_layernorm(hidden)

            hidden_shape = (1, total, -1, attn.head_dim)
            q = attn.q_norm(attn.q_proj(normed).view(hidden_shape)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normed).view(hidden_shape)).transpose(1, 2)
            v = attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
            q, k = qb.apply_rotary_pos_emb(q, k, cos, sin)

            cq, bq = split_heads(q)
            ck, bk = split_heads(k)
            cv, bv = split_heads(v)

            num_heads, head_dim = q.shape[1], q.shape[-1]
            attn_buf = torch.empty((total, num_heads * head_dim), device=hidden.device, dtype=hidden.dtype)

            # Right-padded real queries never see padded keys under a plain causal mask; padded
            # rows are zeroed by token_mask below.
            context_attn = F.scaled_dot_product_attention(
                cq, ck, cv, is_causal=True, scale=attn.scaling, enable_gqa=True
            )
            attn_buf[:n_ctx].view(B, P, num_heads, head_dim).copy_(context_attn.transpose(1, 2))

            _triton_shared_prefix_branch_attention(
                attn,
                bq,
                bk,
                bv,
                ck,
                cv,
                context_lengths,
                branch_lengths,
                branch_to_example,
                out=attn_buf[n_ctx:].view(M, L, num_heads, head_dim),
            )

            hidden = residual + attn.o_proj(attn_buf).unsqueeze(0)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            hidden = hidden * token_mask

        hidden = self.norm(hidden) * token_mask
        context_hidden = hidden[0, :n_ctx].view(B, P, D)
        branch_hidden = hidden[0, n_ctx:].view(M, L, D)

        return BranchedQwen3Output(
            context_last_hidden_state=context_hidden,
            branch_last_hidden_state=branch_hidden,
        )


# --------------------------------------------------------------------------- #
# Hierarchical decision head (unchanged)
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    """Standard bidirectional Transformer block with an explicit padding mask (Pre-Norm)."""

    def __init__(self, d_model, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid_mask, return_attention=False):
        key_padding_mask = ~valid_mask
        
        # Pre-Norm applied BEFORE Attention
        norm_x1 = self.norm1(x)
        attn_out, attn_weights = self.attn(
            norm_x1,
            norm_x1,
            norm_x1,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        # Residual connection added to the original un-normalized x
        x = x + self.drop1(attn_out)
        
        # Pre-Norm applied BEFORE Feed-Forward
        norm_x2 = self.norm2(x)
        # Residual connection added to x
        x = x + self.ff(norm_x2)
        
        # Explicit padding mask zeroing
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)
        
        return x, attn_weights


class LabelTokenReasoner(nn.Module):
    """K Transformer layers over [shared learned CLS] + candidate label tokens only."""

    def __init__(self, d_model, max_label_length, num_layers=2, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.num_layers = int(num_layers)
        self.max_label_length = int(max_label_length)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.empty(1, self.max_label_length + 1, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ff_mult, dropout) for _ in range(self.num_layers)]
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, branch_states, branch_attention_mask, return_attention=False):
        if branch_states.ndim != 3:
            raise ValueError(f"branch_states must be [M,L,D], got {tuple(branch_states.shape)}")
        if branch_attention_mask.ndim != 2:
            raise ValueError("branch_attention_mask must be [M,L]")
        if branch_states.shape[:2] != branch_attention_mask.shape:
            raise ValueError("branch state/mask shapes do not agree")

        if self.num_layers == 0:
            return gather_last_branch_state(branch_states, branch_attention_mask), []

        M, L, D = branch_states.shape
        if L > self.max_label_length:
            raise ValueError(
                f"Observed label length {L} exceeds configured max_label_length={self.max_label_length}"
            )

        cls = self.cls_token.to(dtype=branch_states.dtype).expand(M, 1, D)
        x = torch.cat([cls, branch_states], dim=1)
        cls_mask = torch.ones(M, 1, dtype=torch.bool, device=branch_attention_mask.device)
        valid_mask = torch.cat([cls_mask, branch_attention_mask.to(torch.bool)], dim=1)

        pos = self.position_embedding[:, : L + 1].to(dtype=x.dtype, device=x.device)
        x = x + pos
        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, valid_mask, return_attention=return_attention)
            if return_attention:
                attention_maps.append(weights)
        return x[:, 0], attention_maps


class DirectVariableChoiceQwenClassifier(nn.Module):
    """Branched Qwen3 -> label-token reasoner -> candidate reasoner -> scalar scorer.

    Existing checkpoint parameter names are preserved.  The only new behavior is the optional
    inference-only Triton implementation inside ``self.qwen``.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-0.6B-Base",
        torch_dtype=None,
        max_label_length=96,
        label_layers=2,
        label_heads=8,
        label_ff_mult=4,
        candidate_layers=2,
        candidate_heads=8,
        candidate_ff_mult=4,
        dropout=0.1,
        use_inference_kernel: bool = True,
    ):
        super().__init__()
        self.qwen = InferenceOptimizedQwen3BranchedModel.from_pretrained(model_name, dtype=torch_dtype)
        self.qwen.set_inference_kernel(use_inference_kernel)
        d = self.qwen.config.hidden_size

        self.label_reasoner = LabelTokenReasoner(
            d_model=d,
            max_label_length=max_label_length,
            num_layers=label_layers,
            num_heads=label_heads,
            ff_mult=label_ff_mult,
            dropout=dropout,
        )
        self.pre_candidate_norm = nn.LayerNorm(d)
        self.candidate_blocks = nn.ModuleList(
            [
                TransformerBlock(d, candidate_heads, candidate_ff_mult, dropout)
                for _ in range(candidate_layers)
            ]
        )
        dropout = 0.1
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            # nn.Linear(d, d // 2),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(d // 2, 1),
            nn.Linear(d, 1),
        )

    @property
    def inference_backend(self) -> str:
        return self.qwen.inference_backend

    @property
    def triton_available(self) -> bool:
        return _TRITON_AVAILABLE

    def set_inference_kernel(self, enabled: bool = True) -> None:
        """Enable/disable the Triton inference branch kernel without changing weights."""
        self.qwen.set_inference_kernel(enabled)

    def forward(
        self,
        context_input_ids,
        context_attention_mask,
        branch_input_ids,
        branch_attention_mask,
        branch_to_example,
        choice_index,
        candidate_mask,
        return_attention=False,
        disable_candidate_transformer=False,
    ):
        encoded = self.qwen.forward_branches(
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            branch_input_ids=branch_input_ids,
            branch_attention_mask=branch_attention_mask,
            branch_to_example=branch_to_example,
        )

        flat_candidate_states, label_attention_maps = self.label_reasoner(
            encoded.branch_last_hidden_state,
            branch_attention_mask,
            return_attention=return_attention,
        )

        B, N = candidate_mask.shape
        D = flat_candidate_states.shape[-1]
        candidate_states = flat_candidate_states.new_zeros(B, N, D)
        candidate_states[branch_to_example, choice_index] = flat_candidate_states
        # candidate_states = self.pre_candidate_norm(candidate_states)
        candidate_states = candidate_states * candidate_mask.unsqueeze(-1).to(candidate_states.dtype)

        candidate_attention_maps = []
        x = candidate_states
        # if not disable_candidate_transformer:
        #     for block in self.candidate_blocks:
        #         x, weights = block(x, candidate_mask, return_attention=return_attention)
        #         if return_attention:
        #             candidate_attention_maps.append(weights)

        logits = self.scorer(x).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1e9)
        probabilities = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "flat_candidate_states": flat_candidate_states,
            "candidate_states_before_set": candidate_states,
            "candidate_states_after_set": x,
            "context_states": encoded.context_last_hidden_state,
            "label_attention_maps": label_attention_maps,
            "candidate_attention_maps": candidate_attention_maps,
            "attention_maps": candidate_attention_maps,
            "inference_backend": self.inference_backend,
        }


__all__ = [
    "DirectVariableChoiceQwenClassifier",
    "InferenceOptimizedQwen3BranchedModel",
    "LabelTokenReasoner",
    "TransformerBlock",
]

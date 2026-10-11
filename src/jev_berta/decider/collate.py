"""Inference-time batching for the hierarchical and BEV decision models.

Examples are dicts with `context`, `query`, `labels` and `primitive`. Candidate order is preserved.
"""

from __future__ import annotations

import torch

from .text import TASK_PROMPTS, TASK_TYPES


def _ensure_padding(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


# --------------------------------------------------------------------------- #
# Hierarchical: one shared prefix per example, one branch per candidate
# --------------------------------------------------------------------------- #

def hierarchical_prefix(context, query):
    context, query = str(context or ""), str(query or "")
    if context.strip():
        return f"Context: {context}\nQuery: {query}\nCandidate:"
    return f"Query: {query}\nCandidate:"


def collate_hierarchical(tokenizer, examples, max_length=512, max_label_length=96):
    _ensure_padding(tokenizer)
    if tokenizer.eos_token_id is None:
        raise ValueError("Qwen tokenizer must define eos_token_id")
    if max_length <= max_label_length:
        raise ValueError("max_length must be larger than max_label_length")

    max_n = max(len(ex["labels"]) for ex in examples)
    candidate_mask = torch.zeros(len(examples), max_n, dtype=torch.bool)
    prefixes, branch_token_lists, branch_to_example, choice_index = [], [], [], []

    for b, ex in enumerate(examples):
        prefixes.append(hierarchical_prefix(ex.get("context", ""), ex.get("query", "")))
        for j, label in enumerate(ex["labels"]):
            # A terminal EOS state is gathered as the fixed-width candidate embedding.
            ids = tokenizer(
                " " + str(label), add_special_tokens=False, truncation=True, max_length=max_label_length - 1,
            )["input_ids"]
            branch_token_lists.append(list(ids) + [tokenizer.eos_token_id])
            branch_to_example.append(b)
            choice_index.append(j)
            candidate_mask[b, j] = True

    max_branch_len = max(len(ids) for ids in branch_token_lists)
    max_context_length = max_length - max_branch_len
    if max_context_length < 1:
        raise ValueError("Current candidate branches leave no room for context tokens")

    context_tok = tokenizer(
        prefixes, padding=True, truncation=True, max_length=max_context_length,
        return_tensors="pt", add_special_tokens=True,
    )
    branch_input_ids = torch.full(
        (len(branch_token_lists), max_branch_len), int(tokenizer.pad_token_id), dtype=torch.long
    )
    branch_attention_mask = torch.zeros((len(branch_token_lists), max_branch_len), dtype=torch.bool)
    for i, ids in enumerate(branch_token_lists):
        branch_input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        branch_attention_mask[i, : len(ids)] = True

    return {
        "context_input_ids": context_tok["input_ids"],
        "context_attention_mask": context_tok["attention_mask"].to(torch.bool),
        "branch_input_ids": branch_input_ids,
        "branch_attention_mask": branch_attention_mask,
        "branch_to_example": torch.tensor(branch_to_example, dtype=torch.long),
        "choice_index": torch.tensor(choice_index, dtype=torch.long),
        "candidate_mask": candidate_mask,
    }


# --------------------------------------------------------------------------- #
# BEV: one packed sequence per example with block-causal masking and shared positions
# --------------------------------------------------------------------------- #

def _truncate(tokenizer, text, max_tokens):
    tokens = tokenizer.encode(str(text))
    if len(tokens) <= max_tokens:
        return str(text)
    return tokenizer.decode(tokens[:max_tokens])


def _bev_encode(tokenizer, state, instructions, choices, task_type, max_state_tokens, max_choice_tokens):
    question = (
        f"{TASK_PROMPTS[task_type]}\n"
        f"Question: {instructions}\n\n"
        f"{_truncate(tokenizer, state, max_state_tokens)}\n"
    )
    choices = [_truncate(tokenizer, c, max_choice_tokens) for c in choices]

    begin = tokenizer.encode("<option>")
    end = tokenizer.encode("</option>")
    question_tokens = tokenizer.encode(question)
    answer_tokens = tokenizer.encode("The answer is:")

    tokens = list(question_tokens)
    question_end = len(question_tokens) - 1
    choices_end = []
    for choice in choices:
        tokens.extend(begin)
        tokens.extend(tokenizer.encode(choice))
        tokens.extend(end)
        choices_end.append(len(tokens) - 1)
    tokens.extend(answer_tokens)
    answer_end = len(tokens) - 1

    # Positions do not advance between choices, so every choice starts at the same position.
    position_ids = list(range(question_end + 1))
    start_idx = question_end + 1
    next_begin = start_idx
    for choice_end in choices_end:
        position_ids.extend(start_idx + i for i in range(choice_end - next_begin + 1))
        next_begin = choice_end + 1
    answer_start = max(position_ids) + 1
    position_ids.extend(answer_start + i for i in range(answer_end + 1 - len(position_ids)))

    length = answer_end + 1
    mask = torch.zeros((length, length))
    q_len = question_end + 1
    mask[:q_len, :q_len] = torch.tril(torch.ones(q_len, q_len))
    spans, choice_start = [], q_len
    for choice_end in choices_end:
        spans.append((choice_start, choice_end))
        choice_start = choice_end + 1
    for s, e in spans:
        n = e - s + 1
        mask[s : e + 1, s : e + 1] = torch.tril(torch.ones(n, n))
    mask[q_len:, :q_len] = 1
    answer_begin = spans[-1][1] + 1
    mask[answer_begin:, :answer_begin] = 1
    answer_n = length - answer_begin
    mask[answer_begin:, answer_begin:] = torch.tril(torch.ones(answer_n, answer_n))

    # Readout is the last content token, immediately before </option>.
    read_idx = [e - len(end) for e in choices_end]
    return {
        "input_ids": tokens, "position_ids": position_ids, "attention_mask": mask,
        "choice_read_idx": read_idx, "answer_end_idx": answer_end,
    }


def collate_bev(tokenizer, examples, max_state_tokens=512, max_choice_tokens=96, max_packed_tokens=32768):
    _ensure_padding(tokenizer)
    encoded = []
    for ex in examples:
        primitive = str(ex.get("primitive", "choice")).lower()
        if primitive not in TASK_TYPES:
            raise ValueError(f"Unsupported BEV task type {primitive!r}")
        item = _bev_encode(
            tokenizer, str(ex.get("context", "")), str(ex.get("query", "")), list(ex["labels"]),
            primitive, max_state_tokens, max_choice_tokens,
        )
        if len(item["input_ids"]) > max_packed_tokens:
            raise ValueError(
                f"packed BEV example has {len(item['input_ids']):,} tokens > max_packed_tokens={max_packed_tokens:,}"
            )
        encoded.append((primitive, item))

    B = len(encoded)
    L = max(len(item["input_ids"]) for _, item in encoded)
    C = max(len(item["choice_read_idx"]) for _, item in encoded)

    input_ids = torch.full((B, L), int(tokenizer.pad_token_id), dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    # Padded query rows self-attend so no attention row is all -inf.
    attention_mask = torch.eye(L).repeat(B, 1, 1)
    choice_read_idx = torch.zeros((B, C), dtype=torch.long)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)
    answer_end_idx = torch.zeros(B, dtype=torch.long)
    task_type = torch.zeros(B, dtype=torch.long)

    for b, (primitive, item) in enumerate(encoded):
        n, c = len(item["input_ids"]), len(item["choice_read_idx"])
        input_ids[b, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        position_ids[b, :n] = torch.tensor(item["position_ids"], dtype=torch.long)
        attention_mask[b, :n, :n] = item["attention_mask"]
        choice_read_idx[b, :c] = torch.tensor(item["choice_read_idx"], dtype=torch.long)
        choice_mask[b, :c] = True
        answer_end_idx[b] = int(item["answer_end_idx"])
        task_type[b] = TASK_TYPES[primitive]

    attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "task_type": task_type,
        "candidate_mask": choice_mask,
    }

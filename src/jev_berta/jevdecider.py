"""Inference for Qwen-based JEV decision models (hierarchical or BEV architectures).

Two surfaces over one checkpoint, each with a single-item and a batched form:
  * raw:       choice / noul / score          and  choice_batch / noul_batch / score_batch
  * JEV-style: decide(state, questions)       and  decide_batch(requests)

Noul and score questions are answered by averaging over the label surfaces seen in training
(yes/no, true/false, correct/incorrect, ... in both orders), which removes surface-form bias.
Batched forms flatten every variant of every item into one stream of model batches.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from .decider.collate import collate_bev, collate_hierarchical
from .decider.text import (
    NOUL_SURFACE_PAIRS, SCORE_SURFACE_STYLES, criteria_as_key_desc, criterion_description,
    format_key_description, format_score_labels, humanize_state, make_noul_query, parse_jsonish,
)
from .jevberta import _as_device, _resolve_checkpoint_dir

DEFAULT_DECIDER_REPO = "leobitz/decision-master-base-label-encode-0.1"
ARCHITECTURES = ("hierarchical", "bev")
QUESTION_TYPES = ("choice", "score", "noul")


def _state_text(state: Any) -> str:
    if state is None:
        text = ""
    else:
        text = state.strip() if isinstance(state, str) else humanize_state(state)
    return text or "No additional state was provided."


def _keys_and_descs(options: Any, what: str) -> tuple[list[str], list[str]]:
    """Accept a list of labels or a {key: description} mapping; return parallel (keys, descs)."""
    options = parse_jsonish(options)
    if isinstance(options, dict):
        keys, descs = criteria_as_key_desc(options)
    elif isinstance(options, (list, tuple)):
        keys, descs = [str(o) for o in options], [""] * len(options)
    else:
        keys = []
    if len(keys) < 2:
        raise ValueError(f"{what} needs at least two options")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{what} option keys must be unique: {keys}")
    return keys, descs


def _distribution(probs) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("model returned a non-finite or empty distribution")
    return probs / total


def _example(context: str, query: str, labels: list[str], primitive: str) -> dict[str, Any]:
    return {"context": context, "query": query, "labels": labels, "primitive": primitive}


def _plan_choice(context, query, choices, order_tta=True):
    """Return (examples, finish); finish maps the examples' probabilities to the answer."""
    keys, descs = _keys_and_descs(choices, "choice")
    labels = [format_key_description(k, d) for k, d in zip(keys, descs)]
    context = _state_text(context)

    perms = [list(range(len(keys)))]
    if order_tta:
        perms.append(perms[0][::-1])
    examples = [_example(context, query, [labels[i] for i in perm], "choice") for perm in perms]

    def finish(plist):
        probs = np.zeros(len(keys))
        for perm, p in zip(perms, plist):
            probs[perm] += p
        probs = _distribution(probs)
        best = int(probs.argmax())
        return {
            "type": "choice", "choice": keys[best],
            "probabilities": dict(zip(keys, probs.tolist())), "confidence": float(probs[best]),
        }

    return examples, finish


def _plan_noul(context, query, criteria=None):
    true_desc = false_desc = ""
    criteria = parse_jsonish(criteria)
    for k, v in (criteria.items() if isinstance(criteria, dict) else ()):
        if str(k).strip().lower() in {"true", "1", "yes"}:
            true_desc = criterion_description(v)
        elif str(k).strip().lower() in {"false", "0", "no"}:
            false_desc = criterion_description(v)

    context, query = _state_text(context), make_noul_query(query)
    examples, pos_index = [], []
    for pos, neg in NOUL_SURFACE_PAIRS:
        pos_label = format_key_description(pos, true_desc)
        neg_label = format_key_description(neg, false_desc)
        for labels, idx in (([pos_label, neg_label], 0), ([neg_label, pos_label], 1)):
            examples.append(_example(context, query, labels, "noul"))
            pos_index.append(idx)

    def finish(plist):
        return {"type": "noul", "noul": float(np.mean([p[i] for p, i in zip(plist, pos_index)]))}

    return examples, finish


def _plan_score(context, query, levels):
    keys, descs = _keys_and_descs(levels, "score")
    try:
        values = [float(k) for k in keys]
    except ValueError:
        values = [float(i) for i in range(len(keys))]
    context = _state_text(context)

    examples, seen = [], set()
    for style in SCORE_SURFACE_STYLES:
        labels = format_score_labels(keys, descs, style)
        if tuple(labels) not in seen:
            seen.add(tuple(labels))
            examples.append(_example(context, query, labels, "score"))

    def finish(plist):
        probs = _distribution(np.mean(plist, axis=0))
        return {
            "type": "score", "score": float(np.dot(probs, values)),
            "legend": dict(zip(keys, descs)),
            "probabilities": dict(zip(keys, probs.tolist())), "confidence": float(probs.max()),
        }

    return examples, finish


def _numbers(answer):
    for v in answer.values():
        if isinstance(v, float):
            yield v
        elif isinstance(v, dict):
            yield from (x for x in v.values() if isinstance(x, float))


def _plan_question(context, qid, question):
    """Validate one JEV question and plan it against an already-rendered state."""
    if not isinstance(question, Mapping):
        raise ValueError(f"question {qid!r} must be an object")
    kind = str(question.get("type", "")).lower()
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid!r}: type must be one of {QUESTION_TYPES}, got {kind!r}")
    instructions = str(question.get("instructions") or "").strip()
    if not instructions:
        raise ValueError(f"question {qid!r}: instructions are required")

    planner = {"choice": _plan_choice, "score": _plan_score, "noul": _plan_noul}[kind]
    examples, finish = planner(context, instructions, question.get("criteria"))

    def checked_finish(plist):
        answer = finish(plist)
        if not all(math.isfinite(v) for v in _numbers(answer)):
            raise ValueError(f"question {qid!r}: non-finite value in answer")
        return answer

    return examples, checked_finish


def _plan_request(state, questions):
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("questions must be a non-empty mapping of question id -> question")
    context = _state_text(state)
    plans = {qid: _plan_question(context, qid, q) for qid, q in questions.items()}
    examples = [e for ex, _ in plans.values() for e in ex]

    def finish(plist):
        answers, i = {}, 0
        for qid, (ex, fin) in plans.items():
            answers[qid] = fin(plist[i:i + len(ex)])
            i += len(ex)
        return {"answers": answers}

    return examples, finish


def _resolve_dtype(device: torch.device, requested: str | None) -> torch.dtype:
    if requested in ("float32", "bfloat16", "float16"):
        return getattr(torch, requested)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _build_model(config: Mapping[str, Any], dtype: torch.dtype):
    arch = config["architecture"]
    if arch == "hierarchical":
        from .decider.hierarchical import DirectVariableChoiceQwenClassifier

        return DirectVariableChoiceQwenClassifier(
            config["model_name"],
            torch_dtype=dtype,
            max_label_length=int(config.get("max_label_length", 96)),
            label_layers=int(config.get("label_layers", 2)),
            label_heads=int(config.get("label_heads", 8)),
            label_ff_mult=int(config.get("label_ff_mult", 4)),
            candidate_layers=int(config.get("candidate_layers", 2)),
            candidate_heads=int(config.get("candidate_heads", 8)),
            candidate_ff_mult=int(config.get("candidate_ff_mult", 4)),
            dropout=float(config.get("dropout", 0.0)),
        )
    if arch == "bev":
        from .decider.bev import BEVQwenDecisionNetwork

        return BEVQwenDecisionNetwork(
            config["model_name"],
            torch_dtype=dtype,
            head_dim=int(config.get("head_dim", 512)),
            head_layers=int(config.get("head_layers", 2)),
        )
    raise ValueError(f"Unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


class JevDecider:
    def __init__(
        self,
        model_source: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        model_dtype: str | None = None,
        batch_size: int = 32,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | None = None,
    ):
        """`model_source`: a checkpoint dir, an output dir containing `best_val`, or a Hugging Face repo id."""
        self.device = _as_device(device)
        self.batch_size = max(1, int(batch_size))
        path = _resolve_checkpoint_dir(
            model_source or DEFAULT_DECIDER_REPO,
            cache_dir=cache_dir, local_files_only=local_files_only, token=token,
        )
        if not (path / "model.pt").exists() and (path / "best_val" / "model.pt").exists():
            path = path / "best_val"
        if not (path / "model.pt").exists():
            raise FileNotFoundError(f"Missing model.pt in checkpoint directory: {path}")
        self.checkpoint_dir = path

        self.config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.architecture = self.config.get("architecture")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"{path} is not a decision-model checkpoint (architecture={self.architecture!r}); "
                "use JevBerta for encoder checkpoints"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        dtype = _resolve_dtype(self.device, model_dtype or self.config.get("model_dtype"))
        self.model = _build_model(self.config, dtype)
        self.model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()

        self.use_amp = self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    @classmethod
    def from_pretrained(cls, model_source: str | Path | None = None, **kwargs: Any) -> "JevDecider":
        return cls(model_source=model_source, **kwargs)

    def _collate(self, examples):
        cfg = self.config
        if self.architecture == "hierarchical":
            return collate_hierarchical(
                self.tokenizer, examples,
                max_length=int(cfg.get("max_length", 512)),
                max_label_length=int(cfg.get("max_label_length", 96)),
            )
        return collate_bev(
            self.tokenizer, examples,
            max_state_tokens=int(cfg.get("max_state_tokens", 512)),
            max_choice_tokens=int(cfg.get("max_choice_tokens", 96)),
            max_packed_tokens=int(cfg.get("max_packed_tokens", 32768)),
        )

    def _forward(self, batch):
        if self.architecture == "hierarchical":
            return self.model(
                context_input_ids=batch["context_input_ids"],
                context_attention_mask=batch["context_attention_mask"],
                branch_input_ids=batch["branch_input_ids"],
                branch_attention_mask=batch["branch_attention_mask"],
                branch_to_example=batch["branch_to_example"],
                choice_index=batch["choice_index"],
                candidate_mask=batch["candidate_mask"],
            )
        return self.model(
            input_ids=batch["input_ids"],
            position_ids=batch["position_ids"],
            attention_mask=batch["attention_mask"],
            choice_read_idx=batch["choice_read_idx"],
            choice_mask=batch["choice_mask"],
            answer_end_idx=batch["answer_end_idx"],
            task_type=batch["task_type"],
        )

    @torch.no_grad()
    def _probabilities(self, examples: Sequence[dict[str, Any]]) -> list[np.ndarray]:
        """Run examples through the model; one probability array per example (label order)."""
        out = []
        for i in range(0, len(examples), self.batch_size):
            chunk = examples[i:i + self.batch_size]
            batch = {k: v.to(self.device) for k, v in self._collate(chunk).items()}
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                result = self._forward(batch)
            probs = result["probabilities"].float().cpu().numpy()
            out.extend(_distribution(p[:len(ex["labels"])]) for p, ex in zip(probs, chunk))
        return out

    def _run(self, plans):
        """Evaluate every plan's examples in one pass, then split results back per plan."""
        probs = self._probabilities([e for ex, _ in plans for e in ex])
        results, i = [], 0
        for ex, finish in plans:
            results.append(finish(probs[i:i + len(ex)]))
            i += len(ex)
        return results

    # Batched: items are dicts. choice: {context, query, choices}; noul: {context, query, criteria?};
    # score: {context, query, levels}.
    def choice_batch(self, items, order_tta=True):
        return self._run([_plan_choice(it.get("context"), it["query"], it["choices"], order_tta) for it in items])

    def noul_batch(self, items):
        return self._run([_plan_noul(it.get("context"), it["query"], it.get("criteria")) for it in items])

    def score_batch(self, items):
        return self._run([_plan_score(it.get("context"), it["query"], it["levels"]) for it in items])

    def decide_batch(self, requests):
        """requests: [{"state": ..., "questions": {...}}, ...] -> [{"answers": {...}}, ...]."""
        return self._run([_plan_request(r.get("state"), r.get("questions")) for r in requests])

    # Single-item forms.
    def choice(self, context, query, choices, order_tta=True):
        """`choices`: list of labels or {key: description}. Averages original and reversed order."""
        return self.choice_batch([{"context": context, "query": query, "choices": choices}], order_tta)[0]

    def noul(self, context, query, criteria=None):
        """P(yes) averaged over every yes/no surface pair in both orders.

        `criteria` may hold true/false descriptions, e.g. {"true": "...", "false": "..."}.
        """
        return self.noul_batch([{"context": context, "query": query, "criteria": criteria}])[0]

    def score(self, context, query, levels):
        """Probability-weighted score over ordered `levels` ({key: description} or list of labels).

        The score is the expected level value: numeric keys are used as values, otherwise 0-based indices.
        """
        return self.score_batch([{"context": context, "query": query, "levels": levels}])[0]

    def decide(self, state, questions):
        """JEV-style API: evaluate every question against one shared state.

        questions = {id: {"type": "choice"|"score"|"noul", "instructions": str, "criteria": {...}}}
        """
        return self.decide_batch([{"state": state, "questions": questions}])[0]


__all__ = ["DEFAULT_DECIDER_REPO", "JevDecider"]

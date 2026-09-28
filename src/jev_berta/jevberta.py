from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_REPO = "leobitz/jev-berta-base-zeroshot-classifier"


def _as_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _as_dtype(dtype: str | torch.dtype | None, *, device: torch.device) -> torch.dtype:
    if dtype is None:
        resolved = torch.float32
    elif isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        key = str(dtype).lower()
        if key not in _DTYPE_ALIASES:
            raise ValueError(
                f"Unknown dtype '{dtype}'. Expected one of {sorted(_DTYPE_ALIASES)} or a torch.dtype."
            )
        resolved = _DTYPE_ALIASES[key]

    if resolved is torch.float16 and device.type == "cpu":
        import warnings

        warnings.warn(
            "float16 was requested on CPU, where it is unreliable and often slower "
            "than float32. Falling back to float32. Pass device='cuda' to use fp16, "
            "or dtype='bfloat16' if you need reduced precision on CPU.",
            stacklevel=2,
        )
        resolved = torch.float32

    return resolved


def _assert_uniform_dtype(model: nn.Module, *, expected: torch.dtype) -> None:
    offenders = sorted(
        {f"{name} ({param.dtype})" for name, param in model.named_parameters() if param.dtype != expected}
    )
    if offenders:
        raise RuntimeError(
            "JevBerta expected every parameter to be loaded in "
            f"{expected}, but found mismatched parameters: {', '.join(offenders[:10])}"
            + (f" (+{len(offenders) - 10} more)" if len(offenders) > 10 else "")
            + ". This usually means a newer `transformers`/`torch` release changed how "
            "checkpoint dtype is inferred. Pass an explicit dtype= to JevBerta(...) as a workaround."
        )


def _resolve_checkpoint_dir(
    model_source: str | Path | None,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    token: str | None = None,
) -> Path:
    if model_source is None:
        model_source = DEFAULT_MODEL_REPO

    candidate = Path(model_source).expanduser()
    if candidate.exists():
        return candidate.resolve()

    snapshot_path = snapshot_download(
        repo_id=str(model_source),
        cache_dir=None if cache_dir is None else str(cache_dir),
        local_files_only=local_files_only,
        token=token,
        allow_patterns=[
            "config.json",
            "model.pt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spm.model",
            "added_tokens.json",
        ],
    )
    return Path(snapshot_path)


class CandidateSetBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
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

    def forward(self, x: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=~candidate_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.ff(x))
        return x * candidate_mask.unsqueeze(-1).to(x.dtype)


class DirectVariableChoiceClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        set_layers: int,
        set_heads: int,
        ff_mult: int,
        dropout: float,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
        self.accepts_token_type_ids = "token_type_ids" in inspect.signature(self.bert.forward).parameters
        hidden_size = self.bert.config.hidden_size
        self.pre_set_norm = nn.LayerNorm(hidden_size)
        self.set_blocks = nn.ModuleList(
            [CandidateSetBlock(hidden_size, set_heads, ff_mult, dropout) for _ in range(set_layers)]
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        total = input_ids.shape[0]
        if chunk_size is None or chunk_size <= 0 or chunk_size >= total:
            return self._encode_chunk(input_ids, attention_mask, token_type_ids)

        cls_chunks = []
        for start in range(0, total, chunk_size):
            end = start + chunk_size
            cls_chunks.append(
                self._encode_chunk(
                    input_ids[start:end],
                    attention_mask[start:end],
                    None if token_type_ids is None else token_type_ids[start:end],
                )
            )
        return torch.cat(cls_chunks, dim=0)

    def _encode_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        encoder_kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.accepts_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids
        return self.bert(**encoder_kwargs).last_hidden_state[:, 0]

    def score(
        self,
        cls_states: torch.Tensor,
        batch_index: torch.Tensor,
        choice_index: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_choices = candidate_mask.shape
        hidden_size = cls_states.shape[-1]
        candidate_states = cls_states.new_zeros(batch_size, num_choices, hidden_size)
        candidate_states[batch_index, choice_index] = cls_states
        candidate_states = self.pre_set_norm(candidate_states)

        hidden = candidate_states
        for block in self.set_blocks:
            hidden = block(hidden, candidate_mask)

        logits = self.scorer(hidden).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -1e9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_index: torch.Tensor,
        choice_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        encode_chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_states = self.encode(input_ids, attention_mask, token_type_ids, encode_chunk_size)
        logits = self.score(cls_states, batch_index, choice_index, candidate_mask)
        return {"logits": logits}


@dataclass(frozen=True)
class CandidateEncoding:
    key: str
    rendered_choice: str


class JevBerta:
    def __init__(
        self,
        model_source: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        temperature: float = 1.0,
        batch_size: int = 8,
        max_length: int | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | None = None,
    ):
        self.device = _as_device(device)
        self.dtype = _as_dtype(dtype, device=self.device)
        self.temperature = float(temperature)
        self.batch_size = max(1, int(batch_size))
        self.checkpoint_dir = _resolve_checkpoint_dir(
            model_source,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

        config_path = self.checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in checkpoint directory: {self.checkpoint_dir}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        self.max_length = int(max_length or self.config.get("max_length", 512))
        self.model = DirectVariableChoiceClassifier(
            model_name=self.config["model_name"],
            set_layers=int(self.config.get("set_layers", 2)),
            set_heads=int(self.config.get("set_heads", 8)),
            ff_mult=int(self.config.get("set_ff_mult", 2)),
            dropout=float(self.config.get("dropout", 0.1)),
            dtype=self.dtype,
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        state_path = self.checkpoint_dir / "model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing model.pt in checkpoint directory: {self.checkpoint_dir}")
        state = torch.load(state_path, map_location="cpu")
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        _assert_uniform_dtype(self.model, expected=self.dtype)

    @classmethod
    def from_pretrained(cls, model_source: str | Path | None = None, **kwargs: Any) -> "JevBerta":
        return cls(model_source=model_source, **kwargs)

    def _prepare_batch(
        self, items: Sequence[tuple[str, str, Sequence[str]]]
    ) -> tuple[dict[str, list[list[int]]], torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        text_a: list[str] = []
        text_b: list[str] = []
        batch_index_list: list[int] = []
        choice_index_list: list[int] = []
        num_choices: list[int] = []
        for item_index, (context, query, choices) in enumerate(items):
            normalized_context = "" if context is None else str(context)
            normalized_query = "" if query is None else str(query)
            normalized_choices = [str(choice) for choice in choices]
            if len(normalized_choices) < 2:
                raise ValueError("At least two choices are required.")

            prefix = (
                f"Context: {normalized_context}\nQuery: {normalized_query}"
                if normalized_context.strip()
                else f"Query: {normalized_query}"
            )
            for choice_pos, choice in enumerate(normalized_choices):
                text_a.append(prefix)
                text_b.append(f"Candidate: {choice}")
                batch_index_list.append(item_index)
                choice_index_list.append(choice_pos)
            num_choices.append(len(normalized_choices))

        encoded = self.tokenizer(
            text_a,
            text_b,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        max_choices = max(num_choices)
        candidate_mask = torch.zeros(len(num_choices), max_choices, dtype=torch.bool)
        for item_index, count in enumerate(num_choices):
            candidate_mask[item_index, :count] = True

        batch_index = torch.tensor(batch_index_list, dtype=torch.long, device=self.device)
        choice_index = torch.tensor(choice_index_list, dtype=torch.long, device=self.device)
        return dict(encoded), batch_index, choice_index, candidate_mask.to(self.device), num_choices

    @torch.inference_mode()
    def _score_items(
        self, items: Sequence[tuple[str, str, Sequence[str]]]
    ) -> list[torch.Tensor]:
        encoded, batch_index, choice_index, candidate_mask, num_choices = self._prepare_batch(items)
        cls_states = self._encode_rows(encoded)
        logits = self.model.score(cls_states, batch_index, choice_index, candidate_mask)
        return [logits[item_index, :count] for item_index, count in enumerate(num_choices)]

    def _encode_rows(self, encoded: Mapping[str, list[list[int]]]) -> torch.Tensor:
        input_ids = encoded["input_ids"]
        total = len(input_ids)
        # Group rows of similar length so each chunk pads only to its own longest row.
        order = sorted(range(total), key=lambda row: len(input_ids[row]))
        hidden_size = self.model.bert.config.hidden_size
        param_dtype = next(self.model.parameters()).dtype
        cls_states = torch.empty(total, hidden_size, device=self.device, dtype=param_dtype)
        for start in range(0, total, self.batch_size):
            rows = order[start : start + self.batch_size]
            chunk = {key: [values[row] for row in rows] for key, values in encoded.items()}
            padded = self.tokenizer.pad(chunk, padding='longest', return_tensors="pt")
            token_type_ids = padded.get("token_type_ids")
            cls_chunk = self.model.encode(
                padded["input_ids"].to(self.device),
                padded["attention_mask"].to(self.device),
                None if token_type_ids is None else token_type_ids.to(self.device),
            )
            cls_states[torch.tensor(rows, device=self.device)] = cls_chunk
        return cls_states

    def _decode_logits(self, choices: Sequence[str], logits_row: torch.Tensor) -> dict[str, Any]:
        logits_row = logits_row / self.temperature
        probabilities = F.softmax(logits_row, dim=-1).detach().cpu().tolist()
        best_index = int(torch.argmax(logits_row).item())

        if len(set(choices)) != len(choices):
            import warnings

            warnings.warn(
                "Duplicate choice strings were passed to predict(); the "
                "'probabilities'/'scores' dicts sum duplicate entries together "
                "under one key. Use the returned 'probabilities_by_index' list "
                "(aligned with 'choices') if you need each entry separately.",
                stacklevel=3,
            )

        probability_map: dict[str, float] = {}
        for choice, probability in zip(choices, probabilities):
            probability_map[choice] = probability_map.get(choice, 0.0) + float(probability)
        return {
            "choice": choices[best_index],
            "choice_index": best_index,
            "choices": list(choices),
            "probabilities": probability_map,
            "probabilities_by_index": [float(p) for p in probabilities],
            "scores": dict(probability_map),
        }

    def predict(self, context: str, query: str, choices: Sequence[str]) -> dict[str, Any]:
        normalized_choices = [str(choice) for choice in choices]
        logits_row = self._score_items([(context, query, normalized_choices)])[0]
        return self._decode_logits(normalized_choices, logits_row)

    def _normalize_question_criteria(self, criteria: Any) -> list[CandidateEncoding]:
        if isinstance(criteria, Mapping):
            encodings = []
            for key, value in criteria.items():
                rendered_choice = str(key) if value is None else f"{key}: {value}"
                encodings.append(CandidateEncoding(key=str(key), rendered_choice=rendered_choice))
            return encodings
        if isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
            return [CandidateEncoding(key=str(value), rendered_choice=str(value)) for value in criteria]
        raise TypeError("Question criteria must be either a mapping or a sequence of labels.")

    def predict_jev(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        state = str(payload.get("state", ""))
        questions = payload.get("questions")
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("Payload must contain a non-empty 'questions' mapping.")

        question_specs: list[tuple[str, Mapping[str, Any], str, list[CandidateEncoding]]] = []
        batch_items: list[tuple[str, str, list[str]]] = []
        for question_name, spec in questions.items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"Question '{question_name}' must be a mapping.")
            query = str(spec.get("instructions", ""))
            encodings = self._normalize_question_criteria(spec.get("criteria"))
            question_specs.append((str(question_name), spec, query, encodings))
            batch_items.append((state, query, [encoding.rendered_choice for encoding in encodings]))

        logits_rows = self._score_items(batch_items)

        predictions: dict[str, Any] = {}
        for (question_name, spec, query, encodings), logits_row in zip(question_specs, logits_rows):
            rendered_choices = [encoding.rendered_choice for encoding in encodings]
            raw_prediction = self._decode_logits(rendered_choices, logits_row)
            key_by_rendered_choice = {
                encoding.rendered_choice: encoding.key for encoding in encodings
            }
            predictions[question_name] = {
                "type": spec.get("type"),
                "query": query,
                "choice": key_by_rendered_choice[raw_prediction["choice"]],
                "choice_index": raw_prediction["choice_index"],
                "probabilities": {
                    key_by_rendered_choice[rendered_choice]: probability
                    for rendered_choice, probability in raw_prediction["probabilities"].items()
                },
                "scores": {
                    key_by_rendered_choice[rendered_choice]: score
                    for rendered_choice, score in raw_prediction["scores"].items()
                },
                "rendered_choices": rendered_choices,
            }

        return {"state": state, "predictions": predictions}


__all__ = ["DEFAULT_MODEL_REPO", "JevBerta"]

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
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_index: torch.Tensor,
        choice_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and self.accepts_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids

        cls_states = self.bert(**encoder_kwargs).last_hidden_state[:, 0]
        batch_size, num_choices = candidate_mask.shape
        hidden_size = cls_states.shape[-1]
        candidate_states = cls_states.new_zeros(batch_size, num_choices, hidden_size)
        candidate_states[batch_index, choice_index] = cls_states
        candidate_states = self.pre_set_norm(candidate_states)

        hidden = candidate_states
        for block in self.set_blocks:
            hidden = block(hidden, candidate_mask)

        logits = self.scorer(hidden).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, -1e9)
        probabilities = F.softmax(logits, dim=-1)
        return {"logits": logits, "probabilities": probabilities}


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
        temperature: float = 1.0,
        max_length: int | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | None = None,
    ):
        self.device = _as_device(device)
        self.temperature = float(temperature)
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
        ).to(self.device)

        state_path = self.checkpoint_dir / "model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing model.pt in checkpoint directory: {self.checkpoint_dir}")
        state = torch.load(state_path, map_location="cpu")
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_source: str | Path | None = None, **kwargs: Any) -> "JevBerta":
        return cls(model_source=model_source, **kwargs)

    def _prepare_inputs(self, context: str, query: str, choices: Sequence[str]) -> dict[str, torch.Tensor]:
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
        tokenized = self.tokenizer(
            [prefix] * len(normalized_choices),
            [f"Candidate: {choice}" for choice in normalized_choices],
            padding=True,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {
            **tokenized,
            "batch_index": torch.zeros(len(normalized_choices), dtype=torch.long),
            "choice_index": torch.arange(len(normalized_choices), dtype=torch.long),
            "candidate_mask": torch.ones(1, len(normalized_choices), dtype=torch.bool),
        }
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch.items()}

    @torch.no_grad()
    def predict(self, context: str, query: str, choices: Sequence[str]) -> dict[str, Any]:
        normalized_choices = [str(choice) for choice in choices]
        batch = self._prepare_inputs(context=context, query=query, choices=normalized_choices)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            batch_index=batch["batch_index"],
            choice_index=batch["choice_index"],
            candidate_mask=batch["candidate_mask"],
        )
        logits = outputs["logits"][0] / self.temperature
        probabilities = F.softmax(logits, dim=-1).detach().cpu().tolist()
        best_index = int(torch.argmax(logits).item())
        probability_map = {
            choice: float(probability) for choice, probability in zip(normalized_choices, probabilities)
        }
        return {
            "choice": normalized_choices[best_index],
            "choice_index": best_index,
            "choices": normalized_choices,
            "probabilities": probability_map,
            "scores": dict(probability_map),
        }

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

        predictions: dict[str, Any] = {}
        for question_name, spec in questions.items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"Question '{question_name}' must be a mapping.")
            query = str(spec.get("instructions", ""))
            encodings = self._normalize_question_criteria(spec.get("criteria"))
            raw_prediction = self.predict(
                context=state,
                query=query,
                choices=[encoding.rendered_choice for encoding in encodings],
            )

            key_by_rendered_choice = {
                encoding.rendered_choice: encoding.key for encoding in encodings
            }
            predictions[str(question_name)] = {
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
                "rendered_choices": [encoding.rendered_choice for encoding in encodings],
            }

        return {"state": state, "predictions": predictions}


__all__ = ["DEFAULT_MODEL_REPO", "JevBerta"]

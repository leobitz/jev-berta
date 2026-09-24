# JevBerta

JevBerta is a small Python package for running a variable-choice zero-shot classifier over structured decision tasks.

The model is published at `leobitz/jev-berta-base-zeroshot-classifier`. In the earlier evaluation notebooks and result exports, the same model appears under the checkpoint name `best_ood`. In this README, `best_ood` is referred to by its product name: JevBerta.

## What It Does

JevBerta scores a set of candidate answers conditioned on:

- a `context` string
- a `query` string
- a variable-length list of candidate choices

It also exposes a JEV-style interface where:

- `state` is treated as the context
- each question's `instructions` is treated as the query
- each question's `criteria` defines the candidate labels

All supported question types are reduced to classification over the provided criteria.

## Install

```bash
pip install -e .
```

## Quick Start

```python
from jev_berta import JevBerta

model = JevBerta.from_pretrained("leobitz/jev-berta-base-zeroshot-classifier")

result = model.predict(
    context="Customer says they were billed twice for the same order.",
    query="What is the best category?",
    choices=["billing", "shipping", "technical"],
)

print(result["choice"])
print(result["probabilities"])
print(result["scores"])
```

`probabilities` and `scores` are both softmax-normalized probabilities over the provided choices.

## JEV-Style API

```python
from jev_berta import JevBerta

model = JevBerta.from_pretrained("leobitz/jev-berta-base-zeroshot-classifier")

payload = {
    "state": "A customer says they were charged twice for the same order, already emailed support twice without getting a reply, and now wants the duplicate charge refunded immediately.",
    "questions": {
        "refund_requested": {
            "type": "noul",
            "instructions": "Is the customer explicitly asking for a refund?",
            "criteria": {
                "true": "The customer clearly wants money returned or a charge reversed.",
                "false": "The customer is not asking for a refund."
            }
        },
        "owner_team": {
            "type": "choice",
            "instructions": "Which team should take ownership of this case?",
            "criteria": {
                "billing": "Handles duplicate charges, refunds, invoices, and payment disputes.",
                "support": "Handles follow-up communication and general customer assistance.",
                "technical": "Handles bugs, outages, and product malfunctions.",
                "other": "Use when none of the main teams fit the case."
            }
        },
        "priority": {
            "type": "score",
            "instructions": "How urgent is this case?",
            "criteria": ["Low", "Medium", "High"]
        }
    }
}

result = model.predict_jev(payload)
print(result["predictions"])
```

## Architecture

The runtime package is centered on `JevBerta` in `src/jev_berta/jevberta.py`.

### Input formulation

For each candidate label, the model builds a pair of texts:

- sequence A: `Context: ...\nQuery: ...`
- sequence B: `Candidate: ...`

Each candidate is encoded independently by the Transformer encoder, then grouped back into a candidate set for joint reasoning.

### Encoder backbone

JevBerta uses a pretrained Hugging Face encoder loaded through `transformers.AutoModel`. The exported checkpoint configuration determines:

- base encoder model name
- maximum sequence length
- number of set-attention layers
- number of set-attention heads
- feed-forward expansion factor
- dropout

### Candidate-set reasoning

After encoding, the model:

1. takes the CLS representation for each candidate pair
2. reassembles candidates into a `batch_size x num_choices x hidden_size` tensor
3. applies layer normalization
4. runs several candidate-set self-attention blocks

Each candidate-set block contains:

- multi-head self-attention across the candidate dimension
- residual connections
- layer normalization
- a feed-forward network

This lets the model score each option while conditioning on the full set of alternatives, rather than treating each option fully independently.

### Scoring head

A small MLP converts each candidate state into a scalar logit. A masked softmax over the candidate dimension produces the final probability distribution.

### Public APIs

The package exposes two main inference calls:

- `predict(context, query, choices)`
- `predict_jev(payload)`

`predict_jev` is a thin adapter over `predict` that:

- maps `state -> context`
- maps `instructions -> query`
- converts `criteria` into the candidate list
- returns one prediction block per question key

## Package Layout

```text
jev-berta/
  pyproject.toml
  README.md
  src/jev_berta/
    __init__.py
    jevberta.py
  demo/
    demo_usage.ipynb
```

## Performance

The metrics below are taken from `combined_results.csv` at the workspace root.

Important caveat:

- JevBerta and OpenJev were often evaluated on the same filtered subsets.
- `kev-latest` and `laya-typed-decisions` sometimes used much larger sample counts.
- Because of that, the table is useful for orientation, but not every row is a perfectly controlled apples-to-apples comparison.

### Accuracy Snapshot

| Dataset | JevBerta | OpenJev | kev-latest | laya-typed-decisions |
| --- | ---: | ---: | ---: | ---: |
| validation | 0.854 | 0.556 | 0.713 | 0.503 |
| ood_eval | 0.627 | 0.391 | 0.640 | 0.431 |
| truthfulqa | 0.480 | 0.252 | 0.481 | 0.132 |
| mmlu | 0.276 | 0.327 | 0.458 | 0.274 |
| type_decision | 0.374 | 0.390 | 0.501 | 0.332 |

### JevBerta Detailed Metrics

| Dataset | Examples | Accuracy | Top-2 Acc | NLL | Brier | ECE | Mean Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 998 | 0.854 | 0.948 | 0.372 | 0.187 | 0.025 | 0.851 |
| ood_eval | 976 | 0.627 | 0.831 | 1.198 | 0.523 | 0.124 | 0.745 |
| truthfulqa | 817 | 0.480 | 0.690 | 1.386 | 0.669 | 0.086 | 0.561 |
| mmlu | 983 | 0.276 | 0.532 | 1.408 | 0.759 | 0.070 | 0.345 |
| type_decision | not recorded | 0.374 | not recorded | not recorded | not recorded | not recorded | not recorded |

### Reading The Results

A few clear patterns show up in the current benchmark export.

- JevBerta is strongest on the builder-style validation set, where it posts the best accuracy in the file at `0.854` and also the best calibration there with `ECE = 0.025`.
- On `ood_eval`, JevBerta remains competitive at `0.627` accuracy, trailing `kev-latest` slightly at `0.640` while staying well ahead of OpenJev and Laya on the recorded runs.
- On `truthfulqa`, JevBerta is materially better than OpenJev and Laya, and effectively tied with `kev-latest` on accuracy.
- On `mmlu`, JevBerta is weaker. That benchmark looks more like broad factual multiple-choice knowledge than the decision-routing tasks the model is best at.
- On the custom `type_decision` evaluation row, the exported file records only accuracy. There JevBerta is below OpenJev and `kev-latest`, so that slice is currently a weaker point.

### Summary

Based on the recorded results, JevBerta is best viewed as a strong task-specific decision model rather than a general-purpose knowledge QA model.

It performs best when:

- the task is naturally framed as choosing among candidate actions, labels, or intents
- the context and query are closely tied to operational decision making
- calibration matters as much as top-1 accuracy

It performs less strongly when:

- the benchmark tests broad factual world knowledge, such as MMLU
- the target task differs from the training/evaluation style used in the builder pipeline

## Demo Notebook

See `demo/demo_usage.ipynb` for a runnable walkthrough that loads `leobitz/jev-berta-base-zeroshot-classifier` and demonstrates both APIs.

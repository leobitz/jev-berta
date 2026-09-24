# JevBerta

JevBerta is a small Python package for running the JevBerta zero-shot classifier against variable-choice decision tasks.

It supports two inference APIs:

- direct classification with `context`, `query`, and `choices`
- JEV-style payload inference with `state` and `questions`

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
```

## JEV-Style Payload

```python
from jev_berta import JevBerta

model = JevBerta.from_pretrained("leobitz/jev-berta-base-zeroshot-classifier")

payload = {
    "state": "You are a woman who is old and frail. I can't work with you anymore.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Is there a bias here?",
            "criteria": {
                "true": "There is a bias",
                "false": "No bias detected"
            }
        },
        "department": {
            "type": "choice",
            "instructions": "what kind of bias is it?",
            "criteria": {
                "gender": "Prejudice based on gender",
                "racial": "Prejudice based on race",
                "age": "Prejudice based on age"
            }
        },
        "frustration": {
            "type": "score",
            "instructions": "how strong is the bias?",
            "criteria": ["Mild", "Moderate", "Severe"]
        }
    }
}

result = model.predict_jev(payload)
print(result["predictions"])
```

## Notes

- `instructions` is used as the query.
- `state` is used as the context.
- All question types are reduced to classification over the provided criteria.
- `criteria` may be either a mapping or a list.

## Demo Notebook

See `notebooks/demo_usage.ipynb` for a runnable walkthrough that loads `leobitz/jev-berta-base-zeroshot-classifier` and demonstrates both APIs.
# jev-berta

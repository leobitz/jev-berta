"""Text rendering shared with training: state humanization and label surface forms.

Must stay in sync with the training data pipeline, since the models were trained on these exact surfaces.
"""

from __future__ import annotations

import ast
import json

NOUL_SURFACE_PAIRS = (
    ("yes", "no"),
    ("true", "false"),
    ("correct", "incorrect"),
    ("affirmative", "negative"),
    ("supported", "not supported"),
    ("yes / true", "no / false"),
)

SCORE_SURFACE_STYLES = (
    "key_description",
    "score_key_description",
    "level_key_description",
    "rating_position_description",
    "description_only",
    "option_position_description",
)

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}


def parse_jsonish(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(text)
            except Exception:
                pass
        return value
    return value


def _humanize_key(key: str) -> str:
    key = str(key).strip().replace("_", " ")
    replacements = {
        " id": " ID",
        " url": " URL",
        " uri": " URI",
        " api": " API",
        " ms": " (milliseconds)",
        " s": " (seconds)",
    }
    low = " " + key.lower()
    for suffix, repl in replacements.items():
        if low.endswith(suffix):
            key = key[: -len(suffix.strip())] + repl
            break
    return key[:1].upper() + key[1:] if key else key


def _scalar_text(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def humanize_state(value, heading=None, level=0) -> str:
    """Deterministically render arbitrary JSON-like state as readable structured text."""
    value = parse_jsonish(value)
    if isinstance(value, str):
        if heading:
            return f"{_humanize_key(heading)}:\n{value.strip()}"
        return value.strip()

    lines = []
    if heading is not None:
        lines.append(f"{_humanize_key(heading)}:")

    if isinstance(value, dict):
        for key, child in value.items():
            pretty = _humanize_key(key)
            if isinstance(child, dict):
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(f"{pretty}:")
                nested = humanize_state(child, heading=None, level=level + 1)
                lines.extend(nested.splitlines())
            elif isinstance(child, list):
                if lines and lines[-1] != "":
                    lines.append("")
                lines.append(f"{pretty}:")
                if not child:
                    lines.append("- none")
                else:
                    for item in child:
                        if isinstance(item, (dict, list)):
                            rendered = humanize_state(item, heading=None, level=level + 1).splitlines()
                            if rendered:
                                lines.append(f"- {rendered[0]}")
                                lines.extend(f"  {x}" for x in rendered[1:])
                        else:
                            lines.append(f"- {_scalar_text(item)}")
            else:
                lines.append(f"{pretty}: {_scalar_text(child)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = humanize_state(item, heading=None, level=level + 1).splitlines()
                if rendered:
                    lines.append(f"- {rendered[0]}")
                    lines.extend(f"  {x}" for x in rendered[1:])
            else:
                lines.append(f"- {_scalar_text(item)}")
    else:
        lines.append(_scalar_text(value))

    out = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return "\n".join(out).strip()


def criterion_description(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("description", "text", "label", "name"):
            if value.get(key) is not None:
                return str(value[key]).strip()
        return humanize_state(value)
    return str(value).strip()


def criteria_as_key_desc(criteria):
    """Return (keys, descriptions) preserving criterion order."""
    criteria = parse_jsonish(criteria)
    if isinstance(criteria, dict):
        keys = [str(k) for k in criteria.keys()]
        descs = [criterion_description(criteria[k]) for k in criteria.keys()]
        return keys, descs
    if isinstance(criteria, (list, tuple)):
        keys = [str(i) for i in range(len(criteria))]
        descs = [criterion_description(x) for x in criteria]
        return keys, descs
    return [], []


def format_key_description(key, desc):
    key = str(key).strip()
    desc = str(desc).strip()
    if not desc or desc.lower() == key.lower():
        return key
    return f"{key} - {desc}"


def format_score_labels(keys, descs, style_name):
    k_total = len(keys)
    labels = []
    for i, (key, desc) in enumerate(zip(keys, descs)):
        desc = str(desc).strip()
        if style_name == "key_description":
            text = format_key_description(key, desc)
        elif style_name == "score_key_description":
            text = f"score {key} - {desc}" if desc else f"score {key}"
        elif style_name == "level_key_description":
            text = f"level {key} - {desc}" if desc else f"level {key}"
        elif style_name == "rating_position_description":
            text = f"rating {i + 1} of {k_total} - {desc}" if desc else f"rating {i + 1} of {k_total}"
        elif style_name == "description_only":
            text = desc or str(key)
        elif style_name == "option_position_description":
            text = f"option {i + 1} of {k_total} - {desc}" if desc else f"option {i + 1} of {k_total}"
        else:
            raise ValueError(style_name)
        labels.append(text)
    # Description-only collides when descriptions repeat; fall back to the key form.
    if len(set(labels)) != len(labels):
        labels = [format_key_description(k, d) for k, d in zip(keys, descs)]
    return labels


def make_noul_query(instruction: str) -> str:
    instruction = str(instruction).strip()
    if instruction.endswith("?"):
        return instruction
    return f"Determine whether the following statement is correct:\n{instruction}"

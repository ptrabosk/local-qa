from __future__ import annotations
from typing import Any, Dict
import json
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "prompts"
_LLM_ONLY_TEMPLATE = _TEMPLATES_DIR / "llm_only_prompt.txt"
_LLM_DETAILED_TEMPLATE = _TEMPLATES_DIR / "llm_detailed_prompt.txt"


def _read_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_llm_only_prompt(payload: Dict[str, Any]) -> str:
    schema = {
        "understandable": 0,
        "preferred_tone_followed": 0,
        "empathy": 0,
        "personalization": 0,
        "finding": "one sentence"
    }

    template = _read_template(_LLM_ONLY_TEMPLATE)
    return template.format(
        output_schema_json=json.dumps(schema, ensure_ascii=False),
        payload_json=json.dumps(payload, ensure_ascii=False),
    )


def build_llm_detailed_prompt(payload: Dict[str, Any]) -> str:
    category_schema = {
        "score": 0,
        "justification": "one sentence",
        "flagged_words_or_phrases": ["exact phrase from audited agent message"]
    }
    schema = {
        "understandable": category_schema,
        "preferred_tone_followed": category_schema,
        "empathy": category_schema,
        "personalization": category_schema,
    }

    template = _read_template(_LLM_DETAILED_TEMPLATE)
    return template.format(
        output_schema_json=json.dumps(schema, ensure_ascii=False),
        payload_json=json.dumps(payload, ensure_ascii=False),
    )

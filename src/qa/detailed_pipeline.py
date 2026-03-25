from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Any, Dict, List

import yaml
from sentence_transformers import SentenceTransformer

from .conversation import (
    get_audited_agent_message,
    last_customer_message,
    prior_agent_messages,
    strip_system,
    trailing_customer_messages_before_audited,
)
from .grammar_typos import count_grammar_and_typos
from .schema import AuditInput
from .llm_ollama import OllamaClient, _parse_ollama_json_response
from .prompts import build_llm_detailed_prompt
from .repetition_st import repetition_check

CATEGORIES = ["understandable", "preferred_tone_followed", "empathy", "personalization"]


@lru_cache(maxsize=8)
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=16)
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _get_st_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _contains_exact_phrase(text: str, phrase: str) -> bool:
    p = phrase.strip()
    if not p:
        return False
    pattern = re.escape(p).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){pattern}(?!\w)", text, flags=re.IGNORECASE) is not None


def _find_phrase_hits(text: str, phrases: List[str]) -> List[str]:
    return [phrase for phrase in phrases if isinstance(phrase, str) and _contains_exact_phrase(text, phrase)]


def _require_int01(v: Any, path: str) -> int:
    if isinstance(v, bool):
        v = int(v)
    if not isinstance(v, int) or v not in (0, 1):
        raise ValueError(f"{path} must be 0 or 1, got: {repr(v)}")
    return v


def _require_sentence(v: Any, path: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{path} must be a string")
    txt = v.strip()
    if not txt:
        raise ValueError(f"{path} must not be empty")
    return txt


def _require_phrase_list(v: Any, path: str) -> List[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"{path} must be a list of strings")

    out: List[str] = []
    for i, item in enumerate(v):
        if not isinstance(item, str):
            raise ValueError(f"{path}[{i}] must be a string")
        phrase = item.strip()
        if phrase:
            out.append(phrase)
    return out


def _extract_detailed_scores(llm_raw: Dict[str, Any], blocklist_hits: List[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}

    for category in CATEGORIES:
        if category not in llm_raw or not isinstance(llm_raw[category], dict):
            raise ValueError(f"LLM output missing object for category '{category}'")

        cat = llm_raw[category]
        score = _require_int01(cat.get("score"), f"{category}.score")
        justification = _require_sentence(cat.get("justification"), f"{category}.justification")
        flagged = _require_phrase_list(
            cat.get("flagged_words_or_phrases"), f"{category}.flagged_words_or_phrases"
        )

        if blocklist_hits and not flagged:
            flagged = list(blocklist_hits)

        result[category] = {
            "score": score,
            "justification": justification,
            "flagged_words_or_phrases": flagged,
        }

    return result


def _generate_detailed_json(client: OllamaClient, prompt: str) -> Dict[str, Any]:
    resp = client._generate(prompt, as_json=True)
    try:
        return _parse_ollama_json_response(resp)
    except Exception:
        repair_prompt = (
            "Rewrite the text below as VALID JSON only. Return exactly one JSON object with keys "
            "understandable, preferred_tone_followed, empathy, personalization. Each key must map to an object "
            "with: score (0 or 1), justification (one sentence), flagged_words_or_phrases (array of strings).\n\n"
            f"TEXT:\n{str(resp)}"
        )
        repaired = client._generate(repair_prompt, as_json=True)
        return _parse_ollama_json_response(repaired)


def run_audit_detailed(
    audit_in: AuditInput,
    config_path: str = "config/config.yaml",
    tone_rules_path: str = "config/tone_rules.json",
    empathy_rules_path: str = "config/empathy_rules.json",
    personalization_rules_path: str = "config/personalization_rules.json",
) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    tone_rules = _load_json(tone_rules_path)
    empathy_rules = _load_json(empathy_rules_path)
    personalization_rules = _load_json(personalization_rules_path)

    conv_ns = strip_system(audit_in.conversation)
    audited_msg, prior = get_audited_agent_message(conv_ns)

    grammar_errors, typo_count = count_grammar_and_typos(
        audited_msg, language=cfg.get("grammar_tool", {}).get("language", "en-US")
    )

    st_name = cfg["models"]["repetition_embeddings"]["model"]
    st_model = _get_st_model(st_name)
    prior_agent = prior_agent_messages(prior)
    rep_max_cos, rep_examples = repetition_check(st_model, prior_agent, audited_msg)

    g_max = int(cfg["thresholds"]["correct_grammar_max_grammar_errors"])
    t_max = int(cfg["thresholds"]["no_typos_max_typos"])
    rep_thr = float(cfg["thresholds"]["no_repetition_max_cosine"])

    blocklist_hits = [w for w in audit_in.blocklisted_words if _contains_exact_phrase(audited_msg, w)]
    non_apology_patterns = empathy_rules.get("non_apology_patterns", [])
    non_apology_hits = _find_phrase_hits(audited_msg, non_apology_patterns)

    personalization_customer_messages = trailing_customer_messages_before_audited(prior)
    last_customer = (
        personalization_customer_messages[-1]
        if personalization_customer_messages
        else last_customer_message(prior)
    )

    llm_payload = {
        "id": audit_in.id,
        "preferred_tone": audit_in.preferred_tone,
        "blocklisted_words": audit_in.blocklisted_words,
        "blocklist_hits": blocklist_hits,
        "conversation": [{"role": m.role, "text": m.text} for m in conv_ns],
        "audited_agent_message": audited_msg,
        "prior_customer_message": last_customer,
        "personalization_customer_messages": personalization_customer_messages,
        "tone_rules.json": tone_rules,
        "empathy_rules.json": empathy_rules,
        "non_apology_hits": non_apology_hits,
        "personalization_rules.json": personalization_rules,
        "local_signals": {
            "repetition_max_cosine": rep_max_cos,
            "repetition_hit_examples": rep_examples,
            "no_repetition_max_cosine_threshold": rep_thr,
        },
    }

    llm_cfg = cfg["models"]["llm"]
    client = OllamaClient(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        temperature=float(llm_cfg.get("temperature", 0.0)),
        timeout_s=int(llm_cfg.get("timeout_s", 180)),
        keep_alive=llm_cfg.get("keep_alive"),
        options=llm_cfg.get("options", {}),
    )

    prompt = build_llm_detailed_prompt(llm_payload)
    llm_raw = _generate_detailed_json(client, prompt)
    detailed_scores = _extract_detailed_scores(llm_raw, blocklist_hits)
    if non_apology_hits:
        empathy_bucket = detailed_scores["empathy"]
        empathy_bucket["score"] = 0
        empathy_bucket["justification"] = (
            f"Empathy marked down due to non-apology pattern(s): {', '.join(non_apology_hits)}."
        )
        flagged = empathy_bucket.get("flagged_words_or_phrases", [])
        empathy_bucket["flagged_words_or_phrases"] = list(dict.fromkeys([*flagged, *non_apology_hits]))
        llm_raw["empathy_non_apology_hits"] = non_apology_hits
        if isinstance(llm_raw.get("empathy"), dict):
            llm_raw["empathy"]["score"] = 0
            llm_raw["empathy"]["justification"] = empathy_bucket["justification"]
            llm_raw["empathy"]["flagged_words_or_phrases"] = empathy_bucket["flagged_words_or_phrases"]

    return {
        "id": audit_in.id,
        "message_text": audited_msg,
        "preferred_tone": audit_in.preferred_tone,
        "local": {
            "grammar_error_count": grammar_errors,
            "typo_count": typo_count,
            "repetition_max_cosine": rep_max_cos,
            "repetition_hit_examples": rep_examples,
            "correct_grammar": 1 if grammar_errors <= g_max else 0,
            "no_typos": 1 if typo_count <= t_max else 0,
            "no_repetition": 1 if rep_max_cos <= rep_thr else 0,
        },
        "category_results": detailed_scores,
        "blocklist_hits": blocklist_hits,
        "llm_raw": llm_raw,
    }


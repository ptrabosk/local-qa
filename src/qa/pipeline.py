from __future__ import annotations
from typing import Any, Dict, List
import json
import re
import yaml
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from .schema import AuditInput, AuditOutput, LocalFindings, LLMFindings
from .conversation import (
    strip_system,
    get_audited_agent_message,
    prior_agent_messages,
    last_customer_message,
    trailing_customer_messages_before_audited,
)
from .grammar_typos import count_grammar_and_typos
from .repetition_st import repetition_check
from .llm_ollama import OllamaClient
from .prompts import build_llm_only_prompt

@lru_cache(maxsize=8)
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@lru_cache(maxsize=16)
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _require_int01(obj: Dict[str, Any], key: str) -> int:
    if key not in obj:
        raise ValueError(f"LLM output missing key: {key}")
    v = obj[key]
    if isinstance(v, bool):
        v = int(v)
    if not isinstance(v, int) or v not in (0, 1):
        raise ValueError(f"LLM output key '{key}' must be 0 or 1, got: {repr(obj[key])}")
    return v

def _require_one_sentence(s: Any) -> str:
    if not isinstance(s, str):
        raise ValueError(f"LLM output 'finding' must be a string, got: {type(s)}")
    txt = s.strip()
    if not txt:
        raise ValueError("LLM output 'finding' is empty")
    return txt


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

def run_audit(
    audit_in: AuditInput,
    config_path: str = "config/config.yaml",
    tone_rules_path: str = "config/tone_rules.json",
    empathy_rules_path: str = "config/empathy_rules.json",
    personalization_rules_path: str = "config/personalization_rules.json",
) -> AuditOutput:
    cfg = _load_yaml(config_path)
    tone_rules = _load_json(tone_rules_path)
    empathy_rules = _load_json(empathy_rules_path)
    personalization_rules = _load_json(personalization_rules_path)

    conv_ns = strip_system(audit_in.conversation)
    audited_msg, prior = get_audited_agent_message(conv_ns)

    # Local: grammar + typos counts
    grammar_errors, typo_count = count_grammar_and_typos(
        audited_msg, language=cfg.get("grammar_tool", {}).get("language", "en-US")
    )

    # Local: repetition (semantic) vs prior agent messages only
    st_name = cfg["models"]["repetition_embeddings"]["model"]
    st_model = _get_st_model(st_name)
    prior_agent = prior_agent_messages(prior)
    rep_max_cos, rep_examples = repetition_check(st_model, prior_agent, audited_msg)

    # Local authoritative 0/1
    g_max = int(cfg["thresholds"]["correct_grammar_max_grammar_errors"])
    t_max = int(cfg["thresholds"]["no_typos_max_typos"])
    rep_thr = float(cfg["thresholds"]["no_repetition_max_cosine"])

    local = LocalFindings(
        grammar_error_count=grammar_errors,
        typo_count=typo_count,
        repetition_max_cosine=rep_max_cos,
        repetition_hit_examples=rep_examples,
        correct_grammar=1 if grammar_errors <= g_max else 0,
        no_typos=1 if typo_count <= t_max else 0,
        no_repetition=1 if rep_max_cos <= rep_thr else 0,
    )

    # Blocklist hits (deterministic)
    blocklist_hits = [w for w in audit_in.blocklisted_words if _contains_exact_phrase(audited_msg, w)]
    non_apology_patterns = empathy_rules.get("non_apology_patterns", [])
    non_apology_hits = _find_phrase_hits(audited_msg, non_apology_patterns)

    # LLM-only payload
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
        # optional local context (do not grade these)
        "local_signals": {
            "repetition_max_cosine": rep_max_cos,
            "repetition_hit_examples": rep_examples,
            "no_repetition_max_cosine_threshold": rep_thr
        }
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

    prompt = build_llm_only_prompt(llm_payload)
    llm_raw = client.generate_json(prompt)

    empathy_score = _require_int01(llm_raw, "empathy")
    finding = _require_one_sentence(llm_raw.get("finding"))
    if non_apology_hits:
        empathy_score = 0
        llm_raw["empathy"] = 0
        llm_raw["empathy_non_apology_hits"] = non_apology_hits
        finding = (
            f"Empathy marked down due to non-apology pattern(s): {', '.join(non_apology_hits)}."
        )
        llm_raw["finding"] = finding

    llm = LLMFindings(
        understandable=_require_int01(llm_raw, "understandable"),
        preferred_tone_followed=_require_int01(llm_raw, "preferred_tone_followed"),
        empathy=empathy_score,
        personalization=_require_int01(llm_raw, "personalization"),
        finding=finding,
    )

    return AuditOutput(
        id=audit_in.id,
        preferred_tone=audit_in.preferred_tone,
        blocklisted_words=audit_in.blocklisted_words,
        conversation=conv_ns if cfg["output"]["include_full_conversation_json"] else [],
        audited_agent_message=audited_msg,
        local=local,
        llm=llm,
        llm_raw=llm_raw,
    )

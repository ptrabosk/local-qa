from __future__ import annotations
from typing import List, Tuple
from .schema import Message
from .normalize import normalize_text

def strip_system(conversation: List[Message]) -> List[Message]:
    return [m.model_copy(update={"text": normalize_text(m.text)}) for m in conversation if m.role != "system"]

def get_audited_agent_message(conversation_no_system: List[Message]) -> Tuple[str, List[Message]]:
    last_agent_idx = None
    for i in range(len(conversation_no_system) - 1, -1, -1):
        if conversation_no_system[i].role == "agent":
            last_agent_idx = i
            break
    if last_agent_idx is None:
        raise ValueError("No agent message found to audit.")
    audited = conversation_no_system[last_agent_idx].text
    prior = conversation_no_system[:last_agent_idx]
    return audited, prior

def prior_agent_messages(prior: List[Message]) -> List[str]:
    return [m.text for m in prior if m.role == "agent"]

def last_customer_message(messages: List[Message]) -> str:
    for m in reversed(messages):
        if m.role == "customer":
            return m.text
    return ""


def trailing_customer_messages_before_audited(prior_messages: List[Message]) -> List[str]:
    trailing: List[str] = []
    for m in reversed(prior_messages):
        if m.role == "customer":
            trailing.append(m.text)
            continue
        if trailing:
            break
    trailing.reverse()
    return trailing

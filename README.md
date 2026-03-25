### Local QA (deterministic + LLM-only categories) with Llama3.1:8b

This project audits the **last agent message** in a conversation (system messages ignored).

Authoritative local categories (LLM cannot override):
- correct_grammar (LanguageTool non-spelling matches)
- no_typos (LanguageTool spelling matches)
- no_repetition (sentence-transformers semantic similarity vs prior agent messages)

LLM-only categories (Llama3.1:8b via Ollama):
- understandable
- preferred_tone_followed (uses config/tone_rules.json)
- empathy (uses config/empathy_rules.json + prior customer message)
- personalization (uses config/personalization_rules.json)

Output:
- echoes inputs (id, preferred_tone, blocklisted_words, conversation)
- local findings + 0/1 authoritative scores
- llm 0/1 scores for the four LLM categories + **one sentence** finding
- llm_raw included for debugging

Detailed output option (separate script):
- `python scripts/run_audit_detailed.py ...`
- returns `id`, audited `message_text`, `preferred_tone`
- returns per-category details for `understandable`, `preferred_tone_followed`, `empathy`, `personalization`:
  - `score` (0/1)
  - `justification` (separate sentence per category)
  - `flagged_words_or_phrases` (separate list per category)

Requirements:
- Python 3.10+
- Ollama running locally
- Pull the model:
  - ollama run llama3.1:8b

Install:
- python -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt

Run:
- pip install -r requirements.txt
- python scripts/run_audit.py examples/input.csv
- python scripts/run_audit.py examples/input.csv --row 2
- python scripts/run_audit.py examples/input.csv --send-id 0192d5dd-50f3-46a9-f000-00009aac14d9
- python scripts/run_audit_detailed.py examples/input.csv
- python scripts/run_audit_detailed.py examples/input.csv --row 2

Config:
- config/config.yaml controls:
  - LLM model name/url/temperature
  - sentence-transformers model for repetition
  - thresholds for local pass/fail

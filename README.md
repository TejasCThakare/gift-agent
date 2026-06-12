# Gift Agent — Hyper-Personalised AI Gift Recommendation System

An AI workflow that takes enriched LinkedIn-style contact data and recommends the top 3 personalised gifts with real purchasable links, evidence-grounded reasoning, and a human-in-the-loop review step.

---

## Quick Start

```bash
# 1. Install Ollama (primary LLM — fully local, no API key)
curl -fsSL https://ollama.ai/install.sh | sh    # macOS/Linux
# Windows: https://ollama.ai/download

# 2. Pull Qwen3
ollama pull qwen3

# 3. Clone and set up
git clone <repo> gift-agent && cd gift-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Ollama needs no key — just have it running

# 5. Start API server
uvicorn api.app:app --reload --port 8000

# 6. Open review UI
open http://localhost:8000

# 7. Run CLI
python main.py --input data/sample_input.json

# 8. Run tests
pytest tests/ -v
```

---

## LLM Providers

The system uses a **priority-order auto-fallback** provider chain:

| Priority | Provider | Requirement | Cost |
|---|---|---|---|
| 1 (primary) | Ollama + Qwen3 | Ollama running locally | Free |
| 2 (fallback #1) | Groq | `GROQ_API_KEY` in `.env` | Free tier |
| 3 (fallback #2) | Gemini Flash | `GEMINI_API_KEY` in `.env` | Free tier |

**The project runs entirely with Ollama.** Groq and Gemini are optional.

```bash
# Get free Groq key: https://console.groq.com
# Get free Gemini key: https://aistudio.google.com
```

---

## Architecture

### Workflow (9 nodes + 2 branch nodes)

```
ingest → extract_and_query → filter_signals → search_products → validate_products
                                                                        ↓
                                              [conditional: ≥3 valid products?]
                                                   YES ↓          NO → retry_widen
                                              score_products          (max 2 retries)
                                                   ↓            → escalate if exhausted
                                              rank_gifts
                                                   ↓
                                           generate_messages
                                                   ↓
                                         human_review [interrupt]
                                           ↓    ↓    ↓    ↓
                                        done reject edit regen
```

### Node responsibilities

| Node | LLM | Purpose |
|---|---|---|
| `ingest` | No | Parse, validate, normalize contact |
| `extract_and_query` | Yes (#1) | Extract evidence-grounded signals + search queries |
| `filter_signals` | No | Blocklist safety filter (religion, health, politics...) |
| `search_products` | No | DuckDuckGo search (no API key) |
| `validate_products` | No | 6 weighted validation rules |
| `score_products` | No | Deterministic confidence formula |
| `rank_gifts` | Yes (#2) | Select top 3 with grounded reasoning |
| `generate_messages` | Yes (#3) | Personalised gift messages |
| `human_review` | No | LangGraph interrupt — approve/reject/edit/regenerate |

### Key design decisions

**Hallucination prevention for URLs**: The LLM never constructs URLs. All URLs come from DuckDuckGo search results. The `rank_gifts` LLM selects from a pre-built list of scored products and copies URLs exactly.

**Deterministic confidence scores**: Confidence is a weighted formula computed in `score_products` (no LLM). The formula is:
```
confidence = signal×0.35 + budget×0.25 + country×0.20 + search_quality×0.10 + relationship×0.10
```
LLMs receive confidence as read-only context and cannot modify it.

**Evidence grounding**: `evidence_map` maps every signal to exact profile quotes. `rank_gifts` and `generate_messages` prompts require citing from `evidence_map`. No unsupported inferences.

**Weighted validation**: Products are scored on 6 rules (not all-or-nothing). Products pass at ≥0.75, partial pass at ≥0.40. This prevents discarding good products because of one minor failure.

**Human review routing**: Rejection routes to `rank_gifts` (not `ingest`). Regeneration routes to `score_products`. The profile doesn't change — no need to re-run extraction or search.

**Relationship scoring**: Rule-based lookup table for 7 relationship types. Deterministic, explainable, not hidden in prompts.

---

## API

```
POST /run                    Start workflow for a contact
POST /review/{thread_id}     Submit review action (approve/reject/edit/regenerate)
GET  /status/{thread_id}     Get current workflow state
GET  /runs                   List all past runs
GET  /health                 LLM provider availability check
GET  /docs                   Swagger UI
```

### Example: start workflow

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d @data/sample_input.json
```

Response:
```json
{
  "thread_id": "run_abc123def456",
  "status": "awaiting_review",
  "review_payload": { ... },
  "message": "Recommendations ready for review"
}
```

### Example: approve

```bash
curl -X POST http://localhost:8000/review/run_abc123def456 \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}'
```

### Example: reject with reason

```bash
curl -X POST http://localhost:8000/review/run_abc123def456 \
  -H "Content-Type: application/json" \
  -d '{"action": "reject", "reason": "All gifts are tech accessories — try books instead"}'
```

---

## Output Schema

```json
{
  "contact_name": "Aarav Mehta",
  "profile_signals": {
    "strong_signals": ["interest in cricket", "reading The Challenger Sale"],
    "weak_signals": ["may appreciate business books"],
    "signals_to_avoid": ["..."]
  },
  "search_trace": {
    "queries_used": ["cricket gift India under 5000 rupees", "..."],
    "products_considered_count": 12,
    "validated_count": 7,
    "escalation_triggered": false,
    "retry_count": 0
  },
  "recommended_gifts": [
    {
      "rank": 1,
      "gift_name": "SG Cricket Bat",
      "product_url": "https://www.amazon.in/dp/B09ABC12345",
      "store": "Amazon.in",
      "estimated_price": "₹3,499",
      "why_this_gift": "...",
      "personalisation_reasoning": "...",
      "evidence_citations": ["What a match! MI vs CSK — cricket is truly a religion"],
      "personalised_message": "...",
      "confidence_score": 0.82,
      "risk_level": "low",
      "assumptions": ["..."]
    }
  ],
  "human_review": {
    "status": "approved",
    "available_actions": []
  }
}
```

---

## Artifacts

All workflow artifacts are persisted to `artifacts/{thread_id}/`:

| File | Contents |
|---|---|
| `state.json` | Complete workflow state snapshot |
| `recommendations.json` | Final output matching assignment schema |
| `review_history.json` | All review actions with timestamps |
| `trace.json` | Evidence map, signals, queries, products |
| `logs.json` | Per-node latency, tokens, LLM calls |

---

## Tests

```bash
# All tests
pytest tests/ -v

# Specific test files
pytest tests/test_ingest.py -v
pytest tests/test_filter_signals.py -v
pytest tests/test_score_products.py -v
pytest tests/test_validate_products.py -v
pytest tests/test_relationship_scoring.py -v
```

Tests cover:
- Contact validation and normalization
- Safety filter correctness (blocklist coverage)
- Confidence formula determinism
- Weight sum validation
- Price extraction edge cases
- Domain trust checking
- Relationship scoring table completeness

---

## Evaluation Note

This project was built to demonstrate:

1. **Multi-step AI orchestration** with LangGraph — typed state, conditional edges, retry/escalation branches, human-in-the-loop interrupt
2. **Grounding** — every recommendation cites exact profile evidence; no hallucinated signals
3. **Hallucination prevention** — URLs from search only, confidence scores from deterministic formula only
4. **Safety** — deterministic blocklist filter for 6 sensitive categories, not LLM-based
5. **Auditability** — every decision traced to evidence_map, all artifacts persisted as JSON
6. **Human-in-the-loop** — full approve/reject/edit/regenerate cycle with review history
7. **Provider flexibility** — Ollama (local), Groq, Gemini fallback chain; runs entirely locally

---

## File Structure

```
gift-agent/
├── agent/           LangGraph nodes and graph definition
├── prompts/         LLM prompt templates
├── models/          Pydantic data models
├── services/        LLM providers + DuckDuckGo search
├── utils/           Pricing, validation, logging utilities
├── api/             FastAPI endpoints
├── ui/              Single-file review UI
├── storage/         JSON artifact persistence
├── data/            Sample input/output
├── tests/           Unit tests
└── artifacts/       Runtime-generated workflow artifacts
```

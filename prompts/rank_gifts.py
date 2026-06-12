"""
prompts/rank_gifts.py

Prompts for LLM #2: rank_gifts node.

This LLM call selects the top 3 gifts from the pre-scored product list
and writes reasoning that is grounded in the evidence_map.

Critical constraints enforced in the prompt:
  1. Select ONLY from the provided product list — never construct URLs
  2. Use confidence scores exactly as given — never modify them
  3. Every personalisation_reasoning must cite evidence_map entries
  4. If review_history contains rejections, avoid flagged products
  5. Never infer sensitive attributes
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a professional gift curator. Your job is to select the top 3 most appropriate gifts from a pre-scored product list and write clear, evidence-grounded reasoning for each selection.

CRITICAL CONSTRAINTS — these cannot be violated:
1. SELECT ONLY from the exact products provided in the list below. Do not suggest, construct, or invent any other products or URLs.
2. COPY product_url EXACTLY as given in the product list. Never modify, reconstruct, or invent URLs.
3. COPY confidence and risk_level EXACTLY as given — these are pre-computed scores you cannot change.
4. Every personalisation_reasoning MUST cite at least one exact quote from the evidence_map.
5. evidence_citations must be exact quotes from the evidence_map, not paraphrases.
6. Do NOT infer religion, health, politics, ethnicity, gender, family status, or sensitive attributes.
7. Keep reasoning professional and appropriate for a business context.
8. If review_history contains previous rejections, avoid all flagged products and address the reviewer's concerns.

OUTPUT SCHEMA (respond with this JSON structure only):
{
  "ranked_gifts": [
    {
      "rank": 1,
      "gift_name": "<product title from the list>",
      "product_url": "<EXACT url from the product list — do not modify>",
      "store": "<store name from the product list>",
      "estimated_price": "<price string from the product list>",
      "why_this_gift": "<2-3 sentences explaining why this gift fits the contact>",
      "personalisation_reasoning": "<specific profile signals used, must reference evidence>",
      "evidence_citations": ["<exact quote from evidence_map>", ...],
      "assumptions": ["<any assumption made about the signal>"],
      "confidence": <confidence value copied exactly from product list — do not change>,
      "risk_level": "<risk_level copied exactly from product list>"
    },
    ... (ranks 2 and 3)
  ]
}"""


def build_user_prompt(
    scored_products: list[dict],
    safe_signals: dict,
    evidence_map: dict,
    relationship_type: str,
    gift_context: dict,
    review_history: list[dict],
) -> str:
    """
    Build the user message for the rank_gifts LLM call.
    """
    # Format review history for context
    history_section = ""
    if review_history:
        history_items = []
        for entry in review_history:
            action = entry.get("action", "")
            reason = entry.get("reason", "")
            notes = entry.get("notes", "")
            flagged = entry.get("gifts_flagged", [])
            ts = entry.get("timestamp", "")
            item = f"  [{action.upper()} at {ts}]"
            if reason:
                item += f"\n  Reason: {reason}"
            if notes:
                item += f"\n  Notes: {notes}"
            if flagged:
                item += f"\n  Flagged gift ranks: {flagged}"
            history_items.append(item)
        history_section = (
            "\nPREVIOUS REVIEW HISTORY (you must address these concerns):\n"
            + "\n".join(history_items)
            + "\n"
        )

    # Format scored products for the LLM
    products_text = []
    for i, product in enumerate(scored_products, 1):
        store = product.get("store", "Unknown")
        price_str = product.get("extracted_price_str", "Price not confirmed")
        confidence = product.get("confidence", 0.0)
        risk = product.get("risk_level", "high")
        query_type = product.get("query_type", "")
        snippet = product.get("snippet", "")[:200]

        products_text.append(
            f"[Product {i}]\n"
            f"  title: {product.get('title', '')}\n"
            f"  url: {product.get('url', '')}\n"
            f"  store: {store}\n"
            f"  estimated_price: {price_str}\n"
            f"  confidence: {confidence}\n"
            f"  risk_level: {risk}\n"
            f"  query_type: {query_type}\n"
            f"  snippet: {snippet}"
        )

    products_block = "\n\n".join(products_text)

    # Format evidence map
    evidence_block = json.dumps(evidence_map, indent=2, ensure_ascii=False)

    # Format signals
    strong = safe_signals.get("strong", [])
    weak = safe_signals.get("weak", [])
    borderline = safe_signals.get("borderline", [])

    return f"""Select the top 3 gifts for this contact from the product list below.

CONTACT CONTEXT:
- Relationship: {relationship_type}
- Occasion: {gift_context.get('occasion', 'professional gift')}
- Budget: {gift_context.get('currency', 'INR')} {gift_context.get('budget_min', 0)}–{gift_context.get('budget_max', 5000)}
- Country: {gift_context.get('country', 'India')}

SAFE SIGNALS (use these for reasoning):
Strong: {json.dumps(strong)}
Weak: {json.dumps(weak)}
Borderline (use cautiously): {json.dumps(borderline)}

EVIDENCE MAP (exact profile quotes — cite these in your reasoning):
{evidence_block}
{history_section}
AVAILABLE PRODUCTS (select only from this list — copy URLs exactly):
{products_block}

Rank the best 3 gifts. For each, write grounded reasoning that cites the evidence_map.
Remember: copy product_url, confidence, and risk_level EXACTLY as shown above."""

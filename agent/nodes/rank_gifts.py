"""
agent/nodes/rank_gifts.py

Node 7: rank_gifts (LLM #2)

Responsibilities:
  - Select top 3 gifts from scored_products list
  - Write evidence-grounded reasoning for each selection
  - Cite exact evidence_map entries in personalisation_reasoning
  - Copy confidence and risk_level from scored_products (never modify)
  - Copy product_url from scored_products (never construct)
  - Read review_history to avoid previously rejected products

The LLM receives:
  - scored_products (pre-scored, sorted by confidence)
  - safe_signals + evidence_map (for grounding)
  - review_history (if any previous rejections)

The LLM cannot:
  - Change confidence scores
  - Construct or modify URLs
  - Introduce signals not in evidence_map
  - Select products not in scored_products
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from agent.state import GiftAgentState
from models.recommendations import RankedGift
from prompts.rank_gifts import SYSTEM_PROMPT, build_user_prompt
from services.llm.factory import get_provider
from services.llm.base import LLMProviderError
from utils.logging import get_logger, log_node_start, log_node_end

logger = get_logger("nodes.rank_gifts")

MAX_RETRIES = 2


def rank_gifts(state: GiftAgentState) -> dict:
    """
    Use LLM to rank top 3 gifts with evidence-grounded reasoning.
    Returns partial state update dict.
    """
    logs = log_node_start(state.get("logs", {}), "rank_gifts")

    scored_products = state.get("scored_products", [])
    safe_signals = state.get("safe_signals", {})
    evidence_map = state.get("evidence_map", {})
    relationship_type = state.get("relationship_type", "unknown")
    contact = state.get("contact", {})
    gift_context = contact.get("gift_context", {})
    review_history = state.get("review_history", [])

    if not scored_products:
        logger.error("No scored products available for ranking")
        logs = log_node_end(logs, "rank_gifts", error="No scored products")
        return {
            "ranked_gifts": [],
            "logs": logs,
        }

    provider = get_provider()

    user_prompt = build_user_prompt(
        scored_products=scored_products,
        safe_signals=safe_signals,
        evidence_map=evidence_map,
        relationship_type=relationship_type,
        gift_context=gift_context,
        review_history=[rh if isinstance(rh, dict) else rh for rh in review_history],
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    ranked_gifts: list[RankedGift] = []
    total_tokens = 0
    last_error = ""

    for attempt in range(MAX_RETRIES + 1):
        try:
            raw_response, tokens = provider.complete_json(
                messages=messages,
                temperature=0.3,
                max_tokens=3000,
            )
            total_tokens += tokens

            cleaned = _strip_markdown_json(raw_response)
            data = json.loads(cleaned)

            raw_gifts = data.get("ranked_gifts", [])
            if not raw_gifts:
                raise ValueError("LLM returned empty ranked_gifts list")

            # Build and validate RankedGift objects
            # IMPORTANT: confidence and risk_level are taken from scored_products,
            # not from LLM output, to prevent hallucination.
            ranked_gifts = _build_ranked_gifts(
                llm_gifts=raw_gifts,
                scored_products=scored_products,
            )

            if not ranked_gifts:
                raise ValueError("Could not match any LLM gifts to scored products")

            logger.info(
                "Ranked %d gifts. Top: %s (conf=%.3f)",
                len(ranked_gifts),
                ranked_gifts[0].gift_name[:40] if ranked_gifts else "N/A",
                ranked_gifts[0].confidence if ranked_gifts else 0.0,
            )
            break

        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"Parse/validation error (attempt {attempt + 1}): {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw_response if "raw_response" in dir() else ""})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Error: {e}. Please try again. "
                        "Ensure you select exactly 3 gifts from the product list "
                        "and copy URLs and confidence values exactly as provided."
                    ),
                })

        except LLMProviderError as e:
            last_error = f"LLM error: {e}"
            logger.error(last_error)
            break

    # Fallback: if LLM failed, take top 3 scored products directly
    if not ranked_gifts:
        logger.warning(
            "rank_gifts LLM failed — using top 3 by confidence score as fallback"
        )
        ranked_gifts = _fallback_ranking(scored_products)
        last_error = f"LLM ranking failed — using score-based fallback. Last error: {last_error}"

    logs = log_node_end(
        logs, "rank_gifts",
        tokens_used=total_tokens,
        llm_calls=1,
        error=last_error if last_error else "",
    )

    return {
        "ranked_gifts": [g.model_dump() for g in ranked_gifts],
        "logs": logs,
    }


def _build_ranked_gifts(
    llm_gifts: list[dict],
    scored_products: list[dict],
) -> list[RankedGift]:
    """
    Build RankedGift objects, enforcing that confidence and URL
    come from scored_products rather than LLM output.

    Matches LLM selections to scored_products by URL or title.
    """
    # Build lookup index for scored products
    url_to_product = {p["url"]: p for p in scored_products}
    title_to_product = {p["title"].lower(): p for p in scored_products}

    ranked: list[RankedGift] = []

    for llm_gift in llm_gifts[:3]:
        # Find the matching scored product
        llm_url = llm_gift.get("product_url", "")
        llm_title = llm_gift.get("gift_name", "").lower()

        scored = (
            url_to_product.get(llm_url)
            or title_to_product.get(llm_title)
            or _fuzzy_match(llm_title, scored_products)
        )

        if scored is None:
            # Try to use the best unranked product
            used_urls = {r.product_url for r in ranked}
            for p in scored_products:
                if p["url"] not in used_urls:
                    scored = p
                    break

        if scored is None:
            logger.warning("Could not match LLM gift '%s' to any scored product", llm_title)
            continue

        # Build the RankedGift — confidence and URL from scored_products
        price_str = (
            scored.get("extracted_price_str")
            or llm_gift.get("estimated_price", "Price not confirmed")
        )

        ranked.append(RankedGift(
            rank=llm_gift.get("rank", len(ranked) + 1),
            gift_name=scored.get("title") or llm_gift.get("gift_name", ""),
            product_url=scored["url"],  # ALWAYS from scored_products
            store=scored.get("store") or llm_gift.get("store", ""),
            estimated_price=price_str,
            why_this_gift=llm_gift.get("why_this_gift", ""),
            personalisation_reasoning=llm_gift.get("personalisation_reasoning", ""),
            evidence_citations=llm_gift.get("evidence_citations", []),
            assumptions=llm_gift.get("assumptions", []),
            confidence=scored["confidence"],  # ALWAYS from scored_products
            risk_level=scored["risk_level"],   # ALWAYS from scored_products
        ))

    # Ensure ranks are sequential
    for i, gift in enumerate(ranked):
        ranked[i] = gift.model_copy(update={"rank": i + 1})

    return ranked


def _fuzzy_match(
    llm_title_lower: str,
    scored_products: list[dict],
) -> dict | None:
    """Find the scored product whose title best matches the LLM title."""
    best_match = None
    best_score = 0

    for p in scored_products:
        product_title_lower = p["title"].lower()
        # Count shared words
        llm_words = set(llm_title_lower.split())
        prod_words = set(product_title_lower.split())
        shared = len(llm_words & prod_words)
        if shared > best_score:
            best_score = shared
            best_match = p

    return best_match if best_score >= 2 else None


def _fallback_ranking(scored_products: list[dict]) -> list[RankedGift]:
    """Generate ranked gifts directly from scored products without LLM reasoning."""
    ranked = []
    for i, product in enumerate(scored_products[:3], 1):
        ranked.append(RankedGift(
            rank=i,
            gift_name=product.get("title", ""),
            product_url=product["url"],
            store=product.get("store", ""),
            estimated_price=product.get("extracted_price_str") or "Price not confirmed",
            why_this_gift=(
                "This product was selected based on the highest confidence score "
                "from the validation and scoring pipeline. "
                "Manual reasoning was unavailable due to a system error."
            ),
            personalisation_reasoning=(
                "Selected by deterministic scoring. Manual reasoning unavailable."
            ),
            evidence_citations=[],
            assumptions=["Automated fallback ranking — human review strongly recommended"],
            confidence=product["confidence"],
            risk_level=product["risk_level"],
        ))
    return ranked


def _strip_markdown_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()

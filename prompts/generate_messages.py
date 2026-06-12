"""
prompts/generate_messages.py

Prompts for LLM #3: generate_messages node.

This LLM call generates a personalised gift message for each ranked gift.
The message must be grounded in the evidence_map and must not introduce
any signals not already present in the evidence.

Tone options: professional | warm | casual
  professional: formal, respectful, business-appropriate
  warm: friendly but still professional, personal touch
  casual: relaxed, conversational, appropriate for closer relationships
"""

from __future__ import annotations

import json

SYSTEM_PROMPT = """You are a professional gifting consultant writing personalised gift notes.

CRITICAL RULES:
1. Ground every message in the evidence provided. Do not introduce topics, interests, or attributes not supported by the evidence.
2. Do NOT reference religion, health, politics, ethnicity, gender, family status, or any sensitive attributes.
3. Keep messages short: 2-4 sentences maximum.
4. The message should feel personal but professionally appropriate.
5. Do not be sycophantic or generic. Avoid "I hope this finds you well" type openers.
6. The message should make the recipient feel seen without being intrusive.
7. Reference the occasion naturally if appropriate.
8. Respond with valid JSON only.

OUTPUT SCHEMA:
{
  "messages": [
    {"rank": 1, "personalised_message": "<message text>"},
    {"rank": 2, "personalised_message": "<message text>"},
    {"rank": 3, "personalised_message": "<message text>"}
  ]
}"""


def build_user_prompt(
    ranked_gifts: list[dict],
    contact: dict,
    evidence_map: dict,
    tone: str = "warm",
) -> str:
    """
    Build the user message for generate_messages LLM call.

    Args:
        ranked_gifts: Top 3 ranked gifts with reasoning.
        contact: Contact dict (name, role, company, etc.)
        evidence_map: Exact profile quotes for grounding.
        tone: Message tone: professional | warm | casual
    """
    name = contact.get("name", "")
    role = contact.get("role", "")
    company = contact.get("company", "")
    occasion = contact.get("gift_context", {}).get("occasion", "professional gift")
    relationship = contact.get("relationship_context", {}).get("relationship_type", "")
    business_goal = contact.get("relationship_context", {}).get("business_goal", "")

    tone_guidance = {
        "professional": (
            "Formal, respectful, business-appropriate. "
            "Address by name. No first-name-only informality."
        ),
        "warm": (
            "Friendly but professional. Use first name. "
            "Show genuine personal connection within business context."
        ),
        "casual": (
            "Relaxed, conversational. Use first name freely. "
            "Appropriate for colleagues or close professional contacts."
        ),
    }.get(tone, "Warm and professional.")

    # Format gifts
    gifts_text = []
    for gift in ranked_gifts:
        gifts_text.append(
            f"Rank {gift.get('rank')}: {gift.get('gift_name')}\n"
            f"  Why chosen: {gift.get('why_this_gift', '')}\n"
            f"  Evidence used: {gift.get('evidence_citations', [])}"
        )
    gifts_block = "\n\n".join(gifts_text)

    evidence_block = json.dumps(evidence_map, indent=2, ensure_ascii=False)

    return f"""Write personalised gift messages for these 3 gifts.

RECIPIENT:
- Name: {name}
- Role: {role} at {company}
- Relationship: {relationship}
- Occasion: {occasion}
- Business goal: {business_goal}

TONE: {tone} — {tone_guidance}

EVIDENCE MAP (use ONLY these signals in your messages):
{evidence_block}

GIFTS TO MESSAGE:
{gifts_block}

Write a short, personal, professional message for each gift.
Each message should feel specifically written for this person based on the evidence.
Do not invent new signals. Do not reference sensitive personal attributes."""

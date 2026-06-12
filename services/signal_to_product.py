"""
services/signal_to_product.py

Deterministic signal-to-product-category mapping layer.
Every query is scoped to amazon.in or flipkart.com to ensure
results are actual product pages, not articles or blog posts.
"""

from __future__ import annotations

from typing import Optional


INTEREST_TO_PRODUCT: list[tuple[list[str], str, list[str]]] = [
    (["cricket", "ipl", "csk", "mi ", "rcb", "kkr", "bat", "wicket"],
     "cricket",
     ["SG cricket bat site:amazon.in",
      "cricket kit set site:amazon.in",
      "SS cricket bat site:flipkart.com"]),

    (["football", "soccer", "fifa"],
     "football",
     ["football gift site:amazon.in",
      "football accessories site:flipkart.com"]),

    (["tennis", "badminton", "squash"],
     "racket sport",
     ["badminton racket gift site:amazon.in",
      "yonex badminton racket site:flipkart.com"]),

    (["running", "marathon", "triathlon"],
     "running",
     ["fitness tracker gift site:amazon.in",
      "running shoes gift site:flipkart.com"]),

    (["chess", "board game", "strategy game"],
     "board games",
     ["chess set premium site:amazon.in",
      "strategy board game site:flipkart.com"]),

    (["cycling", "bike", "bicycle"],
     "cycling",
     ["cycling accessories site:amazon.in",
      "cycling gear gift site:flipkart.com"]),

    (["python", "programming", "developer", "software engineer",
      "coding", "open source", "github", "kubernetes", "linux",
      "devops", "backend", "frontend"],
     "tech/programming",
     ["mechanical keyboard site:amazon.in",
      "programming book site:amazon.in",
      "tech accessories gift site:flipkart.com"]),

    (["machine learning", "ml ", "ai ", "data science",
      "deep learning", "nlp", "computer vision", "neural network"],
     "ML/AI",
     ["artificial intelligence book site:amazon.in",
      "data science book gift site:amazon.in",
      "tech notebook premium site:amazon.in"]),

    (["cloud", "aws", "gcp", "azure", "infrastructure",
      "kubernetes", "docker", "microservices"],
     "cloud/infra",
     ["mechanical keyboard gift site:amazon.in",
      "tech book cloud site:amazon.in",
      "premium notebook professional site:amazon.in"]),

    (["reading", "books", "book club", "author", "novel",
      "non-fiction", "business book", "challenger", "leadership book"],
     "books",
     ["business book bestseller site:amazon.in",
      "leadership book gift site:amazon.in",
      "book set premium site:flipkart.com"]),

    (["podcast", "speaker", "conference", "keynote", "public speaking"],
     "speaker/thought leader",
     ["leather notebook premium site:amazon.in",
      "professional diary 2025 site:amazon.in"]),

    (["music", "guitar", "piano", "musician", "singer", "concert"],
     "music",
     ["headphones premium site:amazon.in",
      "bluetooth speaker gift site:flipkart.com"]),

    (["photography", "camera", "photographer"],
     "photography",
     ["camera accessories site:amazon.in",
      "photography book site:amazon.in"]),

    (["art", "painting", "sketch", "design", "creative"],
     "art/design",
     ["art supplies premium site:amazon.in",
      "sketchbook professional site:flipkart.com"]),

    (["sales", "revenue", "crm", "business development"],
     "sales professional",
     ["business book sales site:amazon.in",
      "premium pen set site:amazon.in",
      "leather notebook site:flipkart.com"]),

    (["startup", "founder", "entrepreneur", "venture"],
     "founder/entrepreneur",
     ["startup book site:amazon.in",
      "premium notebook site:amazon.in",
      "business card holder leather site:flipkart.com"]),

    (["finance", "investment", "trading", "stocks", "equity", "banking"],
     "finance",
     ["finance book site:amazon.in",
      "premium pen set site:amazon.in"]),

    (["marketing", "brand", "content", "social media", "growth"],
     "marketing",
     ["marketing book site:amazon.in",
      "premium notebook site:flipkart.com"]),

    (["product management", "product manager", "pm "],
     "product management",
     ["product management book site:amazon.in",
      "premium notebook site:amazon.in"]),

    (["travel", "backpacking", "adventure", "trekking", "hiking"],
     "travel",
     ["travel accessories site:amazon.in",
      "travel wallet leather site:flipkart.com"]),

    (["coffee", "café", "espresso"],
     "coffee",
     ["coffee gift set site:amazon.in",
      "coffee maker premium site:flipkart.com"]),

    (["tea", "chai"],
     "tea",
     ["tea gift set premium site:amazon.in",
      "tea collection site:flipkart.com"]),

    (["cooking", "chef", "food", "culinary", "baking"],
     "cooking",
     ["kitchen accessories gift site:amazon.in",
      "cooking gift set site:flipkart.com"]),

    (["yoga", "meditation", "mindfulness", "wellness"],
     "wellness",
     ["yoga mat premium site:amazon.in",
      "wellness gift set site:flipkart.com"]),

    (["gaming", "gamer", "esports", "video game"],
     "gaming",
     ["gaming headset site:amazon.in",
      "gaming accessories site:flipkart.com"]),

    (["sustainability", "environment", "climate", "green", "eco"],
     "sustainability",
     ["eco friendly gift set site:amazon.in",
      "sustainable gift premium site:flipkart.com"]),
]

ROLE_TO_PRODUCT: list[tuple[list[str], list[str]]] = [
    (["cto", "chief technology", "vp engineering",
      "head of engineering", "engineering director"],
     ["mechanical keyboard site:amazon.in",
      "tech book gift site:amazon.in",
      "desk accessory premium site:amazon.in"]),

    (["ceo", "chief executive", "managing director", "md ", "president"],
     ["leather notebook premium site:amazon.in",
      "luxury pen set site:amazon.in",
      "executive gift set site:flipkart.com"]),

    (["cfo", "chief financial", "finance director", "vp finance"],
     ["premium pen set site:amazon.in",
      "executive desk accessory site:amazon.in"]),

    (["vp sales", "sales director", "head of sales",
      "sales manager", "account executive"],
     ["business book site:amazon.in",
      "premium notebook leather site:amazon.in",
      "pen set gift site:flipkart.com"]),

    (["designer", "ux", "ui ", "product design", "creative director"],
     ["sketchbook premium site:amazon.in",
      "design book site:amazon.in"]),

    (["data scientist", "data analyst", "ml engineer",
      "ai engineer", "research scientist"],
     ["data science book site:amazon.in",
      "mechanical keyboard site:amazon.in",
      "tech notebook premium site:amazon.in"]),

    (["doctor", "physician", "surgeon", "medical"],
     ["premium pen set site:amazon.in",
      "professional organiser site:amazon.in"]),

    (["lawyer", "advocate", "legal", "counsel"],
     ["premium notebook leather site:amazon.in",
      "law book site:amazon.in"]),

    (["teacher", "professor", "educator", "academic"],
     ["book set gift site:amazon.in",
      "premium notebook site:amazon.in"]),

    (["journalist", "writer", "author", "editor"],
     ["premium notebook leather site:amazon.in",
      "fountain pen site:amazon.in"]),
]

OCCASION_TO_PRODUCT: dict[str, list[str]] = {
    "birthday": ["premium gift set site:amazon.in"],
    "diwali": ["diwali gift hamper site:amazon.in"],
    "new year": ["premium gift set site:amazon.in"],
    "onboarding": ["welcome kit site:amazon.in"],
    "promotion": ["premium gift set site:amazon.in"],
    "farewell": ["farewell gift premium site:amazon.in"],
    "work anniversary": ["work anniversary gift site:amazon.in"],
    "thank you": ["thank you gift premium site:amazon.in"],
}


def get_product_queries(
    strong_signals: list[str],
    weak_signals: list[str],
    role: str,
    occasion: str,
    budget_max: float,
    relationship_type: str = "unknown",
) -> list[dict]:
    """
    Generate site-scoped product queries from signals and role.
    All queries target amazon.in or flipkart.com directly.
    """
    queries = []
    budget_str = f"under {int(budget_max)} rupees"
    seen_categories: set[str] = set()

    # 1. Strong signals
    for signal in strong_signals:
        category, search_terms = _signal_to_category(signal)
        if category and category not in seen_categories:
            seen_categories.add(category)
            for term in search_terms[:2]:
                queries.append({
                    "query": f"{term} {budget_str}",
                    "type": "strong",
                    "signal_used": signal,
                    "product_category": category,
                })
            if len(queries) >= 4:
                break

    # 2. Weak signals
    for signal in weak_signals:
        category, search_terms = _signal_to_category(signal)
        if category and category not in seen_categories:
            seen_categories.add(category)
            queries.append({
                "query": f"{search_terms[0]} {budget_str}",
                "type": "weak",
                "signal_used": signal,
                "product_category": category,
            })
            if len(queries) >= 6:
                break

    # 3. Role-based fallback
    role_terms = _role_to_search_terms(role)
    for term in role_terms[:2]:
        queries.append({
            "query": f"{term} {budget_str}",
            "type": "fallback",
            "signal_used": f"role: {role}",
            "product_category": "professional gift",
        })

    # 4. Occasion override
    occasion_lower = occasion.lower() if occasion else ""
    for occ_key, occ_terms in OCCASION_TO_PRODUCT.items():
        if occ_key in occasion_lower:
            queries.insert(0, {
                "query": f"{occ_terms[0]} {budget_str}",
                "type": "strong",
                "signal_used": f"occasion: {occasion}",
                "product_category": occ_key,
            })
            break

    # 5. Universal safe fallback
    queries.append({
        "query": f"premium professional gift site:amazon.in {budget_str}",
        "type": "fallback",
        "signal_used": "universal fallback",
        "product_category": "professional gift",
    })

    return queries


def _signal_to_category(signal: str) -> tuple[Optional[str], list[str]]:
    signal_lower = signal.lower()
    for keywords, category, search_terms in INTEREST_TO_PRODUCT:
        for kw in keywords:
            if kw in signal_lower:
                return category, search_terms
    return None, []


def _role_to_search_terms(role: str) -> list[str]:
    role_lower = role.lower() if role else ""
    for role_keywords, search_terms in ROLE_TO_PRODUCT:
        for kw in role_keywords:
            if kw in role_lower:
                return search_terms
    return [
        "premium leather notebook site:amazon.in",
        "executive desk accessories site:amazon.in",
    ]
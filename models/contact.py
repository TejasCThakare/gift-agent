"""
models/contact.py

Pydantic models for the contact input schema.
These models validate and type-check the incoming contact JSON before
any processing begins. All fields that could be missing are Optional
with sensible defaults to handle real-world imperfect data.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class Experience(BaseModel):
    title: str = ""
    company: str = ""
    description: str = ""


class LinkedInProfile(BaseModel):
    headline: str = ""
    about: str = ""
    experience: list[Experience] = Field(default_factory=list)
    recent_posts: list[str] = Field(default_factory=list)
    recent_comments: list[str] = Field(default_factory=list)
    engaged_topics: list[str] = Field(default_factory=list)


class RelationshipContext(BaseModel):
    relationship_type: str = "unknown"
    last_interaction: str = ""
    business_goal: str = ""

    @field_validator("relationship_type", mode="before")
    @classmethod
    def normalize_relationship_type(cls, v: str) -> str:
        """Normalize relationship type to a standard key."""
        if not v:
            return "unknown"
        mapping = {
            "prospective customer": "prospective_customer",
            "prospect": "prospective_customer",
            "lead": "prospective_customer",
            "existing customer": "existing_customer",
            "customer": "existing_customer",
            "client": "existing_customer",
            "colleague": "colleague",
            "peer": "colleague",
            "coworker": "colleague",
            "executive": "executive",
            "c-suite": "executive",
            "founder": "founder",
            "co-founder": "founder",
            "partner": "partner",
            "vendor": "partner",
        }
        normalized = v.lower().strip()
        for key, value in mapping.items():
            if key in normalized:
                return value
        return normalized.replace(" ", "_")


class GiftContext(BaseModel):
    occasion: str = "general"
    budget_min: float = 0.0
    budget_max: float = 5000.0
    currency: str = "INR"
    country: str = "India"

    @field_validator("budget_min", "budget_max", mode="before")
    @classmethod
    def ensure_float(cls, v: Any) -> float:
        return float(v)

    @field_validator("country", mode="before")
    @classmethod
    def normalize_country(cls, v: str) -> str:
        return v.strip().title() if v else "India"


class ContactInput(BaseModel):
    """
    Top-level contact model. Validates the entire input contact dict.
    Used by the ingest node to ensure all downstream nodes receive
    clean, type-checked data.
    """
    name: str
    role: str = ""
    company: str = ""
    location: str = ""
    linkedin_profile: LinkedInProfile = Field(default_factory=LinkedInProfile)
    relationship_context: RelationshipContext = Field(default_factory=RelationshipContext)
    gift_context: GiftContext = Field(default_factory=GiftContext)

    def to_profile_text(self) -> str:
        """
        Flatten the LinkedIn profile into a single text block.
        Used as input context for the extract_and_query LLM call.
        """
        lp = self.linkedin_profile
        parts = [
            f"Name: {self.name}",
            f"Role: {self.role} at {self.company}",
            f"Location: {self.location}",
            f"Headline: {lp.headline}",
            f"About: {lp.about}",
        ]

        if lp.experience:
            parts.append("\nExperience:")
            for exp in lp.experience:
                parts.append(f"  - {exp.title} at {exp.company}: {exp.description}")

        if lp.recent_posts:
            parts.append("\nRecent posts:")
            for post in lp.recent_posts:
                parts.append(f'  - "{post}"')

        if lp.recent_comments:
            parts.append("\nRecent comments:")
            for comment in lp.recent_comments:
                parts.append(f'  - "{comment}"')

        if lp.engaged_topics:
            parts.append(f"\nEngaged topics: {', '.join(lp.engaged_topics)}")

        parts.append(f"\nRelationship: {self.relationship_context.relationship_type}")
        parts.append(f"Last interaction: {self.relationship_context.last_interaction}")
        parts.append(f"Business goal: {self.relationship_context.business_goal}")
        parts.append(f"\nGift occasion: {self.gift_context.occasion}")
        parts.append(
            f"Budget: {self.gift_context.currency} "
            f"{self.gift_context.budget_min}–{self.gift_context.budget_max}"
        )
        parts.append(f"Country: {self.gift_context.country}")

        return "\n".join(parts)

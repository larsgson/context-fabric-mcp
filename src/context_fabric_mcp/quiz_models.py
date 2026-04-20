"""Quiz data models — JSON-based quiz definitions."""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field


class FeatureVisibility(str, Enum):
    """How a feature is treated in a quiz."""

    show = "show"  # Visible to student as context
    request = "request"  # Student must answer this
    hide = "hide"  # Not shown at all


class FeatureConfig(BaseModel):
    """Configuration for a single feature in a quiz."""

    name: str  # Feature name (e.g. "part_of_speech", "verbal_stem")
    visibility: FeatureVisibility = FeatureVisibility.hide


class QuizDefinition(BaseModel):
    """A complete quiz definition stored as JSON."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "Untitled Quiz"
    description: str = ""
    corpus: str = "hebrew"

    # Passage scope
    book: str = "Genesis"
    chapter_start: int = 1
    chapter_end: int = 1
    verse_start: int | None = None  # None = entire chapter(s)
    verse_end: int | None = None

    # What to quiz on — Text-Fabric search template
    # e.g. "word sp=verb" to find all verbs
    search_template: str = "word sp=verb"

    # Feature configuration
    features: list[FeatureConfig] = Field(
        default_factory=lambda: [
            FeatureConfig(name="gloss", visibility=FeatureVisibility.show),
            FeatureConfig(name="part_of_speech", visibility=FeatureVisibility.request),
        ]
    )

    # Quiz settings
    randomize: bool = True
    max_questions: int = 10  # 0 = all matches
    time_limit_seconds: int = 0  # 0 = no limit
    context_verses: int = 0  # Verses of context before/after


class QuizQuestion(BaseModel):
    """A single generated quiz question."""

    index: int
    book: str
    chapter: int
    verse: int
    word_text: str
    word_text_utf8: str = ""
    shown_features: dict[str, str]  # feature_name -> value (context)
    requested_features: dict[str, str]  # feature_name -> correct answer


class QuizSession(BaseModel):
    """A generated quiz session ready to be taken."""

    quiz_id: str
    quiz_title: str
    questions: list[QuizQuestion]
    time_limit_seconds: int = 0

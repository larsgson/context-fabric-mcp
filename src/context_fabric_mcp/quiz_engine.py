"""Quiz engine — CRUD for quiz definitions + question generation."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from context_fabric_mcp.cf_engine import WORD_FEATURES, WORD_TYPE, CFEngine
from context_fabric_mcp.quiz_models import (
    FeatureVisibility,
    QuizDefinition,
    QuizQuestion,
    QuizSession,
)

logger = logging.getLogger(__name__)

# Default quiz storage directory (override with QUIZ_DIR env var)
QUIZ_DIR = Path(
    os.environ.get("QUIZ_DIR", Path(__file__).parent.parent.parent / "quizzes")
)


class QuizStore:
    """JSON file-based quiz storage."""

    def __init__(self, directory: Path = QUIZ_DIR) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, quiz_id: str) -> Path:
        return self.directory / f"{quiz_id}.json"

    def save(self, quiz: QuizDefinition) -> QuizDefinition:
        path = self._path(quiz.id)
        path.write_text(json.dumps(quiz.model_dump(), indent=2, ensure_ascii=False))
        return quiz

    def load(self, quiz_id: str) -> QuizDefinition:
        path = self._path(quiz_id)
        if not path.exists():
            raise FileNotFoundError(f"Quiz not found: {quiz_id}")
        data = json.loads(path.read_text())
        return QuizDefinition(**data)

    def delete(self, quiz_id: str) -> None:
        path = self._path(quiz_id)
        if path.exists():
            path.unlink()

    def list_all(self) -> list[dict[str, str]]:
        quizzes = []
        for p in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                quizzes.append(
                    {
                        "id": data.get("id", p.stem),
                        "title": data.get("title", "Untitled"),
                        "corpus": data.get("corpus", "hebrew"),
                        "book": data.get("book", ""),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return quizzes


# Canonical feature name -> TF feature name mapping
# This maps quiz feature names to what CFEngine._word_info produces
QUIZ_FEATURE_TO_WORD_INFO = {
    "gloss": "gloss",
    "part_of_speech": "part_of_speech",
    "gender": "gender",
    "number": "number",
    "person": "person",
    "state": "state",
    "verbal_stem": "verbal_stem",
    "verbal_tense": "verbal_tense",
    "lexeme": "lexeme",
    "language": "language",
}


def generate_session(
    quiz: QuizDefinition,
    engine: CFEngine,
) -> QuizSession:
    """Generate a quiz session with questions from the quiz definition.

    Executes the search template against the corpus and builds questions
    from matching words.
    """
    api = engine._ensure_loaded(quiz.corpus)
    feat_map = WORD_FEATURES.get(quiz.corpus, WORD_FEATURES["hebrew"])
    wtype = WORD_TYPE.get(quiz.corpus, "word")

    # Build the scoped search template
    scope_lines = []
    if quiz.verse_start is not None and quiz.verse_end is not None:
        # Verse-level scope — search within chapter, filter by verse after
        scope_lines.append(f"book book={quiz.book}")
        scope_lines.append(f"  chapter chapter={quiz.chapter_start}")
        # Indent the user template under chapter
        for line in quiz.search_template.strip().splitlines():
            scope_lines.append(f"    {line}")
    elif quiz.chapter_start == quiz.chapter_end:
        scope_lines.append(f"book book={quiz.book}")
        scope_lines.append(f"  chapter chapter={quiz.chapter_start}")
        for line in quiz.search_template.strip().splitlines():
            scope_lines.append(f"    {line}")
    else:
        # Multi-chapter — just scope to book, filter chapters after
        scope_lines.append(f"book book={quiz.book}")
        for line in quiz.search_template.strip().splitlines():
            scope_lines.append(f"  {line}")

    template = "\n".join(scope_lines) + "\n"
    logger.info("Quiz search template:\n%s", template)

    results = list(api.S.search(template))
    logger.info("Search returned %d results", len(results))

    # Determine which features to show and request
    show_features = []
    request_features = []
    for fc in quiz.features:
        if fc.visibility == FeatureVisibility.show:
            show_features.append(fc.name)
        elif fc.visibility == FeatureVisibility.request:
            request_features.append(fc.name)

    # Build questions from results
    questions: list[QuizQuestion] = []
    for result_tuple in results:
        # The last element matching wtype is our target word
        w = result_tuple[-1]

        # Check it's actually a word node
        if api.F.otype.v(w) != wtype:
            # Find the word node in the tuple
            for node in reversed(result_tuple):
                if api.F.otype.v(node) == wtype:
                    w = node
                    break
            else:
                continue

        section = api.T.sectionFromNode(w)
        if len(section) < 3:
            continue

        book_name, ch, vs = section

        # Filter by verse range if specified
        if quiz.verse_start is not None and vs < quiz.verse_start:
            continue
        if quiz.verse_end is not None and vs > quiz.verse_end:
            continue

        # Filter by chapter range
        if ch < quiz.chapter_start or ch > quiz.chapter_end:
            continue

        # Get word info
        word_info = engine._word_info(api, w, feat_map)
        word_dict = word_info.model_dump()

        shown = {}
        for fname in show_features:
            key = QUIZ_FEATURE_TO_WORD_INFO.get(fname, fname)
            val = word_dict.get(key, "")
            if val:
                shown[fname] = val

        requested = {}
        for fname in request_features:
            key = QUIZ_FEATURE_TO_WORD_INFO.get(fname, fname)
            val = word_dict.get(key, "")
            if val:
                requested[fname] = val

        # Skip if nothing to request
        if not requested:
            continue

        questions.append(
            QuizQuestion(
                index=len(questions),
                book=book_name,
                chapter=ch,
                verse=vs,
                word_text=word_info.text,
                word_text_utf8=word_info.lexeme_utf8,
                shown_features=shown,
                requested_features=requested,
            )
        )

    # Randomize and limit
    if quiz.randomize:
        random.shuffle(questions)

    if quiz.max_questions > 0:
        questions = questions[: quiz.max_questions]

    # Re-index after shuffle/limit
    for i, q in enumerate(questions):
        q.index = i

    return QuizSession(
        quiz_id=quiz.id,
        quiz_title=quiz.title,
        questions=questions,
        time_limit_seconds=quiz.time_limit_seconds,
    )

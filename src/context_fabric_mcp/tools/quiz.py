"""MCP tool for building quiz definitions (read-only, no server-side storage)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from context_fabric_mcp.cf_engine import CFEngine
from context_fabric_mcp.quiz_engine import generate_session
from context_fabric_mcp.quiz_models import FeatureConfig, FeatureVisibility, QuizDefinition


def register(mcp: FastMCP, engine: CFEngine) -> None:
    @mcp.tool()
    def build_quiz(
        title: str,
        book: str,
        chapter_start: int,
        chapter_end: int | None = None,
        verse_start: int | None = None,
        verse_end: int | None = None,
        corpus: str = "hebrew",
        search_template: str = "word sp=verb",
        show_features: list[str] | None = None,
        request_features: list[str] | None = None,
        max_questions: int = 10,
        randomize: bool = True,
        description: str = "",
    ) -> dict:
        """Build and validate a quiz definition, returning it as JSON.

        Creates a quiz definition from the given parameters, runs the search
        template against Text-Fabric to verify it produces results, and returns
        the complete quiz definition with a preview of how many questions
        would be generated. Nothing is stored on the server.

        Use search_words and get_passage first to explore what's available in
        the passage, then use this tool to package a quiz.

        Available feature names for show_features and request_features:
        - gloss, part_of_speech, verbal_stem, verbal_tense
        - gender, number, person, state, lexeme, language

        Each feature is either shown (given to the student as context),
        requested (the student must answer), or hidden.

        Args:
            title: Quiz title
            book: Book name (e.g. "Genesis", "Psalms")
            chapter_start: Starting chapter
            chapter_end: Ending chapter (default same as start)
            verse_start: Starting verse (None = entire chapter)
            verse_end: Ending verse (None = entire chapter)
            corpus: "hebrew" or "greek"
            search_template: Text-Fabric search template (e.g. "word sp=verb vs=qal")
            show_features: Features shown to student as context (e.g. ["gloss"])
            request_features: Features the student must answer (e.g. ["verbal_stem", "verbal_tense"])
            max_questions: Max questions (0 = all matches)
            randomize: Shuffle question order
            description: Optional quiz description
        """
        if chapter_end is None:
            chapter_end = chapter_start
        if show_features is None:
            show_features = ["gloss"]
        if request_features is None:
            request_features = ["part_of_speech"]

        features = []
        for f in show_features:
            features.append(FeatureConfig(name=f, visibility=FeatureVisibility.show))
        for f in request_features:
            features.append(FeatureConfig(name=f, visibility=FeatureVisibility.request))

        quiz = QuizDefinition(
            title=title,
            description=description,
            corpus=corpus,
            book=book,
            chapter_start=chapter_start,
            chapter_end=chapter_end,
            verse_start=verse_start,
            verse_end=verse_end,
            search_template=search_template,
            features=features,
            randomize=randomize,
            max_questions=max_questions,
        )

        # Validate by generating a session — this runs the search
        session = generate_session(quiz, engine)

        return {
            "quiz_definition": quiz.model_dump(),
            "validation": {
                "total_questions_generated": len(session.questions),
                "sample_questions": [q.model_dump() for q in session.questions[:3]],
            },
        }

"""MCP Server — extends Context-Fabric's MCP with biblical text tools.

Imports the Context-Fabric MCP server (which provides 11 generic corpus tools)
and registers our additional domain-specific tools on top:
- get_vocabulary / get_lexeme_info  (vocabulary analysis)
- get_word_context                  (syntactic hierarchy)
- search_words                      (simplified morphological search)
- build_quiz                        (quiz generation)
"""

from __future__ import annotations

import argparse
import logging
import os

from cfabric_mcp import corpus_manager, mcp

from context_fabric_mcp.cf_engine import CORPORA, CFEngine, _find_corpus_path
from context_fabric_mcp.quiz_engine import generate_session
from context_fabric_mcp.quiz_models import FeatureConfig, FeatureVisibility, QuizDefinition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared engine instance used by our custom tools
engine = CFEngine()


# ============================================================================
# Custom tools registered on Context-Fabric's MCP server
# ============================================================================


@mcp.tool()
def search_words(
    corpus: str = "hebrew",
    book: str | None = None,
    chapter: int | None = None,
    features: dict[str, str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Search for words matching morphological feature constraints.

    A simplified interface that builds a search template for you.
    For complex structural queries, use the built-in search() tool instead.

    Example features for Hebrew: {"sp": "verb", "vs": "hif", "vt": "perf"}
    Example features for Greek: {"sp": "verb", "tense": "aorist"}

    Common Hebrew feature names:
    - sp: part of speech (verb, subs, prep, adjv, advb, conj, art, nmpr)
    - vs: verbal stem (qal, nif, piel, pual, hif, hof, hit)
    - vt: verbal tense (perf, impf, wayq, impv, infa, infc, ptca, ptcp)
    - gn: gender (m, f)
    - nu: number (sg, pl, du)
    - ps: person (p1, p2, p3)
    - st: state (a=absolute, c=construct, e=emphatic)

    Args:
        corpus: "hebrew" or "greek"
        book: Limit search to a specific book (e.g. "Genesis")
        chapter: Limit search to a specific chapter (requires book)
        features: Dict of feature name -> value constraints
        limit: Max results to return (default 100)
    """
    return engine.search_words(corpus, book, chapter, features, limit)


@mcp.tool()
def get_word_context(
    book: str,
    chapter: int,
    verse: int,
    word_index: int = 0,
    corpus: str = "hebrew",
) -> dict:
    """Get the linguistic hierarchy (phrase, clause, sentence) for a specific word.

    Useful for understanding the syntactic context of a word.

    Args:
        book: Book name (e.g. "Genesis")
        chapter: Chapter number
        verse: Verse number
        word_index: 0-based index of the word within the verse
        corpus: "hebrew" or "greek"
    """
    return engine.get_context(book, chapter, verse, word_index, corpus)


@mcp.tool()
def get_lexeme_info(
    lexeme: str,
    corpus: str = "hebrew",
    limit: int = 50,
) -> dict:
    """Look up a lexeme and return its gloss, part of speech, and occurrences.

    The lexeme should be in transliterated form (e.g. "BRJ[" for the Hebrew
    verb "to create", "HLK[" for "to walk"). Use the 'lex' values returned
    by get_passages or search.

    Args:
        lexeme: Lexeme identifier (transliterated, e.g. "BRJ[", "HLK[")
        corpus: "hebrew" or "greek"
        limit: Max occurrences to return (default 50)
    """
    return engine.get_lexeme_info(lexeme, corpus, limit)


@mcp.tool()
def get_vocabulary(
    book: str,
    chapter: int,
    verse_start: int = 1,
    verse_end: int | None = None,
    corpus: str = "hebrew",
) -> list[dict]:
    """Get unique lexemes in a passage with frequency and gloss.

    Returns a deduplicated list of lexemes appearing in the specified verses,
    sorted by frequency (most common first).

    Args:
        book: Book name (e.g. "Genesis")
        chapter: Chapter number
        verse_start: Starting verse
        verse_end: Ending verse (default same as start)
        corpus: "hebrew" or "greek"
    """
    return engine.get_vocabulary(book, chapter, verse_start, verse_end, corpus)


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
    template to verify it produces results, and returns the complete quiz
    definition with a preview. Nothing is stored on the server.

    Available feature names for show_features and request_features:
    - gloss, part_of_speech, verbal_stem, verbal_tense
    - gender, number, person, state, lexeme, language

    Args:
        title: Quiz title
        book: Book name (e.g. "Genesis", "Psalms")
        chapter_start: Starting chapter
        chapter_end: Ending chapter (default same as start)
        verse_start: Starting verse (None = entire chapter)
        verse_end: Ending verse (None = entire chapter)
        corpus: "hebrew" or "greek"
        search_template: Search template (e.g. "word sp=verb vs=qal")
        show_features: Features shown to student as context (e.g. ["gloss"])
        request_features: Features the student must answer (e.g. ["verbal_stem"])
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

    session = generate_session(quiz, engine)

    return {
        "quiz_definition": quiz.model_dump(),
        "validation": {
            "total_questions_generated": len(session.questions),
            "sample_questions": [q.model_dump() for q in session.questions[:3]],
        },
    }


# ============================================================================
# Entry point
# ============================================================================


def main():
    """Run the MCP server with biblical corpora pre-loaded."""
    parser = argparse.ArgumentParser(
        description="Text-Fabric MCP Server (powered by Context-Fabric)",
    )
    parser.add_argument(
        "--sse",
        type=int,
        metavar="PORT",
        help="Run with SSE transport on specified port",
    )
    parser.add_argument(
        "--http",
        type=int,
        metavar="PORT",
        help="Run with Streamable HTTP transport on specified port",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine transport
    if args.sse:
        transport = "sse"
        port = args.sse
    elif args.http:
        transport = "http"
        port = args.http
    else:
        transport = "stdio"
        port = None

    logger.info("Starting Text-Fabric MCP server (transport: %s)...", transport)

    # Load biblical corpora via Context-Fabric's corpus manager
    for corpus_id, (org_repo, display_name) in CORPORA.items():
        try:
            path = _find_corpus_path(org_repo)
            corpus_manager.load(path, name=corpus_id)
            logger.info("Loaded %s as '%s'", display_name, corpus_id)
        except FileNotFoundError:
            logger.warning("Corpus %s not found, skipping", org_repo)

    # Also ensure our engine has them loaded (for custom tools)
    for corpus_id in corpus_manager.list_corpora():
        try:
            engine._ensure_loaded(corpus_id)
        except Exception as e:
            logger.warning("Engine pre-load for %s failed: %s", corpus_id, e)

    # Run the MCP server
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = port
        mcp.run(transport="sse")
    elif transport == "http":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

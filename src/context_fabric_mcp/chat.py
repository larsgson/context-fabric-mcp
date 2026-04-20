"""LLM chat backend — calls Google Gemini API with CFEngine tools in an agentic loop."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from context_fabric_mcp.cf_engine import CFEngine
from context_fabric_mcp.quiz_engine import generate_session
from context_fabric_mcp.quiz_models import FeatureConfig, FeatureVisibility, QuizDefinition

logger = logging.getLogger(__name__)

# Look for system prompts: first relative to source (local dev), then in /app (Docker)
_PROMPTS_DIR = Path(__file__).parent.parent.parent
if not (_PROMPTS_DIR / "system_prompt.md").exists():
    _PROMPTS_DIR = Path("/app")
SYSTEM_PROMPT = (_PROMPTS_DIR / "system_prompt.md").read_text()
SYSTEM_PROMPT_QUIZ = (_PROMPTS_DIR / "system_prompt_quiz.md").read_text()

# ---------------------------------------------------------------------------
# Tool declarations shared by both chat modes
# ---------------------------------------------------------------------------

_EXPLORATION_TOOLS = [
    {
        "name": "list_corpora",
        "description": "List available corpora.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_books",
        "description": "List books with chapter counts for a corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "corpus": {
                    "type": "string",
                    "description": "Corpus name (hebrew or greek)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_passage",
        "description": "Get biblical text for a verse range with full morphological annotations.",
        "parameters": {
            "type": "object",
            "properties": {
                "book": {"type": "string", "description": "Book name"},
                "chapter": {"type": "integer", "description": "Chapter number"},
                "verse_start": {"type": "integer", "description": "Start verse"},
                "verse_end": {"type": "integer", "description": "End verse"},
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": ["book", "chapter"],
        },
    },
    {
        "name": "get_schema",
        "description": "Return object types and their features for a corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": [],
        },
    },
    {
        "name": "search_words",
        "description": "Search for words matching morphological feature constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "corpus": {"type": "string", "description": "Corpus name"},
                "book": {"type": "string", "description": "Book name"},
                "chapter": {"type": "integer", "description": "Chapter number"},
                "features": {
                    "type": "object",
                    "description": "Feature name/value pairs",
                    "properties": {},
                },
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": [],
        },
    },
    {
        "name": "search_constructions",
        "description": "Search for structural/syntactic patterns using Text-Fabric search templates.",
        "parameters": {
            "type": "object",
            "properties": {
                "template": {"type": "string", "description": "Search template"},
                "corpus": {"type": "string", "description": "Corpus name"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["template"],
        },
    },
    {
        "name": "get_lexeme_info",
        "description": "Look up a lexeme and return its gloss, part of speech, and occurrences.",
        "parameters": {
            "type": "object",
            "properties": {
                "lexeme": {"type": "string", "description": "Lexeme identifier"},
                "corpus": {"type": "string", "description": "Corpus name"},
                "limit": {"type": "integer", "description": "Max occurrences"},
            },
            "required": ["lexeme"],
        },
    },
    {
        "name": "get_vocabulary",
        "description": "Get unique lexemes in a passage with frequency and gloss.",
        "parameters": {
            "type": "object",
            "properties": {
                "book": {"type": "string", "description": "Book name"},
                "chapter": {"type": "integer", "description": "Chapter number"},
                "verse_start": {"type": "integer", "description": "Start verse"},
                "verse_end": {"type": "integer", "description": "End verse"},
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": ["book", "chapter"],
        },
    },
    {
        "name": "get_word_context",
        "description": "Get the linguistic hierarchy (phrase, clause, sentence) for a specific word.",
        "parameters": {
            "type": "object",
            "properties": {
                "book": {"type": "string", "description": "Book name"},
                "chapter": {"type": "integer", "description": "Chapter number"},
                "verse": {"type": "integer", "description": "Verse number"},
                "word_index": {"type": "integer", "description": "Word index in verse"},
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": ["book", "chapter", "verse"],
        },
    },
    {
        "name": "search_syntax_guide",
        "description": "Get search template syntax documentation. Call without section for overview, or with a section name for details.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": "Section: basics, structure, relations, quantifiers, or examples. Omit for summary.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "describe_feature",
        "description": "Get detailed info about a feature including sample values sorted by frequency. Use to understand what values a feature can have before searching.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "Feature name (e.g. 'sp', 'vs', 'vt')",
                },
                "sample_limit": {
                    "type": "integer",
                    "description": "Max sample values (default 20)",
                },
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": ["feature"],
        },
    },
    {
        "name": "list_features",
        "description": "List features with optional filtering by kind (node/edge) and node type. Lightweight catalog for discovery.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "'all', 'node', or 'edge'"},
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to features for these types (e.g. ['word'])",
                },
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": [],
        },
    },
    {
        "name": "search_advanced",
        "description": "Search with advanced return types. Use return_type='statistics' for feature distributions, 'count' for totals, 'passages' for formatted text, or 'results' for node details.",
        "parameters": {
            "type": "object",
            "properties": {
                "template": {"type": "string", "description": "Search template"},
                "return_type": {
                    "type": "string",
                    "description": "results, count, statistics, or passages",
                },
                "aggregate_features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For statistics: features to aggregate",
                },
                "group_by_section": {
                    "type": "boolean",
                    "description": "Include distribution by book",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max values per feature distribution",
                },
                "limit": {
                    "type": "integer",
                    "description": "Page size for results/passages",
                },
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": ["template"],
        },
    },
    {
        "name": "search_comparative",
        "description": "Search the same or adapted pattern across both Hebrew and Greek corpora. Best with return_type='statistics' or 'count' for comparison.",
        "parameters": {
            "type": "object",
            "properties": {
                "template_hebrew": {
                    "type": "string",
                    "description": "Search template for Hebrew",
                },
                "template_greek": {
                    "type": "string",
                    "description": "Search template for Greek",
                },
                "return_type": {"type": "string", "description": "count or statistics"},
                "limit": {"type": "integer", "description": "Max results per corpus"},
            },
            "required": ["template_hebrew", "template_greek"],
        },
    },
    {
        "name": "list_edge_features",
        "description": "List available edge features (relationships between nodes) for a corpus.",
        "parameters": {
            "type": "object",
            "properties": {
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": [],
        },
    },
    {
        "name": "get_edge_features",
        "description": "Get edges (relationships) for a specific node using an edge feature like 'mother' or 'functional_parent'.",
        "parameters": {
            "type": "object",
            "properties": {
                "node": {"type": "integer", "description": "Node ID"},
                "edge_feature": {"type": "string", "description": "Edge feature name"},
                "direction": {
                    "type": "string",
                    "description": "'from' (outgoing) or 'to' (incoming)",
                },
                "corpus": {"type": "string", "description": "Corpus name"},
            },
            "required": ["node", "edge_feature"],
        },
    },
    {
        "name": "compare_distribution",
        "description": "Compare feature value distributions across books or sections. E.g. compare verb stem usage in Genesis vs Exodus.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature": {
                    "type": "string",
                    "description": "Feature name (e.g. 'vs', 'sp')",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of {book, chapter?, corpus?} dicts",
                },
                "node_type": {
                    "type": "string",
                    "description": "Object type (default 'word')",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max values per distribution",
                },
            },
            "required": ["feature", "sections"],
        },
    },
]

_BUILD_QUIZ_TOOL = {
    "name": "build_quiz",
    "description": (
        "Build and validate a quiz definition. Runs the search template against "
        "Text-Fabric to verify it produces results and returns the complete quiz "
        "definition JSON with a preview. Nothing is stored on the server."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Quiz title"},
            "book": {"type": "string", "description": "Book name"},
            "chapter_start": {"type": "integer", "description": "Starting chapter"},
            "chapter_end": {"type": "integer", "description": "Ending chapter"},
            "verse_start": {
                "type": "integer",
                "description": "Starting verse (omit for entire chapter)",
            },
            "verse_end": {
                "type": "integer",
                "description": "Ending verse (omit for entire chapter)",
            },
            "corpus": {"type": "string", "description": "hebrew or greek"},
            "search_template": {
                "type": "string",
                "description": "Text-Fabric search template (e.g. 'word sp=verb vs=qal')",
            },
            "show_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features shown as context (e.g. ['gloss', 'lexeme'])",
            },
            "request_features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Features the student must answer (e.g. ['verbal_stem', 'verbal_tense'])",
            },
            "max_questions": {
                "type": "integer",
                "description": "Max questions (0 = all)",
            },
            "randomize": {"type": "boolean", "description": "Shuffle question order"},
            "description": {"type": "string", "description": "Quiz description"},
        },
        "required": [
            "title",
            "book",
            "chapter_start",
            "search_template",
            "request_features",
        ],
    },
}

# Tool sets for each chat mode
GENERAL_TOOLS = types.Tool(function_declarations=_EXPLORATION_TOOLS)
QUIZ_TOOLS = types.Tool(function_declarations=_EXPLORATION_TOOLS + [_BUILD_QUIZ_TOOL])


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(engine: CFEngine, name: str, args: dict[str, Any]) -> Any:
    """Execute a tool call against the CFEngine and return the result."""
    if name == "list_corpora":
        return engine.list_corpora()
    elif name == "list_books":
        books = engine.list_books(args.get("corpus", "hebrew"))
        return [b.model_dump() for b in books]
    elif name == "get_passage":
        result = engine.get_passage(
            book=args["book"],
            chapter=args["chapter"],
            verse_start=args.get("verse_start", 1),
            verse_end=args.get("verse_end"),
            corpus=args.get("corpus", "hebrew"),
        )
        return result.model_dump()
    elif name == "get_schema":
        result = engine.get_schema(args.get("corpus", "hebrew"))
        return result.model_dump()
    elif name == "search_words":
        return engine.search_words(
            corpus=args.get("corpus", "hebrew"),
            book=args.get("book"),
            chapter=args.get("chapter"),
            features=args.get("features"),
            limit=args.get("limit", 100),
        )
    elif name == "search_constructions":
        return engine.search_constructions(
            template=args["template"],
            corpus=args.get("corpus", "hebrew"),
            limit=args.get("limit", 50),
        )
    elif name == "get_lexeme_info":
        return engine.get_lexeme_info(
            lexeme=args["lexeme"],
            corpus=args.get("corpus", "hebrew"),
            limit=args.get("limit", 50),
        )
    elif name == "get_vocabulary":
        return engine.get_vocabulary(
            book=args["book"],
            chapter=args["chapter"],
            verse_start=args.get("verse_start", 1),
            verse_end=args.get("verse_end"),
            corpus=args.get("corpus", "hebrew"),
        )
    elif name == "get_word_context":
        return engine.get_context(
            book=args["book"],
            chapter=args["chapter"],
            verse=args["verse"],
            word_index=args.get("word_index", 0),
            corpus=args.get("corpus", "hebrew"),
        )
    elif name == "search_syntax_guide":
        return engine.get_search_syntax_guide(args.get("section"))
    elif name == "describe_feature":
        return engine.describe_feature(
            features=args["feature"],
            sample_limit=args.get("sample_limit", 20),
            corpus=args.get("corpus", "hebrew"),
        )
    elif name == "list_features":
        return engine.list_features(
            kind=args.get("kind", "all"),
            node_types=args.get("node_types"),
            corpus=args.get("corpus", "hebrew"),
        )
    elif name == "search_advanced":
        return engine.search_advanced(
            template=args["template"],
            return_type=args.get("return_type", "results"),
            aggregate_features=args.get("aggregate_features"),
            group_by_section=args.get("group_by_section", False),
            top_n=args.get("top_n", 50),
            limit=args.get("limit", 100),
            corpus=args.get("corpus", "hebrew"),
        )
    elif name == "search_comparative":
        return engine.search_comparative(
            template_hebrew=args["template_hebrew"],
            template_greek=args["template_greek"],
            return_type=args.get("return_type", "count"),
            limit=args.get("limit", 50),
        )
    elif name == "list_edge_features":
        return engine.list_edge_features(args.get("corpus", "hebrew"))
    elif name == "get_edge_features":
        return engine.get_edge_features(
            node=args["node"],
            edge_feature=args["edge_feature"],
            direction=args.get("direction", "from"),
            corpus=args.get("corpus", "hebrew"),
        )
    elif name == "compare_distribution":
        return engine.compare_feature_distribution(
            feature=args["feature"],
            sections=args["sections"],
            node_type=args.get("node_type", "word"),
            top_n=args.get("top_n", 20),
        )
    elif name == "build_quiz":
        return _execute_build_quiz(engine, args)
    else:
        return {"error": f"Unknown tool: {name}"}


def _execute_build_quiz(engine: CFEngine, args: dict[str, Any]) -> Any:
    """Build and validate a quiz definition."""
    chapter_end = args.get("chapter_end") or args["chapter_start"]
    show_features = args.get("show_features") or ["gloss"]
    request_features = args.get("request_features") or ["part_of_speech"]

    features = []
    for f in show_features:
        features.append(FeatureConfig(name=f, visibility=FeatureVisibility.show))
    for f in request_features:
        features.append(FeatureConfig(name=f, visibility=FeatureVisibility.request))

    quiz = QuizDefinition(
        title=args["title"],
        description=args.get("description", ""),
        corpus=args.get("corpus", "hebrew"),
        book=args["book"],
        chapter_start=args["chapter_start"],
        chapter_end=chapter_end,
        verse_start=args.get("verse_start"),
        verse_end=args.get("verse_end"),
        search_template=args["search_template"],
        features=features,
        randomize=args.get("randomize", True),
        max_questions=args.get("max_questions", 10),
    )

    session = generate_session(quiz, engine)

    return {
        "quiz_definition": quiz.model_dump(),
        "validation": {
            "total_questions_generated": len(session.questions),
            "sample_questions": [q.model_dump() for q in session.questions[:3]],
        },
    }


# ---------------------------------------------------------------------------
# Chat loop (shared by both modes)
# ---------------------------------------------------------------------------


def _chat_loop(
    engine: CFEngine,
    message: str,
    history: list[dict[str, Any]] | None,
    system_prompt: str,
    tools: types.Tool,
    model: str,
    max_turns: int,
) -> dict[str, Any]:
    """Run a chat turn with tool use loop.

    Returns:
        {"reply": str, "tool_calls": [...]}
    """
    client = genai.Client()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tools],
    )

    contents: list[types.Content] = []
    if history:
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg["content"])])
            )
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    tool_calls_log: list[dict[str, Any]] = []

    for _ in range(max_turns):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Guard against empty responses (safety filters, overloaded model, etc.)
        if (
            not response.candidates
            or response.candidates[0].content is None
            or not response.candidates[0].content.parts
        ):
            reason = (
                getattr(response.candidates[0], "finish_reason", "unknown")
                if response.candidates
                else "no candidates"
            )
            logger.warning("Gemini returned empty content: %s", reason)
            return {
                "reply": "I wasn't able to generate a response. Please try rephrasing your question.",
                "tool_calls": tool_calls_log,
            }

        candidate = response.candidates[0]

        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if part.function_call is not None
        ]

        if not function_calls:
            text_parts = [part.text for part in candidate.content.parts if part.text]
            return {
                "reply": "\n".join(text_parts)
                or "I wasn't able to generate a response.",
                "tool_calls": tool_calls_log,
            }

        contents.append(candidate.content)

        function_response_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            logger.info("Tool call: %s(%s)", fc.name, json.dumps(args)[:200])

            try:
                result = _execute_tool(engine, fc.name, args)
                result_str = json.dumps(result, ensure_ascii=False, default=str)
                if len(result_str) > 20000:
                    result_str = result_str[:20000] + "... (truncated)"
                result_data = (
                    json.loads(result_str)
                    if not result_str.endswith("(truncated)")
                    else result_str
                )
            except Exception as e:
                logger.error("Tool error: %s", e)
                result_data = {"error": str(e)}

            tool_calls_log.append(
                {
                    "name": fc.name,
                    "input": args,
                    "result": result_data,
                }
            )

            function_response_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result_data},
                )
            )

        contents.append(types.Content(role="user", parts=function_response_parts))

    return {
        "reply": "I've reached the maximum number of tool calls. Please try a more specific question.",
        "tool_calls": tool_calls_log,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat(
    engine: CFEngine,
    message: str,
    history: list[dict[str, Any]] | None = None,
    model: str = "gemini-2.5-flash",
    max_turns: int = 10,
) -> dict[str, Any]:
    """General biblical text chat."""
    return _chat_loop(
        engine, message, history, SYSTEM_PROMPT, GENERAL_TOOLS, model, max_turns
    )


def chat_quiz(
    engine: CFEngine,
    message: str,
    history: list[dict[str, Any]] | None = None,
    model: str = "gemini-2.5-flash",
    max_turns: int = 10,
) -> dict[str, Any]:
    """Quiz-builder chat — has access to build_quiz tool."""
    return _chat_loop(
        engine, message, history, SYSTEM_PROMPT_QUIZ, QUIZ_TOOLS, model, max_turns
    )

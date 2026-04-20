"""FastAPI HTTP layer wrapping CFEngine."""

from __future__ import annotations

import hmac
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from context_fabric_mcp.cf_engine import CFEngine
from context_fabric_mcp.quiz_engine import QuizStore, generate_session
from context_fabric_mcp.quiz_models import QuizDefinition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Context-Fabric MCP API",
    description="Biblical text analysis API powered by Context-Fabric",
    version="0.1.0",
)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.warning("API_KEY not set. All requests will be allowed.")


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if API_KEY and request.url.path != "/health":
        key = request.headers.get("x-api-key", "")
        if not hmac.compare_digest(key, API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


@app.get("/health")
def health():
    """Unauthenticated health check for Railway/Fly.io."""
    return {"status": "ok"}


engine = CFEngine()
quiz_store = QuizStore()


# --- Request models ---


class WordSearchRequest(BaseModel):
    corpus: str = "hebrew"
    book: str | None = None
    chapter: int | None = None
    features: dict[str, str] | None = None
    limit: int = 100


class ConstructionSearchRequest(BaseModel):
    template: str
    corpus: str = "hebrew"
    limit: int = 50


class AdvancedSearchRequest(BaseModel):
    template: str
    return_type: str = "results"
    aggregate_features: list[str] | None = None
    group_by_section: bool = False
    top_n: int = 50
    limit: int = 100
    corpus: str = "hebrew"


class SearchContinueRequest(BaseModel):
    cursor_id: str
    offset: int = 0
    limit: int = 100


class ComparativeSearchRequest(BaseModel):
    template_hebrew: str
    template_greek: str
    return_type: str = "count"
    limit: int = 50


class CompareDistributionRequest(BaseModel):
    feature: str
    sections: list[dict]
    node_type: str = "word"
    top_n: int = 20


class ChatRequest(BaseModel):
    message: str
    history: list[dict] | None = None


# --- Endpoints ---


@app.get("/api/corpora")
def list_corpora():
    """List available corpora."""
    return engine.list_corpora()


@app.get("/api/books")
def list_books(corpus: str = "hebrew"):
    """List books with chapter counts for a corpus."""
    books = engine.list_books(corpus)
    return [b.model_dump() for b in books]


@app.get("/api/passage")
def get_passage(
    book: str,
    chapter: int,
    verse_start: int = 1,
    verse_end: int | None = None,
    corpus: str = "hebrew",
):
    """Get biblical text for a verse range with morphological annotations."""
    result = engine.get_passage(book, chapter, verse_start, verse_end, corpus)
    return result.model_dump()


@app.get("/api/schema")
def get_schema(corpus: str = "hebrew"):
    """Return object types and their features for a corpus."""
    result = engine.get_schema(corpus)
    return result.model_dump()


@app.post("/api/search/words")
def search_words(req: WordSearchRequest):
    """Search for words matching morphological feature constraints."""
    return engine.search_words(
        corpus=req.corpus,
        book=req.book,
        chapter=req.chapter,
        features=req.features,
        limit=req.limit,
    )


@app.post("/api/search/constructions")
def search_constructions(req: ConstructionSearchRequest):
    """Search for structural patterns using search templates."""
    return engine.search_constructions(
        template=req.template,
        corpus=req.corpus,
        limit=req.limit,
    )


@app.get("/api/search/syntax-guide")
def search_syntax_guide(section: str | None = None):
    """Get search template syntax documentation."""
    return engine.get_search_syntax_guide(section)


@app.post("/api/search/advanced")
def search_advanced(req: AdvancedSearchRequest):
    """Search with advanced return types (results, count, statistics, passages)."""
    return engine.search_advanced(
        template=req.template,
        return_type=req.return_type,
        aggregate_features=req.aggregate_features,
        group_by_section=req.group_by_section,
        top_n=req.top_n,
        limit=req.limit,
        corpus=req.corpus,
    )


@app.post("/api/search/continue")
def search_continue_endpoint(req: SearchContinueRequest):
    """Continue paginated search results using a cursor ID."""
    return engine.search_continue(
        cursor_id=req.cursor_id,
        offset=req.offset,
        limit=req.limit,
    )


@app.post("/api/search/comparative")
def search_comparative(req: ComparativeSearchRequest):
    """Search same pattern across Hebrew and Greek corpora."""
    return engine.search_comparative(
        template_hebrew=req.template_hebrew,
        template_greek=req.template_greek,
        return_type=req.return_type,
        limit=req.limit,
    )


@app.get("/api/features")
def list_features(
    kind: str = "all",
    node_types: str | None = None,
    corpus: str = "hebrew",
):
    """List available features with optional filtering."""
    nt_list = node_types.split(",") if node_types else None
    return engine.list_features(kind=kind, node_types=nt_list, corpus=corpus)


@app.get("/api/features/{feature}")
def describe_feature(
    feature: str,
    sample_limit: int = 20,
    corpus: str = "hebrew",
):
    """Get detailed info about a feature with sample values."""
    return engine.describe_feature(feature, sample_limit, corpus)


@app.get("/api/edges")
def list_edge_features(corpus: str = "hebrew"):
    """List available edge features."""
    return engine.list_edge_features(corpus)


@app.get("/api/edges/{edge_feature}")
def get_edges(
    edge_feature: str,
    node: int,
    direction: str = "from",
    corpus: str = "hebrew",
):
    """Get edges from/to a node using a specific edge feature."""
    return engine.get_edge_features(node, edge_feature, direction, corpus)


@app.post("/api/compare/distribution")
def compare_distribution(req: CompareDistributionRequest):
    """Compare feature distributions across books/sections."""
    return engine.compare_feature_distribution(
        feature=req.feature,
        sections=req.sections,
        node_type=req.node_type,
        top_n=req.top_n,
    )


@app.get("/api/lexeme/{lexeme}")
def get_lexeme_info(
    lexeme: str,
    corpus: str = "hebrew",
    limit: int = 50,
):
    """Look up a lexeme and return its gloss, part of speech, and occurrences."""
    return engine.get_lexeme_info(lexeme, corpus, limit)


@app.get("/api/vocabulary")
def get_vocabulary(
    book: str,
    chapter: int,
    verse_start: int = 1,
    verse_end: int | None = None,
    corpus: str = "hebrew",
):
    """Get unique lexemes in a passage with frequency and gloss."""
    return engine.get_vocabulary(book, chapter, verse_start, verse_end, corpus)


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Send a message to the LLM with biblical text tools available."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise HTTPException(
            status_code=503,
            detail="Chat unavailable: GOOGLE_API_KEY not configured",
        )
    try:
        from context_fabric_mcp.chat import chat

        return chat(engine, req.message, req.history)
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat-quiz")
def chat_quiz_endpoint(req: ChatRequest):
    """AI-assisted quiz builder. Describe the quiz you want in natural language."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise HTTPException(
            status_code=503,
            detail="Chat unavailable: GOOGLE_API_KEY not configured",
        )
    try:
        from context_fabric_mcp.chat import chat_quiz

        return chat_quiz(engine, req.message, req.history)
    except Exception as e:
        logger.error("Chat-quiz error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/context")
def get_context(
    book: str,
    chapter: int,
    verse: int,
    word_index: int = 0,
    corpus: str = "hebrew",
):
    """Get the linguistic hierarchy for a specific word."""
    return engine.get_context(book, chapter, verse, word_index, corpus)


# --- Quiz endpoints ---


@app.get("/api/quizzes")
def list_quizzes():
    """List all saved quizzes."""
    return quiz_store.list_all()


@app.post("/api/quizzes")
def create_quiz(quiz: QuizDefinition):
    """Create a new quiz."""
    return quiz_store.save(quiz).model_dump()


@app.get("/api/quizzes/{quiz_id}")
def get_quiz(quiz_id: str):
    """Load a quiz by ID."""
    try:
        return quiz_store.load(quiz_id).model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")


@app.put("/api/quizzes/{quiz_id}")
def update_quiz(quiz_id: str, quiz: QuizDefinition):
    """Update an existing quiz."""
    quiz.id = quiz_id
    return quiz_store.save(quiz).model_dump()


@app.delete("/api/quizzes/{quiz_id}")
def delete_quiz(quiz_id: str):
    """Delete a quiz."""
    quiz_store.delete(quiz_id)
    return {"status": "deleted"}


@app.post("/api/quizzes/{quiz_id}/generate")
def generate_quiz_session(quiz_id: str):
    """Generate a quiz session with questions from the quiz definition."""
    try:
        quiz = quiz_store.load(quiz_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Quiz not found: {quiz_id}")
    session = generate_session(quiz, engine)
    return session.model_dump()


if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
    logger.warning("GOOGLE_API_KEY not set. Chat endpoint will be unavailable.")


def _provision_corpus_data():
    """Copy pre-downloaded corpus data from Docker image to the persistent volume.

    During Docker build, corpus data (.tf files) and pre-compiled caches
    (.cfm directories) are prepared at /root/text-fabric-data/. At runtime
    on Railway, the volume is mounted at /data and HOME=/data. This function
    copies everything so Context-Fabric can load instantly without recompiling.
    """
    import shutil
    from pathlib import Path

    data_dir = Path("/data")
    if not data_dir.exists():
        return  # Local dev — no volume

    # Marker version: bump to force re-provisioning after image changes.
    marker = data_dir / "text-fabric-data" / ".cache-v5"
    for d in ["/data/text-fabric-data", "/data/quizzes"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    src = Path("/root/text-fabric-data")
    dst = data_dir / "text-fabric-data"

    if not marker.exists() and src.exists():
        logger.info("Provisioning corpus data from Docker image...")

        # Log what we're copying (especially .cfm caches)
        cfm_dirs = list(src.rglob(".cfm"))
        logger.info("Source .cfm cache dirs: %s", [str(d) for d in cfm_dirs])

        github_dst = dst / "github"
        if github_dst.exists():
            shutil.rmtree(github_dst)
        # Copy .tf source files AND pre-compiled .cfm caches
        shutil.copytree(src, dst, dirs_exist_ok=True)

        # Remove stale TF .tf compile caches (directories named ".tf",
        # NOT the .tf source files which are regular files)
        for tf_cache in dst.rglob(".tf"):
            if tf_cache.is_dir():
                logger.info("Removing TF compile cache: %s", tf_cache)
                shutil.rmtree(tf_cache)

        # Remove stale markers from previous versions
        for old in dst.glob(".cache-v*"):
            if old != marker:
                old.unlink(missing_ok=True)

        marker.touch()

        # Verify .cfm caches survived
        cfm_result = list(dst.rglob(".cfm"))
        logger.info("Provisioned .cfm cache dirs: %s", [str(d) for d in cfm_result])
        logger.info("Corpus data provisioned.")
    elif marker.exists():
        logger.info("Corpus data already provisioned (marker: %s).", marker.name)
    elif not src.exists():
        logger.warning(
            "No pre-downloaded corpus data found at %s. "
            "Corpora will need to be available at runtime.",
            src,
        )


def main():
    """Run the API server."""
    import uvicorn

    _provision_corpus_data()

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

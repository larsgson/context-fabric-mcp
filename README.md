# Context-Fabric MCP

Biblical text analysis server powered by [Context-Fabric](https://context-fabric.ai/), providing morphologically annotated Hebrew Bible (BHSA) and Greek New Testament (Nestle 1904) data via both an MCP (Model Context Protocol) interface and a REST API.

## Features

- **Hebrew Bible** -- 39 books (Genesis--2 Chronicles) with full morphological annotation from [ETCBC/bhsa](https://github.com/ETCBC/bhsa)
- **Greek New Testament** -- 27 books (Matthew--Revelation) from [ETCBC/nestle1904](https://github.com/ETCBC/nestle1904)
- **Passage retrieval** -- verse-level text with per-word lexeme, gloss, part of speech, and morphological features
- **Morphological search** -- find words by part of speech, verbal stem, tense, gender, number, person, state, and more
- **Structural search** -- find syntactic patterns (clauses, phrases) using search templates
- **Vocabulary extraction** -- unique lexemes in a passage sorted by corpus frequency
- **LLM chat** -- agentic conversation powered by Groq (Llama 3.3 70B, free tier) with optional OpenAI fallback; can call all the above tools
- **Quiz generation** -- configurable quiz engine for Hebrew morphology drills
- **AI-assisted quiz builder** -- teachers describe a quiz in natural language and the AI builds a validated quiz definition
- **Dual interface** -- same engine exposed as MCP tools (for AI assistants) and as a FastAPI HTTP API (for web frontends)

## Quick Start

```bash
# Clone and set up
git clone https://github.com/larsgson/context-fabric-mcp.git
cd context-fabric-mcp
cp .env.example .env   # configure API keys

# Start the REST API server
uv run cf-api
```

The API runs at `http://localhost:8000`. The first request loads corpus data into memory (~2s with cached data).

### Run as MCP server

```bash
uv run cf-mcp
```

This starts the MCP (Model Context Protocol) server for use with Claude Desktop or other MCP-compatible clients.

## Client Demo

A working client application that uses this server as its backend is available at [biblical-text-quizz-generator](https://github.com/larsgson/biblical-text-quizz-generator). It is a web-based quiz tool for biblical Hebrew and Greek morphology, demonstrating how a frontend can consume the REST API endpoints (passage retrieval, morphological search, quiz generation, and AI-assisted quiz building) provided by this server.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/corpora` | List available corpora |
| GET | `/api/books?corpus=hebrew` | List books with chapter counts |
| GET | `/api/passage?book=Genesis&chapter=1&verse_start=1` | Get annotated text |
| GET | `/api/schema?corpus=hebrew` | Corpus object types and features |
| POST | `/api/search/words` | Search by morphological features |
| POST | `/api/search/constructions` | Structural pattern matching |
| GET | `/api/search/syntax-guide` | Search template syntax documentation |
| POST | `/api/search/advanced` | Search with statistics, count, or passage grouping |
| POST | `/api/search/continue` | Cursor-based pagination for search results |
| POST | `/api/search/comparative` | Cross-corpus search (Hebrew + Greek) |
| GET | `/api/features?node_types=word` | List features with optional filtering |
| GET | `/api/features/{feature}` | Feature details with sample values |
| GET | `/api/edges` | List edge features (linguistic relationships) |
| GET | `/api/edges/{feature}?node=1` | Get edges from/to a node |
| POST | `/api/compare/distribution` | Compare feature distributions across sections |
| GET | `/api/lexeme/{lexeme}` | Lexeme lookup with occurrences |
| GET | `/api/vocabulary?book=Genesis&chapter=1` | Unique lexemes in a passage |
| GET | `/api/context?book=Genesis&chapter=1&verse=1` | Syntactic hierarchy for a word |
| POST | `/api/chat` | LLM-powered biblical analysis |
| POST | `/api/chat-quiz` | AI-assisted quiz builder |
| GET/POST | `/api/quizzes` | Quiz CRUD |
| POST | `/api/quizzes/{id}/generate` | Generate a quiz session |

## MCP Tools

When running as an MCP server, the following tools are available to AI assistants:

- `list_corpora`, `list_books`, `get_schema` -- corpus introspection
- `get_passage`, `get_word_context` -- text retrieval
- `search_words`, `search_constructions` -- linguistic search
- `search_advanced`, `search_comparative` -- advanced search with statistics and cross-corpus comparison
- `search_syntax_guide` -- search template syntax documentation
- `describe_feature`, `list_features` -- feature discovery with sample values
- `list_edge_features`, `get_edge_features` -- linguistic relationship exploration
- `compare_distribution` -- feature distribution comparison across sections
- `get_lexeme_info`, `get_vocabulary` -- vocabulary queries
- `build_quiz` -- build and validate a quiz definition (returns JSON, no server-side storage)

## Project Structure

```
src/context_fabric_mcp/
├── server.py          # MCP server entry point
├── api.py             # FastAPI HTTP layer
├── cf_engine.py       # Context-Fabric data access
├── chat.py            # Groq + OpenAI fallback LLM integration
├── models.py          # Pydantic data models
├── quiz_engine.py     # Quiz generation engine
├── quiz_models.py     # Quiz data models
└── tools/             # MCP tool definitions
    ├── passage.py
    ├── schema.py
    ├── search.py
    ├── vocab.py
    └── quiz.py
tests/
├── test_api.py
├── test_cf_engine.py
└── test_quiz.py
```

## AI-Assisted Quiz Builder

Teachers can create quizzes in two ways:

### Via the REST API (for web frontends)

`POST /api/chat-quiz` accepts a natural language description and returns a validated quiz definition:

```json
{
  "message": "Create a quiz on qal perfect verbs in Genesis 1-3, showing the gloss and asking for verbal stem and tense"
}
```

The AI explores the text, builds a `QuizDefinition`, validates it against the corpus, and returns the definition with a preview. The frontend can then save it via `POST /api/quizzes`.

### Via MCP (for AI assistants like Claude Desktop)

The `build_quiz` tool lets an AI assistant build quiz definitions interactively. The teacher describes what they want, the AI explores passages with `search_words` and `get_passage`, then calls `build_quiz` to produce a validated definition. The result is returned as portable JSON -- nothing is stored on the server.

## Testing

```bash
uv run pytest
```

Tests require corpus data on first run (cached in `~/text-fabric-data/`).

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `API_KEY` | Recommended | Shared secret for API authentication. All requests must include `x-api-key` header. If unset, all requests are allowed. |
| `PORT` | No | API server port (default: 8000) |
| `GROQ_API_KEY` | For chat | Primary LLM provider for `/api/chat` and `/api/chat-quiz`. Free tier at [console.groq.com/keys](https://console.groq.com/keys). |
| `OPENAI_API_KEY` | No | Optional fallback provider. Used automatically if Groq is rate-limited/unavailable. |
| `OPENAI_FALLBACK_DAILY_LIMIT` | No | Max OpenAI fallback calls per day to prevent surprise bills. Default: 50. |
| `GROQ_MODEL` | No | Override Groq model (default: `llama-3.3-70b-versatile`). |
| `OPENAI_MODEL` | No | Override OpenAI model (default: `gpt-4o-mini`). |

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions on deploying to Fly.io, Railway, or Docker.

```bash
# Docker
docker build -t context-fabric-mcp .
docker run -p 8000:8000 context-fabric-mcp
```

## License

[MIT](LICENSE)

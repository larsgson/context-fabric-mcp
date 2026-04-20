"""MCP tools for vocabulary and lexeme queries."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from context_fabric_mcp.cf_engine import CFEngine


def register(mcp: FastMCP, engine: CFEngine) -> None:
    @mcp.tool()
    def get_lexeme_info(
        lexeme: str,
        corpus: str = "hebrew",
        limit: int = 50,
    ) -> dict:
        """Look up a lexeme and return its gloss, part of speech, and occurrences.

        The lexeme should be in transliterated form (e.g. "BRJ[" for the Hebrew
        verb "to create", "HLK[" for "to walk"). Use the 'lex' values returned
        by get_passage or search_words.

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

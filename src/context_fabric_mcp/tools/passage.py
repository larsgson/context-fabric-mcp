"""MCP tools for passage retrieval."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from context_fabric_mcp.cf_engine import CFEngine


def register(mcp: FastMCP, engine: CFEngine) -> None:
    @mcp.tool()
    def get_passage(
        book: str,
        chapter: int,
        verse_start: int = 1,
        verse_end: int | None = None,
        corpus: str = "hebrew",
    ) -> dict:
        """Get biblical text for a verse range with full morphological annotations.

        Returns each word with: surface text, lexeme, gloss, part of speech,
        gender, number, person, state, verbal stem, and verbal tense.

        Args:
            book: Book name (e.g. "Genesis", "Matthew")
            chapter: Chapter number
            verse_start: Starting verse (default 1)
            verse_end: Ending verse (default same as verse_start)
            corpus: "hebrew" for OT (BHSA) or "greek" for NT (Nestle 1904)
        """
        result = engine.get_passage(book, chapter, verse_start, verse_end, corpus)
        return result.model_dump()

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

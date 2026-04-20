"""MCP tools for schema introspection."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from context_fabric_mcp.cf_engine import CFEngine


def register(mcp: FastMCP, engine: CFEngine) -> None:
    @mcp.tool()
    def list_corpora() -> list[dict]:
        """List available biblical text corpora.

        Returns corpus IDs and display names. Use the corpus ID in other tools.
        """
        return engine.list_corpora()

    @mcp.tool()
    def list_books(corpus: str = "hebrew") -> list[dict]:
        """List all books in a corpus with chapter counts.

        Args:
            corpus: "hebrew" for OT (BHSA) or "greek" for NT (Nestle 1904)
        """
        return [b.model_dump() for b in engine.list_books(corpus)]

    @mcp.tool()
    def get_schema(corpus: str = "hebrew") -> dict:
        """Get the database schema: object types (word, phrase, clause, etc.) and their features.

        Useful for discovering what features are available for querying.

        Args:
            corpus: "hebrew" or "greek"
        """
        return engine.get_schema(corpus).model_dump()

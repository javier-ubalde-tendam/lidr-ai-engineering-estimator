class RagError(Exception):
    """Base class for RAG pipeline failures."""


class RetrievalError(RagError):
    """Retrieval could not run (database or reranker failure). Not the same as "no results"."""

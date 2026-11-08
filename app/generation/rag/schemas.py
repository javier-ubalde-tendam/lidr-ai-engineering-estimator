from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchMode = Literal["vector", "hybrid"]


class RetrievedChunk(BaseModel):
    id: int = Field(description="BIGINT id of the chunk row")
    content: str
    sector: str = Field(description="Client sector of the parent budget")
    project_year: int = Field(description="Year of the parent budget")
    chunk_type: str
    distance: float = Field(
        description="Cosine distance to the query. 1.0 is a sentinel for chunks found only by "
        "the lexical branch (no vector distance); fusion and reranking never use it"
    )
    budget_id: str | None = Field(description="Parent budget (metadata->>'budget_id')")


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk] = Field(description="Best first")
    low_confidence: bool = Field(description="True when nothing was retrieved (soft-fail)")
    candidates_evaluated: int = Field(description="Chunks with an embedding considered by k-NN")
    search_mode: SearchMode = "vector"
    rerank: bool = False
    search_time_ms: int = Field(default=0, description="Retrieval + fusion + rerank, no embedding")
    vector_hits: int = Field(default=0, description="Rows returned by the vector branch")
    lexical_hits: int = Field(default=0, description="Rows returned by the lexical branch")


class RetrievalRequest(BaseModel):
    # extra="forbid": metadata filters are deliberately not exposed to clients
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0, le=2, description="Cosine distance")
    search_mode: SearchMode | None = Field(
        default=None, description="None = RETRIEVAL_SEARCH_MODE setting"
    )
    rerank: bool | None = Field(default=None, description="None = RERANKER_ENABLED setting")

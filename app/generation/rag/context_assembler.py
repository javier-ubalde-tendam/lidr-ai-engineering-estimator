"""Augmentation: assemble retrieved chunks into a delimited context block.

Two anti-patterns this module avoids:

* ``"\\n\\n".join(chunk.content ...)``: the model cannot tell where one source ends and the next
  begins, nor which id to cite. Each chunk is wrapped in a ``<source>`` element carrying its id
  and metadata so the generator can ground every claim in a specific, citable source.
* Splitting a chunk when the context overflows. ``truncate_to_token_budget`` drops whole chunks
  from the tail (least relevant first): half a budget line item is worse than one fewer.
"""

from typing import Protocol

from app.generation.rag.schemas import RetrievedChunk


class TokenEncoder(Protocol):
    """What the budget needs from a tokenizer (a ``tiktoken`` encoding satisfies it)."""

    def encode(self, text: str) -> list[int]: ...


def _wrap_chunk(chunk: RetrievedChunk) -> str:
    """Render one chunk as a self-describing ``<source>`` element.

    ``id`` is what an estimate line cites (``SourceReference.chunk_id``); ``document_id`` is the
    parent budget (copied into ``SourceReference.document_id``).
    """
    document_id = chunk.budget_id or "unknown"
    return (
        f'<source id="{chunk.id}" document_id="{document_id}" sector="{chunk.sector}" '
        f'project_year="{chunk.project_year}" chunk_type="{chunk.chunk_type}" '
        f'distance="{chunk.distance:.4f}">\n'
        f"{chunk.content}\n"
        f"</source>"
    )


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Concatenate the ``<source>`` elements, most relevant first (empty string if no chunks)."""
    return "\n".join(_wrap_chunk(chunk) for chunk in chunks)


def truncate_to_token_budget(
    chunks: list[RetrievedChunk],
    max_context_tokens: int,
    encoder: TokenEncoder,
) -> list[RetrievedChunk]:
    """Keep the leading chunks that fit within ``max_context_tokens``; never split a chunk.

    Tokens are counted over the chunk already wrapped in its ``<source>`` XML: the delimiters are
    real tokens the model will see. The first chunk that would overflow stops the loop.
    """
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(encoder.encode(_wrap_chunk(chunk)))
        if used + cost > max_context_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept

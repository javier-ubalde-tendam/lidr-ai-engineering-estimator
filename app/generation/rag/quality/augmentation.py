"""Session 11 - context augmentation: shape the retrieved chunks before assembly.

Two deterministic passes applied to the token-truncated chunks BEFORE they become the ``<source>``
block (``context_assembler``), each independently switchable so it can be measured in isolation:

* ``extract_key_points`` / ``compress_chunk``: extractive compression. Keeps the lines that carry a
  figure or an id (component name + hours, what the generator must ground on) and drops the
  filler, so the token budget buys more distinct sources.
* ``reorder_edge_loaded``: a countermeasure to *lost-in-the-middle*: the strongest sources go to
  both ends of the context and the weakest to the middle.

Pure functions: no LLM, no cost. Chunk ids are preserved, so citation verification is unaffected.
"""

from app.generation.rag.schemas import RetrievedChunk


def extract_key_points(chunk: RetrievedChunk) -> str:
    """Keep only the lines of the chunk that carry a digit or a ``::`` code-like token."""
    lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
    kept = [line for line in lines if any(ch.isdigit() for ch in line) or "::" in line]
    # Never compress to nothing: fall back to the first non-empty line
    if not kept:
        kept = lines[:1]
    return "\n".join(kept)


def compress_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    """Copy of ``chunk`` whose content is its extracted key points."""
    return chunk.model_copy(update={"content": extract_key_points(chunk)})


def reorder_edge_loaded(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder best-first chunks so the strongest sit at BOTH ends of the context.

    Even ranks go to the head (ascending) and odd ranks to the tail (reversed), so the 2nd best
    lands last and the weakest end up in the middle.
    """
    head = chunks[0::2]
    tail = chunks[1::2]
    return head + tail[::-1]


def augment_chunks(
    chunks: list[RetrievedChunk], *, compress: bool = True, reorder: bool = True
) -> list[RetrievedChunk]:
    """Apply the augmentation passes: compress first (cheaper tokens), then reorder by position."""
    out = [compress_chunk(chunk) for chunk in chunks] if compress else list(chunks)
    return reorder_edge_loaded(out) if reorder else out

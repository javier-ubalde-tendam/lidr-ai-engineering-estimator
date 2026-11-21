"""Post-generation checks for grounded estimates.

Two independent guards run after the LLM returns an ``Estimate``:

* ``verify_citations``: every cited ``chunk_id`` (per-line ``SourceReference`` and the
  estimate-global ``SourceCitation``) must resolve to a chunk that was actually retrieved. Ids
  that were never in the context are *dangling*: a hallucination wearing the costume of rigor.
* ``check_coherence``: the ``insufficient`` confidence level has a strict shape (no numbers, an
  explanation present); a violation is a malformed response, not a valid estimate.
"""

from app.generation.rag.schemas import CitationReport, Estimate, LineCitation


def verify_citations(estimate: Estimate, retrieved_chunk_ids: set[str]) -> CitationReport:
    """Verify every citation against the chunk ids actually handed to the LLM.

    ``dangling_citations`` in the report is the sorted, de-duplicated set of cited ids that were
    never retrieved; an empty list means every citation is real.
    """
    lines: list[LineCitation] = []
    verified = 0
    dangling: set[str] = set()
    grounded_lines = dangling_lines = insufficient_lines = 0

    for module in estimate.modules:
        for task in module.tasks:
            cited = [ref.chunk_id for ref in task.sources]
            line_dangling = [cid for cid in cited if cid not in retrieved_chunk_ids]
            verified += len(cited) - len(line_dangling)
            dangling.update(line_dangling)

            if not task.grounded:
                status = "insufficient"
                insufficient_lines += 1
            elif line_dangling:
                status = "dangling"
                dangling_lines += 1
            else:
                status = "grounded"
                grounded_lines += 1

            lines.append(
                LineCitation(
                    module=module.name,
                    component=task.name,
                    status=status,
                    cited_chunk_ids=cited,
                    dangling_chunk_ids=line_dangling,
                )
            )

    # The estimate-global citations count too: a fabricated id there is just as bad
    for citation in estimate.sources:
        cid = str(citation.source_id)
        if cid in retrieved_chunk_ids:
            verified += 1
        else:
            dangling.add(cid)

    return CitationReport(
        total_lines=len(lines),
        grounded_lines=grounded_lines,
        dangling_lines=dangling_lines,
        insufficient_lines=insufficient_lines,
        verified_citations=verified,
        dangling_citations=sorted(dangling),
        lines=lines,
    )


def check_coherence(estimate: Estimate) -> bool:
    """Return whether the estimate's confidence level matches its content.

    With ``confidence == "insufficient"`` both totals must be None, ``modules`` empty and
    ``insufficient_context_explanation`` non-empty. Any other level is coherent here (numeric
    checks belong to the schema/business rules, not to this guard).
    """
    if estimate.confidence != "insufficient":
        return True
    return (
        estimate.total_engineer_days is None
        and estimate.duration_weeks is None
        and not estimate.modules
        and bool(estimate.insufficient_context_explanation)
    )

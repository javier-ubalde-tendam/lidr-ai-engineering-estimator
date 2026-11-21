#!/usr/bin/env python3
"""Sub-block 2.5 demo: pseudonymization with Presidio (Spanish) + an in-memory mapping table.

Builds ``build_analyzer()`` wired to the Spanish spaCy model + the custom recognizers, then runs a
``ConsistentPseudonymizer`` over a seed transcript and prints the result. Free and offline; it
needs the spaCy model installed (see the README).

It does NOT run the "English fails" contrast on purpose: forcing the default English engine
triggers a ~382 MB ``en_core_web_lg`` download. The rationale is documented in
``app/ingestion/pii/analyzer.py``; here the fix is shown working.

Usage::

    uv run python scripts/demo_pii_s06.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Make ``app`` importable when this script runs directly from ``scripts/``
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.ingestion.pii import (
    ConsistentPseudonymizer,
    InMemoryMappingStore,
    build_analyzer,
)

TRANSCRIPT = ROOT / "data" / "seed" / "transcripts" / "transcripcion_2025-02-03_betanorte.txt"


async def main() -> int:
    if not TRANSCRIPT.exists():
        print(f"ERROR: {TRANSCRIPT} does not exist")
        return 1

    settings = get_settings()
    pseudonymizer = ConsistentPseudonymizer(
        analyzer=build_analyzer(),  # Spanish model + BUDGET_ID/CLIENT_CODE recognizers
        mapping_store=InMemoryMappingStore(),
        salt=settings.PSEUDONYM_HASH_SALT,
        faker_locale=settings.PSEUDONYM_FAKER_LOCALE,
    )
    result = await pseudonymizer.pseudonymize(TRANSCRIPT.read_text(encoding="utf-8"))

    print(f"Transcript: {TRANSCRIPT.name}")
    print("\n--- Pseudonymized text (first 400 chars) ---")
    print(result.pseudonymized_text[:400])

    print(f"\n--- Applied mappings ({len(result.applied)}) ---")
    for mapping in result.applied[:12]:
        print(f"  {mapping.entity_type:14} -> {mapping.pseudonym}")
    print("\nSame original value -> same pseudonym across the whole document (consistency per value).")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

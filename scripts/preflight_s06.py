#!/usr/bin/env python3
"""Session 6 pre-flight verification.

Run it BEFORE using the ingestion endpoints to catch the usual setup pitfalls: missing Spanish
spaCy model, broken catalog, missing seed data, Postgres unreachable or not migrated. Prints OK /
FAIL per check; the exit code is non-zero if anything fails. Free and offline (the optional
``/health`` probe only talks to localhost).

Usage::

    uv run python scripts/preflight_s06.py
    # or, inside docker:
    docker compose exec ai_service python scripts/preflight_s06.py
"""

import asyncio
import importlib.metadata as importlib_metadata
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

OK = "OK  "
FAIL = "FAIL"

ROOT = Path(__file__).resolve().parent.parent
# Make ``app`` importable when this script runs directly from ``scripts/``
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `unstructured` and `openpyxl` are NOT checked: nothing in this project's ingestion reads
# XLSX/DOCX/PDF yet (only JSON and TXT parsers are registered)
_MIN_VERSIONS = {
    "pandas": "2.2.0",
    "pandera": "0.20.0",
    "pyyaml": "6.0",
    "sqlalchemy": "2.0.0",
    "asyncpg": "0.29.0",
    "alembic": "1.13.0",
    "presidio-analyzer": "2.2.0",
    "presidio-anonymizer": "2.2.0",
    "spacy": "3.7.0",
    "faker": "24.0.0",
}


class Check:
    def __init__(self, name: str, fn: Callable[[], str | None]) -> None:
        self.name = name
        self._fn = fn

    def run(self) -> bool:
        try:
            detail = self._fn()
        except Exception as exc:  # noqa: BLE001 - a failing check is reported, not raised
            print(f"{FAIL}  {self.name}: {type(exc).__name__}: {exc}")
            return False
        print(f"{OK}  {self.name}" + (f": {detail}" if detail else ""))
        return True


def check_python_version() -> str:
    major, minor, *_ = sys.version_info
    if (major, minor) < (3, 11):
        raise RuntimeError(f"Python 3.11+ required, found {major}.{minor}")
    return f"{major}.{minor}.{sys.version_info.micro}"


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def check_packages() -> str:
    missing: list[str] = []
    too_old: list[str] = []
    for package, minimum in _MIN_VERSIONS.items():
        try:
            installed = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            missing.append(package)
            continue
        if _parse_version(installed) < _parse_version(minimum):
            too_old.append(f"{package} {installed} < {minimum}")
    if missing or too_old:
        raise RuntimeError("missing=" + ",".join(missing) + " too_old=" + ",".join(too_old))
    return f"{len(_MIN_VERSIONS)} packages OK"


def check_spacy_model() -> str:
    import spacy

    from app.config import get_settings

    model = get_settings().PRESIDIO_SPACY_MODEL
    nlp = spacy.load(model)
    persons = [e.text for e in nlp("Laura Fernández firmó el contrato.").ents if e.label_ == "PER"]
    if "Laura Fernández" not in " ".join(persons):
        raise RuntimeError(f"{model} loaded but did NOT detect 'Laura Fernández' as PER")
    return f"{model} loaded; detected PER entities: {persons}"


def check_corpus_seed() -> str:
    from app.config import get_settings

    seed = ROOT / get_settings().INGESTION_DATA_ROOT
    budgets = list((seed / "budgets").glob("*.json"))
    transcripts = list((seed / "transcripts").glob("*.txt"))
    if not budgets or not transcripts:
        raise RuntimeError(f"corpus incomplete: budgets={len(budgets)}, transcripts={len(transcripts)}")
    return f"{len(budgets)} json + {len(transcripts)} txt under {seed}"


def check_catalog() -> str:
    from app.config import get_settings
    from app.ingestion.catalog import load_catalog

    catalog = load_catalog(ROOT / get_settings().CATALOG_PATH)
    return f"version {catalog.version}, {len(catalog.included_sources())} included"


def check_health_endpoint() -> str:
    from app.config import get_settings

    url = f"{get_settings().API_BASE_URL}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return f"{response.status} OK"
    except Exception as exc:
        raise RuntimeError(f"GET {url} failed: is the API running? ({exc})") from exc


def check_postgres() -> str:
    """Connect to Postgres and confirm the ingestion migration (0005) has been applied."""
    from sqlalchemy import inspect

    from app.foundation.persistence.database import get_engine

    async def table_names() -> set[str]:
        async with get_engine().connect() as connection:
            return set(await connection.run_sync(lambda sync: inspect(sync).get_table_names()))

    expected = {"pseudonym_mappings", "ingestion_jobs", "alembic_version"}
    missing = expected - asyncio.run(table_names())
    if missing:
        raise RuntimeError(f"missing tables: {sorted(missing)}; run `alembic upgrade head`")
    return f"tables OK: {sorted(expected)}"


CHECKS = [
    Check("Python version", check_python_version),
    Check("Required packages", check_packages),
    Check("spaCy Spanish model", check_spacy_model),
    Check("Corpus seed present", check_corpus_seed),
    Check("Catalog validates", check_catalog),
    Check("API /health", check_health_endpoint),
    Check("Postgres + migration", check_postgres),
]


def main() -> int:
    print("Pre-flight Session 06\n")
    failed = sum(not check.run() for check in CHECKS)
    print()
    if failed:
        print(f"{FAIL}  {failed} check(s) failed")
        return 1
    print(f"{OK}  All checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

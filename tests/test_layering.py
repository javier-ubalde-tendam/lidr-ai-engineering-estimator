"""Enforces the dependency rules of ARCHITECTURE.md §3 by parsing every import under ``app/``."""

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"

# Known, documented deviations: (importing module prefix, imported module prefix).
# Keep this list short; every entry is technical debt with a reason.
ALLOWED_EXCEPTIONS: list[tuple[str, str]] = [
    # Output guardrail builds the EstimationResult contract it filters.
    ("app.foundation.guardrails.output", "app.domain.schemas"),
    # Prompt rendering is typed against the contract schemas and the conversation value objects.
    ("app.foundation.prompts.loader", "app.domain.schemas"),
    ("app.foundation.prompts.loader", "app.generation.conversation.models"),
]


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(APP_DIR.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _layer(module: str) -> tuple[str, str | None]:
    """Return (layer, sibling) where sibling is only set for ``generation.<x>``."""
    parts = module.split(".")
    top = parts[1] if len(parts) > 1 else ""
    if top in {"config", "logging_config", "dependencies", "main"}:
        return top, None
    if top == "foundation":
        return "foundation", None
    if top == "domain":
        # domain/graph (Session 13) composes generation + foundation: same layer as the conductor
        if len(parts) > 2 and parts[2] in {"estimation_service", "graph"}:
            return "conductor", None
        return "schemas", None
    if top == "generation":
        return "generation", parts[2] if len(parts) > 2 else None
    if top in {"ingestion", "api"}:
        return top, None
    return "unknown", None


_BASE = {"config", "logging_config", "foundation"}
ALLOWED_LAYERS: dict[str, set[str]] = {
    "config": set(),
    "logging_config": {"config"},
    "foundation": _BASE,
    "schemas": _BASE | {"schemas"},
    "generation": _BASE | {"schemas", "generation"},
    "conductor": _BASE | {"schemas", "generation", "conductor"},
    "ingestion": _BASE | {"schemas", "ingestion", "generation"},
    "api": _BASE | {"schemas", "conductor", "api", "dependencies", "generation", "ingestion"},
    "main": _BASE | {"api"},
    "dependencies": {"config", "logging_config", "foundation", "schemas", "conductor", "generation",
                     "ingestion", "dependencies"},
}
# Only agentic may reach into conversation (never the other way round).
ALLOWED_SIBLING_EDGES = {("agentic", "conversation")}


def _imports(path: Path) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
            # `from app.generation import rag` imports the submodule app.generation.rag
            found += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return [(line, name) for line, name in found if name.split(".")[0] == "app"]


def _violations() -> list[str]:
    problems: list[str] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        module = _module_name(path)
        src_layer, src_sibling = _layer(module)
        for line, target in _imports(path):
            if any(module.startswith(a) and target.startswith(b) for a, b in ALLOWED_EXCEPTIONS):
                continue
            dst_layer, dst_sibling = _layer(target)
            if dst_layer == "unknown":
                continue
            if src_layer == dst_layer == "generation":
                same = src_sibling == dst_sibling or src_sibling is None or dst_sibling is None
                if not same and (src_sibling, dst_sibling) not in ALLOWED_SIBLING_EDGES:
                    problems.append(
                        f"{module}:{line} imports sibling {target} "
                        f"(generation/{src_sibling} -> generation/{dst_sibling})"
                    )
                continue
            if dst_layer == src_layer:
                continue
            if dst_layer not in ALLOWED_LAYERS[src_layer]:
                problems.append(f"{module}:{line} ({src_layer}) must not import {target} ({dst_layer})")
    return problems


def test_imports_respect_layer_rules() -> None:
    problems = _violations()
    assert not problems, "Layering violations:\n" + "\n".join(problems)


def test_generation_siblings_do_not_import_each_other() -> None:
    sibling_problems = [p for p in _violations() if "imports sibling" in p]
    assert not sibling_problems, "\n".join(sibling_problems)


@pytest.mark.parametrize("module", ["app.config", "app.logging_config"])
def test_root_modules_are_classified(module: str) -> None:
    assert _layer(module)[0] == module.split(".")[1]

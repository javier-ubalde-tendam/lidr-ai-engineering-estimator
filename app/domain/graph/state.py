"""Typed state of the estimation graph.

Every node receives the whole state and returns only a *partial update*; LangGraph merges it.
Plain fields are overwritten by the last writer. The two ``Annotated[..., operator.add]``
fields are accumulators: see the docstring of ``EstimationState``.
"""

import operator
from typing import Annotated

# typing_extensions.TypedDict: Pydantic/LangGraph introspection requires it on Python < 3.12
from typing_extensions import TypedDict


class Component(TypedDict):
    name: str
    category: str


class BudgetMatch(TypedDict):
    component: str
    reference_budget_id: str | None
    # Engineer-HOURS of the historical task (the corpus stores hours, not euros)
    amount: float
    # Cosine distance to the query: lower is more similar
    distance: float


class EstimationState(TypedDict, total=False):
    """State shared by all nodes.

    Two reducers (``operator.add``) make ``budget_matches`` and ``errors`` *accumulate*:

    - ``budget_matches``: each component search returns only its own matches and LangGraph
      concatenates them, so a node never has to read-modify-write the whole list.
    - ``errors``: any node (a soft retrieval failure, a validation issue, the review flag) can
      report a problem without overwriting what an earlier node reported.
    """

    transcript: str
    estimation_id: str
    requirements: list[str]
    components: list[Component]
    budget_matches: Annotated[list[BudgetMatch], operator.add]
    estimate: dict | None
    # "validated" | "needs_review"
    status: str | None
    errors: Annotated[list[str], operator.add]

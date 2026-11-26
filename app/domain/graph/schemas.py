"""Private Pydantic I/O models of the LLM nodes. The estimate reuses ``AgentEstimate`` (hours)."""

from pydantic import BaseModel, Field


class RequirementsExtraction(BaseModel):
    requirements: list[str] = Field(
        description="Atomic, self-contained project requirements stated in the transcript."
    )


class ComponentModel(BaseModel):
    name: str = Field(description="Short component name, e.g. a payment gateway integration.")
    category: str = Field(description="Broad category, e.g. backend, frontend, integration, data.")


class ComponentClassification(BaseModel):
    components: list[ComponentModel]

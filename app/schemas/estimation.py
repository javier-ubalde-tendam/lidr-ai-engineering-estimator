from pydantic import BaseModel, Field

class EstimationRequest(BaseModel):
    transcription: str = Field(..., min_length=50, description="The transcription of the meeting")

class EstimationResponse(BaseModel):
    estimation: str = Field(..., description="The generated estimation")
    model: str = Field(..., description="The model used for estimation")
    provider: str = Field(..., description="The provider of the model")
    tokens_input: int = Field(..., description="The number of input tokens")
    tokens_output: int = Field(..., description="The number of output tokens")
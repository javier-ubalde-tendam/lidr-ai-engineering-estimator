from pydantic import BaseModel

class EstimationRequest(BaseModel):
    transcription: str

class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    tokens_input: int
    tokens_output: int
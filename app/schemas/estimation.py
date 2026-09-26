from pydantic import BaseModel

class EstimationRequest(BaseModel):
    transcription: str

class EstimationResponse(BaseModel):
    estimation: str
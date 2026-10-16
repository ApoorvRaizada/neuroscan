from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    disclaimer: str = "Research demo only. Not a medical device. Not for diagnosis."


class HealthResponse(BaseModel):
    status: str
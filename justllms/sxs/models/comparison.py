from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ResponseStatus(Enum):
    """Status of a model response."""
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ModelResponse:
    """Response from a single model."""
    provider: str
    model: str
    content: str
    status: ResponseStatus
    latency: float
    tokens: int
    cost: float
    error: Optional[str] = None

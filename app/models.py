# app/models.py
from pydantic import BaseModel
from typing import List

class VisemeWeight(BaseModel):
    name: str
    weight: float  # 0..1

class VisemeSegment(BaseModel):
    timestamp: float
    weights: List[VisemeWeight]

class VisemeResponse(BaseModel):
    segments: List[VisemeSegment]
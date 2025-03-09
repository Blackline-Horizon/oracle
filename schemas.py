# schemas.py

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, date

class AlertPrediction(BaseModel):
    alert_id: int
    coordinates: str  # WKT representation of coordinates
    prediction: str  # Dummy prediction

class PredictionResponse(BaseModel):
    total_alerts: int
    predictions: List[AlertPrediction]

class GetReport(BaseModel):
    resolutions: List[int] = Field(...)
    devices: List[int] = Field(...)
    sensors: List[int] = Field(...)
    events: List[int] = Field(...)
    industry: Optional[List[int]] = Field(default=None)
    date_start: date = Field(...)
    date_end: date = Field(...)
    countries: Optional[List[int]] = Field(default=None) 
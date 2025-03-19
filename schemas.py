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
    resolution_reason: Optional[List[str]] = None
    device_type: Optional[List[str]] = None
    sensor_type: Optional[List[str]] = None
    event_type: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    continent: Optional[List[str]] = None
    date_start: datetime = Field(...)
    date_end: datetime = Field(...)
    country: Optional[List[str]] = None
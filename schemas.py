# schemas.py

from pydantic import BaseModel
from typing import List
from datetime import datetime

class AlertPrediction(BaseModel):
    alert_id: int
    coordinates: str  # WKT representation of coordinates
    prediction: str  # Dummy prediction

class PredictionResponse(BaseModel):
    total_alerts: int
    predictions: List[AlertPrediction]
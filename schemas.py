# schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date


class GetReport(BaseModel):
    resolution_reason: Optional[List[str]] = None
    device_type: Optional[List[str]] = None
    sensor_type: Optional[List[str]] = None
    event_type: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    continent: Optional[List[str]] = None
    date_end: datetime = Field(...)
    country: Optional[List[str]] = None
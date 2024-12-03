# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2.functions import ST_AsText
from dotenv import load_dotenv
import os

from models import Base, Alert, Type
from schemas import AlertPrediction, PredictionResponse

# Load environment variables
load_dotenv()

# Get values from environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 3007))  # Use a different port for this service
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create synchronous engine
engine = create_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
)

# Create sessionmaker with autoflush enabled
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=True,
    bind=engine
)

@app.on_event("startup")
def startup_event():
    # Create the 'map' schema if it doesn't exist
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS map"))
    # Create tables within the 'map' schema
    Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Dummy Prediction Service is running!"}

@app.get("/predictions", response_model=PredictionResponse)
def get_predictions():
    db = SessionLocal()
    try:
        # Query alerts and types
        results = db.query(Alert, Type).join(Type).all()

        # Generate dummy predictions
        predictions = []
        for alert, alert_type in results:
            predictions.append(
                AlertPrediction(
                    alert_id=alert.alert_id,
                    coordinates=db.scalar(func.ST_AsText(alert.coordinates)),
                    prediction=f"Dummy prediction for {alert_type.name}"
                )
            )

        response = PredictionResponse(
            total_alerts=len(predictions),
            predictions=predictions
        )

        return response
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
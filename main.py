# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import pickle
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas as pd

from models import Base, Alert
from schemas import GetReport

# Load environment variables
load_dotenv(dotenv_path=".env")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 3001))
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

# Create synchronous engine and sessionmaker
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=True, bind=engine)

@app.on_event("startup")
def startup_event():
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS map"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Prediction Service is up and running!"}

def create_input_sequence(data: np.ndarray, timesteps: int = 3) -> np.ndarray:
    """
    Create a 3D array of sequences from the 2D data array.
    For prediction purposes, we will flatten the last timesteps into one vector.
    """
    X = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i : i + timesteps])
    return np.array(X)

@app.post("/report_data")
def get_report_info(filters: GetReport):
    try:
        # Load the pre-trained SVR model and scaler from pickle files.
        with open("models/svr_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Define the time range for the last 4 months.
        end_date = filters.date_end
        start_date = end_date - relativedelta(months=4)

        # Start a new database session.
        session: Session = SessionLocal()
        query = session.query(Alert).filter(
            Alert.date_created >= start_date,
            Alert.date_created <= end_date
        )

        # Apply additional filters if provided.
        if filters.resolution_reason:
            query = query.filter(Alert.resolution_reason.in_(filters.resolution_reason))
        if filters.device_type:
            query = query.filter(Alert.device_type.in_(filters.device_type))
        if filters.sensor_type:
            query = query.filter(Alert.sensor_type.in_(filters.sensor_type))
        if filters.event_type:
            query = query.filter(Alert.event_type.in_(filters.event_type))
        if filters.industry:
            query = query.filter(Alert.industry.in_(filters.industry))
        if filters.country:
            query = query.filter(Alert.country.in_(filters.country))
        # (If needed, add continent filters here using your own logic.)

        alerts = query.all()
        session.close()

        # Convert query results to a DataFrame.
        # Exclude the '_sa_instance_state' attribute.
        df = pd.DataFrame([
            {k: v for k, v in a.__dict__.items() if k != '_sa_instance_state'}
            for a in alerts
        ])
        if df.empty:
            raise HTTPException(status_code=404, detail="No alert data found for the given filters.")

        # Ensure the date_created column is datetime and add a helper count column.
        df['date_created'] = pd.to_datetime(df['date_created'])
        df["count"] = 1
        df.set_index('date_created', inplace=True)

        # --- Reconstruct the feature engineering pipeline used in training ---
        # 1. Aggregate total alerts by month end (~4 weeks)
        monthly_total = df.resample("ME").sum()[["count"]]
        monthly_total.rename(columns={"count": "total_alerts"}, inplace=True)

        # 2. Create categorical dummy columns for the categorical features
        cat_columns = ["device_type", "sensor_type", "event_type", "resolution_reason", "industry"]
        df_dummies = pd.get_dummies(df[cat_columns])
        monthly_cats = df_dummies.resample("ME").sum()

        # 3. Join the aggregated total alerts and the categorical aggregates
        base_monthly_data = monthly_total.join(monthly_cats)

        # 4. Create additional time features as in training.
        base_monthly_data['year_fraction'] = (base_monthly_data.index.dayofyear - 1) / 365.0
        base_monthly_data['time_idx'] = (
            (base_monthly_data.index.year - base_monthly_data.index.year.min()) * 12 +
            base_monthly_data.index.month
        )
        # Create a scaled time index if that feature was used.
        min_time = base_monthly_data['time_idx'].min()
        max_time = base_monthly_data['time_idx'].max()
        base_monthly_data['time_idx_scaled'] = (
            (base_monthly_data['time_idx'] - min_time) / (max_time - min_time)
            if max_time != min_time else 0
        )

        # 5. Ensure that the dataset has exactly the features the scaler was fitted on.
        # It is assumed that the scaler has the attribute `feature_names_in_`.
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in base_monthly_data.columns:
                base_monthly_data[col] = 0  # populate missing features with zeros

        # Reorder columns to match the expected order.
        features_df = base_monthly_data[expected_features]

        # Scale the features.
        scaled_features = scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_features, index=features_df.index, columns=expected_features)

        # For prediction, we need at least (TIME_STEPS + 1) data points.
        TIME_STEPS = 3
        if len(scaled_df) < TIME_STEPS + 1:
            raise HTTPException(status_code=400, detail="Not enough historical data to generate predictions.")

        # Convert the scaled DataFrame to a NumPy array.
        data_array = scaled_df.values

        # Construct input sequences:
        # - Use the first TIME_STEPS rows to predict the 4th period (predicted_last_4w).
        # - Use the last TIME_STEPS rows to predict the next period (predicted_next_4w).
        X_pred_last = data_array[0:TIME_STEPS].reshape(1, -1)
        X_pred_next = data_array[-TIME_STEPS:].reshape(1, -1)

        # Make predictions (scaled values).
        pred_last_scaled = model.predict(X_pred_last)
        pred_next_scaled = model.predict(X_pred_next)

        # Inverse transform the predictions to the original scale.
        dummy_last = np.zeros((1, scaled_df.shape[1]))
        dummy_last[0, 0] = pred_last_scaled  # assuming 'total_alerts' is at index 0
        inv_pred_last = scaler.inverse_transform(dummy_last)[0, 0]

        dummy_next = np.zeros((1, scaled_df.shape[1]))
        dummy_next[0, 0] = pred_next_scaled
        inv_pred_next = scaler.inverse_transform(dummy_next)[0, 0]

        # Get the actual total alerts for the last period from the aggregated data.
        actual_last = base_monthly_data["total_alerts"].iloc[-1]

        return {
            "predicted_last_4w": float(inv_pred_last),
            "actual_last_4w": int(actual_last),
            "predicted_next_4w": float(inv_pred_next)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
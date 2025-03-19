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


@app.post("/report_data")
def get_report_data(filters: GetReport):
    try:
        # Load the pre-trained SVR model and scaler from pickle files.
        with open("models/svr_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Define the time range for the last 4 months.
        end_date = filters.date_end.replace(tzinfo=None)
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
        # (Add continent filters if needed.)

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

        # --- Aggregate data into 4-week (28-day) buckets aligned with end_date ---
        # Use resample with a 28-day frequency. The "origin" parameter aligns the buckets so that the last bucket ends at end_date.
        total_agg = df.resample("28D", origin=end_date, label='right', closed='right').sum()[["count"]]
        total_agg.rename(columns={"count": "total_alerts"}, inplace=True)

        # Create categorical dummy columns for selected features.
        cat_columns = ["device_type", "sensor_type", "event_type", "resolution_reason", "industry"]
        df_dummies = pd.get_dummies(df[cat_columns])
        cat_agg = df_dummies.resample("28D", origin=end_date, label='right', closed='right').sum()

        # Join the total alerts and categorical aggregates.
        base_data = total_agg.join(cat_agg)

        # --- Create additional time features ---
        # Use the bucket end date to compute a fractional representation of the year.
        base_data['year_fraction'] = (base_data.index.dayofyear - 1) / 365.0
        # Use a sequential index for the buckets.
        base_data['time_idx'] = np.arange(len(base_data))
        # Optionally add a rolling average of total_alerts.
        base_data['rolling_avg'] = base_data['total_alerts'].rolling(window=3, min_periods=1).mean()
        # Create a scaled time index.
        min_time = base_data['time_idx'].min()
        max_time = base_data['time_idx'].max()
        base_data['time_idx_scaled'] = ((base_data['time_idx'] - min_time) / (max_time - min_time)
                                         if max_time != min_time else 0)

        # --- Ensure the feature set matches what the scaler was fitted on ---
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in base_data.columns:
                base_data[col] = 0  # populate missing features with zeros
        # Reorder columns to match the expected order.
        features_df = base_data[expected_features]

        # Scale the features.
        scaled_features = scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_features, index=features_df.index, columns=expected_features)

        # --- Build prediction inputs ---
        # We require at least 4 buckets (4 periods) within the 4-month window.
        TIME_STEPS = 3  # number of buckets to feed as input
        if len(scaled_df) < TIME_STEPS + 1:
            raise HTTPException(status_code=400, detail="Not enough historical data to generate predictions.")

        # Select the most recent 4 buckets from the aggregated data.
        recent_scaled = scaled_df.iloc[-(TIME_STEPS + 1):]  # shape: (4, number_of_features)
        data_array = recent_scaled.values

        # According to the requirement:
        # - "predicted_last_4w" uses data from 4 months ago to 1 month ago: that is the oldest 3 buckets.
        # - "predicted_next_4w" uses data from 3 months ago to today: that is the most recent 3 buckets.
        X_pred_last = data_array[0:TIME_STEPS].reshape(1, -1)   # buckets 0,1,2 (oldest 3)
        X_pred_next = data_array[1:TIME_STEPS+1].reshape(1, -1)   # buckets 1,2,3 (most recent 3)

        # Make predictions (these are scaled values).
        pred_last_scaled = model.predict(X_pred_last)
        pred_next_scaled = model.predict(X_pred_next)

        # Inverse transform the predictions.
        dummy_last = np.zeros((1, scaled_df.shape[1]))
        dummy_last[0, 0] = pred_last_scaled  # assuming the target 'total_alerts' is in column index 0
        inv_pred_last = scaler.inverse_transform(dummy_last)[0, 0]

        dummy_next = np.zeros((1, scaled_df.shape[1]))
        dummy_next[0, 0] = pred_next_scaled
        inv_pred_next = scaler.inverse_transform(dummy_next)[0, 0]

        # Get the actual total alerts for the last observed bucket (i.e. the bucket ending 1 month ago).
        # Since our buckets are ordered chronologically, the 4th bucket (index -2) corresponds to the last complete bucket
        # if we assume that the very last bucket might be a partial bucket. Here we take the second-to-last bucket.
        recent_base = base_data.iloc[-(TIME_STEPS + 1):]
        actual_last = recent_base["total_alerts"].iloc[-2]  # second-to-last bucket

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
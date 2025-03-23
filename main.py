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
from mappings import (
    DEVICE_TYPE_MAPPING, EVENT_TYPE_MAPPING, RESOLUTION_REASON_MAPPING,
    INDUSTRY_MAPPING, SENSOR_TYPE_MAPPING, COUNTRY_MAPPING, CONTINENT_MAPPING
)

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


def convert_filter(values, mapping: dict):
    """Convert a list of filter strings into their corresponding int IDs."""
    if values:
        converted = [mapping[val] for val in values if val in mapping]
        return converted if converted else None
    return None

def combine_country_filters(country_list, continent_list):
    """
    Combine country and continent filters.
    Convert the list of country names (if any) and the countries mapped from continents,
    then return a single list of country IDs.
    """
    combined_ids = set()
    if country_list:
        ids = convert_filter(country_list, COUNTRY_MAPPING)
        if ids:
            combined_ids.update(ids)
    if continent_list:
        continent_countries = []
        for cont in continent_list:
            continent_countries.extend(CONTINENT_MAPPING.get(cont, []))
        ids = convert_filter(continent_countries, COUNTRY_MAPPING)
        if ids:
            combined_ids.update(ids)
    return list(combined_ids) if combined_ids else None


@app.post("/report_data")
def get_report_data(filters: GetReport):
    try:
        # Load the pre-trained linear regression model and its experiment configuration.
        with open("models/best_lr_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/best_lr_experiment.pkl", "rb") as f:
            best_exp_lr = pickle.load(f)

        # Get expected features and time_steps from the experiment details.
        expected_features = best_exp_lr["features"]
        TIME_STEPS = best_exp_lr["time_steps"]
        print(expected_features)
        print(TIME_STEPS)
        # Define the time range for the last 4 months.
        end_date = filters.date_end.replace(tzinfo=None)
        start_date = end_date - relativedelta(months=TIME_STEPS)

        # Start a new database session.
        session: Session = SessionLocal()
        # Build the base query with date filters and any additional filters.
        query = session.query(Alert)
        query = query.filter(
            Alert.date_created >= start_date,
            Alert.date_created <= end_date
        )
        res_ids = convert_filter(filters.resolution_reason, RESOLUTION_REASON_MAPPING)
        if res_ids:
            query = query.filter(Alert.resolution_reason_id.in_(res_ids))
        dev_ids = convert_filter(filters.device_type, DEVICE_TYPE_MAPPING)
        if dev_ids:
            query = query.filter(Alert.device_type_id.in_(dev_ids))
        sens_ids = convert_filter(filters.sensor_type, SENSOR_TYPE_MAPPING)
        if sens_ids:
            query = query.filter(Alert.sensor_type_id.in_(sens_ids))
        event_ids = convert_filter(filters.event_type, EVENT_TYPE_MAPPING)
        if event_ids:
            query = query.filter(Alert.event_type_id.in_(event_ids))
        ind_ids = convert_filter(filters.industry, INDUSTRY_MAPPING)
        if ind_ids:
            query = query.filter(Alert.industry_id.in_(ind_ids))

        # Combine country and continent filters if provided.
        combined_country_ids = combine_country_filters(filters.country, filters.continent)
        if combined_country_ids:
            query = query.filter(Alert.country_id.in_(combined_country_ids))

        alerts = query.all()
        session.close()

        # Convert query results to a DataFrame (excluding SQLAlchemy internal state).
        df = pd.DataFrame([
            {k: v for k, v in a.__dict__.items() if k != '_sa_instance_state'}
            for a in alerts
        ])
        if df.empty:
            raise HTTPException(status_code=404, detail="No alert data found for the given filters.")

        # Prepare the raw data.
        df['date_created'] = pd.to_datetime(df['date_created'])
        df["count"] = 1
        df.set_index('date_created', inplace=True)

        # --- Aggregate data into 28-day buckets aligned with end_date ---
        total_agg = df.resample("28D", origin=end_date, label='right', closed='right').sum()[["count"]]
        total_agg.rename(columns={"count": "total_alerts"}, inplace=True)

        # Create categorical dummy columns for selected features.
        cat_columns = ["device_type", "sensor_type", "event_type", "resolution_reason", "industry"]
        df_dummies = pd.get_dummies(df[cat_columns])
        cat_agg = df_dummies.resample("28D", origin=end_date, label='right', closed='right').sum()

        # Join total alerts and categorical aggregates.
        base_data = total_agg.join(cat_agg)

        # --- Create additional time features (consistent with your training configuration) ---
        # Year fraction encoding.
        base_data['year_fraction'] = (base_data.index.dayofyear - 1) / 365.0
        # Time index: using (year difference * 12 + month) to capture trend over time.
        base_data['time_idx'] = (base_data.index.year - base_data.index.year.min()) * 12 + base_data.index.month
        # Add rolling average if it was used during training.
        if best_exp_lr["config"].get("use_rolling_avg", False):
            base_data['rolling_avg'] = base_data['total_alerts'].rolling(window=3, min_periods=1).mean()

        # --- Ensure the feature set matches what the model was trained on ---
        for col in expected_features:
            if col not in base_data.columns:
                base_data[col] = 0  # populate missing features with zeros
        # Reorder columns to match the expected order.
        features_df = base_data[expected_features]

        # Check that there are enough data points to create the required time steps.
        if len(features_df) < TIME_STEPS + 1:
            raise HTTPException(status_code=404, detail="Not enough historical data to generate predictions.")

        # --- Build prediction inputs ---
        # Select the most recent TIME_STEPS + 1 buckets.
        recent_data = features_df.iloc[-(TIME_STEPS + 1):]
        data_array = recent_data.values
        # According to requirements:
        # - "predicted_last_4w" uses buckets 0 to TIME_STEPS-1 (oldest TIME_STEPS buckets).
        # - "predicted_next_4w" uses buckets 1 to TIME_STEPS (most recent TIME_STEPS buckets).
        X_pred_last = data_array[0:TIME_STEPS].reshape(1, -1)
        X_pred_next = data_array[1:TIME_STEPS+1].reshape(1, -1)

        # Make predictions.
        # Note: The linear regression model is a TransformedTargetRegressor whose predict()
        # method returns predictions in the original target space.
        pred_last = model.predict(X_pred_last)
        pred_next = model.predict(X_pred_next)

        # Get the actual total alerts for the last observed complete bucket.
        # We take the second-to-last bucket from the original (non-flattened) base_data.
        recent_base = base_data.iloc[-(TIME_STEPS + 1):]
        actual_last = recent_base["total_alerts"].iloc[-2]

        return {
            "predicted_last_4w": int(pred_last),
            "actual_last_4w": int(actual_last),
            "predicted_next_4w": int(pred_next)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=HOST, port=PORT, log_level="info")
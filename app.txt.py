

AIRLINES = [
    "Jet Airways",
    "IndiGo",
    "Air India",
    "Multiple carriers",
    "SpiceJet",
    "Vistara",
    "GoAir",
    "Air Asia",
    "Multiple carriers Premium economy",
    "Jet Airways Business",
    "Trujet",
    "Vistara Premium economy",
]

SOURCES = ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"]

DESTINATIONS = ["Cochin", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Banglore"]

TOTAL_STOPS_OPTIONS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]

TOTAL_STOPS_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


AIRLINE_TO_CODE = {name: i for i, name in enumerate(AIRLINES)}
DEST_TO_CODE = {name: i for i, name in enumerate(DESTINATIONS)}



--------------------


from __future__ import annotations

import numpy as np
import pandas as pd


def build_feature_row(
    *,
    model,
    airline_code: int,
    destination_code: int,
    total_stops: int,
    date_day: int,
    date_month: int,
    dep_hour: int,
    dep_min: int,
    arr_hour: int,
    arr_min: int,
    duration_hours: int,
    duration_minutes: int,
    source: str,
) -> pd.DataFrame:


    
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        
        feature_names = np.array(
            [
                "Airline",
                "Destination",
                "Total_Stops",
                "Date_of_Journey_day",
                "Date_of_Journey_month",
                "Dep_time_hour",
                "Dep_time_minutes",
                "Arrival_Time_hour",
                "Arrival_Time_minutes",
                "Duration_hours",
                "Duration_minutes",
                
                "Source_Banglore",
                "Source_Chennai",
                "Source_Delhi",
                "Source_Kolkata",
                "Source_Mumbai",
            ],
            dtype=object,
        )

    row = {col: 0 for col in feature_names}

    
    for k, v in {
        "Airline": airline_code,
        "Destination": destination_code,
        "Total_Stops": total_stops,
        "Date_of_Journey_day": date_day,
        "Date_of_Journey_month": date_month,
        "Dep_time_hour": dep_hour,
        "Dep_time_minutes": dep_min,
        "Arrival_Time_hour": arr_hour,
        "Arrival_Time_minutes": arr_min,
        "Duration_hours": duration_hours,
        "Duration_minutes": duration_minutes,
    }.items():
        if k in row:
            row[k] = v

    
    
    source_col = f"Source_{source}"
    if source_col in row:
        row[source_col] = 1
    else:
        
        pass

    return pd.DataFrame([row], columns=feature_names)




----------------------


import os
import pickle
import streamlit as st

from src.config import (
    AIRLINES,
    DESTINATIONS,
    SOURCES,
    TOTAL_STOPS_OPTIONS,
    TOTAL_STOPS_MAP,
    AIRLINE_TO_CODE,
    DEST_TO_CODE,
)
from src.preprocess import build_feature_row


APP_TITLE = "Flight Fare Prediction (Random Forest)"


@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)

    model_path = os.path.join("models", "rd_random.pkl")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}\n\nPut your pkl here: models/rd_random.pkl")
        st.stop()

    model = load_model(model_path)

    st.write("Enter trip details and get a predicted fare.")

    
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Airline", AIRLINES, index=0)
        source = st.selectbox("Source", SOURCES, index=0)
        total_stops_label = st.selectbox("Total Stops", TOTAL_STOPS_OPTIONS, index=1)

    with col2:
        destination = st.selectbox("Destination", DESTINATIONS, index=0)
        date_day = st.number_input("Date_of_Journey_day", min_value=1, max_value=31, value=15, step=1)
        date_month = st.number_input("Date_of_Journey_month", min_value=1, max_value=12, value=6, step=1)

    
    with st.expander("More inputs (needed by your model)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            dep_hour = st.slider("Dep_time_hour", 0, 23, 10)
            dep_min = st.slider("Dep_time_minutes", 0, 59, 30)
        with c2:
            arr_hour = st.slider("Arrival_Time_hour", 0, 23, 12)
            arr_min = st.slider("Arrival_Time_minutes", 0, 59, 0)
        with c3:
            duration_hours = st.slider("Duration_hours", 0, 24, 2)
            duration_minutes = st.slider("Duration_minutes", 0, 59, 30)

    airline_code = AIRLINE_TO_CODE.get(airline, 0)
    destination_code = DEST_TO_CODE.get(destination, 0)
    total_stops = TOTAL_STOPS_MAP[total_stops_label]

    if st.button("Predict Fare"):
        X = build_feature_row(
            model=model,
            airline_code=airline_code,
            destination_code=destination_code,
            total_stops=total_stops,
            date_day=int(date_day),
            date_month=int(date_month),
            dep_hour=int(dep_hour),
            dep_min=int(dep_min),
            arr_hour=int(arr_hour),
            arr_min=int(arr_min),
            duration_hours=int(duration_hours),
            duration_minutes=int(duration_minutes),
            source=source,
        )

        try:
            pred = model.predict(X)[0]
            st.success(f"Predicted Fare: ₹ {pred:,.0f}")
            st.caption("If your Airline/Destination encodings differ from training, update src/config.py mappings.")
            with st.expander("Show model input row"):
                st.dataframe(X)
        except Exception as e:
            st.error("Prediction failed. This usually means your feature columns don't match training.")
            st.code(str(e))
            st.write("If you see a column mismatch error, check:")
            st.write("- `model.feature_names_in_` exists and matches your training features")
            st.write("- Source dummy column names match exactly (e.g., `Source_Delhi` etc.)")
            st.write("- Airline/Destination numeric encoding matches what you used while training")


if __name__ == "__main__":
    main()


----------------------------


streamlit==1.37.1
pandas==2.2.2
numpy==2.0.1
scikit-learn==1.5.1






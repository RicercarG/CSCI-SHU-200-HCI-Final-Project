import os
import random
import time
from functools import partial
from datetime import datetime

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sqlalchemy import text

CAR_MODELS = [
    "Tesla Model 3", "Tesla Model S", "Tesla Model X", "Tesla Model Y",
    "Nissan Leaf", "Chevrolet Bolt EV", "BMW i3", "Audi e-tron",
    "Porsche Taycan", "Hyundai Kona Electric", "Kia Niro EV", "Volkswagen ID.4",
    "Ford Mustang Mach-E", "Polestar 2", "Rivian R1T", "Lucid Air"
]

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Stormy"]


def ui_landing_page():
    """
    The main landing page of the app.
    """

    title_placeholder = st.empty()

    ########## car switch ##########
    cars_df = conn.query(
        f"SELECT * FROM user_car_table WHERE user_id = {st.session_state.user.id}",
        ttl=0,
    )

    car_options = [
        row["user_car_id"]
        for _, row in cars_df.iterrows()
    ]

    user_car_id = st.segmented_control("Car ID", options=car_options, selection_mode="single", default=car_options[0], label_visibility="collapsed")
    car_row = cars_df[cars_df["user_car_id"] == user_car_id].iloc[0]
    st.session_state.current_car.user_car_id = car_row["user_car_id"]
    st.session_state.current_car.car_model = car_row["car_model"]
    st.session_state.current_car.battery_level = get_battery_level(user_car_id)

    title_placeholder.title(st.session_state.current_car.car_model)

    ride_history_df = conn.query(
        f"""
        SELECT * FROM user_car_history_table 
        WHERE user_car_id = {st.session_state.current_car.user_car_id}
        ORDER BY history_id DESC
        """,
        ttl=0,
    )

    with st.container(border=True, key="car_switch_container"):

        prediction = model_inference(ride_history_df[ride_history_df["user_car_id"] == user_car_id])

        st.progress(st.session_state.current_car.battery_level / 100, text=f"Battery Level: {st.session_state.current_car.battery_level}%")

        if prediction:
            if st.session_state.current_car.battery_level > 50:
                text = "The car is full of juice. However, we highly "
            else:
                text = "We "

            text += "recommend you charge this car before driving. "
        else:
            text = "No need to charge this car. Have fun driving! "

        # if st.session_state.current_car.battery_level > 80:
        #     text = "This car is full of juice. "
        # elif 80 > st.session_state.current_car.battery_level >= 30:
        #     text = "This car is half full. "
        # else:
        #     text = "This car is low on juice. Please charge it before driving."
        func = partial(stream_text, text=text)
        st.write_stream(func)
    

    
    ########## UI for logging rides ##########
    with st.container(border=True):
        ride_logger()
    

    ########## UI for ride history ##########
    st.write("Ride History")

    history_selection = st.dataframe(
        ride_history_df, 
        hide_index=True,
        selection_mode="single-row",
        column_order=["type", "start_date", "start_time", "end_date", "end_time", "weather", "paid", "end_battery_level"],
        )


def stream_text(text):
    """
    Used with st.write_stream() to display text in a streaming manner
    """

    for char in text:
        yield char
        time.sleep(0.01)


def get_location_coordinates():
    """
    Randomly generate latitude and longitude coordinates for DEMO usage
    """
    latitude = random.uniform(30.70, 31.53)
    longitude = random.uniform(120.85, 122.12)

    return (latitude, longitude)


def get_battery_level(user_car_id):
    # get the latest battery level from the database
    history_df = conn.query(
        f"""
        SELECT `end_battery_level` 
        FROM user_car_history_table 
        WHERE user_car_id = {user_car_id} 
        ORDER BY `history_id` DESC 
        LIMIT 1
        """,
        ttl=0,
    )
    return history_df.iloc[0]["end_battery_level"]


def form_params_template(operation):
    """
    A uniform component for the ride/charge forms
    :param operation: "Ride" or "Charge" or "Change"
    """

    assert operation in ["Ride", "Charge", "Change"]

    form_params = {}

    form_params["user_car_id"] = st.session_state.current_car.user_car_id
    form_params["username"] = st.session_state.user.username
    form_params["car_model"] = st.session_state.current_car.car_model
    
    if operation != "Change":
        form_params["type"] = operation
    else:
        form_params["type"] = st.selectbox(
            "Operation",
            ["Charge", "Ride"]
        )

    with st.expander(label="Configure Start Time & Location"):
        form_params["start_date"] = st.date_input("Start Date", value=st.session_state.operation_start_date)
        form_params["start_time"] = st.time_input("Start Time", value=st.session_state.operation_start_time)
        form_params["start_location_latitude"] = st.text_input("Start Location", value=st.session_state.operation_start_location[0])
        form_params["start_location_longitude"] = st.text_input("Start Location", value=st.session_state.operation_start_location[1])
    
    with st.expander(label="Configure End Time & Location"):
        form_params["end_date"] = st.date_input("End Date", value=st.session_state.operation_end_date)
        form_params["end_time"] = st.time_input("End Time", value=st.session_state.operation_end_time)
        form_params["end_location_latitude"] = st.text_input("Start Location", value=st.session_state.operation_end_location[0])
        form_params["end_location_longitude"] = st.text_input("Start Location", value=st.session_state.operation_end_location[1])
    
    form_params["weather"] = st.selectbox("Weather", WEATHER_CONDITIONS)

    if operation == "Ride":
        form_params["paid"] = None
    else:
        form_params["paid"] = st.number_input("Paid", value=0)

    form_params["end_battery_level"] = st.number_input("End Battery Level", min_value=0, max_value=100)

    return form_params


def commit_data_callback(form_params):
    """
    Sends the form data to the database.
    """
    with conn.session as session:
        raw_columns = form_params.keys()

        columns = ", ".join(raw_columns)
        placeholders = ", ".join([f":{col}" for col in raw_columns])

        sql = text(f"INSERT INTO user_car_history_table ({columns}) VALUES ({placeholders})")
        session.execute(sql, form_params)
        session.commit()


def start_operation_callback(operation):
    st.session_state.operation = operation
    st.session_state.operation_start_date = datetime.now().date()
    st.session_state.operation_start_time = datetime.now().time()
    st.session_state.operation_start_location = get_location_coordinates()


@st.dialog("Log the ride")
def stop_ride_callback():
    st.session_state.operation = None
    st.session_state.operation_end_date = datetime.now().date()
    st.session_state.operation_end_time = datetime.now().time()
    st.session_state.operation_end_location = get_location_coordinates()
    st.success("Ride stopped")

    form_params = form_params_template(operation="Ride")

    if st.button("Log this ride!", type="primary", use_container_width=True):
        commit_data_callback(form_params=form_params)
        st.rerun()

@st.dialog("Log the charge")
def stop_charge_callback():
    st.session_state.operation = None
    st.session_state.operation_end_date = datetime.now().date()
    st.session_state.operation_end_time = datetime.now().time()
    st.session_state.operation_end_location = get_location_coordinates()
    st.write("Charge stopped")

    form_params = form_params_template(operation="Charge")

    if st.button("Log this charge!", type="primary", use_container_width=True):
        commit_data_callback(form_params=form_params)
        st.rerun()

def ride_logger():
    """
    The actual UI for logging a ride/charge.
    """
    if st.session_state.operation == "Ride":
        st.warning("Ride in progress", width="stretch")
        st.button("Stop", use_container_width=True, on_click=stop_ride_callback)
    
    elif st.session_state.operation == "Charge":
        st.warning("Chargeing in progress", width="stretch")
        st.button("Stop", use_container_width=True, on_click=stop_charge_callback)
    else:
        st.info("Click to start a new ride", width="stretch", icon="🚗")
        col_left, col_right = st.columns(2, vertical_alignment="top")
        with col_left:
            st.button("Start A Ride", use_container_width=True, on_click=start_operation_callback, args=("Ride",), type="primary")
        with col_right:
            st.button("Begin Charging", use_container_width=True, on_click=start_operation_callback, args=("Charge", ), type="secondary")


def model_inference(car_df):

    print(car_df)
    # save the df to cache
    csv_save_path = "cached_table/car_data.csv"
    car_df.to_csv(csv_save_path, index=False)

    checkpoint_path = "best_model.pt"

    os.system(
        f"python3 ev_charging_predictor/main.py --mode predict --model_dir {checkpoint_path} --data_dir {csv_save_path}"
    )

    results = pd.read_csv("models/predictions.csv")
    # print(results)
    prediction = results["predicted_should_charge"].iloc[0]
    # print(prediction)
    # return prediction
    return prediction == 1


if __name__ == "__main__":
    conn = st.connection("mysql", type="sql", ttl=0)
    ui_landing_page()
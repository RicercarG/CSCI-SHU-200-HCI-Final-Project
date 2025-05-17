import os
import asyncio
import random
import time
import urllib.request
import json
from functools import partial
from datetime import datetime

import python_weather
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sqlalchemy import text

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

        if st.session_state.operation == None:
            st.progress(st.session_state.current_car.battery_level / 100, text=f"Battery Level: {st.session_state.current_car.battery_level}%")

            prediction = model_inference(ride_history_df[ride_history_df["user_car_id"] == user_car_id])
            if prediction:
                st.warning("Based on your usage pattern, we recommend you charge this car before driving. ")
            else:
                st.success("We don't think you need to charge. Happy driving! ")

            # st.info("Click to start a new ride", width="stretch", icon="🚗")
            col_left, col_right = st.columns(2, vertical_alignment="top")
            with col_left:
                st.button("🚙 Start A New Ride", use_container_width=True, on_click=start_operation_callback, args=("Ride",), type="primary")
            with col_right:
                st.button("🔋 Begin Charging", use_container_width=True, on_click=start_operation_callback, args=("Charge", ), type="secondary")

        elif st.session_state.operation == "Ride":
            st.info("Ride in progress", width="stretch")
            # st.button("Stop", use_container_width=True, on_click=stop_ride_callback)
            if st.button("Stop", use_container_width=True):
                stop_ride_callback()
        
        elif st.session_state.operation == "Charge":
            st.info("Chargeing in progress", width="stretch")
            st.button("Stop", use_container_width=True, on_click=stop_charge_callback)
        
        else:
            raise NotImplementedError
    

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
    url = "https://ipapi.co/json/"  # Free IP geolocation API
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
    
    return data

async def get_weather(city):
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
    # fetch a weather forecast from a city
        weather = await client.get(city)
    return weather.kind


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
        form_params["end_location_latitude"] = st.text_input("End Location", value=st.session_state.operation_end_location[0])
        form_params["end_location_longitude"] = st.text_input("End Location", value=st.session_state.operation_end_location[1])
    
    weather = st.session_state.weather
    weather_options = [str(k) for k in python_weather.enums.Kind]
    form_params["weather"] = st.selectbox("Weather (Auto Fetched)", weather_options, index=weather_options.index(weather))

    if operation == "Ride":
        form_params["paid"] = None
    else:
        form_params["paid"] = st.number_input("Paid", value=0)

    end_battery_level = st.session_state.end_battery_level
    form_params["end_battery_level"] = st.number_input("End Battery Level (Auto Fetched From Car)", min_value=0, max_value=100, value=end_battery_level)

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
    # first try to clearup the session state
    remove_session_state("operation_end_date")
    remove_session_state("operation_end_time")
    remove_session_state("operation_end_city")
    remove_session_state("operation_end_location")
    remove_session_state("end_battery_level")
    remove_session_state("weather")

    st.session_state.operation = operation
    st.session_state.operation_start_date = datetime.now().date()
    st.session_state.operation_start_time = datetime.now().time()
    
    location_data = get_location_coordinates()
    st.session_state.operation_start_city = location_data.get("city")
    st.session_state.operation_start_location = (location_data.get("latitude"), location_data.get("longitude"))

def register_session_state(key, value, overwrite=False):
    if overwrite or key not in st.session_state:
        st.session_state[key] = value

def remove_session_state(key):
    if key in st.session_state.keys():
        del st.session_state[key]


@st.dialog("Log the ride")
def stop_ride_callback():
    st.session_state.operation = None

    overwrite = False
    register_session_state("operation_end_date", datetime.now().date(), overwrite=overwrite)
    register_session_state("operation_end_time", datetime.now().time(), overwrite=overwrite)

    location_data = get_location_coordinates()
    register_session_state("operation_end_city", location_data.get("city"), overwrite=overwrite)
    register_session_state("operation_end_location", (location_data.get("latitude"), location_data.get("longitude")), overwrite=overwrite)
    
    register_session_state("end_battery_level", random.randint(10, 90), overwrite=overwrite)

    if overwrite or "weather" not in st.session_state:
        with st.spinner("Fetching Data"):
            st.session_state.weather = str(asyncio.run(get_weather(st.session_state.operation_end_city)))

    st.success("Ride stopped")
    form_params = form_params_template(operation="Ride")

    if st.button(
        "Log this ride!", 
        type="primary", 
        use_container_width=True,
    ):
        commit_data_callback(form_params=form_params)
        st.rerun()


@st.dialog("Log the charge")
def stop_charge_callback():
    st.session_state.operation = None
    
    overwrite = False
    register_session_state("operation_end_date", datetime.now().date(), overwrite=overwrite)
    register_session_state("operation_end_time", datetime.now().time(), overwrite=overwrite)

    location_data = get_location_coordinates()
    register_session_state("operation_end_city", location_data.get("city"), overwrite=overwrite)
    register_session_state("operation_end_location", (location_data.get("latitude"), location_data.get("longitude")), overwrite=overwrite)
    
    register_session_state("end_battery_level", random.randint(10, 90), overwrite=overwrite)

    if overwrite or "weather" not in st.session_state:
        with st.spinner("Fetching Data"):
            st.session_state.weather = str(asyncio.run(get_weather(st.session_state.operation_end_city)))

    
    st.success("Charge stopped")

    form_params = form_params_template(operation="Charge")

    if st.button("Log this charge!", type="primary", use_container_width=True):
        commit_data_callback(form_params=form_params)
        st.rerun()


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
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
  
    conn = st.connection("mysql", type="sql", ttl=0)
    ui_landing_page()
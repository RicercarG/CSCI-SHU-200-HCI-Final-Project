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

    title_placeholder = st.empty()
    # car switch
    cars_df = conn.query(
        f"SELECT * FROM user_car_table WHERE user_id = {st.session_state.user.id}",
        ttl=0,
    )

    car_options = [
        # f"{row['car_model']} - {row['user_car_id']}"
        row["user_car_id"]
        for _, row in cars_df.iterrows()
    ]

    user_car_id = st.segmented_control("Car ID", options=car_options, selection_mode="single", default=car_options[0], label_visibility="collapsed")
    # user_car_id = int(selected.split(" - ")[1])
    car_row = cars_df[cars_df["user_car_id"] == user_car_id].iloc[0]
    st.session_state.current_car.user_car_id = car_row["user_car_id"]
    st.session_state.current_car.car_model = car_row["car_model"]
    st.session_state.current_car.battery_level = get_battery_level(user_car_id)

    title_placeholder.title(st.session_state.current_car.car_model)


    with st.container(border=True, key="car_switch_container"):

        st.progress(st.session_state.current_car.battery_level / 100, text=f"Battery Level: {st.session_state.current_car.battery_level}%")

        if st.session_state.current_car.battery_level > 80:
            # st.write(f"This car is full of juice. Have fun driving!")
            text = "This car is full of juice. Have fun driving!"
            

        elif 80 > st.session_state.current_car.battery_level >= 30:
            text = "This car is half full. Watchout for the range!"
            # st.write(f"This car is almost full. Have fun driving!")
        else:
            text = "This car is low on juice. Please charge it before driving."
            # st.write(f"This car is low on juice. Please charge it before driving.")
        func = partial(stream_text, text=text)
        st.write_stream(func)

    # with st.container(border=False):
    #     if st.button("Nearby Charging Locations", use_container_width=True):
    #         st.switch_page("pages/map.py")
    
    # button for logging each ride
    with st.container(border=True):
        ride_logger()
    

    # show a table
    st.write("Ride History")
    ride_history_df = conn.query(
        f"""
        SELECT * FROM user_car_history_table 
        WHERE user_car_id = {st.session_state.current_car.user_car_id}
        ORDER BY history_id DESC
        """,
        ttl=0,
    )
    history_selection = st.dataframe(
        ride_history_df, 
        hide_index=True,
        selection_mode="single-row",
        column_order=["type", "start_date", "start_time", "end_date", "end_time", "weather", "paid", "end_battery_level"],
        # on_select=stop_ride_callback,
        )
    
    st.write(history_selection)

def stream_text(text):
    for char in text:
        yield char
        time.sleep(0.01)


def get_location_coordinates():
    # random generater latitude and longitude
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
    with conn.session as session:
        raw_columns = form_params.keys()

        columns = ", ".join(raw_columns)
        placeholders = ", ".join([f":{col}" for col in raw_columns])

        sql = text(f"INSERT INTO user_car_history_table ({columns}) VALUES ({placeholders})")
        session.execute(sql, form_params)
        session.commit()
        # print("Data committed to database")

        # # add this dict as a row of the pd dataframe
        # if st.session_state.history is None:
        #     st.session_state.history = pd.DataFrame([form_params])
        # else:
        #     st.session_state.history.loc[len(st.session_state.history)] = form_params


def start_operation_callback(operation):
    # st.session_state.operation_started = True
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

    # form_params = {}

    # form_params["user_car_id"] = st.session_state.current_car.user_car_id
    # form_params["username"] = st.session_state.user.username
    # form_params["car_model"] = st.session_state.current_car.car_model
    
    # form_params["type"] = "Ride"

    # with st.expander(label="Configure Ride Start Time & Location"):
    #     form_params["start_date"] = st.date_input("Start Date", value=st.session_state.ride_start_date)
    #     form_params["start_time"] = st.time_input("Start Time", value=st.session_state.ride_start_time)
    #     form_params["start_location_latitude"] = st.text_input("Start Location", value=st.session_state.ride_start_location[0])
    #     form_params["start_location_longitude"] = st.text_input("Start Location", value=st.session_state.ride_start_location[1])
    
    # with st.expander(label="Configure Ride End Time & Location"):
    #     form_params["end_date"] = st.date_input("End Date")
    #     form_params["end_time"] = st.time_input("End Time")
    #     form_params["end_location_latitude"] = st.text_input("Start Location", value=st.session_state.ride_end_location[0])
    #     form_params["end_location_longitude"] = st.text_input("Start Location", value=st.session_state.ride_end_location[1])
    # form_params["weather"] = st.selectbox("Weather", WEATHER_CONDITIONS)

    # form_params["paid"] = None
    # form_params["end_battery_level"] = st.number_input("Current Battery Level", min_value=0, max_value=100)

    if st.button("Log this ride!", type="primary", use_container_width=True):
        commit_data_callback(form_params=form_params)
        st.rerun()



# def start_charge_callback():
#     st.session_state.charge_started = True
#     st.session_state.charge_start_date = datetime.now().date()
#     st.session_state.charge_start_time = datetime.now().time()
#     st.session_state.charge_start_location = get_location_coordinates()

@st.dialog("Log the charge")
def stop_charge_callback():
    st.session_state.operation = None
    st.session_state.operation_end_date = datetime.now().date()
    st.session_state.operation_end_time = datetime.now().time()
    st.session_state.operation_end_location = get_location_coordinates()
    # st.session_state.charge_started = False
    st.write("Charge stopped")

    form_params = form_params_template(operation="Charge")

    if st.button("Log this charge!", type="primary", use_container_width=True):
        commit_data_callback(form_params=form_params)
        st.rerun()

    # form_params = {}
    # form_params["type"] = "Ride"
    # form_params["start_date"] = st.date_input("Start Date", value=st.session_state.charge_start_date)
    # form_params["start_time"] = st.time_input("Start Time", value=st.session_state.charge_start_time)
    # form_params["start_location"] = st.text_input("Start Location", value=st.session_state.charge_start_location)
    # form_params["end_date"] = st.date_input("End Date")
    # form_params["end_time"] = st.time_input("End Time")
    # form_params["end_location"] = st.text_input("End Location", value=get_location_coordinates())
    # form_params["weather"] = st.selectbox("weather", ["Sunny", "Rainy"])
    # form_params["car"] = st.session_state.current_car
    # form_params["paid"] = st.number_input("paid", value=0)

    # if st.button("Log this charge!", type="primary", use_container_width=True):
    #     # add this dict as a row of the pd dataframe
    #     if st.session_state.history is None:
    #         st.session_state.history = pd.DataFrame([form_params])
    #     else:
    #         st.session_state.history.loc[len(st.session_state.history)] = form_params
        
    #     st.rerun()

def ride_logger():
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

        

if __name__ == "__main__":
    conn = st.connection("mysql", type="sql", ttl=0)
    ui_landing_page()
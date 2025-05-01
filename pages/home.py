import random
from datetime import datetime

import streamlit as st
import pandas as pd


def ui_landing_page():
    # car switch
    with st.container(border=True, key="car_switch_container"):
        st.session_state.current_car = st.selectbox("Current Car", ["BMW", "Tesla", "Audi"])
        st.write(f"This car is full of juice. Have fun driving!")
    
    # button for logging each ride
    with st.container(border=True):
        ride_logger()
    
    # make two columns
    col_charger_loc, col_money_paid = st.columns(2)
    with col_charger_loc:
        with st.container(border=False):
            if st.button("Nearby Charging Locations", use_container_width=True):
                st.switch_page("pages/map.py")

    with col_money_paid:
        with st.container(border=True):
            st.write(f"Total Paid This Month")

    # show a table
    st.write("Ride History")
    st.data_editor(st.session_state.history)

def get_location_coordinates():
    # random generater latitude and longitude
    latitude = random.uniform(-90, 90)
    longitude = random.uniform(-180, 180)

    return (latitude, longitude)


def start_ride_callback():
    st.session_state.ride_started = True
    st.session_state.ride_start_date = datetime.now().date()
    st.session_state.ride_start_time = datetime.now().time()
    st.session_state.ride_start_location = get_location_coordinates()

@st.dialog("Log the ride")
def stop_ride_callback():
    st.session_state.ride_started = False
    st.write("Ride stopped")

    form_params = {}

    form_params["type"] = "Ride"
    form_params["start_date"] = st.date_input("Start Date", value=st.session_state.ride_start_date)
    form_params["start_time"] = st.time_input("Start Time", value=st.session_state.ride_start_time)
    form_params["start_location"] = st.text_input("Start Location", value=st.session_state.ride_start_location)
    form_params["end_date"] = st.date_input("End Date")
    form_params["end_time"] = st.time_input("End Time")
    form_params["end_location"] = st.text_input("End Location", value=get_location_coordinates())
    form_params["weather"] = st.selectbox("weather", ["Sunny", "Rainy"])
    form_params["car"] = st.session_state.current_car
    form_params["paid"] = None

    if st.button("Log this ride!", type="primary", use_container_width=True):
        # add this dict as a row of the pd dataframe
        if st.session_state.history is None:
            st.session_state.history = pd.DataFrame([form_params])
        else:
            st.session_state.history.loc[len(st.session_state.history)] = form_params
        
        st.rerun()


def start_charge_callback():
    st.session_state.charge_started = True
    st.session_state.charge_start_date = datetime.now().date()
    st.session_state.charge_start_time = datetime.now().time()
    st.session_state.charge_start_location = get_location_coordinates()

@st.dialog("Log the charge")
def stop_charge_callback():
    st.session_state.charge_started = False
    st.write("Charge stopped")

    form_params = {}
    form_params["type"] = "Ride"
    form_params["start_date"] = st.date_input("Start Date", value=st.session_state.charge_start_date)
    form_params["start_time"] = st.time_input("Start Time", value=st.session_state.charge_start_time)
    form_params["start_location"] = st.text_input("Start Location", value=st.session_state.charge_start_location)
    form_params["end_date"] = st.date_input("End Date")
    form_params["end_time"] = st.time_input("End Time")
    form_params["end_location"] = st.text_input("End Location", value=get_location_coordinates())
    form_params["weather"] = st.selectbox("weather", ["Sunny", "Rainy"])
    form_params["car"] = st.session_state.current_car
    form_params["paid"] = st.number_input("paid", value=0)

    if st.button("Log this charge!", type="primary", use_container_width=True):
        # add this dict as a row of the pd dataframe
        if st.session_state.history is None:
            st.session_state.history = pd.DataFrame([form_params])
        else:
            st.session_state.history.loc[len(st.session_state.history)] = form_params
        
        st.rerun()

def ride_logger():
    if st.session_state.ride_started:
        st.warning("Ride in progress", width="stretch")
        st.button("Stop", use_container_width=True, on_click=stop_ride_callback)
    
    elif st.session_state.charge_started:
        st.warning("Chargeing in progress", width="stretch")
        st.button("Stop", use_container_width=True, on_click=stop_charge_callback)
    else:
        st.info("Click to start a new ride", width="stretch", icon="🚗")
        col_left, col_right = st.columns(2, vertical_alignment="top")
        with col_left:
            st.button("Start A Ride", use_container_width=True, on_click=start_ride_callback, type="primary")
        with col_right:
            st.button("Begin Charging", use_container_width=True, on_click=start_charge_callback, type="secondary")

        


def main():
    tab1, tab2 = st.tabs(["Your Rides", "Data Analyze"])
    with tab1:
        ui_landing_page()
    with tab2:
        st.write("This is the content of Tab 2")

if __name__ == "__main__":
    main()
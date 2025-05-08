import streamlit as st

def init_state():
    if "user" not in st.session_state:
        st.session_state.user = User()

    if "current_car" not in st.session_state:
        st.session_state.current_car = Car()

    if "operation" not in st.session_state:
        st.session_state.operation = None

    if "history" not in st.session_state:
        st.session_state.history = None

    # if "current_page" not in st.session_state:
    #     st.session_state.current_page = "login"

    if "page_to_switch" not in st.session_state:
        st.session_state.page_to_switch = None

class User():
    def __init__(self):
        self.is_logged_in = False
        self.id = None
        self.username = None
        self.password = None

class Car():
    def __init__(self):
        self.user_car_id = None
        self.car_model = None
        self.battery_level = 100
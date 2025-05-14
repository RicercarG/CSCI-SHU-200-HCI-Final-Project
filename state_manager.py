import streamlit as st

def init_state():
    # to manage the values of the session state
    if "user" not in st.session_state:
        st.session_state.user = User()

    if "current_car" not in st.session_state:
        st.session_state.current_car = Car()

    if "operation" not in st.session_state:
        st.session_state.operation = None

    if "history" not in st.session_state:
        st.session_state.history = None

    if "page_to_switch" not in st.session_state:
        st.session_state.page_to_switch = None

def reset_state():
    st.session_state.user.is_logged_in = False
    st.session_state.user = User()
    st.session_state.current_car = Car()
    st.session_state.operation = None
    st.session_state.history = None
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
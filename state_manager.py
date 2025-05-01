import streamlit as st

def init_state():
    if "user" not in st.session_state:
        st.session_state.user = User()

    if "current_car" not in st.session_state:
        st.session_state.current_car = None
    
    if "ride_started" not in st.session_state:
        st.session_state.ride_started = False

    if "charge_started" not in st.session_state:
        st.session_state.charge_started = False

    if "history" not in st.session_state:
        st.session_state.history = None

class User():
    def __init__(self):
        self.is_logged_in = False
        self.username = None
        self.password = None
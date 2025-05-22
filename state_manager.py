import streamlit as st
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get session expiry time from environment (default 8 hours)
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", 8))

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
        
    if "csrf_token" not in st.session_state:
        st.session_state.csrf_token = None
        
    if "login_attempts" not in st.session_state:
        st.session_state.login_attempts = 0
        
    if "last_activity" not in st.session_state:
        st.session_state.last_activity = time.time()
    
    # Update last activity time
    st.session_state.last_activity = time.time()
    
    # Check for session expiry
    check_session_expiry()

def reset_state():
    st.session_state.user.is_logged_in = False
    st.session_state.user = User()
    st.session_state.current_car = Car()
    st.session_state.operation = None
    st.session_state.history = None
    st.session_state.page_to_switch = None
    st.session_state.csrf_token = None
    st.session_state.login_attempts = 0

def check_session_expiry():
    """Check if the session has expired due to inactivity"""
    if hasattr(st.session_state, "last_activity") and st.session_state.user.is_logged_in:
        current_time = time.time()
        # If no activity for 30 minutes, log out the user
        if current_time - st.session_state.last_activity > 1800:  # 30 minutes
            st.session_state.user.is_logged_in = False
            st.warning("Your session has expired due to inactivity. Please log in again.")
            st.session_state.redirect_to_login = True

class User():
    def __init__(self):
        self.is_logged_in = False
        self.id = None
        self.username = None
        self.session_token = None
        self.session_expiry = None
        self.role = "user"  # Default role
        self.last_password_change = None
        self.failed_login_attempts = 0
        self.account_locked = False
        self.account_locked_until = None
        
    def logout(self):
        """Securely log out the user"""
        self.is_logged_in = False
        self.id = None
        self.username = None
        self.session_token = None
        self.session_expiry = None
        
    def is_session_valid(self):
        """Check if the user's session is valid"""
        if not self.is_logged_in or not self.session_expiry:
            return False
        return time.time() < self.session_expiry
        
    def extend_session(self):
        """Extend the user's session"""
        if self.is_logged_in:
            self.session_expiry = time.time() + (SESSION_EXPIRY_HOURS * 3600)
            st.session_state.last_activity = time.time()
            
    def is_admin(self):
        """Check if the user has admin role"""
        return self.role == "admin"

class Car():
    def __init__(self):
        self.user_car_id = None
        self.car_model = None
        self.battery_level = 100
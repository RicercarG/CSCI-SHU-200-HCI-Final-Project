import os
import time
import functools
import hmac
import hashlib
from collections import defaultdict
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Rate limiting configuration
RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 100))
RATE_PERIOD = int(os.getenv("API_RATE_PERIOD", 3600))  # in seconds
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key_for_development_only")

# Rate limiting storage
rate_limit_store = defaultdict(list)

def csrf_protect(view_func):
    """
    Decorator to protect against CSRF attacks
    """
    @functools.wraps(view_func)
    def wrapped_view(*args, **kwargs):
        # Check if we're handling a form submission (POST-like action)
        if "formSubmitter" in st.session_state:
            if "csrf_token" not in st.session_state:
                st.error("CSRF token missing. Please reload the page.")
                return None
                
            # Verify the CSRF token
            if not verify_csrf_token(st.session_state.csrf_token):
                st.error("Invalid CSRF token. Please reload the page.")
                return None
                
        # Generate a new CSRF token if one doesn't exist
        if "csrf_token" not in st.session_state:
            st.session_state.csrf_token = generate_csrf_token()
            
        return view_func(*args, **kwargs)
    return wrapped_view


def generate_csrf_token():
    """
    Generate a new CSRF token using HMAC
    """
    # Use user session ID and timestamp for token uniqueness
    user_id = getattr(st.session_state.user, "id", "anonymous")
    timestamp = str(int(time.time()))
    
    # Create the token using HMAC
    msg = f"{user_id}:{timestamp}".encode('utf-8')
    signature = hmac.new(SECRET_KEY.encode('utf-8'), msg, hashlib.sha256).hexdigest()
    
    return f"{timestamp}:{signature}"


def verify_csrf_token(token):
    """
    Verify a CSRF token
    """
    try:
        # Parse the token
        timestamp, signature = token.split(":")
        
        # Check if token is expired (tokens valid for 2 hours)
        current_time = int(time.time())
        if current_time - int(timestamp) > 7200:  # 2 hours
            return False
            
        # Recreate the message and verify the signature
        user_id = getattr(st.session_state.user, "id", "anonymous")
        msg = f"{user_id}:{timestamp}".encode('utf-8')
        expected_signature = hmac.new(SECRET_KEY.encode('utf-8'), msg, hashlib.sha256).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False


def rate_limit(max_requests=None, period=None):
    """
    Decorator to implement rate limiting
    
    Args:
        max_requests (int): Maximum number of requests in the time period
        period (int): Time period in seconds
    """
    # Use default values if not provided
    max_requests = max_requests or RATE_LIMIT
    period = period or RATE_PERIOD
    
    def decorator(view_func):
        @functools.wraps(view_func)
        def wrapped_view(*args, **kwargs):
            # Get client identifier (user_id or IP address)
            user_id = getattr(st.session_state.user, "id", "anonymous")
            client_id = user_id if user_id != "anonymous" else st.session_state.get("client_ip", "unknown")
            
            # Clean up expired timestamps
            current_time = time.time()
            rate_limit_store[client_id] = [t for t in rate_limit_store[client_id] if current_time - t < period]
            
            # Check if rate limit exceeded
            if len(rate_limit_store[client_id]) >= max_requests:
                st.error("Rate limit exceeded. Please try again later.")
                return None
                
            # Add current request timestamp
            rate_limit_store[client_id].append(current_time)
            
            return view_func(*args, **kwargs)
        return wrapped_view
    return decorator


def validate_session():
    """
    Validate user session
    
    Returns:
        bool: True if session is valid, False otherwise
    """
    if not hasattr(st.session_state, "user") or not st.session_state.user.is_logged_in:
        return False
        
    # Check if session has expired
    if hasattr(st.session_state.user, "session_expiry"):
        if st.session_state.user.session_expiry < time.time():
            st.session_state.user.is_logged_in = False
            return False
            
    return True


def session_required(view_func):
    """
    Decorator to require a valid session for a view
    """
    @functools.wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not validate_session():
            st.warning("Please log in to access this page")
            st.session_state.redirect_to_login = True
            return None
            
        return view_func(*args, **kwargs)
    return wrapped_view 
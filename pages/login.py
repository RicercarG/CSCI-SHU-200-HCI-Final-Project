import streamlit as st
import time
from database.db_config import init_streamlit_connection
from database.security import sanitize_input, verify_password, generate_secure_token


def login_callback(username, password):
    # Sanitize inputs to prevent XSS
    username = sanitize_input(username)
    
    if not username or not password:
        st.error("Username and password are required")
        return
    
    # For testing purposes: hardcoded password "abcd" for test users
    # In a real app, you would remove this and rely solely on database authentication
    if password == "abcd" and username in ["bmoore29", "amandacraig67", "andrewlawson51", "austin3479"]:
        st.success(f"Welcome {username}! (Test Mode)")
        
        # Set session state
        st.session_state.user.is_logged_in = True
        st.session_state.user.id = 1  # Default ID for test users
        st.session_state.user.username = username
        
        # Generate a session token for the user
        session_token = generate_secure_token()
        st.session_state.user.session_token = session_token
        
        # Set a session expiry time (8 hours from login)
        st.session_state.user.session_expiry = time.time() + (8 * 60 * 60)
        
        st.session_state.current_page = "Home"
        st.rerun()
        return
    
    # Try to get database connection
    try:
        conn = init_streamlit_connection()
        
        # Check if connection was successful
        if conn is None:
            st.error("Database connection failed. Please try again later.")
            return
        
        # Use parameterized query to prevent SQL injection
        users = conn.query(
            "SELECT * FROM user_table WHERE username = :username;",
            params={"username": username},
            ttl=0  # Don't cache authentication queries
        )

        if len(users) == 0:
            # Use consistent error message to prevent username enumeration
            st.error("Invalid username or password")
            # Add a small delay to prevent timing attacks
            time.sleep(0.5)
            return
        else:
            user = users.iloc[0]
            stored_hash = user["password"]
            
            # Verify password using secure verification
            if verify_password(stored_hash, password):
                st.success(f"Welcome {username}!")
                
                # Set session state
                st.session_state.user.is_logged_in = True
                st.session_state.user.id = user["user_id"]
                st.session_state.user.username = username
                
                # Generate a session token for the user
                session_token = generate_secure_token()
                st.session_state.user.session_token = session_token
                
                # Set a session expiry time (8 hours from login)
                st.session_state.user.session_expiry = time.time() + (8 * 60 * 60)
                
                st.session_state.current_page = "Home"
            else:
                # Use consistent error message to prevent username enumeration
                st.error("Invalid username or password")
                # Add a small delay to prevent timing attacks
                time.sleep(0.5)
                return
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        st.info("Try using test credentials (username from list, password: 'abcd')")
        return

    st.rerun()


def ui_login():
    """
    Control the login page.
    """
    st.title("Login")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Login")
        if submitted:
            login_callback(username, password)
    
    # Display test user information
    st.markdown("---")
    st.markdown("### Test Users")
    st.markdown("For testing, you can use any of these usernames with password `abcd`:")
    st.markdown("- bmoore29")
    st.markdown("- amandacraig67")
    st.markdown("- andrewlawson51")
    st.markdown("- austin3479")
    
    st.markdown("---")
    st.write("Don't have an account? [Sign up](signup)")


def check_session_expired():
    """
    Check if the user session has expired
    Returns True if expired, False otherwise
    """
    # Make sure the user object exists and is properly initialized
    if not hasattr(st.session_state, "user"):
        return False
    
    # Make sure the user is logged in
    if not getattr(st.session_state.user, "is_logged_in", False):
        return False
    
    # Check if the session_expiry attribute exists
    if not hasattr(st.session_state.user, "session_expiry"):
        return False
    
    # Now it's safe to check if the session has expired
    try:
        return st.session_state.user.session_expiry < time.time()
    except (TypeError, ValueError):
        # Handle any unexpected issues with the comparison
        return False


if __name__ == "__main__":
    # Check if user session has expired
    if check_session_expired():
        st.session_state.user.is_logged_in = False
        st.warning("Your session has expired. Please log in again.")
    
    # Make sure the user object exists and is properly initialized
    if not hasattr(st.session_state, "user") or not getattr(st.session_state.user, "is_logged_in", False):
        ui_login()
    else:
        st.write("You are already logged in")
        if st.button("Go to Home"):
            st.switch_page("pages/home.py")
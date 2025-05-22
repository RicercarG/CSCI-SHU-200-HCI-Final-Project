import streamlit as st
import re
from database.db_config import init_streamlit_connection
from database.security import sanitize_input, hash_password, validate_query_params


def validate_password(password):
    """
    Validate password strength
    
    Args:
        password (str): Password to validate
        
    Returns:
        bool: True if password is strong enough, False otherwise
    """
    # Password must be at least 8 characters long, contain uppercase, lowercase, digit and special char
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"


def signup_callback(username, password, confirm_password):
    # Sanitize inputs
    username = sanitize_input(username)
    
    # Basic validation
    if not username or not password or not confirm_password:
        st.error("All fields are required")
        return
    
    if password != confirm_password:
        st.error("Passwords do not match")
        return
    
    # Validate password strength
    is_valid, message = validate_password(password)
    if not is_valid:
        st.error(message)
        return
    
    # Get database connection
    conn = init_streamlit_connection()
    
    # Check if username already exists
    existing_users = conn.query(
        "SELECT * FROM user_table WHERE username = :username;",
        params={"username": username},
        ttl=0
    )
    
    if len(existing_users) > 0:
        st.error("Username already exists. Please choose another one.")
        return
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Insert the new user
    try:
        # Use raw connection for insert operation
        connection = conn.session
        cursor = connection.execute(
            "INSERT INTO user_table (username, password) VALUES (:username, :password)",
            {"username": username, "password": hashed_password}
        )
        connection.commit()
        
        st.success("Account created successfully! You can now log in.")
        st.session_state.redirect_to_login = True
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def ui_signup():
    """
    Control the signup page.
    """
    st.title("Create Account")
    
    with st.form("signup"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Password strength indicator
        if password:
            is_valid, message = validate_password(password)
            if is_valid:
                st.success(message)
            else:
                st.warning(message)
        
        submitted = st.form_submit_button("Create Account")
        if submitted:
            signup_callback(username, password, confirm_password)
    
    st.markdown("---")
    st.write("Already have an account? [Log in](login)")


if __name__ == "__main__":
    if "redirect_to_login" in st.session_state and st.session_state.redirect_to_login:
        st.session_state.redirect_to_login = False
        st.switch_page("pages/login.py")
    
    if not st.session_state.user.is_logged_in:
        ui_signup()
    else:
        st.write("You are already logged in")
        if st.button("Go to Home"):
            st.switch_page("pages/home.py") 
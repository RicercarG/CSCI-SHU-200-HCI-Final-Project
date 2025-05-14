import streamlit as st

def logout_callback():
    st.session_state.user.is_logged_in = None


def ui_account_page():
    """
    Account management page.
    """
    
    st.title(st.session_state.user.username)

    st.logout_button = st.button("Logout", on_click=logout_callback)
    

if __name__ == "__main__":
    ui_account_page()
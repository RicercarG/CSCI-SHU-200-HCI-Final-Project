import streamlit as st

from state_manager import reset_state

def logout_callback():
    reset_state()
    # st.session_state.user.is_logged_in = None
    # st.switch_page("pages/login")


def ui_account_page():
    """
    Account management page.
    """
    
    st.title(st.session_state.user.username)

    st.logout_button = st.button("Logout", on_click=logout_callback)
    

if __name__ == "__main__":
    ui_account_page()
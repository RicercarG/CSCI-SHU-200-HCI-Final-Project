import streamlit as st

from state_manager import init_state

if __name__ == "__main__":
    init_state()

    if not st.session_state.user.is_logged_in:
        pages = [
            st.Page("pages/login.py", title="Login"),
        ]
        
    else:
        pages = [
            st.Page("pages/home.py", title="Home"),
            st.Page("pages/map.py", title="Map"),
            st.Page("pages/account.py", title="Account")
        ]

    pg = st.navigation(pages)
    pg.run()
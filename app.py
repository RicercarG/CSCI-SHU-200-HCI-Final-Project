import streamlit as st
from streamlit_option_menu import option_menu

from state_manager import init_state


def on_change(key):
    st.session_state.page_to_switch = st.session_state[key]


def switch_page(selection):
# def on_change(key):
    st.session_state.page_to_switch = None

    if selection == "Home":
        st.switch_page("pages/home.py")
    elif selection == "Data":
        st.switch_page("pages/data.py")
    elif selection == "Charge":
        print("Charge Selected")
        print("-" * 20)
        st.switch_page("pages/map.py")
    elif selection == "Account":
        st.switch_page("pages/account.py")
    else:
        st.rerun()


if __name__ == "__main__":

    init_state()

    if not st.session_state.user.is_logged_in:
        pages = [
            st.Page("pages/login.py", title="Login"),
        ]
        
    else:
        pages = [
            st.Page("pages/home.py", title="Home"),
            st.Page("pages/data.py", title="Data"),
            st.Page("pages/map.py", title="Map"),
            st.Page("pages/account.py", title="Account")
        ]

        # make a header for navigation
        selection = option_menu(None, ["Home", "Data", "Charge", 'Account'], 
            icons=['house', 'bi-bar-chart', "bi-lightning-charge", 'bi-person'], 
            menu_icon="cast", default_index=0, orientation="horizontal",
            styles={
                "nav-link": {"font-size": "12px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            },
            key="top_nav",
            on_change=on_change,
        )

        if st.session_state.page_to_switch is not None:
            switch_page(st.session_state.page_to_switch)
    

    pg = st.navigation(pages)
    pg.run()
import streamlit as st

def logout_callback():
    st.session_state.user.is_logged_in = None
    # st.switch_page("pages/login.py")


def ui_account_page():
    # Set the title of your app
    # st.title("My Account")

    # st.write(f"Welcome to your account, {st.session_state.user.username}!")

    st.title(st.session_state.user.username)

    st.logout_button = st.button("Logout", on_click=logout_callback)
    

if __name__ == "__main__":
    ui_account_page()
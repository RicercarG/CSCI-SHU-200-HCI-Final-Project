import streamlit as st


def login_callback(username, password):
    # TODO: add password check logic
    st.session_state.user.is_logged_in = True
    st.session_state.user.username = username
    st.session_state.user.password = password

    st.rerun()


def ui_login():
    st.title("Login")

    with st.form("login"):
        # TODO: Add login logic
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        submitted = st.form_submit_button("Login")
        if submitted:
            login_callback(username, password)


if __name__ == "__main__":
    if not st.session_state.user.is_logged_in:
        ui_login()
    else:
        # TODO: Add logout logic
        st.write("You are already logged in")
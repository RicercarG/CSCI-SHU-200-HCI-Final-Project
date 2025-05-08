import streamlit as st


def login_callback(username, password):
    users = conn.query(
        f'SELECT * from user_table WHERE username="{username}";', 
        ttl=600
    )

    if len(users) == 0:
        st.error(f"User {username} does not exist.")
        return

    else:
        # find the row in the df where the password matches
        target_users = users[users["password"] == password]
        assert len(target_users) <= 1
        if len(target_users) == 1:
            st.success(f"Welcome {username}!")
            st.session_state.user.is_logged_in = True
            # set the user info in session state
            st.session_state.user.id = target_users.iloc[0]["user_id"]
            st.session_state.user.username = username
            st.session_state.user.password = password
            st.session_state.current_page = "Home"

        else:
            st.error(f"Password is incorrect.")
            return


    # # TODO: add password check logic
    # st.session_state.user.is_logged_in = True
    # st.session_state.user.username = username
    # st.session_state.user.password = password

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
    conn = st.connection("mysql", type="sql")

    if not st.session_state.user.is_logged_in:
        ui_login()
    else:
        # TODO: Add logout logic
        st.write("You are already logged in")
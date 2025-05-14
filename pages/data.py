import streamlit as st

def ui_analysis_page():
    """
    Visualize the data insights.
    """
    
    st.title("Data Insights")

    history_df = conn.query(
        f"SELECT * FROM user_car_history_table WHERE username = '{st.session_state.user.username}'",
        ttl=0,
    )

    history_df["car_model_with_id"] = history_df.apply(lambda row: f"{row['car_model']} - {row['user_car_id']}", axis=1)

    
    with st.container(border=True):
        vis_paid(history_df)

    with st.container(border=True):
        vis_driving_habits(history_df)

    with st.container(border=True):
        vis_charging_loc(history_df)

    st.write(history_df)


def vis_paid(df):
    st.write("Essential Stats")
    paid_col, charge_count_col = st.columns(2)
    with paid_col:
        st.metric(
            label=f"Total Paid",
            value=round(df["paid"].sum(), 2),
            border=True
        )
    with charge_count_col:
        st.metric(
            label=f"Total Charge Count",
            value=len(df[df["type"]=='charge']),
            border=True
        )

def vis_driving_habits(df):
    st.write("Battery Usage Habits")

    st.bar_chart(
        df,
        x="end_date",
        y="end_battery_level",
        color="car_model_with_id",
        stack=False
    )

def vis_charging_loc(df):
    st.write("Charging Locations")
    st.map(
        df[df["type"]=="charge"],
        latitude="start_location_latitude",
        longitude="start_location_longitude",
        # size=10,
        # zoom="auto",
    )


if __name__ == "__main__":
    conn = st.connection("mysql", type="sql")

    ui_analysis_page()
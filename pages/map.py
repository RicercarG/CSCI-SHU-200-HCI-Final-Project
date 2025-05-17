import urllib.request
import json

import streamlit as st
import pandas as pd
import random

from haversine import haversine

def get_charging_stations(current_location):
    """
    return a pandas dataframe with columns "latitude" and "longitude"
    :param current_location: a tuple of (latitude, longitude)
    """

    num_stations = 5
    df = pd.DataFrame({
        'name': ['Station ' + str(i) for i in range(num_stations)],
        'latitude': [random.uniform(current_location[0] - 1, current_location[0] + 1) for _ in range(num_stations)],
        'longitude': [random.uniform(current_location[1] - 1, current_location[1] + 1) for _ in range(num_stations)]
    })

    return df

def ui_map_page():
    st.title("Charging Stations Nearby")

    # get current location
    url = "https://ipapi.co/json/"  # Free IP geolocation API
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
    current_location = (data['latitude'], data['longitude'])
    # get charging stations
    df = get_charging_stations(current_location)
    # display map with charging stations
    st.map(df)

    # calculate a distance between current location and charging stations, and add it to the dataframe
    df['distance'] = [haversine(current_location, (row['latitude'], row['longitude'])) for idx, row in df.iterrows()]
    
    # sort the dataframe by distance
    df = df.sort_values(by='distance')

    # show charging locations:
    for idx, row in df.iterrows():
        distance = round(row['distance'], 2)
        st.metric(label=row['name'], value=f"{distance} km", border=True)



if __name__ == "__main__":
    ui_map_page()
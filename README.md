# CSCI-SHU-200-HCI-Final-Project

## Start the app
```
streamlit run app.py
```


## TODOs

### Database

- [ ] User Table
    - **username**
    - password

- [ ] User-Car Table
    - **username**
    - **car model**

- [ ] User-Car-History Table
    - **username**
    - **car model**
    - type: ["ride", "charge"]
    - start date
    - start time
    - start location: e.g. (25.34, 48.56)
    - end date
    - end time
    - end location
    - weather: ["Sunny","Cloudy","Rainy","Snowy","Windy","Foggy","Stormy"]
    - paid: for charging, how much did the user pay?
    - end batterylevel

    .... (more to come)

### UI
- [ ] Login Logic
- [ ] Nearby maps
- [ ] Data Analysis Page


### ML Model

- [ ] Train a model to predict the user's next event type (charging or driving), and the battery level after the event.

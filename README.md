# CSCI-SHU-200-HCI-Final-Project

## Project Description

This is a simple demo for an app to release mileage anxiety for EV users.

### Problem to solve

### Core Functionality

### Cost and Scalability Estimation



## Quick Start

### Step1: Install python packages (we tested on python 3.10)
```
pip install -r requirements.txt
```

### Step2: Setup the database
Create an mysql database, and run `create_tables.sql` to create tables

### Step3: Populate Data
```
python database/populate_data.py
```
A browser page should be opened. If not please try Local URL: 
`http://localhost:8501`

To login, please select an existing user from the database `user_table`.


### Step4: Start the app
```
streamlit run app.py
```


## TODOs

### Database

- [x] User Table
    - **username**
    - password

- [x] User-Car Table
    - **username**
    - **car model**

- [x] User-Car-History Table
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

### UI
- [x] Login Logic
- [ ] Nearby maps
- [x] Data Analysis Page


### ML Model

- [ ] Train a model to predict the user's next event type (charging or driving), and the battery level after the event.

# CSCI-SHU-200-HCI-Final-Project

## Project Description

This is a simple demo for an app to release mileage anxiety for EV users.

### Problem to Solve

How to release mileage anxiety for EV users, with minimal user effort?

### Our Solution

We propose a simple app that learns vehicle-level battery consumption pattern, and send reminders to users for recharging.

#### Target Users
People with electric vehicles, especially those who have multiple EVs.

#### How it Works
The logic is similar to an accounting software:

1. During each ride/charge, the app records the parameters like weather, location and battery level at the beginning and end of each ride/charge (and charing costs).

2. Our model learns the battery consumption pattern for each vehicle, and predicts when that vehicle will need to be recharged. (Check out our [EV Charging Predictor](https://github.com/RicercarG/CSCI-SHU-200-HCI-Final-Project/blob/main/ev_charging_predictor/README.md) for more details)

3. The app will visualize the data for each user, including the battery consumption pattern for each vehicle, the charging costs, and the frequently visited chargning locations.

To log each ride/charge, except for logging all data manually, we envision that we can utilize the bluetooth connection between the EV and phone. The logging will be automatically triggered when the phone is connected/disconnected to the EV.

### Business Model

#### Costs and scalability
- A data storage service is needed to store the data. We can use AWS RDS or Google Cloud SQL, which are both scalable and cost-effective.

- The online learning algorithm could be done locally on the phone. Thus the user scale is not a concern.

#### Revenue Streams
- We plan to release the app for free.
- We can sell the data to EV manufacturers and charging station operators. The data can be used to optimize the battery consumption pattern for each vehicle, and predict when that vehicle will need to be recharged. The data can also be used to optimize the charging station locations and schedules.


## Quick Start

### Step1: Install python packages (we tested on python 3.10)
```
pip install -r requirements.txt
```

### Step2: Setup the database
Create an mysql database, and run `create_tables.sql` to create tables

### Step3: Populate Data
To ensure proper data instertion, first set the MySQL max_allowed_packet to be 256M
``````
SET GLOBAL max_allowed_packet = 268435456;
```

After that, we can safely populate data
```
python database/populate_data.py
```


### Step4: Start the app
```
streamlit run app.py
```

A browser page should be opened. If not please try Local URL: 
`http://localhost:8501`

To login, please select an existing user from the database `user_table`.


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
- [x] Nearby maps
- [x] Data Analysis Page


### ML Model

- [x] Train a model to predict the user's next event type (charging or driving), and the battery level after the event.

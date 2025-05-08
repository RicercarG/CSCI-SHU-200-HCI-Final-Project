#!/usr/bin/env python3.11
import mysql.connector
import random
import datetime
import decimal
from faker import Faker
import hashlib

# Initialize Faker for generating mock data
fake = Faker()

# Database connection details (adjust if necessary)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # Assuming root user, adjust if you created a specific user
    'password': '',  # Add password if you set one for the root user or specific user
    'database': 'electric_vehicle_assistant'
}

# Constants for data generation
NUM_USERS = 20
NUM_CARS_PER_USER_MAX = 2
NUM_HISTORY_RECORDS_PER_CAR = 50

CAR_MODELS = [
    "Tesla Model 3", "Tesla Model S", "Tesla Model X", "Tesla Model Y",
    "Nissan Leaf", "Chevrolet Bolt EV", "BMW i3", "Audi e-tron",
    "Porsche Taycan", "Hyundai Kona Electric", "Kia Niro EV", "Volkswagen ID.4",
    "Ford Mustang Mach-E", "Polestar 2", "Rivian R1T", "Lucid Air"
]

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy", "Stormy"]

# Helper function to hash passwords (simple SHA256 for simulation)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_user_data(cursor):
    print("Generating user data...")
    users = []
    for i in range(NUM_USERS):
        username = fake.user_name() + str(random.randint(10,99)) # ensure more uniqueness
        # Check if username already exists to avoid collision (simple check for simulation)
        cursor.execute("SELECT user_id FROM user_table WHERE username = %s", (username,))
        if cursor.fetchone():
            username = fake.user_name() + str(random.randint(100,999))
        
        # password = fake.password(length=12)
        # hashed_pwd = hash_password(password)
        # users.append((username, hashed_pwd))
        password = "abcd"
        users.append((username, password))
        print(f"  Generated user: {username}")

    try:
        cursor.executemany("INSERT INTO user_table (username, password) VALUES (%s, %s)", users)
        print(f"{len(users)} users inserted.")
    except mysql.connector.Error as err:
        print(f"Error inserting user data: {err}")
        # If unique constraint fails for username, it will be caught here.
        # For a robust solution, retry with a new username or handle more gracefully.


def generate_user_car_data(cursor):
    print("\nGenerating user car data...")
    cursor.execute("SELECT user_id, username FROM user_table")
    db_users = cursor.fetchall()
    if not db_users:
        print("No users found in user_table. Skipping user_car_table population.")
        return

    user_cars = []
    for user_id, username in db_users:
        num_cars = random.randint(1, NUM_CARS_PER_USER_MAX)
        user_owned_cars = random.sample(CAR_MODELS, num_cars)
        for car_model in user_owned_cars:
            user_cars.append((user_id, car_model))
            print(f"  Generated car for user {username}: {car_model}")
    
    if user_cars:
        cursor.executemany("INSERT INTO user_car_table (user_id, car_model) VALUES (%s, %s)", user_cars)
        print(f"{len(user_cars)} user cars inserted.")
    else:
        print("No user cars generated.")

def generate_user_car_history_data(cursor):
    print("\nGenerating user car history data...")
    cursor.execute("SELECT uc.user_car_id, ut.username, uc.car_model FROM user_car_table uc JOIN user_table ut ON uc.user_id = ut.user_id")
    db_user_cars = cursor.fetchall()
    if not db_user_cars:
        print("No user cars found. Skipping user_car_history_table population.")
        return

    history_records = []
    for user_car_id, username, car_model in db_user_cars:
        # Simulate initial battery level for the car
        current_battery_level = random.randint(70, 100) 
        last_event_end_datetime = datetime.datetime.now() - datetime.timedelta(days=random.randint(90, 180)) # Start history from some time ago

        for _ in range(NUM_HISTORY_RECORDS_PER_CAR):
            event_type = random.choice(["ride", "charge"])
            
            # Start date and time (ensure it's after the last event)
            start_offset_hours = random.uniform(1, 72) # 1 hour to 3 days after last event
            start_datetime = last_event_end_datetime + datetime.timedelta(hours=start_offset_hours)
            start_date = start_datetime.date()
            start_time = start_datetime.time()

            start_lat = decimal.Decimal(fake.latitude())
            start_lon = decimal.Decimal(fake.longitude())
            
            weather = random.choice(WEATHER_CONDITIONS)
            paid_amount = None
            
            # End date and time
            duration_hours = 0
            battery_change = 0

            if event_type == "ride":
                duration_hours = random.uniform(0.2, 5) # 12 mins to 5 hours ride
                # battery_consumed = random.randint(max(5, current_battery_level), min(current_battery_level, 40)) # Consume up to 40% or current level
                battery_consumed = random.randint(0, min(current_battery_level-10, 40)) # Consume up to 40% or current level
                battery_change = -battery_consumed
                paid_amount = None
            elif event_type == "charge":
                duration_hours = random.uniform(0.3, 8) # 18 mins to 8 hours charge
                # battery_gained = random.randint(10, max(10, 100 - current_battery_level)) if current_battery_level < 100 else 0
                battery_gained = random.randint(min(5, 100-current_battery_level), 100-current_battery_level)
                battery_change = battery_gained
                paid_amount = decimal.Decimal(random.uniform(5.0, 50.0)).quantize(decimal.Decimal('0.01'))

            end_datetime = start_datetime + datetime.timedelta(hours=duration_hours)
            end_date = end_datetime.date()
            end_time = end_datetime.time()
            
            end_lat = decimal.Decimal(fake.latitude())
            end_lon = decimal.Decimal(fake.longitude())
            
            end_battery_level = max(0, min(100, current_battery_level + battery_change))

            history_records.append((
                user_car_id, username, car_model, event_type, 
                start_date, start_time, start_lat, start_lon,
                end_date, end_time, end_lat, end_lon,
                weather, paid_amount, end_battery_level
            ))
            print(f"    Generated history: User {username}, Car {car_model}, Type {event_type}, Start {start_date} {start_time}, End Bat {end_battery_level}%")
            
            # Update for next iteration
            current_battery_level = end_battery_level
            last_event_end_datetime = end_datetime
            
            # If battery is very low, next event is more likely a charge
            if current_battery_level < 20 and random.random() < 0.8: # 80% chance to charge if battery < 20%
                # Force next event to be charge if possible, or just continue
                pass # The logic already picks randomly, this just notes the consideration

    if history_records:
        sql = """
        INSERT INTO user_car_history_table (
            user_car_id, username, car_model, type, 
            start_date, start_time, start_location_latitude, start_location_longitude,
            end_date, end_time, end_location_latitude, end_location_longitude,
            weather, paid, end_battery_level
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(sql, history_records)
        print(f"{len(history_records)} history records inserted.")
    else:
        print("No history records generated.")

def main():
    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(f"Successfully connected to MySQL database: {DB_CONFIG['database']}")

        # Clear existing data (optional, for repeatable testing)
        # print("\nClearing existing data (user_car_history_table, user_car_table, user_table)...")
        # cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        # cursor.execute("TRUNCATE TABLE user_car_history_table;")
        # cursor.execute("TRUNCATE TABLE user_car_table;")
        # cursor.execute("TRUNCATE TABLE user_table;")
        # cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        # conn.commit()
        # print("Existing data cleared.")

        generate_user_data(cursor)
        conn.commit()
        
        generate_user_car_data(cursor)
        conn.commit()
        
        generate_user_car_history_data(cursor)
        conn.commit()

        print("\nMock data generation and population complete.")

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL connection closed.")

if __name__ == "__main__":
    main()


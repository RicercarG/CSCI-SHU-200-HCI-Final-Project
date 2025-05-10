SET GLOBAL max_allowed_packet = 16777216; -- Sets it to 16MB

DELETE FROM electric_vehicle_assistant.user_table;
DELETE FROM electric_vehicle_assistant.user_car_table ;
DELETE FROM electric_vehicle_assistant.user_car_history_table ;
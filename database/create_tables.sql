-- SQL Script for creating the database and tables

-- Create the database if it doesn't exist
CREATE DATABASE IF NOT EXISTS `electric_vehicle_assistant` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `electric_vehicle_assistant`;

-- 1. user_table (用户信息表)
CREATE TABLE IF NOT EXISTS `user_table` (
    `user_id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户唯一标识符',
    `username` VARCHAR(255) NOT NULL UNIQUE COMMENT '用户名',
    `password` VARCHAR(255) NOT NULL COMMENT '用户密码 (实际应用中应存储哈希值)'
) ENGINE=InnoDB COMMENT='用户信息表';

-- 2. user_car_table (用户车辆信息表)
CREATE TABLE IF NOT EXISTS `user_car_table` (
    `user_car_id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户车辆唯一标识符',
    `user_id` INT NOT NULL COMMENT '关联的用户ID',
    `car_model` VARCHAR(255) NOT NULL COMMENT '车辆型号',
    FOREIGN KEY (`user_id`) REFERENCES `user_table`(`user_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB COMMENT='用户车辆信息表';

-- 3. user_car_history_table (用户车辆使用历史表)
CREATE TABLE IF NOT EXISTS `user_car_history_table` (
    `history_id` INT AUTO_INCREMENT PRIMARY KEY COMMENT '历史记录唯一标识符',
    `user_car_id` INT NOT NULL COMMENT '关联的用户车辆ID',
    `username` VARCHAR(255) NOT NULL COMMENT '用户名 (冗余字段，方便查询)',
    `car_model` VARCHAR(255) NOT NULL COMMENT '车辆型号 (冗余字段，方便查询)',
    `type` ENUM('ride', 'charge') NOT NULL COMMENT '事件类型：ride (行驶) 或 charge (充电)',
    `start_date` DATE NOT NULL COMMENT '启动或开始充电日期',
    `start_time` TIME NOT NULL COMMENT '启动或开始充电时间',
    `start_location_latitude` DECIMAL(9,6) COMMENT '启动或充电位置纬度',
    `start_location_longitude` DECIMAL(9,6) COMMENT '启动或充电位置经度',
    `end_date` DATE COMMENT '熄火或结束充电日期',
    `end_time` TIME COMMENT '熄火或结束充电时间',
    `end_location_latitude` DECIMAL(9,6) COMMENT '熄火或结束充电位置纬度',
    `end_location_longitude` DECIMAL(9,6) COMMENT '熄火或结束充电位置经度',
    `weather` ENUM('Sunny', 'Partly Cloudy', 'Cloudy', 'Very Cloudy', 'Fog', 'Light Showers', 'Light Sleet Showers', 'Light Sleet', 'Thundery Showers', 'Light Snow', 'Heavy Snow', 'Light Rain', 'Heavy Showers', 'Heavy Rain', 'Light Snow Showers', 'Heavy Snow Showers', 'Thundery Heavy Rain', 'Thundery Snow Showers') COMMENT '天气状况',
    `paid` DECIMAL(10,2) NULL COMMENT '充电花费 (仅当 type 为 charge 时有效)',
    `end_battery_level` INT NOT NULL COMMENT '事件结束时剩余电池百分比 (0-100)',
    FOREIGN KEY (`user_car_id`) REFERENCES `user_car_table`(`user_car_id`) ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT `chk_end_battery_level` CHECK (`end_battery_level` >= 0 AND `end_battery_level` <= 100)
) ENGINE=InnoDB COMMENT='用户车辆使用历史表';



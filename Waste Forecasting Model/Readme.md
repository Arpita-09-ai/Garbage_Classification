# Predict daily waste volume and optimize collection schedules and resources using LSTM.

## Overview

This project predicts daily municipal waste for multiple cities using an LSTM neural network and calculates the required trucks, staff, and drivers for collection. The system includes preprocessing, feature engineering, model training, evaluation, and visualization.

**Features**

Load and preprocess historical waste data

Feature engineering & label encoding

Normalize data and create sequences for LSTM

Train-test split and LSTM model training

Predict future waste volumes

Optimize waste collection schedules

Allocate resources (trucks, staff, drivers)

Visualize predictions and resource allocation

Save/load trained model for deployment

##  Dataset: https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset?utm_



Column	Description
ticket_date	Date of waste collection
city	City name
net_weight_kg	Waste collected (in kg)

Example:

city	ticket_date	net_weight_kg
moratuwa	2018-11-21	94360
boralasgamuwa	2018-11-21	48000
## Model

Algorithm: LSTM (Long Short-Term Memory)

Performance on Test Set:

MSE: 0.00

RMSE: 0.05

MAE: 0.03

R²: 0.7857

**Sample Predictions**
Date	Predicted_Waste
2025-08-27	17807.55
2025-08-28	19564.89
2025-08-29	17695.61
Resource Allocation Example
City	Predicted_Waste	Trucks_Required	Staff_Required	Drivers_Required
moratuwa	92000	2	6	2
boralasgamuwa	48000	1	3	1
dehiwala	75000	2	6	2
homagama	60000	2	6	2

Truck capacity: 50,000 kg

Staff per truck: 3

Drivers per truck: 1

## Visualizations

Actual vs Predicted Waste

Resource Allocation per Day

These charts help in understanding trends and planning collection efficiently.
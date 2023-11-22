import numpy as np
import torch
from torch.utils.data import Dataset

def generate_features():
    features = {
        'fuel_age': np.random.uniform(0, 30),
        'wind_speed': np.random.normal(50, 25),
        'temperature': np.random.normal(20, 10),
        'humidity': np.random.normal(50, 20),
        'fuel_moisture_content': np.random.normal(15, 7),
        'slope': np.random.uniform(0, 45),
        'surface_fuel_hazard': np.random.uniform(1, 10),
        'near_surface_fuel_hazard': np.random.uniform(1, 10),
        'near_surface_fuel_height': np.random.normal(50, 25),
        'elevated_fuel_height': np.random.normal(100, 50),
        'flame_height': np.random.normal(25, 12.5),
    }
    return features

def new_generate_features():
    features = {
        'fuel_age': np.random.uniform(0, 30),
        'wind_speed': np.random.normal(50, 25),
        'temperature': np.random.normal(20, 10),
        'humidity': np.random.normal(50, 20),
        'fuel_moisture_content': np.random.normal(15, 7),
        'slope': np.random.uniform(0, 45),
        'surface_fuel_hazard': np.random.uniform(1, 10),
        'near_surface_fuel_hazard': np.random.uniform(1, 10),
        'near_surface_fuel_height': np.random.normal(50, 25),
        'elevated_fuel_height': np.random.normal(100, 50),
        'flame_height': np.random.normal(25, 12.5),
    }
    return features



def calculate_ws10(U, V):
    """
    Calculate the daily absolute wind speed at 10 meters height.

    Parameters:
    - U: Zonal velocity of wind in m/s.
    - V: Meridional velocity of wind in m/s.

    Returns:
    - WS10: Absolute wind speed at 10 meters height in m/s.
    """

    # Calculate the square of the wind components
    U_squared = U ** 2
    V_squared = V ** 2

    # Calculate the sum of squares and take the square root
    WS10 = np.sqrt(U_squared + V_squared)

    return WS10

# Example usage
U = np.array([2, 4, 1, 3])
V = np.array([1, 3, 2, 4])

WS10 = calculate_ws10(U, V)
print("Daily Absolute Wind Speed at 10m Height:", WS10)


def identify_heatwaves(temperature_data, historical_percentile, consecutive_days=3):
    """
    Identify heatwaves based on temperature data.

    Parameters:
    - temperature_data: An array or list containing daily temperature values.
    - historical_percentile: The 95th percentile of maximum temperature for the historical period.
    - consecutive_days: The minimum number of consecutive days for an event to be considered a heatwave.

    Returns:
    - An array of 0s and 1s indicating the presence (1) or absence (0) of heatwaves.
    """

    # Calculate the threshold for a heatwave
    heatwave_threshold = np.percentile(temperature_data, historical_percentile)

    # Identify heatwaves based on the threshold and consecutive days
    heatwave_indicator = np.zeros_like(temperature_data, dtype=int)

    for i in range(len(temperature_data) - consecutive_days + 1):
        if all(temperature_data[i:i+consecutive_days] > heatwave_threshold):
            heatwave_indicator[i:i+consecutive_days] = 1

    return heatwave_indicator

# Example usage
temperature_data = np.array([28, 30, 31, 32, 33, 34, 35, 36, 30, 28])
historical_percentile = 95
heatwave_indicator = identify_heatwaves(temperature_data, historical_percentile)

print("Heatwave Indicator:", heatwave_indicator)


def bushfire_rate_of_spread(features):
    # This is a hypothetical function.
    rate = (
        features['wind_speed'] * 0.3 +
        features['temperature'] * 0.2 -
        features['humidity'] * 0.15 +
        features['fuel_moisture_content'] * 0.1 +
        features['slope'] * 0.05 +
        features['surface_fuel_hazard'] * 0.1 +
        features['near_surface_fuel_hazard'] * 0.1 -
        features['near_surface_fuel_height'] * 0.05 +
        features['elevated_fuel_height'] * 0.05 +
        features['flame_height'] * 0.1
    )
    return np.clip(rate, 0, None)  # Ensuring rate of spread is non-negative

def new_bushfire_rate_of_spread():
    # This is a hypothetical function.
    rate = (
    
    )
    return np.clip(rate, 0, None)  # Ensuring rate of spread is non-negative
def generate_bushfire_data(size):
    data = []
    for _ in range(size):
        features = generate_features()
        rate_of_spread = bushfire_rate_of_spread(features)
        features['rate_of_spread'] = rate_of_spread
        data.append(features)
    return data

# Example usage
bushfire_data = generate_bushfire_data(100000)

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

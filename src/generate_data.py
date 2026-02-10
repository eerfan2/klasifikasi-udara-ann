import pandas as pd
import numpy as np
import os

def generate_dummy_data(n_samples=1000):
    """
    Generates synthetic air quality data suitable for classification.
    Features: PM2.5, PM10, NO2, CO, O3, SO2
    Target: Quality (0: Good, 1: Moderate, 2: Unhealthy)
    """
    np.random.seed(42)
    
    # Generate random features (mimicking real-world ranges)
    n_per_class = n_samples // 3
    
    # Class 0: Good
    pm25_0 = np.random.uniform(0, 35, n_per_class)
    pm10_0 = np.random.uniform(0, 50, n_per_class)
    no2_0 = np.random.uniform(0, 40, n_per_class)
    co_0 = np.random.uniform(0, 4, n_per_class)
    o3_0 = np.random.uniform(0, 50, n_per_class)
    so2_0 = np.random.uniform(0, 20, n_per_class)
    labels_0 = np.zeros(n_per_class)

    # Class 1: Moderate
    pm25_1 = np.random.uniform(35, 75, n_per_class)
    pm10_1 = np.random.uniform(50, 100, n_per_class)
    no2_1 = np.random.uniform(40, 80, n_per_class)
    co_1 = np.random.uniform(4, 9, n_per_class)
    o3_1 = np.random.uniform(50, 100, n_per_class)
    so2_1 = np.random.uniform(20, 80, n_per_class)
    labels_1 = np.ones(n_per_class)

    # Class 2: Unhealthy
    pm25_2 = np.random.uniform(75, 150, n_per_class)
    pm10_2 = np.random.uniform(100, 250, n_per_class)
    no2_2 = np.random.uniform(80, 180, n_per_class)
    co_2 = np.random.uniform(9, 15, n_per_class)
    o3_2 = np.random.uniform(100, 168, n_per_class)
    so2_2 = np.random.uniform(80, 350, n_per_class)
    labels_2 = np.full(n_per_class, 2)

    # Combine
    pm25 = np.concatenate([pm25_0, pm25_1, pm25_2])
    pm10 = np.concatenate([pm10_0, pm10_1, pm10_2])
    no2 = np.concatenate([no2_0, no2_1, no2_2])
    co = np.concatenate([co_0, co_1, co_2])
    o3 = np.concatenate([o3_0, o3_1, o3_2])
    so2 = np.concatenate([so2_0, so2_1, so2_2])
    labels = np.concatenate([labels_0, labels_1, labels_2])

    df = pd.DataFrame({
        'pm_10': pm10,
        'pm_duakomalima': pm25,
        'so2': so2,
        'co': co,
        'o3': o3,
        'no2': no2,
        'Quality': labels.astype(int)
    })
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic air quality data...")
    df = generate_dummy_data(1500)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'air_quality_dummy.csv')
    
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    print(df.head())

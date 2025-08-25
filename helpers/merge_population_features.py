import pandas as pd
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')

# File mapping for each radius
FILES = {
    500: {
        'features': os.path.join(DATA_DIR, 'switzerland_station_features_500m.csv'),
        'population': os.path.join(DATA_DIR, 'station_population_500m.csv'),
        'output': os.path.join(DATA_DIR, 'switzerland_station_features_500m_with_pop.csv'),
    },
    1000: {
        'features': os.path.join(DATA_DIR, 'switzerland_station_features_1000m.csv'),
        'population': os.path.join(DATA_DIR, 'station_population_1000m.csv'),
        'output': os.path.join(DATA_DIR, 'switzerland_station_features_1000m_with_pop.csv'),
    },
    1500: {
        'features': os.path.join(DATA_DIR, 'switzerland_station_features_1500m.csv'),
        'population': os.path.join(DATA_DIR, 'station_population_1500m.csv'),
        'output': os.path.join(DATA_DIR, 'switzerland_station_features_1500m_with_pop.csv'),
    },
}

def merge_population_features(radius):
    print(f"Merging population for {radius}m radius...")
    features_path = FILES[radius]['features']
    pop_path = FILES[radius]['population']
    output_path = FILES[radius]['output']

    features = pd.read_csv(features_path)
    pop = pd.read_csv(pop_path)

    # Merge on station_id
    merged = features.merge(pop[['station_id', f'population_{radius}m']], on='station_id', how='left')
    merged.to_csv(output_path, index=False)
    print(f"Saved merged file to {output_path}")

if __name__ == "__main__":
    for radius in [500, 1000, 1500]:
        merge_population_features(radius)
    print("All merges complete.")

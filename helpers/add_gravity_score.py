import pandas as pd
import sys

# Usage: python add_gravity_score.py <input_csv> <output_csv> <pop_col>
def compute_gravity_score(row, count_col, dist_col, pop_col):
    # Avoid division by zero
    dist = row[dist_col] if row[dist_col] > 0 else 1.0
    return (row[count_col] * row[pop_col]) / dist

def main():
    if len(sys.argv) != 4:
        print("Usage: python add_gravity_score.py <input_csv> <output_csv> <pop_col>")
        sys.exit(1)
    input_csv, output_csv, pop_col = sys.argv[1:]
    df = pd.read_csv(input_csv)

    # Example: gravity score for cafes
    gravity_features = [
        ("cafes_count", "cafes_nearest_distance"),
        ("restaurants_count", "restaurants_nearest_distance"),
        ("supermarkets_count", "supermarkets_nearest_distance"),
        ("schools_count", "schools_nearest_distance"),
        ("parks_count", "parks_nearest_distance"),
        ("bus_stops_count", "bus_stops_nearest_distance"),
        ("train_stations_count", "train_stations_nearest_distance"),
        ("bike_parking_count", "bike_parking_nearest_distance"),
    ]

    for count_col, dist_col in gravity_features:
        gravity_col = f"gravity_{count_col.replace('_count','')}"
        df[gravity_col] = df.apply(lambda row: compute_gravity_score(row, count_col, dist_col, pop_col), axis=1)

    df.to_csv(output_csv, index=False)
    print(f"Gravity-based features added and saved to {output_csv}")

if __name__ == "__main__":
    main()

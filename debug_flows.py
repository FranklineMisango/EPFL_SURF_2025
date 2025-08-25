import json

# Check the predicted flows
with open('predicted_flows.json', 'r') as f:
    flows = json.load(f)

print(f"Total flows: {len(flows)}")
print(f"Flow range: {min(f['predicted_flow'] for f in flows):.6f} to {max(f['predicted_flow'] for f in flows):.6f}")

# Show top 10 flows
top_flows = sorted(flows, key=lambda x: x['predicted_flow'], reverse=True)[:10]
print("\nTop 10 flows:")
for f in top_flows:
    print(f"  {f['origin']} -> {f['destination']}: {f['predicted_flow']:.6f}")

# Check stations
with open('stations.json', 'r') as f:
    stations = json.load(f)

station_ids = {s['station_id'] for s in stations}
flow_stations = {f['origin'] for f in flows} | {f['destination'] for f in flows}

print(f"\nStations in stations.json: {len(station_ids)}")
print(f"Stations in flows: {len(flow_stations)}")
print(f"Missing coordinates: {flow_stations - station_ids}")
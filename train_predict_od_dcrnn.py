import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json
from scipy.spatial.distance import pdist, squareform

# ---- CONSTANTS ----
MAX_STATIONS = 714  # Full dataset
MAX_TIMESTEPS = 48
H = 6
PREDICT_HORIZON = 1
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 1  # Process one sample at a time
CHUNK_SIZE = 100  # Process stations in chunks


# Load station features
station_df = pd.read_csv('data/switzerland_station_features_1000m_with_pop.csv')
station_df = station_df.iloc[:MAX_STATIONS]
station_ids = station_df['station_id'].values
station_id_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}
N = len(station_ids)
print(f"Processing {N} stations in chunks of {CHUNK_SIZE}")

# Node features (OSM, population, etc.)
node_features = station_df.drop(['station_id', 'lat', 'lon', 'coords'], axis=1).values
scaler = MinMaxScaler()
node_features = scaler.fit_transform(node_features)

# Load trips
df_trips = pd.read_csv('data/trips_8days_flat.csv')
df_trips = df_trips[df_trips['start_station_id'].isin(station_ids) & df_trips['end_station_id'].isin(station_ids)]

# Parse time
parse_time = lambda s: datetime.strptime(s, '%Y%m%d_%H%M%S')
df_trips['start_time'] = df_trips['start_time'].apply(parse_time)

# Build OD matrices for each time window (e.g., hourly)
time_min = df_trips['start_time'].min()
time_max = df_trips['start_time'].max()
time_delta = timedelta(hours=1)
time_bins = pd.date_range(time_min, time_max, freq='1h')  # fix warning

od_matrices = []
timestamps = []
step_count = 0
for t_start in time_bins[:-1]:
    if MAX_TIMESTEPS is not None and step_count >= MAX_TIMESTEPS:
        break
    t_end = t_start + time_delta
    od = np.zeros((N, N), dtype=np.float32)
    trips = df_trips[(df_trips['start_time'] >= t_start) & (df_trips['start_time'] < t_end)]
    for _, row in trips.iterrows():
        i = station_id_to_idx[row['start_station_id']]
        j = station_id_to_idx[row['end_station_id']]
        od[i, j] += 1
    od_matrices.append(od)
    timestamps.append(t_start)
    step_count += 1
if len(od_matrices) == 0:
    raise RuntimeError('No OD matrices created. Check your data/time window.')
od_matrices = np.stack(od_matrices)  # [T, N, N]
print(f"Loaded OD matrices shape: {od_matrices.shape} (timesteps, stations, stations)")
print(f"Node features shape: {node_features.shape}")

# ---- CHUNKED PROCESSING ----
def process_chunk(chunk_start, chunk_end, od_matrices, node_features, adj_chunk):
    """Process a chunk of stations"""
    chunk_size = chunk_end - chunk_start
    T, N, _ = od_matrices.shape
    feature_dim = node_features.shape[1] + 2
    
    X_chunk, Y_chunk = [], []
    for t in range(H, T - PREDICT_HORIZON):
        x_seq = []
        for h in range(H):
            od = od_matrices[t - H + h]
            # Only process the chunk
            od_chunk = od[chunk_start:chunk_end, :]
            in_flows = od_chunk.sum(axis=1)
            out_flows = od[:, chunk_start:chunk_end].sum(axis=0)
            
            feat = np.zeros((chunk_size, feature_dim), dtype=np.float32)
            feat[:, :node_features.shape[1]] = node_features[chunk_start:chunk_end]
            feat[:, -2] = in_flows
            feat[:, -1] = out_flows
            x_seq.append(feat)
        X_chunk.append(np.stack(x_seq))
        Y_chunk.append(od_matrices[t + PREDICT_HORIZON][chunk_start:chunk_end, :])
    
    return np.stack(X_chunk), np.stack(Y_chunk)

# ---- MODEL ----
class DiffusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.adj = adj
        self.fc = nn.Linear(in_channels, out_channels)
    def forward(self, x):
        Ax = torch.einsum('ij,bjkd->bikd', self.adj, x)
        Ax = torch.einsum('ij,bikd->bjkd', self.adj, Ax)
        out = self.fc(Ax)
        return out

class DCRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, adj):
        super().__init__()
        self.adj = adj
        self.fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    def forward(self, x, h):
        # Simplified: just use adjacency for message passing
        xh = torch.cat([x, h], dim=-1)
        z = torch.relu(self.fc(xh))
        # Apply adjacency
        z = torch.matmul(self.adj, z)
        h_new = self.gru(z.reshape(-1, z.shape[-1]), h.reshape(-1, h.shape[-1]))
        return h_new.reshape(h.shape)

class DCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, horizon):
        super().__init__()
        self.horizon = horizon
        self.cell = DCRNNCell(input_dim, hidden_dim, adj)
        self.output = nn.Linear(hidden_dim, N)  # Output to all stations
    def forward(self, x):
        batch, H, chunk_size, input_dim = x.shape
        h = torch.zeros(batch, chunk_size, self.cell.gru.hidden_size, device=x.device)
        for t in range(H):
            h = self.cell(x[:, t], h)
        y = self.output(h)  # [batch, chunk_size, N]
        return y

# ---- ADJACENCY ----
coords = station_df[['lat', 'lon']].values
dist_matrix = squareform(pdist(coords))
threshold = np.percentile(dist_matrix[dist_matrix > 0], 5)  # Connect closest 5%
adj_full = (dist_matrix < threshold) & (dist_matrix > 0)

# ---- CHUNKED TRAINING ----
all_predictions = []
for chunk_start in range(0, N, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, N)
    print(f"Processing chunk {chunk_start}-{chunk_end}")
    
    # Get adjacency for this chunk
    adj_chunk = torch.tensor(adj_full[chunk_start:chunk_end, chunk_start:chunk_end], dtype=torch.float32)
    
    # Process chunk data
    X_chunk, Y_chunk = process_chunk(chunk_start, chunk_end, od_matrices, node_features, adj_chunk)
    
    # Train model for this chunk
    model = DCRNN(input_dim=X_chunk.shape[-1], hidden_dim=HIDDEN_DIM, output_dim=1, adj=adj_chunk, horizon=1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_chunk, dtype=torch.float32)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i in range(len(X_tensor)):
            xb = X_tensor[i:i+1]
            yb = Y_tensor[i:i+1]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(X_tensor):.4f}")
    
    # Get predictions for this chunk
    model.eval()
    with torch.no_grad():
        chunk_preds = model(X_tensor[-1:]).cpu().numpy()[0]  # Last sample
    all_predictions.append(chunk_preds)
    
    # Clear memory
    del model, optimizer, X_tensor, Y_tensor, X_chunk, Y_chunk
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---- COMBINE PREDICTIONS & SAVE ----
last_pred = np.vstack(all_predictions)  # Combine all chunk predictions
last_time = timestamps[-1].strftime('%Y-%m-%dT%H:%M')

output = []
for i in range(N):
    for j in range(N):
        if last_pred[i, j] > 0.01:
            output.append({
                'origin': int(station_ids[i]),
                'destination': int(station_ids[j]),
                'timestamp': last_time,
                'predicted_flow': float(last_pred[i, j])
            })

with open('predicted_flows.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'Processed {N} stations in chunks. Saved predicted_flows.json for visualization.')
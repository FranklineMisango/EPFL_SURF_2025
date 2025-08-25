import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import json

# ---- CONSTANTS ----
MAX_STATIONS = 714
MAX_TIMESTEPS = 48
H = 6  
PREDICT_HORIZON = 1
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 8


# Load station features
station_df = pd.read_csv('/Data/switzerland_station_features_1000m_with_pop.csv')
if MAX_STATIONS is not None:
    station_df = station_df.iloc[:MAX_STATIONS]
station_ids = station_df['station_id'].values
station_id_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}
N = len(station_ids)

# Node features (OSM, population, etc.)
node_features = station_df.drop(['station_id', 'lat', 'lon', 'coords'], axis=1).values
scaler = MinMaxScaler()
node_features = scaler.fit_transform(node_features)

# Load trips
df_trips = pd.read_csv('Data/trips_8days_flat.csv')
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

# ---- FEATURE TENSOR BUILDING ----
def build_feature_tensor(od_matrices, node_features, history_len):
    T, N, _ = od_matrices.shape
    feature_dim = 2 * node_features.shape[1] + 1  # origin + dest features + flow
    X, Y = [], []
    
    for t in range(history_len, T - PREDICT_HORIZON):
        x_seq = []
        for h in range(history_len):
            od = od_matrices[t - history_len + h]
            feat = np.zeros((N, N, feature_dim), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    feat[i, j, :node_features.shape[1]] = node_features[i]
                    feat[i, j, node_features.shape[1]:2*node_features.shape[1]] = node_features[j]
                    feat[i, j, -1] = od[i, j]
            x_seq.append(feat)
        X.append(np.stack(x_seq))
        Y.append(od_matrices[t + PREDICT_HORIZON])
    return np.stack(X), np.stack(Y)

X, Y = build_feature_tensor(od_matrices, node_features, H)

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
        self.diff_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, adj)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
    def forward(self, x, h):
        xh = torch.cat([x, h], dim=-1)
        z = torch.relu(self.diff_conv(xh))
        h_new = self.gru(z.reshape(-1, z.shape[-1]), h.reshape(-1, h.shape[-1]))
        return h_new.reshape(h.shape)

class DCRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj, horizon):
        super().__init__()
        self.horizon = horizon
        self.cell = DCRNNCell(input_dim, hidden_dim, adj)
        self.output = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        batch, H, N, _, input_dim = x.shape
        h = torch.zeros(batch, N, N, self.cell.gru.hidden_size, device=x.device)
        for t in range(H):
            h = self.cell(x[:, t], h)
        y = self.output(h)
        return y.squeeze(-1)

# ---- ADJACENCY ----
# Simple adjacency: connect all stations (can be improved with real connectivity)
adj = torch.ones(N, N)
adj.fill_diagonal_(0)

# ---- TRAINING ----
model = DCRNN(input_dim=X.shape[-1], hidden_dim=HIDDEN_DIM, output_dim=1, adj=adj, horizon=1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(X_tensor))
    total_loss = 0
    for i in range(0, len(X_tensor), BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        xb = X_tensor[idx]
        yb = Y_tensor[idx]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(X_tensor):.4f}")

# ---- PREDICTION & SAVE ----
model.eval()
preds = model(X_tensor).detach().cpu().numpy()

# Save last prediction for visualization
last_pred = preds[-1]  # [N, N]
last_time = timestamps[-1].strftime('%Y-%m-%dT%H:%M')

output = []
for i in range(N):
    for j in range(N):
        if last_pred[i, j] > 0.01:  # lowered threshold for visualization
            output.append({
                'origin': int(station_ids[i]),
                'destination': int(station_ids[j]),
                'timestamp': last_time,
                'predicted_flow': float(last_pred[i, j])
            })

with open('predicted_flows.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Saved predicted_flows.json for JS visualization.')
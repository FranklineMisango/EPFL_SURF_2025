import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDCRNNCell(nn.Module):
    """A minimal DCRNN cell for demonstration (not production-grade)."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, h):
        # x: [batch, input_dim], h: [batch, hidden_dim]
        return self.gru(x, h)

class MinimalDCRNN(nn.Module):
    """A minimal DCRNN for static graphs and short sequences."""
    def __init__(self, node_features, hidden_dim, seq_len=4, out_dim=1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.cell = SimpleDCRNNCell(node_features, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_seq):
        # x_seq: [batch, seq_len, node_features]
        batch_size = x_seq.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        for t in range(self.seq_len):
            h = self.cell(x_seq[:, t, :], h)
        out = self.fc(h)
        return out.squeeze(-1)

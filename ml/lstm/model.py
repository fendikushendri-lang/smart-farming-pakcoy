import torch
import torch.nn as nn

class PakcoyMultivariateLSTM(nn.Module):
    """
    Multivariate LSTM untuk Peramalan Deret Waktu Multi-Step (4 Hari / 12 Timestep).
    Input: [soil_moisture, temperature, maturity_level] (3 fitur)
    Sequence input: 12 timestep (4 hari historis)
    Output: 12 timestep x 3 fitur (4 hari ke depan: Day+1 s.d. Day+4)
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_steps=12, output_features=3, dropout=0.2):
        super(PakcoyMultivariateLSTM, self).__init__()
        self.output_steps = output_steps
        self.output_features = output_features
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_steps * output_features)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use last hidden state: hn[-1] shape: (batch_size, hidden_size)
        last_hidden = hn[-1]
        out = self.fc(last_hidden)
        # Reshape to (batch_size, output_steps, output_features)
        out = out.view(-1, self.output_steps, self.output_features)
        return out

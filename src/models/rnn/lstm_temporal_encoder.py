import torch.nn as nn


class LSTMTemporalEncoder(nn.Module):
    """
    Drop-in LSTM alternative to transformer.temporal_encoder.TemporalEncoder.
    Same call signature/shapes; final hidden state stands in for the
    transformer's CLS-token summary.
    """

    def __init__(self, d_in, d_model=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, time_maks, veh_maks=None):
        """
        x: [B, N1, T, d_in]
        time_maks: [B, N1, T]  (True=pad, False=keep). Required.
        veh_maks:  [B, N1]     (True=pad). Optional; if given, will zero invalid outputs.
        """
        B, N1, T, d_in = x.shape
        assert time_maks.shape == (B, N1, T), "time_valid must be [B,N1,T] bool"

        x = self.proj_in(x).view(B * N1, T, -1)  # [B*N1, T, D]
        pad_t = time_maks.view(B * N1, T)  # True = pad

        # Valid (non-pad) timestep count per sequence. Clamped to >=1 so
        # pack_padded_sequence never sees a zero-length sequence - fully-padded
        # vehicles are zeroed out afterwards via veh_maks instead.
        lengths = (~pad_t).sum(dim=1).clamp(min=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        h_last = h_n[-1]  # [B*N1, D] - top layer's final hidden state
        h_veh = self.ln_out(h_last).view(B, N1, -1)  # [B, N1, D]

        if veh_maks is not None:
            assert veh_maks.shape == (B, N1)
            h_veh = h_veh * ~veh_maks.unsqueeze(-1)

        return h_veh

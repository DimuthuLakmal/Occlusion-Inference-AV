import torch
import math
import torch.nn as nn


# --- Sinusoidal 1D positional encoding (time) ---
class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, D]
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a parameter

    def forward(self, x):
        # x: [B*, T, D]  -> add PE for first T positions
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)  # [1,T,D] + [B*,T,D]


class TemporalEncoder(nn.Module):
    def __init__(self, d_in, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
        self.posenc = SinusoidalPosEnc(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # One learnable CLS token shared across vehicles (expanded per sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x, time_maks, veh_maks=None):
        """
        x: [B, N1, T, d_in]
        time_valid: [B, N1, T]  (True=pad, False=keep). Required.
        veh_valid:  [B, N1]     (True=keep). Optional; if given, will zero invalid outputs.
        """
        B, N1, T, d_in = x.shape
        assert time_maks.shape == (B, N1, T), "time_valid must be [B,N1,T] bool"

        # Flatten vehicles into batch axis for per-vehicle temporal encoding
        x = self.proj_in(x).view(B * N1, T, -1)  # [B*N1, T, D]
        pad_t = time_maks.view(B * N1, T)  # True = pad (Transformer convention)

        # Prepend CLS (not padded)
        cls = self.cls_token.expand(B * N1, 1, -1)  # [B*N1, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B*N1, T+1, D]

        # Build padding mask for T+1 (CLS is always valid => False)
        cls_pad = torch.zeros(B * N1, 1, dtype=torch.bool, device=pad_t.device)
        src_key_padding_mask = torch.cat([cls_pad, pad_t], dim=1)  # [B*N1, T+1], True=pad

        # Add positional encodings (T+1 because of CLS at position 0)
        x = self.posenc(x)

        # Encode (self-attn over time per vehicle)
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B*N1, T+1, D]

        # Take CLS as per-vehicle summary
        h_cls = h[:, 0, :]  # [B*N1, D]
        h_veh = self.ln_out(h_cls).view(B, N1, -1)  # [B, N1, D]

        if veh_maks is not None:
            assert veh_maks.shape == (B, N1)
            h_veh = h_veh * ~veh_maks.unsqueeze(-1)

        return h_veh
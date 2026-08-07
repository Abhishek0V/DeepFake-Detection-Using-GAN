import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Model -----------------------------------------------


class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate=16000,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.sample_rate  = sample_rate
        self.min_low_hz   = min_low_hz
        self.min_band_hz  = min_band_hz
        low_hz  = 30.0
        high_hz = sample_rate / 2 - (min_low_hz + min_band_hz)
        mel = np.linspace(self._hz2mel(low_hz), self._hz2mel(high_hz), out_channels + 1)
        hz  = self._mel2hz(mel)
        self.low_hz_  = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).unsqueeze(1))
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).unsqueeze(1))
        n = (self.kernel_size - 1) / 2.0
        self.register_buffer('n_',
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate)
        self.register_buffer('window_',
            torch.hamming_window(self.kernel_size)[: self.kernel_size // 2])

    @staticmethod
    def _hz2mel(hz):  return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def _mel2hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        low  = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),
                           self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]
        f_l  = low  * self.n_
        f_h  = high * self.n_
        bp_l = ((torch.sin(f_h) - torch.sin(f_l)) / (self.n_ / 2)) * self.window_
        bp_c = 2 * band.view(-1, 1)
        bp_r = torch.flip(bp_l, dims=[1])
        filt = torch.cat([bp_l, bp_c, bp_r], dim=1) / (2 * band[:, None])
        return F.conv1d(x, filt.view(self.out_channels, 1, self.kernel_size),
                        stride=1, padding=self.kernel_size // 2)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super().__init__()
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a     = nn.Parameter(torch.zeros(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(alpha)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        h = self.W(x)
        B, N, D = h.shape
        e = torch.cat([h.unsqueeze(2).expand(-1, -1, N, -1),
                       h.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1)
        e = self.leaky((e @ self.a).squeeze(-1))
        return F.elu(self.drop(F.softmax(e, dim=-1)) @ h)


class HeteroGraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        assert out_dim % heads == 0
        hd = out_dim // heads
        self.spec_gat = nn.ModuleList([GraphAttentionLayer(in_dim, hd) for _ in range(heads)])
        self.temp_gat = nn.ModuleList([GraphAttentionLayer(in_dim, hd) for _ in range(heads)])
        self.proj     = nn.Linear(out_dim * 2, out_dim)

    def forward(self, spec_nodes, temp_nodes):
        s = torch.cat([h(spec_nodes) for h in self.spec_gat], dim=-1)
        t = torch.cat([h(temp_nodes) for h in self.temp_gat], dim=-1)
        return self.proj(torch.cat([s, t], dim=-1))


def _make_res_block(in_ch, out_ch, kernel, stride):
    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                                   padding=kernel // 2, bias=False)
            self.bn1   = nn.BatchNorm1d(out_ch)
            self.act1  = nn.SiLU()
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, stride=1,
                                   padding=kernel // 2, bias=False)
            self.bn2   = nn.BatchNorm1d(out_ch)
            self.act2  = nn.SiLU()
            self.skip  = (
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch),
                ) if (in_ch != out_ch or stride != 1) else nn.Identity()
            )
        def forward(self, x):
            return self.act2(
                self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))
                + self.skip(x)
            )
    return _Block()


class AASIST(nn.Module):
    def __init__(self, sinc_out=70, sinc_kernel=128,
                 conv_channels=64, gat_dim=64, num_classes=2):
        super().__init__()
        self.sinc = SincConv(sinc_out, sinc_kernel)
        self.bn0  = nn.BatchNorm1d(sinc_out)

        # ----- spec encoder: 4 residual blocks -----
        self._spec_blk0 = _make_res_block(sinc_out,      conv_channels, kernel=3, stride=3)
        self._spec_drop0 = nn.Dropout(0.1)
        self._spec_blk1 = _make_res_block(conv_channels, conv_channels, kernel=3, stride=3)
        self._spec_drop1 = nn.Dropout(0.1)
        self._spec_blk2 = _make_res_block(conv_channels, conv_channels, kernel=3, stride=3)
        self._spec_blk3 = _make_res_block(conv_channels, conv_channels, kernel=3, stride=3)

        # Keep Sequential alias so checkpoint keys still match
        self.spec_encoder = nn.Sequential(
            self._spec_blk0, self._spec_drop0,
            self._spec_blk1, self._spec_drop1,
            self._spec_blk2, self._spec_blk3,
        )

        # ----- temp encoder: 4 residual blocks -----
        self._temp_blk0 = _make_res_block(sinc_out,      conv_channels, kernel=5, stride=5)
        self._temp_drop0 = nn.Dropout(0.1)
        self._temp_blk1 = _make_res_block(conv_channels, conv_channels, kernel=5, stride=5)
        self._temp_drop1 = nn.Dropout(0.1)
        self._temp_blk2 = _make_res_block(conv_channels, conv_channels, kernel=3, stride=3)
        self._temp_blk3 = _make_res_block(conv_channels, conv_channels, kernel=3, stride=3)

        self.temp_encoder = nn.Sequential(
            self._temp_blk0, self._temp_drop0,
            self._temp_blk1, self._temp_drop1,
            self._temp_blk2, self._temp_blk3,
        )

        self.hgat1 = HeteroGraphAttention(conv_channels, gat_dim)
        self.hgat2 = HeteroGraphAttention(gat_dim,       gat_dim)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Sequential(
            nn.Linear(gat_dim * 2, gat_dim), nn.SiLU(),
            nn.Dropout(0.5), nn.Linear(gat_dim, num_classes)
        )

        # -- Grad-CAM storage --
        self.feature_maps        = None   # final spec_encoder output  (high-level)
        self.gradients           = None   # gradient w.r.t. feature_maps
        self.early_feature_maps  = None   # after block-1 of spec_encoder (higher resolution)
        self.early_gradients     = None   # gradient w.r.t. early_feature_maps
        self.temp_feature_maps   = None   # temp_encoder early features (temporal branch)
        self.temp_gradients      = None   # gradient w.r.t. temp_feature_maps

    # ---------- gradient save-hooks ----------
    def _save_gradient(self, grad):
        self.gradients = grad

    def _save_early_gradient(self, grad):
        self.early_gradients = grad

    def _save_temp_gradient(self, grad):
        self.temp_gradients = grad

    # ---------- public accessors (kept for backward compat) ----------
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.feature_maps

    # ---------- new public accessors ----------
    def get_early_activations(self):
        return self.early_feature_maps

    def get_early_gradients(self):
        return self.early_gradients

    def get_temp_activations(self):
        return self.temp_feature_maps

    def get_temp_gradients(self):
        return self.temp_gradients

    # ---------- forward ----------
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.abs(self.bn0(self.sinc(x)))

        # --- spec encoder with intermediate hook ---
        s = self._spec_blk0(x)
        s = self._spec_drop0(s)
        s = self._spec_blk1(s)          # <-- EARLY hook (after 2nd block, stride=9 total)
        s = self._spec_drop1(s)

        # Register early hook (higher temporal resolution)
        self.early_feature_maps = s
        if s.requires_grad:
            s.register_hook(self._save_early_gradient)

        s = self._spec_blk2(s)
        s = self._spec_blk3(s)

        # Register final hook (deep features for context)
        self.feature_maps = s
        if s.requires_grad:
            s.register_hook(self._save_gradient)

        # --- temp encoder with intermediate hook ---
        t = self._temp_blk0(x)
        t = self._temp_drop0(t)
        t = self._temp_blk1(t)          # <-- EARLY temporal hook

        self.temp_feature_maps = t
        if t.requires_grad:
            t.register_hook(self._save_temp_gradient)

        t = self._temp_drop1(t)
        t = self._temp_blk2(t)
        t = self._temp_blk3(t)

        # --- graph attention ---
        L = min(s.size(2), t.size(2))
        s = s[:, :, :L].permute(0, 2, 1)
        t = t[:, :, :L].permute(0, 2, 1)
        s = self.hgat1(s, t)
        t = self.hgat2(t, s)
        s = self.pool(s.permute(0, 2, 1)).squeeze(-1)
        t = self.pool(t.permute(0, 2, 1)).squeeze(-1)
        return self.fc(torch.cat([s, t], dim=-1))

    # ---------- legacy hook (kept for compatibility) ----------
    def save_gradient(self, grad):
        self.gradients = grad

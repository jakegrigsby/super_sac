import math

import torch
from torch import nn
import torch.nn.functional as F


class PopArtLayer(nn.Module):
    def __init__(self, beta=1e-4, min_steps=1000, init_nu=0):
        super().__init__()
        self.mu = torch.zeros(1)
        # high initialization said to increase stability (PopArt pg 13)
        self.nu = torch.ones(1) * init_nu
        self.beta = beta
        self.w = torch.ones(1)
        self.b = torch.zeros(1)
        self._t = 1
        self._stable = False
        self.min_steps = min_steps

    @property
    def sigma(self):
        return (torch.sqrt(self.nu - self.mu**2) + 1e-5).clamp(1e-4, 1e6)

    def normalize_values(self, val):
        return (val - self.mu) / self.sigma

    def to(self, device):
        self.w = self.w.to(device)
        self.b = self.b.to(device)
        self.mu = self.mu.to(device)
        self.nu = self.nu.to(device)
        return self

    def update_stats(self, val):
        self._t += 1
        old_sigma = self.sigma
        old_mu = self.mu
        # Use adaptive step size to reduce reliance on initialization (pg 13)
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        self.mu = (1.0 - beta_t) * self.mu + beta_t * val.mean()
        self.nu = (1.0 - beta_t) * self.nu + (beta_t * (val**2).mean())

        # heuristic to protect stability early in training
        self._stable = (self._t > self.min_steps) and (
            ((1.0 - old_sigma) / self.sigma) <= 0.1
        )
        # self._stable = self._t > self.min_steps

        if self._stable:
            self.w *= old_sigma / self.sigma
            self.b = (old_sigma * self.b + old_mu - self.mu) / (self.sigma)

    def forward(self, x, normalized=True):
        normalized_out = (self.w * x) + self.b
        if normalized:
            return normalized_out
        else:
            return (self.sigma * normalized_out) + self.mu

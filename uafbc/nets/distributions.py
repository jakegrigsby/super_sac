import math

import torch
import torch.distributions as pyd
import torch.nn.functional as F
import numpy as np

class BetaDist(pyd.transformed_distribution.TransformedDistribution):

    class _BetaDistTransform(pyd.transforms.Transform):
        domain = pyd.constraints.real
        codomain = pyd.constraints.interval(-1.0, 1.0)

        def __init__(self, cache_size=1):
            super().__init__(cache_size=cache_size)

        def __eq__(self, other):
            return isinstance(other, _BetaDistTransform)

        def _inverse(self, y):
            return (y.clamp(-.99, .99) + 1.0) / 2.0

        def _call(self, x):
            return (2.0 * x) - 1.0

        def log_abs_det_jacobian(self, x, y):
            # return log det jacobian |dy/dx| given input and output
            return torch.Tensor([math.log(2.0)]).to(x.device)

    def __init__(self, alpha, beta):
        self.base_dist = pyd.beta.Beta(alpha, beta)
        transforms = [self._BetaDistTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.base_dist.mean
        for tr in self.transforms:
            mu = tr(mu)
        return mu


"""
Credit for actor distribution code: https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py
"""


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y.clamp(-.99, .99))

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

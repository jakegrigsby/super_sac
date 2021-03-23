import numpy as np
from torch import nn
import torch.nn.functional as F

import gym
from minatar import Environment

from uafbc import nets


class MinAtarEnv:
    def __init__(self, *args, **kwargs):
        self.env = Environment(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(6)

    def _state(self):
        return {"obs": self.env.state().astype(np.uint8)}

    def reset(self):
        self.env.reset()
        return self._state()

    def step(self, act):
        if isinstance(act, np.ndarray):
            act = int(act)
        reward, done = self.env.act(act)
        return self._state(), reward, done, {}

    @property
    def num_channels(self):
        return self.env.state_shape()[-1]


class MinAtarEncoder(nets.Encoder):
    def __init__(self, channels, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, stride=1)

        output_height, output_width = nets.cnns.compute_conv_output(
            (10, 10), kernel_size=(3, 3), stride=(1, 1)
        )
        output_height, output_width = nets.cnns.compute_conv_output(
            (output_height, output_width), kernel_size=(2, 2), stride=(1, 1)
        )
        self.fc = nn.Linear(output_height * output_width * 16, emb_dim)
        self.apply(nets.weight_init)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        img = obs_dict["obs"].permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        state = self.fc(x)
        return state

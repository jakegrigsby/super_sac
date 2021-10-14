import numpy as np
from torch import nn
import torch.nn.functional as F

import gym
from minatar import Environment

from uafbc import nets


class MinAtarEnv(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        self.env = Environment(*args, **kwargs)
        self.env.action_space = gym.spaces.Discrete(6)
        self.env.observation_space = None
        self.env.reward_range = None
        self.env.metadata = None
        super().__init__(self.env)

    def reset(self):
        self.env.reset()
        return self.env.state().astype(np.uint8)

    def step(self, act):
        reward, done = self.env.act(act)
        state = self.env.state().astype(np.uint8)
        return state, reward, done, {}

    @property
    def num_channels(self):
        return self.env.state_shape()[-1]

    def render(self, *args, **kwargs):
        self.env.display_state()


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

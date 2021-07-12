import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn

from . import weight_init


def compute_conv_output(
    inp_shape, kernel_size, padding=(0, 0), dilation=(1, 1), stride=(1, 1)
):
    """
    Compute the shape of the output of a torch Conv2d layer using
    the formula from the docs.

    every argument is a tuple corresponding to (height, width), e.g. kernel_size=(3, 4)
    """
    height_out = math.floor(
        (
            (inp_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            / stride[0]
        )
        + 1
    )
    width_out = math.floor(
        (
            (inp_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            / stride[1]
        )
        + 1
    )
    return height_out, width_out


class BigPixelEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim=50):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        output_height, output_width = compute_conv_output(
            obs_shape[1:], kernel_size=(3, 3), stride=(2, 2)
        )
        for _ in range(3):
            output_height, output_width = compute_conv_output(
                (output_height, output_width), kernel_size=(3, 3), stride=(1, 1)
            )

        self.fc = nn.Linear(output_height * output_width * 32, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.apply(weight_init)
        self.embedding_dim = out_dim

    def forward(self, obs):
        obs /= 255.0
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        state = torch.tanh(x)
        return state


class SmallPixelEncoder(nn.Module):
    def __init__(self, obs_shape, out_dim=50):
        super().__init__()
        channels = obs_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        output_height, output_width = compute_conv_output(
            obs_shape[1:], kernel_size=(8, 8), stride=(4, 4)
        )

        output_height, output_width = compute_conv_output(
            (output_height, output_width), kernel_size=(4, 4), stride=(2, 2)
        )

        output_height, output_width = compute_conv_output(
            (output_height, output_width), kernel_size=(3, 3), stride=(1, 1)
        )

        self.fc = nn.Linear(output_height * output_width * 64, out_dim)
        self.apply(weight_init)
        self.embedding_dim = out_dim

    def forward(self, obs):
        obs /= 255.0
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        state = self.fc(x)
        return state

import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import d4rl
import afbc
import gym


class MiniGridEncoder(afbc.nets.AFBCEncoder):
    def __init__(self, img_shape, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        # minigrid is channels last, so switch to channels first
        img_shape = (img_shape[-1],) + img_shape[:-1]
        channels = img_shape[0]
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2, stride=1)

        output_height, output_width = afbc.nets.cnns.compute_conv_output(
            img_shape[1:], kernel_size=(2, 2), stride=(1, 1)
        )

        output_height, output_width = afbc.nets.cnns.compute_conv_output(
            (output_height, output_width), kernel_size=(2, 2), stride=(1, 1)
        )

        self.fc = nn.Linear(output_height * output_width * 32, emb_dim)
        self.apply(afbc.nets.weight_init)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        img = obs_dict["image"].permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        state = self.fc(x)
        return state


def train_d4rl_minigrid(args):
    train_env = afbc.wrappers.KeepKeysWrapper(gym.make(args.env), ["image"])
    test_env = afbc.wrappers.KeepKeysWrapper(gym.make(args.env), ["image"])
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = afbc.AFBCAgent(
        action_space.n,
        encoder=MiniGridEncoder(state_space["image"].shape),
        actor_network_cls=afbc.nets.mlps.DiscreteActor,
        critic_network_cls=afbc.nets.mlps.DiscreteCritic,
        hidden_size=256,
        discrete=True,
    )

    # get offline datset
    dset = d4rl.qlearning_dataset(test_env)
    dset_size = dset["observations"].shape[0]
    # create replay buffer
    buffer = afbc.replay.PrioritizedReplayBuffer(size=dset_size)
    buffer.load_experience(
        {"image": dset["observations"].astype(np.uint8)},
        np.expand_dims(dset["actions"], 1),
        dset["rewards"],
        {"image": dset["next_observations"].astype(np.uint8)},
        dset["terminals"],
    )

    # run training
    afbc.afbc(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        name=args.name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="minigrid-fourrooms-random-v0")
    parser.add_argument("--name", type=str, default="afbc_run")
    args = parser.parse_args()
    train_d4rl_minigrid(args)

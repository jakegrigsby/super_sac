import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import uafbc
import gym
import d4rl_atari


class AtariEncoder(uafbc.nets.Encoder):
    def __init__(self, img_shape, emb_dim=128):
        super().__init__()
        self._emb_dim = emb_dim
        self.cnn = uafbc.nets.cnns.SmallPixelEncoder(img_shape, emb_dim)

    @property
    def embedding_dim(self):
        return self._emb_dim

    def forward(self, obs_dict):
        img = obs_dict["obs"].float()
        return self.cnn(img)


class EnvWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return {"obs": obs[:]}


def train_d4rl_atari(args):
    train_env = EnvWrapper(gym.make(args.env, stack=True))
    test_env = EnvWrapper(gym.make(args.env, stack=True))
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = uafbc.Agent(
        action_space.n,
        encoder=AtariEncoder(state_space.shape),
        actor_network_cls=uafbc.nets.mlps.DiscreteActor,
        critic_network_cls=uafbc.nets.mlps.DiscreteCritic,
        hidden_size=256,
        discrete=True,
        auto_rescale_targets=False,
    )

    # get offline datset
    dset = train_env.get_dataset()
    states = np.array(dset["observations"]).astype(np.uint8)[:-1]
    actions = dset["actions"][:-1]
    rewards = dset["rewards"][:-1]
    dones = dset["terminals"][:-1]
    next_states = np.array(dset["observations"]).astype(np.uint8)[1:]
    dset_size = len(states)

    # create replay buffer
    buffer = uafbc.replay.PrioritizedReplayBuffer(size=dset_size)
    buffer.load_experience(
        {"obs": states},
        np.expand_dims(actions, 1),
        rewards,
        {"obs": next_states},
        dones,
    )

    # run training
    uafbc.uafbc(
        agent=agent,
        buffer=buffer,
        train_env=train_env,
        test_env=test_env,
        num_steps_offline=1_000_000,
        num_steps_online=0,
        batch_size=64,
        pop=False,
        verbosity=1,
        name=args.name,
        weighted_bellman_temp=None,
        weight_type=None,
        max_episode_steps=108_000,
        eval_episodes=1,
        eval_interval=20_000,
        init_alpha=0,
        alpha_lr=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="breakout-mixed-v0")
    parser.add_argument("--name", type=str, default="uafbc_atari_offline")
    args = parser.parse_args()
    train_d4rl_atari(args)

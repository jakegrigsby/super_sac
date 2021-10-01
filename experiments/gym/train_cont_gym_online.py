import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import pybullet
import pybullet_envs

import uafbc
from uafbc.wrappers import SimpleGymWrapper, NormActionSpace, ParallelActors


class IdentityEncoder(uafbc.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        self.dummy = torch.nn.Linear(1, 1)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_cont_gym_online(args):

    def make_env():
        return NormActionSpace(gym.make(args.env))

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.actors))
    test_env = SimpleGymWrapper(make_env())

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        critic_ensemble_size=2,
        hidden_size=256,
        discrete=False,
        auto_rescale_targets=False,
        beta_dist=False,
    )

    buffer = uafbc.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    uafbc.uafbc(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        name=args.name,
        use_pg_update_online=True,
        actor_lr=1e-4,
        critic_lr=1e-4,
        encoder_lr=1e-4,
        batch_size=512,
        weighted_bellman_temp=None,
        weight_type=None,
        use_bc_update_online=False,
        bc_warmup_steps=0,
        num_steps_offline=0,
        num_steps_online=1_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=1000,
        pop=False,
        init_alpha=0.1,
        alpha_lr=1e-4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--name", type=str, default="uafbc_pendulum_run")
    parser.add_argument("--actors", type=int, default=1)
    args = parser.parse_args()
    train_cont_gym_online(args)

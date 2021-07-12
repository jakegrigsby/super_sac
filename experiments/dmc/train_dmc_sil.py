import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dmc2gym

import uafbc
from uafbc.wrappers import SimpleGymWrapper


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


def train_dmc_sil(args):
    train_env = SimpleGymWrapper(dmc2gym.make(args.domain, args.task))
    test_env = SimpleGymWrapper(dmc2gym.make(args.domain, args.task))

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        critic_ensemble_size=2,
        hidden_size=1024,
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
        use_bc_update_online=True,
        actor_lr=1e-4,
        critic_lr=1e-4,
        actor_clip=10.,
        batch_size=512,
        weighted_bellman_temp=None,
        weight_type=None,
        num_steps_offline=0,
        num_steps_online=1_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=1000,
        pop=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--name", type=str, default="uafbc_sil_dmc_run")
    args = parser.parse_args()
    train_dmc_sil(args)

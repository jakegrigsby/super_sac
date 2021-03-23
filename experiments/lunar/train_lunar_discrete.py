import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import afbc
from afbc.wrappers import SimpleGymWrapper, DiscreteActionWrapper
import gym


class IdentityEncoder(afbc.nets.AFBCEncoder):
    def __init__(self, obs_dim):
        super().__init__()
        self._dim = obs_dim
        self.dummy = nn.Linear(1, 1)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_gym_discrete(args):
    train_env = SimpleGymWrapper(DiscreteActionWrapper(gym.make(args.env)))
    test_env = SimpleGymWrapper(DiscreteActionWrapper(gym.make(args.env)))
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = afbc.AFBCAgent(
        action_space.n,
        encoder=IdentityEncoder(state_space.shape[0]),
        actor_network_cls=afbc.nets.mlps.DiscreteActor,
        critic_network_cls=afbc.nets.mlps.DiscreteCritic,
        hidden_size=128,
        discrete=True,
        critic_ensemble_size=2,
        auto_rescale_targets=False,
        beta_dist=False,
    )

    # create replay buffer
    buffer = afbc.replay.PrioritizedReplayBuffer(size=250_000)

    # run training
    afbc.afbc(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        name=args.name,
        use_pg_update_online=True,
        use_bc_update_online=False,
        actor_lr=1e-4,
        critic_lr=1e-4,
        batch_size=128,
        num_steps_offline=0,
        num_steps_online=250_000,
        random_warmup_steps=10_000,
        max_episode_steps=1000,
        weighted_bellman_temp=None,
        weight_type=None,
        encoder_lambda=0.,
        actor_lambda=0.,
        aug_mix=1.,
        augmenter=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--name", type=str, default="afbc_online_discrete")
    args = parser.parse_args()
    train_gym_discrete(args)

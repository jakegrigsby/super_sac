import argparse

import numpy as np
import torch

import d4rl
import afbc
import gym
from afbc.wrappers import SimpleGymWrapper, NormActionSpace


class IdentityEncoder(afbc.nets.AFBCEncoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        self.dummy = torch.nn.Linear(1, 1)

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_d4rl_gym(args):
    train_env = SimpleGymWrapper(NormActionSpace(gym.make(args.env)))
    test_env = SimpleGymWrapper(NormActionSpace(gym.make(args.env)))
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = afbc.AFBCAgent(
        action_space.shape[0],
        encoder=IdentityEncoder(state_space.shape[0]),
        actor_network_cls=afbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=afbc.nets.mlps.ContinuousCritic,
        hidden_size=256,
        beta_dist=True,
        discrete=False,
    )

    # get offline datset
    dset = d4rl.qlearning_dataset(test_env)
    dset_size = dset["observations"].shape[0]
    # create replay buffer
    buffer = afbc.replay.PrioritizedReplayBuffer(size=dset_size)
    buffer.load_experience(
        {"obs": dset["observations"]},
        dset["actions"],
        dset["rewards"],
        {"obs": dset["next_observations"]},
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
    parser.add_argument("--env", type=str, default="hopper-medium-expert-v0")
    parser.add_argument("--name", type=str, default="afbc_run")
    args = parser.parse_args()
    train_d4rl_gym(args)

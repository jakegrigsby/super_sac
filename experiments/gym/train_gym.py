import argparse
import os
import gin

import gymnasium as gym
from torch import nn
import torch.nn.functional as F

import super_sac
from super_sac.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    DiscreteActionWrapper,
)


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


class SharedEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self.fc0 = nn.Linear(dim, 128)
        self.fc1 = nn.Linear(128, dim)
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        x = F.relu(self.fc0(obs_dict["obs"]))
        x = F.relu(self.fc1(x))
        return x


def train_gym(args):
    gin.parse_config_file(args.config)

    discrete = isinstance(gym.make(args.env).action_space, gym.spaces.Discrete)

    def make_env():
        env = gym.make(args.env)
        if discrete:
            env = DiscreteActionWrapper(env)
        else:
            env = NormActionSpace(env)
        env = gym.wrappers.TimeLimit(env, args.max_episode_steps)
        return env

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(make_env())
    if args.render:
        train_env.reset()
        test_env.reset()  # fix common gym render bug

    act_space_size = (
        train_env.action_space.n if discrete else train_env.action_space.shape[0]
    )

    dim = train_env.observation_space.shape[0]
    if args.shared_encoder:
        encoder = SharedEncoder(dim)
    else:
        encoder = IdentityEncoder(dim)

    # create agent
    agent = super_sac.Agent(
        act_space_size=act_space_size,
        encoder=encoder,
    )

    buffer = super_sac.replay.ReplayBuffer(size=1_000_000)
    # buffer = super_sac.replay.TrajectoryBuffer(max_trajectories=100_000, seq_length=1, workers=0, parallel_rollouts=args.parallel_envs)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        logging_method=args.logging,
        render=args.render,
        max_episode_steps=args.max_episode_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--name", type=str, default="super_sac_pendulum_run")
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument(
        "--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )
    parser.add_argument("--shared_encoder", action="store_true")
    args = parser.parse_args()
    train_gym(args)

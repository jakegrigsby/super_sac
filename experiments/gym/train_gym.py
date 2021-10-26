import argparse
import os
import gin

import gym

import pybullet
import pybullet_envs

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


def train_gym(args):
    gin.parse_config_file(args.config)

    discrete = isinstance(gym.make(args.env).action_space, gym.spaces.Discrete)

    def make_env():
        env = gym.make(args.env)
        if discrete:
            env = DiscreteActionWrapper(env)
        else:
            env = NormActionSpace(env)
        return env

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(make_env())
    if args.render:
        train_env.reset()
        test_env.reset()  # fix common gym render bug

    if discrete:
        actor_network_cls = super_sac.nets.mlps.DiscreteActor
        critic_network_cls = super_sac.nets.mlps.DiscreteCritic
        act_space_size = train_env.action_space.n
    else:
        actor_network_cls = super_sac.nets.mlps.ContinuousStochasticActor
        critic_network_cls = super_sac.nets.mlps.ContinuousCritic
        act_space_size = train_env.action_space.shape[0]

    # create agent
    agent = super_sac.Agent(
        act_space_size=act_space_size,
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
        actor_network_cls=actor_network_cls,
        critic_network_cls=critic_network_cls,
        discrete=discrete,
    )

    buffer = super_sac.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        logging_method="wandb",
        render=args.render,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--name", type=str, default="super_sac_pendulum_run")
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--parallel_envs", type=int, default=1)
    args = parser.parse_args()
    train_gym(args)

import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import uafbc
from uafbc.wrappers import ParallelActors, DiscreteActionWrapper, SimpleGymWrapper
import gym
from minatar_utils import MinAtarEnv, MinAtarEncoder


def train_minatar_online(args):
    def make_env():
        return DiscreteActionWrapper(MinAtarEnv(args.game))

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_eval_envs))

    # create agent
    agent = uafbc.Agent(
        act_space_size=6,
        encoder=MinAtarEncoder(channels=make_env().num_channels),
        actor_network_cls=uafbc.nets.mlps.DiscreteActor,
        critic_network_cls=uafbc.nets.mlps.DiscreteCritic,
        hidden_size=256,
        discrete=True,
        critic_ensemble_size=3,
        actor_ensemble_size=args.actors,
        ucb_bonus=0.0,
        auto_rescale_targets=True,
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
        use_bc_update_online=False,
        num_steps_offline=0,
        num_steps_online=5_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=100_000,
        pop=True,
        weighted_bellman_temp=10.0,
        weight_type="softmax",
        target_entropy_mul=0.5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--name", type=str, default="uafbc_minatar_online")
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--parallel_eval_envs", type=int, default=1)
    parser.add_argument("--actors", type=int, default=1)
    args = parser.parse_args()
    train_minatar_online(args)

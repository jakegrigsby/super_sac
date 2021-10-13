import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import pybullet
import pybullet_envs

import uafbc
from uafbc.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    ScaleReward,
)


class IdentityEncoder(uafbc.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_cont_gym_online(args):
    def make_env():
        return ScaleReward(NormActionSpace(gym.make(args.env)), args.r_scale)

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_eval_envs))

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        discrete=False,
        ensemble_size=args.ensemble_size,
        num_critics=args.num_critics,
        hidden_size=args.hidden_size,
        ucb_bonus=args.ucb_bonus,
        auto_rescale_targets=args.popart,
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
        batch_size=args.batch_size,
        target_critic_ensemble_n=2,
        weighted_bellman_temp=None,
        weight_type=None,
        use_bc_update_online=False,
        bc_warmup_steps=0,
        num_steps_offline=0,
        num_steps_online=1_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=1000,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        pop=args.popart,
        init_alpha=0.1,
        alpha_lr=1e-4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--name", type=str, default="uafbc_pendulum_run")
    parser.add_argument("--ucb_bonus", type=float, default=0.0)
    parser.add_argument("--r_scale", type=float, default=1.0)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--num_critics", type=int, default=2)
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--parallel_eval_envs", type=int, default=1)
    parser.add_argument("--popart", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=5_000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    train_cont_gym_online(args)

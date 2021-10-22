import argparse

import numpy as np
import torch

import d4rl
import uafbc
import gym


class IdentityEncoder(uafbc.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_d4rl_gym(args):
    train_env = uafbc.wrappers.SimpleGymWrapper(gym.make(args.env))
    test_env = uafbc.wrappers.SimpleGymWrapper(gym.make(args.env))
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = uafbc.Agent(
        action_space.shape[0],
        encoder=IdentityEncoder(state_space.shape[0]),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        hidden_size=512,
        discrete=False,
        ensemble_size=args.ensemble_size,
        num_critics=args.num_critics,
        auto_rescale_targets=args.popart,
    )

    # get offline datset
    dset = d4rl.qlearning_dataset(test_env)
    dset_size = dset["observations"].shape[0]
    # create replay buffer
    buffer = uafbc.replay.PrioritizedReplayBuffer(size=dset_size)
    buffer.load_experience(
        {"obs": dset["observations"]},
        dset["actions"],
        dset["rewards"],
        {"obs": dset["next_observations"]},
        dset["terminals"],
    )

    # run training
    uafbc.uafbc(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        num_steps_offline=args.steps,
        num_steps_online=0,
        weighted_bellman_temp=None,
        weight_type=None,
        bc_warmup_steps=0,
        name=args.name,
        pop=args.popart,
        init_alpha=0,
        alpha_lr=0,
        logging_method=args.logging_method,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah-medium-expert-v0")
    parser.add_argument("--name", type=str, default="uafbc_d4rl_gym")
    parser.add_argument("--num_critics", type=int, default=2)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--popart", action="store_true")
    parser.add_argument(
        "--logging_method",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    args = parser.parse_args()
    train_d4rl_gym(args)

import argparse
import os

import dmc2gym
import gin

import super_sac
from super_sac.wrappers import SimpleGymWrapper


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_dmc(args):
    gin.parse_config_file(args.config)
    train_env = SimpleGymWrapper(dmc2gym.make(args.domain, args.task))
    test_env = SimpleGymWrapper(dmc2gym.make(args.domain, args.task))

    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
    )

    buffer = super_sac.replay.ReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        logging_method=args.logging,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--name", type=str, default="super_sac_dmc")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )
    args = parser.parse_args()
    for _ in range(args.trials):
        train_dmc(args)

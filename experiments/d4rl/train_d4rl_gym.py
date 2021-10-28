import argparse
import os
import gin

import d4rl
import super_sac
import gym


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_d4rl_gym(args):
    gin.parse_config_file(args.config)

    train_env = super_sac.wrappers.SimpleGymWrapper(gym.make(args.env))
    test_env = super_sac.wrappers.SimpleGymWrapper(gym.make(args.env))
    state_space = test_env.observation_space
    action_space = test_env.action_space

    # create agent
    agent = super_sac.Agent(
        action_space.shape[0],
        encoder=IdentityEncoder(state_space.shape[0]),
    )

    # get offline datset
    dset = d4rl.qlearning_dataset(test_env)
    dset_size = dset["observations"].shape[0]
    # create replay buffer
    buffer = super_sac.replay.ReplayBuffer(size=dset_size)
    buffer.load_experience(
        {"obs": dset["observations"]},
        dset["actions"],
        dset["rewards"],
        {"obs": dset["next_observations"]},
        dset["terminals"],
    )

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
    parser.add_argument("--env", type=str, default="halfcheetah-medium-expert-v0")
    parser.add_argument("--name", type=str, default="super_sac_d4rl_gym")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard")
    args = parser.parse_args()
    train_d4rl_gym(args)

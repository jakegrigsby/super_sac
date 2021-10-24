import argparse
import pickle
import os
import gin

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import super_sac
from super_sac.wrappers import ParallelActors, DiscreteActionWrapper, Uint8Wrapper
import gym
from minatar_utils import MinAtarEnv, MinAtarEncoder


def train_minatar(args):
    gin.parse_config_file(args.config)

    if args.dset is None:
        print("Warning: Running without offline dataset.")

    def make_env():
        return DiscreteActionWrapper(MinAtarEnv(args.game))

    train_env = Uint8Wrapper(ParallelActors(make_env, args.parallel_train_envs))
    test_env = Uint8Wrapper(ParallelActors(make_env, args.parallel_eval_envs))

    # create agent
    agent = super_sac.Agent(
        act_space_size=6,
        encoder=MinAtarEncoder(channels=make_env().num_channels),
    )

    if args.dset is not None:
        with open(args.dset, "rb") as f:
            data = pickle.load(f)
            size = len(data["rewards"])
            buffer = super_sac.replay.PrioritizedReplayBuffer(size=size)
            print(f"Offline Dset Size: {size}")

            def transpose_dict(d):
                state_dict = {key: [] for key in d[0].keys()}
                for state in d:
                    for key, val in state.items():
                        state_dict[key].append(val)
                state_dict = {key: np.array(val) for key, val in state_dict.items()}
                return state_dict

            buffer.load_experience(
                transpose_dict(data["states"]),
                np.expand_dims(np.array(data["actions"]), 1),
                np.array(data["rewards"]),
                transpose_dict(data["next_states"]),
                np.array(data["dones"]),
            )
    else:
        buffer = super_sac.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        random_warmup_steps=10_000 if args.dset is None else 0,
        logging_method="wandb",
        wandb_entity=os.getenv("SSAC_WANDB_ACCOUNT"),
        wandb_project=os.getenv("SSAC_WANDB_PROJECT"),
        base_save_path=os.getenv("SSAC_SAVE"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--name", type=str, default="super_sac_minatar_offline")
    parser.add_argument("--parallel_eval_envs", type=int, default=1)
    parser.add_argument("--parallel_train_envs", type=int, default=1)
    parser.add_argument("--dset", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_minatar(args)

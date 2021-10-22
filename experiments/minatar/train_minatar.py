import argparse
import pickle

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import uafbc
from uafbc.wrappers import ParallelActors, DiscreteActionWrapper, Uint8Wrapper
import gym
from minatar_utils import MinAtarEnv, MinAtarEncoder


def train_minatar(args):
    def make_env():
        return DiscreteActionWrapper(MinAtarEnv(args.game))

    train_env = Uint8Wrapper(ParallelActors(make_env, args.parallel_train_envs))
    test_env = Uint8Wrapper(ParallelActors(make_env, args.parallel_eval_envs))

    # create agent
    agent = uafbc.Agent(
        act_space_size=6,
        encoder=MinAtarEncoder(channels=make_env().num_channels),
        actor_network_cls=uafbc.nets.mlps.DiscreteActor,
        critic_network_cls=uafbc.nets.mlps.DiscreteCritic,
        hidden_size=256,
        discrete=True,
        num_critics=2,
        ensemble_size=args.ensemble_size,
        ucb_bonus=0.0,
        auto_rescale_targets=False,
        beta_dist=False,
    )

    if args.dset is not None:
        with open(args.dset, "rb") as f:
            data = pickle.load(f)
            size = len(data["rewards"])
            buffer = uafbc.replay.PrioritizedReplayBuffer(size=size)
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
        buffer = uafbc.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    uafbc.uafbc(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        name=args.name,
        bc_warmup_steps=args.bc_steps,
        num_steps_offline=args.offline_steps,
        num_steps_online=args.online_steps,
        use_pg_update_online=True,
        use_bc_update_online=False,
        random_warmup_steps=10_000 if args.dset is None else 0,
        max_episode_steps=100_000,
        actor_lr=5e-4,
        critic_lr=5e-4,
        pop=False,
        weighted_bellman_temp=None,
        weight_type=None,
        init_alpha=0,
        alpha_lr=1e-4,
        target_entropy_mul=0.5,
        eval_interval=5000,
        log_interval=50,
        afbc_per=False,
        render=args.render,
        logging_method=args.logging_method,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--name", type=str, default="uafbc_minatar_offline")
    parser.add_argument("--parallel_eval_envs", type=int, default=1)
    parser.add_argument("--parallel_train_envs", type=int, default=1)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--dset", type=str, default=None)
    parser.add_argument("--bc_steps", type=int, default=0)
    parser.add_argument("--offline_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--online_steps", type=int, default=1_000_000)
    parser.add_argument(
        "--logging_method",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    args = parser.parse_args()
    train_minatar(args)

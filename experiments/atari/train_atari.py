import gym

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import uafbc
from uafbc import nets
from uafbc.augmentations import AugmentationSequence, DrqAug


class AtariEncoder(nets.Encoder):
    def __init__(self, img_shape, emb_dim=128):
        super().__init__()
        self._emb_dim = emb_dim
        self.cnn = nets.cnns.SmallPixelEncoder(img_shape, emb_dim)

    @property
    def embedding_dim(self):
        return self._emb_dim

    def forward(self, obs_dict):
        return self.cnn(obs_dict["obs"])


def train_atari(args):
    def make_env():
        return uafbc.wrappers.load_atari(args.game, frame_skip=4)

    train_env = uafbc.wrappers.SimpleGymWrapper(
        uafbc.wrappers.ParallelActors(make_env, args.actors)
    )
    test_env = uafbc.wrappers.SimpleGymWrapper(make_env())

    # create agent
    img_shape = train_env.observation_space.shape
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.n,
        encoder=AtariEncoder(img_shape, emb_dim=128),
        actor_network_cls=uafbc.nets.mlps.DiscreteActor,
        critic_network_cls=uafbc.nets.mlps.DiscreteCritic,
        hidden_size=256,
        discrete=True,
        critic_ensemble_size=2,
        auto_rescale_targets=args.popart,
        beta_dist=False,
    )

    buffer = uafbc.replay.PrioritizedReplayBuffer(size=250_000)

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
        num_steps_online=args.steps,
        random_warmup_steps=10_000,
        max_episode_steps=108_000,
        actor_clip=10.0,
        critic_clip=10.0,
        encoder_clip=10.0,
        init_alpha=0.1,
        batch_size=64,
        pop=args.popart,
        weighted_bellman_temp=None,
        weight_type=None,
        critic_updates_per_step=1,
        eval_episodes=10,
        augmenter=None,
        target_entropy_mul=1.0,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Name of logging dir", required=True)
    parser.add_argument("--game", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--popart", action="store_true")
    parser.add_argument("--actors", type=int, default=1)
    args = parser.parse_args()
    train_atari(args)

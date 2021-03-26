import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dmc_remastered as dmcr

import uafbc
from uafbc.wrappers import SimpleGymWrapper
from uafbc import nets
from uafbc.augmentations import AugmentationSequence, DrqAug


class DMCREncoder(nets.Encoder):
    def __init__(self, inp_shape, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv_block = nets.cnns.BigPixelEncoder(inp_shape, emb_dim)

    def forward(self, obs_dict):
        return self.conv_block(obs_dict["obs"])

    @property
    def embedding_dim(self):
        return self._dim


def train_dmcr_online(args):
    train_env, test_env = dmcr.benchmarks.classic(args.domain, args.task, visual_seed=args.seed)
    train_env = SimpleGymWrapper(train_env)
    test_env = SimpleGymWrapper(test_env)

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=DMCREncoder(train_env.observation_space.shape),
        actor_network_cls=uafbc.nets.mlps.ContinuousStochasticActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        critic_ensemble_size=2,
        hidden_size=1024,
        discrete=False,
        auto_rescale_targets=False,
        beta_dist=False,
    )

    buffer = uafbc.replay.PrioritizedReplayBuffer(size=100_000)

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
        batch_size=512,
        weighted_bellman_temp=None,
        weight_type=None,
        use_bc_update_online=False,
        num_steps_offline=0,
        num_steps_online=1_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=1000,
        pop=False,
        augmenter=AugmentationSequence([DrqAug(512)]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="cartpole")
    parser.add_argument("--task", type=str, default="balance")
    parser.add_argument("--name", type=str, default="uafbc_dmcr_run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train_dmcr_online(args)

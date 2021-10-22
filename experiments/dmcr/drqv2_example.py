"""
Shows hyperparamter settings to (roughly) replicate DrqV2.
See https://arxiv.org/abs/2107.09645.

Notes on Key Changes:
1. init_alpha, alpha_lr = 0 turns automatic entropy off
2. exploration noise is added with decaying gaussian noise.
3. that noise is also used in the online actor and critic updates
    - this was the main code addition needed.
4. faster drqv2 augmentation.
5. drqv2 doesn't use a target encoder. We fake this change by
   setting encoder_tau = 1.0, target_delay = 1, meaning it is fully
   updated to the online encoder every update.

We are still missing n-step returns, and the overall
implementation is much slower due to more replay buffer sampling. No
real plans to add n-step returns anytime soon. I will work on seeing if sunrise or discor-style critic updates can improve learning to a similar extent without
creating off-policy issues.
"""

import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dmc_remastered as dmcr
import gym

import uafbc
from uafbc import nets
from uafbc.augmentations import AugmentationSequence, Drqv2Aug


class DMCREncoder(nets.Encoder):
    def __init__(self, inp_shape, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv_block = nets.cnns.BigPixelEncoder(inp_shape, emb_dim)

    def forward(self, obs_dict):
        img = obs_dict["obs"]
        return self.conv_block(img)

    @property
    def embedding_dim(self):
        return self._dim


class Uint8Image(gym.ObservationWrapper):
    def observation(self, obs):
        return {"obs": obs.astype(np.uint8)}


def train_dmcr_drqv2(args):
    train_env, test_env = dmcr.benchmarks.classic(
        args.domain, args.task, visual_seed=args.seed
    )
    train_env = Uint8Image(train_env)
    test_env = Uint8Image(test_env)

    # create agent
    agent = uafbc.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=DMCREncoder(train_env.observation_space.shape),
        # deterministic actor with exploration noise
        actor_network_cls=uafbc.nets.mlps.ContinuousDeterministicActor,
        critic_network_cls=uafbc.nets.mlps.ContinuousCritic,
        ensemble_size=1,
        num_critics=2,
        hidden_size=1024,
        discrete=False,
        auto_rescale_targets=False,
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
        actor_lr=1e-4,
        critic_lr=1e-4,
        encoder_lr=1e-4,
        # drqv2 does not use target encoders.
        # encoder_tau=1 has the same effect.
        encoder_tau=1.0,
        mlp_tau=0.01,
        target_delay=1,
        batch_size=256,
        # no automatic entropy
        init_alpha=0,
        alpha_lr=0,
        # exploration noise, also used in updates
        use_exploration_process=True,
        # control exploration noise scale schedule
        exploration_param_init=1.0,
        exploration_param_final=0.1,
        exploration_param_anneal=1_000_000,
        weighted_bellman_temp=None,
        weight_type=None,
        use_bc_update_online=False,
        # "upate delay = 2"
        transitions_per_online_step=2,
        num_steps_offline=0,
        num_steps_online=args.steps,
        random_warmup_steps=6_000,
        max_episode_steps=1000,
        eval_interval=5000,
        eval_episodes=10,
        log_interval=500,
        pop=False,
        augmenter=AugmentationSequence([Drqv2Aug(256)]),
        logging_method=args.logging_method,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="cartpole")
    parser.add_argument("--task", type=str, default="balance")
    parser.add_argument("--name", type=str, default="uafbc_dmcr_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument(
        "--logging_method",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    args = parser.parse_args()
    train_dmcr_drqv2(args)

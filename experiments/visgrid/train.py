"""
This script mostly recreates the main grid world experiment figure in
"Learning Markov State Abstractions for Deep Reinforcement Learning",
Allen et al. 2021 (https://arxiv.org/abs/2106.04379)

It takes ~5 mins to run on a GPU and will display a scatter plot similar
to arXiv v3 Figure 3a.
"""
import argparse
import gin
import os
import pickle

import numpy as np
import gym
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import super_sac
from super_sac import nets
from super_sac.augmentations import AugmentationSequence, Drqv2Aug
from super_sac.wrappers import SimpleGymWrapper

try:
    from visgrid.gridworld import GridWorld
    from visgrid.sensors import *
except ImportError:
    raise ImportError(
        "Clone visgrid (https://github.com/camall3n/visgrid/) to this directory first"
    )


class VisgridGymWrapper(gym.Env):
    def __init__(self, visgrid):
        super().__init__()
        self.env = visgrid
        self.sensor = SensorChain(
            [
                OffsetSensor(offset=(0.5, 0.5)),
                NoisySensor(sigma=0.05),
                ImageSensor(
                    range=((0, self.env._rows), (0, self.env._cols)), pixel_density=3
                ),
                BlurSensor(sigma=0.6, truncate=1.0),
                NoisySensor(sigma=0.01),
            ]
        )

        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(18, 18)
        )
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        self.env.reset_agent()
        # no goal... all rews 0
        # self.env.reset_goal()
        return self.create_obs()

    def step(self, action):
        action = int(action[0])
        s, r, done = self.env.step(action)
        return self.create_obs(), r, done, {}

    def create_obs(self):
        state = self.env.get_state()
        return {"obs": self.sensor.observe(state)}


class VisgridEncoder(nets.Encoder):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18 * 18, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, obs_dict):
        img = obs_dict["obs"]
        img = img.view(img.shape[0], -1)
        return torch.tanh(self.fc2(F.relu(self.fc1(img))))

    @property
    def embedding_dim(self):
        return 2


def train_visgrid():
    train_env = VisgridGymWrapper(GridWorld(rows=6, cols=6))
    test_env = VisgridGymWrapper(GridWorld(rows=6, cols=6))
    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.n,
        encoder=VisgridEncoder(),
        actor_network_cls=nets.mlps.DiscreteActor,
        critic_network_cls=nets.mlps.DiscreteCritic,
        discrete=True,
        ensemble_size=1,
        hidden_size=64,
        auto_rescale_targets=False,
    )

    buffer = super_sac.replay.ReplayBuffer(size=100_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name="visgrid_rep_test",
        logging_method="wandb",
        # augmenter=AugmentationSequence([Drqv2Aug(128)]),
        batch_size=256,
        num_steps_offline=0,
        num_steps_online=10_000,
        afbc_actor_updates_per_step=0,
        use_afbc_update_online=False,
        use_pg_update_online=False,
        pg_actor_updates_per_step=0,
        critic_updates_per_step=0,
        init_alpha=0,
        alpha_lr=0,
        pop=False,
        reuse_replay_dicts=False,
        transitions_per_online_step=0,
        inverse_markov_coeff=1.0,
        contrastive_markov_coeff=1.0,
        smoothness_markov_coeff=0.0,
        smoothness_markov_max_dist=0.0,  # N/A
        markov_abstraction_updates_per_step=1,
        random_warmup_steps=30_000,
        log_interval=1000,
        max_episode_steps=100,
        eval_episodes=1,
        verbosity=1,
    )

    (o, *_), _ = buffer.sample_uniform(512)
    s_reps = (
        agent.encoder({key: val.float().to(super_sac.device) for key, val in o.items()})
        .detach()
        .cpu()
        .numpy()
    )
    plt.scatter(s_reps[:, 0], s_reps[:, 1])
    plt.show()


if __name__ == "__main__":
    train_visgrid()

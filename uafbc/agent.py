import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import device, popart, adv_estimator


class Agent:
    def __init__(
        self,
        act_space_size,
        encoder,
        actor_network_cls,
        critic_network_cls,
        discrete=False,
        critic_ensemble_size=5,
        embedding_size=50,
        hidden_size=256,
        auto_rescale_targets=True,
        log_std_low=-10.0,
        log_std_high=2.0,
        adv_method=None,
        beta_dist=False,
    ):

        assert hasattr(encoder, "embedding_dim")

        # adjust network constructor arguments based on action space
        actor_kwargs = {
            "state_size": encoder.embedding_dim,
            "action_size": act_space_size,
            "hidden_size": hidden_size,
        }
        critic_kwargs = {
            "state_size": encoder.embedding_dim,
            "action_size": act_space_size,
            "hidden_size": hidden_size,
        }

        if not discrete:
            actor_kwargs.update(
                {
                    "log_std_low": log_std_low,
                    "log_std_high": log_std_high,
                    "dist_impl": "beta" if beta_dist else "pyd",
                }
            )

        # create networks
        self.encoder = encoder
        self.actor = actor_network_cls(**actor_kwargs)
        self.critics = [
            critic_network_cls(**critic_kwargs) for _ in range(critic_ensemble_size)
        ]

        # create popart layer if we are autoscaling targets
        if auto_rescale_targets:
            self.popart = popart.PopArtLayer()
        else:
            self.popart = False

        # create adv estimator
        if discrete:
            self.adv_estimator = adv_estimator.AdvantageEstimator(
                self.encoder,
                self.actor,
                self.critics,
                self.popart,
                discrete=True,
                discrete_method=adv_method if adv_method else "indirect",
            )
        else:
            self.adv_estimator = adv_estimator.AdvantageEstimator(
                self.encoder,
                self.actor,
                self.critics,
                self.popart,
                discrete=False,
                continuous_method=adv_method if adv_method else "mean",
            )

        self.discrete = discrete

    def to(self, device):
        self.actor = self.actor.to(device)
        self.encoder = self.encoder.to(device)
        if self.popart:
            self.popart = self.popart.to(device)
        for i, critic in enumerate(self.critics):
            self.critics[i] = critic.to(device)

    def eval(self):
        self.actor.eval()
        self.encoder.eval()
        if self.popart:
            self.popart.eval()
        for critic in self.critics:
            critic.eval()

    def train(self):
        self.actor.train()
        self.encoder.train()
        if self.popart:
            self.popart.train()
        for critic in self.critics:
            critic.train()

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        torch.save(self.actor.state_dict(), actor_path)
        encoder_path = os.path.join(path, "encoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)
        if self.popart:
            popart_path = os.path.join(path, "popart.pt")
            torch.save(self.popart.state_dict(), popart_path)
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            torch.save(critic.state_dict(), critic_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        encoder_path = os.path.join(path, "encoder.pt")
        self.encoder.load_state_dict(torch.load(encoder_path))
        if self.popart:
            popart_path = os.path.join(path, "popart.pt")
            self.popart.load_state_dict(torch.load(popart_path))
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            critic.load_state_dict(torch.load(critic_path))

    def discrete_forward(self, obs, from_cpu=True):
        if from_cpu:
            obs = self._process_obs(obs)
        self.eval()
        with torch.no_grad():
            state_rep = self.encoder.forward(obs)
            act_dist = self.actor.forward(state_rep)
            act = torch.argmax(act_dist.probs, dim=1)
        self.actor.train()
        if from_cpu:
            act = self._process_act(act)
        return act

    def continuous_forward(self, obs, from_cpu=True):
        if from_cpu:
            obs = self._process_obs(obs)
        self.actor.eval()
        with torch.no_grad():
            s_rep = self.encoder(obs)
            act_dist = self.actor(s_rep)
            act = act_dist.mean
        self.actor.train()
        if from_cpu:
            act = self._process_act(act)
        return act

    def forward(self, state, from_cpu=True):
        if self.discrete:
            return self.discrete_forward(state, from_cpu)
        else:
            return self.continuous_forward(state, from_cpu)

    def sample_action(self, obs, from_cpu=True):
        if from_cpu:
            obs = self._process_obs(obs)
        self.eval()
        with torch.no_grad():
            state_rep = self.encoder.forward(obs)
            act_dist = self.actor.forward(state_rep)
            act = act_dist.sample()
        self.train()
        if from_cpu:
            act = self._process_act(act)
        return act

    def _process_obs(self, obs):
        return {
            x: torch.from_numpy(y).unsqueeze(0).float().to(device)
            for x, y in obs.items()
        }

    def _process_act(self, act):
        if not self.discrete:
            act = act.squeeze(0).clamp(-1.0, 1.0)
        return act.cpu().numpy()

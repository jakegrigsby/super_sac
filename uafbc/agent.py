import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import device, popart, adv_estimator


class Critic(nn.Module):
    def __init__(self, critic_network_cls, critic_kwargs, num_critics):
        super().__init__()
        self.nets = nn.ModuleList(
            [critic_network_cls(**critic_kwargs) for _ in range(num_critics)]
        )
        self.num_critics = num_critics

    def forward(self, *args, subset=None, return_min=True):
        if subset is not None:
            # compute q vals from a random subset of the networks
            # (used in redq temporal difference target step)
            assert subset > 0 and subset <= self.num_critics
            # can't directly sample ModuleList
            ensemble = [
                self.nets[i] for i in random.sample(range(self.num_critics), k=subset)
            ]
        else:
            ensemble = self.nets

        preds = [q(*args) for q in ensemble]
        if return_min:
            return torch.stack(preds, dim=0).min(0).values
        else:
            return tuple(preds)


class Agent:
    def __init__(
        self,
        act_space_size,
        encoder,
        actor_network_cls,
        critic_network_cls,
        discrete=False,
        ensemble_size=3,
        num_critics=2,
        ucb_bonus=0.0,
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
        self.actors = [actor_network_cls(**actor_kwargs) for _ in range(ensemble_size)]
        self.critics = [
            Critic(critic_network_cls, critic_kwargs, num_critics)
            for _ in range(ensemble_size)
        ]

        self.ensemble_size = ensemble_size
        self.num_critics = num_critics

        # create popart layer if we are autoscaling targets
        if auto_rescale_targets:
            self.popart = [popart.PopArtLayer() for _ in range(ensemble_size)]
        else:
            self.popart = [False for _ in range(ensemble_size)]

        # create adv estimator
        if discrete:
            self.adv_estimator = adv_estimator.AdvantageEstimator(
                encoder=self.encoder,
                actors=self.actors,
                critics=self.critics,
                popart=self.popart,
                discrete=True,
                discrete_method=adv_method if adv_method else "indirect",
            )
        else:
            self.adv_estimator = adv_estimator.AdvantageEstimator(
                encoder=self.encoder,
                actors=self.actors,
                critics=self.critics,
                popart=self.popart,
                discrete=False,
                continuous_method=adv_method if adv_method else "mean",
            )

        self.discrete = discrete
        self.ucb_bonus = ucb_bonus

    @property
    def ensemble(self):
        return zip(self.actors, self.critics)

    def to(self, device):
        for i, actor in enumerate(self.actors):
            self.actors[i] = actor.to(device)
        self.encoder = self.encoder.to(device)
        for i, popart in enumerate(self.popart):
            if popart:
                self.popart[i] = popart.to(device)
        for i, critic in enumerate(self.critics):
            self.critics[i] = critic.to(device)

    def eval(self):
        self.encoder.eval()
        for popart in self.popart:
            if popart:
                popart.eval()
        for critic in self.critics:
            critic.eval()
        for actor in self.actors:
            actor.eval()

    def train(self):
        self.encoder.train()
        for popart in self.popart:
            if popart:
                popart.train()
        for critic in self.critics:
            critic.train()
        for actor in self.actors:
            actor.train()

    def save(self, path):
        encoder_path = os.path.join(path, "encoder.pt")
        torch.save(self.encoder.state_dict(), encoder_path)
        for i, popart in enumerate(self.popart):
            if popart:
                popart_path = os.path.join(path, "popart{i}.pt")
                torch.save(popart.state_dict(), popart_path)
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            torch.save(critic.state_dict(), critic_path)
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor{i}.pt")
            torch.save(actor.state_dict(), actor_path)

    def load(self, path):
        encoder_path = os.path.join(path, "encoder.pt")
        self.encoder.load_state_dict(torch.load(encoder_path))
        for i, popart in enumerate(self.popart):
            if popart:
                popart_path = os.path.join(path, "popart{i}.pt")
                popart.load_state_dict(torch.load(popart_path))
        for i, critic in enumerate(self.critics):
            critic_path = os.path.join(path, f"critic{i}.pt")
            critic.load_state_dict(torch.load(critic_path))
        for i, actor in enumerate(self.actors):
            actor_path = os.path.join(path, f"actor{i}.pt")
            actor.load_state_dict(torch.load(actor_path))

    def discrete_forward(self, obs, from_cpu=True, num_envs=1):
        if from_cpu:
            obs = self._process_obs(obs, num_envs=num_envs)
        self.eval()
        with torch.no_grad():
            state_rep = self.encoder.forward(obs)
            act_probs = torch.stack(
                [actor(state_rep).probs for actor in self.actors], dim=0
            ).mean(0)
            act = torch.argmax(act_probs, dim=-1, keepdim=True)
        self.train()
        if from_cpu:
            act = self._process_act(act, num_envs=num_envs)
        return act

    def continuous_forward(self, obs, from_cpu=True, num_envs=1):
        if from_cpu:
            obs = self._process_obs(obs, num_envs=num_envs)
        self.eval()
        with torch.no_grad():
            s_rep = self.encoder(obs)
            acts = torch.stack([actor(s_rep).mean for actor in self.actors], dim=0)
            act = acts.mean(0)
        self.train()
        if from_cpu:
            act = self._process_act(act, num_envs=num_envs)
        return act

    def forward(self, state, from_cpu=True, num_envs=1):
        if self.discrete:
            return self.discrete_forward(state, from_cpu=from_cpu, num_envs=num_envs)
        else:
            return self.continuous_forward(state, from_cpu=from_cpu, num_envs=num_envs)

    def sample_action(self, obs, from_cpu=True, num_envs=1, return_dist=False):
        if from_cpu:
            obs = self._process_obs(obs, num_envs)
        with torch.no_grad():
            state_rep = self.encoder.forward(obs)

            if self.ucb_bonus > 0:
                # UCB bonus incentivizes taking actions that the ensemble of
                # critics disagree about (from SUNRISE). When using parallel
                # training environments this becomes a huge tensor shape headache...

                # sample one action from each actor in each environment
                act_dists = [actor(state_rep) for actor in self.actors]
                act_candidates = torch.stack(
                    [dist.sample() for dist in act_dists], dim=0
                )
                # act_candidates.shape = (actors, envs, action_dimension)
                act_dist = random.choice(act_dists)  # not important; used for logging

                if self.discrete:
                    act_candidates.unsqueeze_(-1)
                    q_vals = torch.stack(
                        [critic(state_rep) for critic in self.critics],
                        dim=0,
                    )  # q_vals.shape = (critics, actors, envs, action_dimension)

                else:
                    if num_envs > 1:
                        act_candidates.squeeze_(1)
                    state_rep = state_rep.unsqueeze(0).repeat(len(act_candidates), 1, 1)
                    # repeat the state vector for each action candidate
                    # evaluate each candidate action in each env
                    q_vals = torch.stack(
                        [critic(state_rep, act_candidates) for critic in self.critics],
                        dim=0,
                    )  # q_vals.shape = (critics, actors, envs, 1)

                # compute this in a loop for each environment (the -2 axis)
                # TODO: there is probably a better way to do this with `gather` or something...
                act = []
                for q_val_env_i, acts_env_i in zip(
                    q_vals.chunk(num_envs, -2), act_candidates.chunk(num_envs, -2)
                ):
                    q_val_env_i.squeeze_(-2)
                    acts_env_i.squeeze_(-2)
                    if self.discrete:
                        # get q value for the specific candidate actions
                        q_val_env_i = q_val_env_i[..., acts_env_i]
                    ucb_val = q_val_env_i.mean(0) + self.ucb_bonus * q_val_env_i.std(0)
                    argmax_ucb_val = torch.argmax(ucb_val)
                    act.append(acts_env_i[argmax_ucb_val])
                act = torch.stack(act, dim=0)
            else:
                # otherwise pick an action from one of the actors
                act_dist = random.choice(self.actors)(state_rep)
                act = act_dist.sample()
                if self.discrete and num_envs > 1:
                    act = act.unsqueeze(-1)
        if from_cpu:
            act = self._process_act(act, num_envs)
        if return_dist:
            return act, act_dist
        return act

    def _process_obs(self, obs, num_envs=1):
        unsqueeze = lambda tens: tens.unsqueeze(0) if num_envs == 1 else tens
        return {
            x: unsqueeze(torch.from_numpy(y)).float().to(device) for x, y in obs.items()
        }

    def _process_act(self, act, num_envs=1):
        squeeze = lambda tens: tens.squeeze(0) if num_envs == 1 else tens
        act = squeeze(act)
        if not self.discrete:
            act.clamp_(-1.0, 1.0)
        return act.cpu().numpy()

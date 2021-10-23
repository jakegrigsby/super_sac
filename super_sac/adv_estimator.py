import random

import torch
from torch import nn
import torch.nn.functional as F


class AdvantageEstimator(nn.Module):
    def __init__(
        self,
        encoder,
        actors,
        critics,
        popart=False,
        discrete_method="indirect",
        continuous_method="mean",
        discrete=False,
    ):
        super().__init__()
        assert continuous_method in ["mean", "max"]
        assert discrete_method in ["indirect", "direct"]
        self.encoder = encoder
        self.actors = actors
        self.critics = critics
        self.popart = popart
        self.cont_method = continuous_method
        self.discrete = discrete
        self.discrete_method = discrete_method

    def pop(self, ensemble_idx, *args):
        q = self.critics[ensemble_idx](*args)
        if self.popart[ensemble_idx]:
            return self.popart[ensemble_idx](q)
        else:
            return q

    def discrete_direct_forward(self, obs, action):
        # use dueling arch adv
        raise NotImplementedError

    def discrete_indirect_forward(self, obs, action, ensemble_idx):
        state_rep = self.encoder(obs)
        with torch.no_grad():
            # V(s) = E_{a ~ \pi(s)} [Q(s, a)]
            probs = torch.stack(
                [actor(state_rep).probs for actor in self.actors], dim=0
            ).mean(0)
            min_q = self.pop(ensemble_idx, state_rep)
            value = (probs * min_q).sum(1, keepdim=True)

        # Q(s, a)
        q_preds = self.pop(ensemble_idx, state_rep).gather(1, action.long())

        # A(s, a) = Q(s, a) - V(s)
        adv = q_preds - value
        return adv

    def continuous_forward(self, obs, action, ensemble_idx, n=4):
        with torch.no_grad():
            # get an action distribution from the policy
            state_rep = self.encoder(obs)
            policy_actions = [
                self.actors[ensemble_idx](state_rep).sample() for _ in range(n)
            ]
            # get the q value for each of the n actions
            q_a_preds = torch.stack(
                [self.pop(ensemble_idx, state_rep, act) for act in policy_actions],
                dim=0,
            )
            if self.cont_method == "mean":
                # V(s) = E_{a ~ \pi(s)} [Q(s, a)]
                value = q_a_preds.mean(0)
            elif self.cont_method == "max":
                # Optimisitc value estimate: V(s) = max_{a1, a2, a3, ..., aN}(Q(s, a))
                value = q_a_preds.max(0).values
        q_preds = self.pop(ensemble_idx, state_rep, action)
        # A(s, a) = Q(s, a) - V(s)
        adv = q_preds - value
        return adv

    def forward(self, obs, action, ensemble_idx):
        # TODO
        if self.discrete:
            if self.discrete_method == "indirect":
                return self.discrete_indirect_forward(obs, action, ensemble_idx)
            elif self.discrete_method == "direct":
                return self.discrete_direct_forward(obs, action, ensemble_idx)
        else:
            return self.continuous_forward(obs, action, ensemble_idx)

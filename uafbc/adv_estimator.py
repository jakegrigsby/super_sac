import torch
from torch import nn
import torch.nn.functional as F


class AdvantageEstimator(nn.Module):
    def __init__(
        self,
        encoder,
        actor,
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
        self.actor = actor
        self.critics = critics
        self.popart = popart
        self.cont_method = continuous_method
        self.discrete = discrete
        self.discrete_method = discrete_method

    def pop(self, q, *args):
        if self.popart:
            return self.popart(q(*args))
        else:
            return q(*args)

    def discrete_direct_forward(self, obs, action):
        # use dueling arch adv
        raise NotImplementedError

    def discrete_indirect_forward(self, obs, action):
        state_rep = self.encoder(obs)
        # V(s) = E_{a ~ \pi(s)} [Q(s, a)]
        probs = self.actor(state_rep).probs
        min_q = (
            torch.stack([self.pop(q, state_rep) for q in self.critics], dim=0)
            .min(0)
            .values
        )
        value = (probs * min_q).sum(1, keepdim=True)

        # Q(s, a)
        q_preds = (
            torch.stack(
                [self.pop(q, state_rep).gather(1, action.long()) for q in self.critics],
                dim=0,
            )
            .min(0)
            .values
        )

        # A(s, a) = Q(s, a) - V(s)
        adv = q_preds - value
        return adv

    def continuous_forward(self, obs, action, n=4):
        # get an action distribution from the policy
        state_rep = self.encoder(obs)
        act_dist = self.actor(state_rep)
        actions = [act_dist.sample() for _ in range(n)]

        # get the q value for each of the n actions
        qs = []
        for act in actions:
            q_a_preds = (
                torch.stack(
                    [self.pop(critic, state_rep, act) for critic in self.critics], dim=0
                )
                .min(0)
                .values
            )
            qs.append(q_a_preds)
        if self.cont_method == "mean":
            # V(s) = E_{a ~ \pi(s)} [Q(s, a)]
            value = torch.stack(qs, dim=0).mean(0)
        elif self.cont_method == "max":
            # Optimisitc value estimate: V(s) = max_{a1, a2, a3, ..., aN}(Q(s, a))
            value = torch.stack(qs, dim=0).max(0).values
        q_preds = (
            torch.stack(
                [self.pop(critic, state_rep, action) for critic in self.critics], dim=0
            )
            .min(0)
            .values
        )

        # A(s, a) = Q(s, a) - V(s)
        adv = q_preds - value
        if torch.isnan(adv).sum() > 0:
            breakpoint()
        return adv

    def forward(self, obs, action):
        if self.discrete:
            if self.discrete_method == "indirect":
                return self.discrete_indirect_forward(obs, action)
            elif self.discrete_method == "direct":
                return self.discrete_direct_forward(obs, action)
        else:
            return self.continuous_forward(obs, action)

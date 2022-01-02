import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
import gin


from . import distributions, weight_init


@gin.configurable
class ContinuousStochasticActor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        log_std_low=-10.0,
        log_std_high=2.0,
        hidden_size=256,
        dist_impl="pyd",
    ):
        super().__init__()
        assert dist_impl in ["pyd", "beta"]
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2 * action_size)
        self.log_std_low = log_std_low
        self.log_std_high = log_std_high
        self.apply(weight_init)
        self.dist_impl = dist_impl

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        mu, log_std = out.chunk(2, dim=-1)
        if self.dist_impl == "pyd":
            log_std = torch.tanh(log_std)
            log_std = self.log_std_low + 0.5 * (
                self.log_std_high - self.log_std_low
            ) * (log_std + 1)
            std = log_std.exp()
            dist = distributions.SquashedNormal(mu, std)
        elif self.dist_impl == "beta":
            out = 1.0 + F.softplus(out)
            alpha, beta = out.chunk(2, dim=-1)
            dist = distributions.BetaDist(alpha, beta)
        return dist


@gin.configurable
class ContinuousInverseModel(ContinuousStochasticActor):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size,
        **kwargs,
    ):
        super().__init__(state_size, action_size, hidden_size=hidden_size, **kwargs)
        self.fc1 = nn.Linear(state_size * 2, hidden_size)

    def forward(self, state, next_state):
        inp = torch.cat((state, next_state), dim=1)
        return super().forward(inp)


@gin.configurable
class ContinuousDeterministicActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
        self.apply(weight_init)
        self.dist_impl = "deterministic"

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        act = torch.tanh(self.out(x))
        dist = distributions.ContinuousDeterministic(act)
        return dist


@gin.configurable
class ContrastiveModel(nn.Module):
    def __init__(self, state_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.apply(weight_init)

    def forward(self, states, next_states):
        inp = torch.cat((states, next_states), dim=1)
        x = F.relu(self.fc1(inp))
        x = F.relu(self.fc2(x))
        pred = torch.sigmoid(self.out(x))
        return pred


@gin.configurable
class ContinuousCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.features = None
        self.out = nn.Linear(hidden_size, 1)
        self.apply(weight_init)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        self.features = x
        val = self.out(x)
        return val


@gin.configurable
class DiscreteActor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.act_p = nn.Linear(hidden_size, action_size)
        self.apply(weight_init)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # numerical stability improvement??
        # act_p = F.softmax(self.act_p(x), dim=-1)
        # dist = pyd.categorical.Categorical(probs=act_p)
        act_p = self.act_p(x)
        dist = pyd.categorical.Categorical(logits=act_p)
        return dist


@gin.configurable
class DiscreteInverseModel(DiscreteActor):
    def __init__(self, state_size, action_size, hidden_size, **kwargs):
        super().__init__(state_size, action_size, **kwargs)
        self.fc1 = nn.Linear(state_size * 2, hidden_size)

    def forward(self, state, next_state):
        inp = torch.cat((state, next_state), dim=1)
        return super().forward(inp)


@gin.configurable
class DiscreteCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=300):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
        self.features = None
        self.apply(weight_init)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        self.features = x
        vals = self.out(x)
        return vals

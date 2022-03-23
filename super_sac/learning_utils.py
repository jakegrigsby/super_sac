import random
import contextlib
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import device, replay


###########
## UTILS ##
###########


class GaussianExplorationNoise:
    def __init__(
        self,
        action_space,
        start_scale=1.0,
        final_scale=0.1,
        steps_annealed=1000,
        eps=1e-6,
    ):
        assert start_scale >= final_scale
        self.action_space = action_space
        self.start_scale = start_scale
        self.final_scale = final_scale
        self.steps_annealed = steps_annealed
        self.current_scale = start_scale
        self._scale_slope = (start_scale - final_scale) / steps_annealed
        self.eps = eps
        self.act_low_torch = torch.from_numpy(action_space.low).to(device)
        self.act_high_torch = torch.from_numpy(action_space.high).to(device)

    def sample(self, action, clip=None, update_schedule=False):
        if isinstance(action, np.ndarray):
            noise = self.current_scale * np.random.randn(*action.shape)
            if clip is not None:
                noise = np.clip(noise, -clip, clip)
            noisy_action = np.clip(
                action + noise,
                self.action_space.low + self.eps,
                self.action_space.high - self.eps,
            )
        elif isinstance(action, torch.Tensor):
            noise = self.current_scale * torch.randn(*action.shape).to(action.device)
            if clip is not None:
                noise = noise.clamp(-clip, clip)
            noisy_action = action + noise
            noisy_action_clamped = (noisy_action).clamp(
                self.act_low_torch + self.eps, self.act_high_torch - self.eps
            )
            # gradient preservation trick from drqv2???
            noisy_action = (
                noisy_action - noisy_action.detach() + noisy_action_clamped.detach()
            )
        else:
            raise ValueError(f"Unrecognized action array type: {type(action)}")
        if update_schedule:
            self.current_scale = max(
                self.current_scale - self._scale_slope, self.final_scale
            )
        return noisy_action


class EpsilonGreedyExplorationNoise:
    def __init__(
        self, action_space, eps_start=1.0, eps_final=1e-5, steps_annealed=1000
    ):
        assert eps_start >= eps_final
        self.action_space = action_space
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.steps_annealed = steps_annealed
        self.current_scale = eps_start
        self._eps_slope = (eps_start - eps_final) / steps_annealed

    def sample(self, action, clip=None, update_schedule=False):
        if random.random() < self.current_scale:
            rand_action = np.random.randint(
                0, self.action_space.n, size=action.shape, dtype=action.dtype
            )
            action = rand_action
        if update_schedule:
            self.current_scale = max(
                self.current_scale - self._eps_slope,
                self.eps_final,
            )
        return action


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        try:
            param = p.grad.data
        except AttributeError:
            continue
        else:
            param_norm = param.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def warmup_buffer(
    buffer, env, warmup_steps : int, max_episode_steps : int, n_step : int, gamma :float , num_envs : int = 1
):
    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    steps_this_ep = 0
    exp_deque = deque([], maxlen=n_step)
    for step_num in range(warmup_steps):
        if done:
            state = env.reset()
            steps_this_ep = 0
            done = False
            exp_deque.clear()
        rand_action = env.action_space.sample()
        if not isinstance(rand_action, np.ndarray):
            rand_action = np.array(rand_action)
            if len(rand_action.shape) == 0:
                rand_action = np.expand_dims(rand_action, 0)
        if num_envs > 1:
            rand_action = np.array([rand_action for _ in range(num_envs)])
        next_state, reward, done, info = env.step(rand_action)
        exp_deque.append((state, rand_action, reward, next_state, done))
        if len(exp_deque) == exp_deque.maxlen:
            # enough transitions to compute n-step returns
            s, a, r, s1, d = exp_deque.popleft()
            for i, trans in enumerate(exp_deque):
                *_, r_i, s1, d = trans
                r += (gamma ** (i + 1)) * r_i
            # buffer gets n-step transition
            traj_over = d.any() if num_envs > 1 else d
            buffer.push(
                s, a, r, s1, d, terminate_traj=traj_over or step_num >= warmup_steps - 1
            )
        if num_envs > 1:
            done = done.any()
        state = next_state
        steps_this_ep += 1
        if steps_this_ep >= max_episode_steps:
            done = True


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def _move_dict_to_device(dict_):
    return {key: val.to(device) for key, val in dict_.items()}


def sample_move_and_augment(buffer, batch_size, augmenter, aug_mix, per=True):
    assert len(buffer) >= batch_size
    if per:
        batch, imp_weights, priority_idxs = buffer.sample(batch_size)
    else:
        batch, priority_idxs = buffer.sample_uniform(batch_size)
        imp_weights = torch.ones(1).float()
    imp_weights = imp_weights.to(device)

    # "original observation", "action", "original observation t+1"
    oo, a, r, oo1, d = batch
    # move to the appropriate device (probably a gpu)
    oo = _move_dict_to_device(oo)
    a = a.to(device)
    r = r.to(device)
    oo1 = _move_dict_to_device(oo1)
    d = d.to(device)

    # now that data is on gpu, cast to float
    oo = {key: val.float() for key, val in oo.items()}
    a = a.float()
    r = r.float()
    oo1 = {key: val.float() for key, val in oo1.items()}
    d = d.float()

    # augment the observation tensors and mix them into the batch
    aug_mix_idx = int(batch_size * aug_mix)
    ao, ao1 = augmenter(oo, oo1)
    o = {x: y.clone() for x, y in oo.items()}
    o1 = {x: y.clone() for x, y in oo1.items()}
    for lbl in o.keys():
        o[lbl][:aug_mix_idx] = ao[lbl][:aug_mix_idx]
        o1[lbl][:aug_mix_idx] = ao1[lbl][:aug_mix_idx]

    return {
        "primary_batch": (o, a, r, o1, d),
        "augmented_obs": (ao, ao1),
        "original_obs": (oo, oo1),
        "priority_idxs": priority_idxs,
        "imp_weights": imp_weights,
    }


def compute_filter_stats(
    buffer,
    agent,
    augmenter,
    batch_size,
):
    replay_dict = sample_move_and_augment(
        buffer=buffer,
        batch_size=batch_size,
        augmenter=augmenter,
        aug_mix=0.0,
        per=False,
    )
    o, a, *_ = replay_dict["primary_batch"]
    with torch.no_grad():
        # use a random member of the ensemble to compute binary advantage filter stats
        adv = agent.adv_estimator(
            o, a, ensemble_idx=random.choice(range(agent.ensemble_size))
        )
        exp_filter = (adv >= 0.0).float()
    pct_accepted = (exp_filter.sum().float() / exp_filter.shape[0]) * 100.0
    return pct_accepted.item()


def filtered_bc_loss(
    logs,
    replay_dict,
    agent,
    ensemble_idx,
    filter_=True,
    discrete=False,
    need_critic_grad=True,
):
    o, a, *_ = replay_dict["primary_batch"]
    if filter_:
        with torch.no_grad():
            adv = agent.adv_estimator(o, a, ensemble_idx=ensemble_idx)
            # binary filter
            mask = (adv >= 0.0).float()
            adv_weights = mask
    with contextlib.nullcontext() if need_critic_grad else torch.no_grad():
        s_rep = agent.encoder(o)
    dist = agent.actors[ensemble_idx](s_rep)
    if discrete:
        logp_a = dist.log_prob(a.squeeze(1)).unsqueeze(1)
    else:
        logp_a = dist.log_prob(a).sum(-1, keepdim=True)
    if filter_:
        logs[f"losses/adv_weights_mean"] = adv_weights.mean().item()
        logp_a *= adv_weights
    loss = -(logp_a).mean()
    logs[f"losses/filterd_bc_loss_{ensemble_idx}"] = loss.item()
    return loss


def action_invariance_constraint(logs, replay_dict, agent, ensemble_idx, a=None):
    oo, _ = replay_dict["original_obs"]
    ao, _ = replay_dict["augmented_obs"]
    actor = agent.actors[ensemble_idx]
    with torch.no_grad():
        os_rep = agent.encoder(oo)
        o_dist = actor(os_rep)
        if a is None:
            a = o_dist.sample()
        o_logp_a = o_dist.log_prob(a).sum(-1, keepdim=True)
    as_rep = agent.encoder(ao)
    a_dist = actor(as_rep)
    a_logp_a = a_dist.log_prob(a).sum(-1, keepdim=True)
    return F.mse_loss(o_logp_a, a_logp_a)


def adjust_priorities(logs, replay_dict, agent, buffer):
    o, a, *_ = replay_dict["primary_batch"]
    priority_idxs = replay_dict["priority_idxs"]
    with torch.no_grad():
        random_ensemble_member = random.choice(range(agent.ensemble_size))
        adv = agent.adv_estimator(o, a, random_ensemble_member)
    new_priorities = (F.relu(adv) + 1e-4).cpu().detach().squeeze(1).numpy()
    buffer.update_priorities(priority_idxs, new_priorities)


def compute_td_targets(
    logs,
    replay_dict,
    agent,
    target_agent,
    ensemble_idx,
    ensemble_n,
    log_alphas,
    pop,
    gamma,
    random_process,
    noise_clip,
    discrete=False,
):
    o, a, r, o1, d = replay_dict["primary_batch"]

    actor = agent.actors[ensemble_idx]
    target_critic = target_agent.critics[ensemble_idx]
    popart = agent.popart[ensemble_idx]
    log_alpha = log_alphas[ensemble_idx]

    with torch.no_grad():
        s1_rep = target_agent.encoder(o1)
        a_dist_s1 = actor(s1_rep)
        if discrete:
            s1_q_pred = target_critic(s1_rep, subset=ensemble_n)
            probs = a_dist_s1.probs
            log_probs = torch.log_softmax(a_dist_s1.logits, dim=1)
            entropy_bonus = log_alpha.exp() * log_probs
            val_s1 = (probs * (s1_q_pred - entropy_bonus)).sum(1, keepdim=True)
            a_s1 = probs
        else:
            a_s1 = a_dist_s1.sample()
            if random_process is not None:
                a_s1 = random_process.sample(
                    a_s1, clip=noise_clip, update_schedule=False
                )
                entropy_bonus = torch.Tensor([0.0]).to(a_s1.device)
            else:
                logp_a1 = a_dist_s1.log_prob(a_s1).sum(-1, keepdim=True)
                entropy_bonus = log_alpha.exp() * logp_a1
            s1_q_pred = target_critic(s1_rep, a_s1, subset=ensemble_n)
            val_s1 = s1_q_pred - (entropy_bonus)
        if popart and pop:
            # denormalize target
            val_s1 = popart(val_s1, normalized=False)
        td_target = r + gamma * (1.0 - d) * val_s1

        if popart:
            # update popart stats
            popart.update_stats(td_target)
            # normalize TD target
            td_target = popart.normalize_values(td_target)
    logs[f"td_targets/mean_td_target_{ensemble_idx}"] = td_target.mean().item()
    logs[f"td_targets/std_td_target_{ensemble_idx}"] = td_target.std().item()
    logs[f"td_targets/entropy_bonus_{ensemble_idx}"] = entropy_bonus.mean().item()
    return td_target, (s1_rep, a_s1)


def compute_backup_weights(
    logs,
    replay_dict,
    agent,
    target_agent,
    weight_type,
    weight_temp,
    batch_size,
    discrete=False,
):
    if weight_type is None or weight_temp is None or agent.ensemble_size == 1:
        return 1.0

    o, a, _, o1, _ = replay_dict["primary_batch"]
    with torch.no_grad():
        if weight_type == "sunrise":
            s_rep = target_agent.encoder(o)
            if discrete:
                q_std = torch.stack(
                    [q(s_rep).gather(1, a.long()) for q in target_agent.critics], dim=0
                ).std(0)
            else:
                q_std = torch.stack(
                    [q(s_rep, a) for q in target_agent.critics], dim=0
                ).std(0)
            weights = torch.sigmoid(-q_std * weight_temp) + 0.5
        elif weight_type == "softmax":
            s1_rep = target_agent.encoder(o1)
            q1s = []
            for actor, critic in agent.ensemble:
                a1 = actor(s1_rep).sample()
                if discrete:
                    q1s.append(critic(s1_rep).gather(1, a1.unsqueeze(1).long()))
                else:
                    q1s.append(critic(s1_rep, a1))
            q_std = torch.stack(q1s, dim=0).std(0)
            weights = batch_size * F.softmax(-q_std * weight_temp, dim=0)
    logs["bellman_weights/mean"] = weights.mean().item()
    logs["bellman_weights/max"] = weights.max().item()
    logs["bellman_weights/min"] = weights.min().item()
    logs["bellman_weights/std"] = weights.std().item()
    return weights


def encoder_invariance_constraint(logs, replay_dict, agent):
    oo, _ = replay_dict["original_obs"]
    ao, _ = replay_dict["augmented_obs"]
    with torch.no_grad():
        os_rep = agent.encoder(oo)
    as_rep = agent.encoder(ao)
    loss = torch.norm(as_rep - os_rep)
    logs["encoder_constraint_loss"] = loss.mean().item()
    return loss

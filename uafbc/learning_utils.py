import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import device, replay


###########
## UTILS ##
###########


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


def warmup_buffer(buffer, env, warmup_steps, max_episode_steps):
    # use warmp up steps to add random transitions to the buffer
    state = env.reset()
    done = False
    steps_this_ep = 0
    for _ in range(warmup_steps):
        if done:
            state = env.reset()
            steps_this_ep = 0
            done = False
        rand_action = env.action_space.sample()
        if not isinstance(rand_action, np.ndarray):
            rand_action = np.array(float(rand_action))
            if len(rand_action.shape) == 0:
                rand_action = np.expand_dims(rand_action, 0)
        next_state, reward, done, info = env.step(rand_action)
        buffer.push(state, rand_action, reward, next_state, done)
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
    batch_size,
):
    batch, _ = buffer.sample_uniform(batch_size)
    o, a, *_ = batch
    o = _move_dict_to_device(o)
    a = a.to(device)
    with torch.no_grad():
        adv = agent.adv_estimator(o, a)
        exp_filter = (adv >= 0.0).float()
    pct_accepted = (exp_filter.sum().float() / exp_filter.shape[0]) * 100.0
    return pct_accepted.item()


def filtered_bc_loss(logs, replay_dict, agent, filter_=True, discrete=False):
    o, *_ = replay_dict["augmented_obs"]
    _, a, *_ = replay_dict["primary_batch"]
    if filter_:
        with torch.no_grad():
            adv = agent.adv_estimator(o, a)
            # binary filter
            mask = (adv >= 0.0).float()
            adv_weights = mask
    s_rep = agent.encoder(o)
    dist = agent.actor(s_rep)
    if discrete:
        logp_a = dist.log_prob(a.squeeze(1)).unsqueeze(1)
    else:
        logp_a = dist.log_prob(a).sum(-1, keepdim=True)
    if filter_:
        logp_a *= adv_weights
    loss = -(logp_a.clamp(-100.0, 100.0)).mean()
    logs["filterd_bc_loss"] = loss.item()
    return loss


def action_invariance_constraint(logs, replay_dict, agent, a=None):
    oo, _ = replay_dict["original_obs"]
    ao, _ = replay_dict["augmented_obs"]
    with torch.no_grad():
        os_rep = agent.encoder(oo)
        o_dist = agent.actor(os_rep)
        if a is None:
            a = o_dist.sample()
        o_logp_a = o_dist.log_prob(a).sum(-1, keepdim=True)
    as_rep = agent.encoder(ao)
    a_dist = agent.actor(as_rep)
    a_logp_a = a_dist.log_prob(a).sum(-1, keepdim=True)
    return F.mse_loss(o_logp_a, a_logp_a)


def adjust_priorities(logs, replay_dict, agent, buffer):
    o, a, *_ = replay_dict["primary_batch"]
    priority_idxs = replay_dict["priority_idxs"]
    with torch.no_grad():
        adv = agent.adv_estimator(o, a)
    new_priorities = (F.relu(adv) + 1e-5).cpu().detach().squeeze(1).numpy()
    buffer.update_priorities(priority_idxs, new_priorities)


def compute_td_targets(
    logs,
    replay_dict,
    agent,
    target_agent,
    ensemble_n,
    log_alpha,
    pop,
    gamma,
    discrete=False,
):
    _, a, r, _, d = replay_dict["primary_batch"]
    o, o1 = replay_dict["augmented_obs"]
    with torch.no_grad():
        s1_rep = target_agent.encoder(o1)
        a_dist_s1 = agent.actor(s1_rep)
        # REDQ
        ensemble = random.sample(target_agent.critics, ensemble_n)
        if discrete:
            ensemble_preds = torch.stack(
                [critic(s1_rep) for critic in ensemble],
                dim=0,
            )
            s1_q_pred = ensemble_preds.min(0).values
            val_s1 = (a_dist_s1.probs * s1_q_pred).sum(1, keepdim=True) - (
                log_alpha.exp() * a_dist_s1.entropy()
            ).unsqueeze(1)
        else:
            a_s1 = a_dist_s1.sample()
            logp_a1 = a_dist_s1.log_prob(a_s1).sum(-1, keepdim=True)
            ensemble_preds = torch.stack(
                [critic(s1_rep, a_s1) for critic in ensemble], dim=0
            )
            val_s1 = ensemble_preds.min(0).values - (log_alpha.exp() * logp_a1)
        if agent.popart and pop:
            # denormalize target
            val_s1 = agent.popart(val_s1, normalized=False)
        td_target = r + gamma * (1.0 - d) * val_s1

        if agent.popart:
            # update popart stats
            agent.popart.update_stats(td_target)
            # normalize TD target
            td_target = agent.popart.normalize_values(td_target)
    logs["td_target"] = td_target.mean().item()
    return td_target


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
    if weight_type is None or weight_temp is None:
        return 1.0

    _, a, *_ = replay_dict["primary_batch"]
    o, o1 = replay_dict["augmented_obs"]
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
            a1 = agent.actor(s1_rep).sample()
            if discrete:
                q_std = torch.stack(
                    [
                        q(s1_rep).gather(1, a1.unsqueeze(1).long())
                        for q in target_agent.critics
                    ],
                    dim=0,
                ).std(0)
            else:
                q_std = torch.stack(
                    [q(s1_rep, a1) for q in target_agent.critics], dim=0
                ).std(0)
            weights = batch_size * F.softmax(-q_std * weight_temp, dim=0)
    logs["bellman_weight"] = weights.mean().item()
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

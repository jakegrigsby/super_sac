import argparse
import copy
import math
import os
from itertools import chain
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import device, replay

from . import learning_utils as lu


def critic_update(
    buffer,
    agent,
    target_agent,
    critic_optimizer,
    encoder_optimizer,
    log_alpha,
    batch_size,
    gamma,
    critic_clip,
    encoder_clip,
    target_critic_ensemble_n,
    weighted_bellman_temp,
    weight_type,
    pop,
    augmenter,
    encoder_lambda,
    aug_mix=0.75,
    discrete=False,
    per=False,
    update_priorities=False,
):
    logs = {}

    replay_dict = lu.sample_move_and_augment(
        buffer,
        batch_size,
        augmenter,
        aug_mix,
        per=per,
    )

    td_target = lu.compute_td_targets(
        logs=logs,
        replay_dict=replay_dict,
        agent=agent,
        target_agent=target_agent,
        ensemble_n=target_critic_ensemble_n,
        log_alpha=log_alpha,
        pop=pop,
        gamma=gamma,
        discrete=discrete,
    )

    backup_weights = lu.compute_backup_weights(
        logs=logs,
        replay_dict=replay_dict,
        agent=agent,
        target_agent=target_agent,
        weight_type=weight_type,
        weight_temp=weighted_bellman_temp,
        batch_size=batch_size,
        discrete=discrete,
    )

    o, a, *_ = replay_dict["primary_batch"]
    critic_loss = 0.0
    s_rep = agent.encoder(o)
    for i, critic in enumerate(agent.critics):
        if discrete:
            q_pred = critic(s_rep).gather(1, a.long())
        else:
            q_pred = critic(s_rep, a)
        if agent.popart and pop:
            q_pred = agent.popart(q_pred)
        td_error = td_target - q_pred
        critic_loss += td_error ** 2
    critic_loss *= (
        0.5 * backup_weights * replay_dict["imp_weights"] * (1.0 / len(agent.critics))
    )
    critic_loss = critic_loss.mean()

    if encoder_lambda:
        critic_loss += encoder_lambda * lu.encoder_invariance_constraint(
            logs, replay_dict, agent
        )

    critic_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(critic.parameters() for critic in agent.critics)),
            critic_clip,
        )
    if encoder_clip:
        torch.nn.utils.clip_grad_norm_(agent.encoder.parameters(), encoder_clip)
    critic_optimizer.step()
    encoder_optimizer.step()

    logs["random_critic_grad_norm"] = lu.get_grad_norm(random.choice(agent.critics))
    logs["encoder_grad_norm"] = lu.get_grad_norm(agent.encoder)
    logs["critic_loss"] = critic_loss
    if update_priorities:
        lu.adjust_priorities(logs, replay_dict, agent, buffer)
    return logs


def offline_actor_update(
    buffer,
    agent,
    optimizer,
    batch_size,
    clip,
    augmenter,
    actor_lambda,
    aug_mix,
    per=True,
    discrete=False,
    filter_=True,
):
    logs = {}
    replay_dict = lu.sample_move_and_augment(
        buffer=buffer,
        batch_size=batch_size,
        augmenter=augmenter,
        aug_mix=aug_mix,
        per=per,
    )

    # build loss function
    loss = lu.filtered_bc_loss(
        logs, replay_dict, agent, filter_=filter_, discrete=discrete
    )
    if actor_lambda:
        loss += actor_lambda * lu.action_invariance_constraint(logs, replay_dict, agent)

    # take gradient step
    optimizer.zero_grad()
    loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), clip)
    optimizer.step()

    if per:
        lu.adjust_priorities(logs, replay_dict, agent, buffer)
    logs["offline_actor_loss"] = loss.item()
    logs["actor_offline_grad_norm"] = lu.get_grad_norm(agent.actor)
    return logs


def alpha_update(
    buffer,
    agent,
    optimizer,
    batch_size,
    log_alpha,
    augmenter,
    aug_mix,
    target_entropy,
    discrete,
):
    logs = {}
    replay_dict = lu.sample_move_and_augment(
        buffer=buffer,
        batch_size=batch_size,
        augmenter=augmenter,
        per=False,
        aug_mix=aug_mix,
    )
    o, *_ = replay_dict["primary_batch"]
    with torch.no_grad():
        a_dist = agent.actor(agent.encoder(o))
    if discrete:
        alpha_loss = (
            -(
                a_dist.probs
                * (
                    -log_alpha.exp()
                    * (torch.log_softmax(a_dist.logits, dim=1) + target_entropy)
                )
            )
            .sum(1)
            .mean()
        )

    else:
        logp_a = (
            a_dist.log_prob(a_dist.sample()).sum(-1, keepdim=True).clamp(-100.0, 100.0)
        )
        alpha_loss = (-log_alpha.exp() * (logp_a + target_entropy)).mean()
    optimizer.zero_grad()
    alpha_loss.backward()
    optimizer.step()
    logs["alpha_loss"] = alpha_loss.item()
    logs["alpha"] = log_alpha.exp().item()
    return logs


def online_actor_update(
    buffer,
    agent,
    optimizer,
    log_alpha,
    batch_size,
    clip,
    augmenter,
    aug_mix,
    per=False,
    discrete=False,
    use_baseline=False,
):
    logs = {}
    replay_dict = lu.sample_move_and_augment(
        buffer, batch_size, augmenter, aug_mix, per=per
    )

    o, *_ = replay_dict["primary_batch"]
    with torch.no_grad():
        s_rep = agent.encoder(o)
    a_dist = agent.actor(s_rep)
    if discrete:
        vals = torch.stack([q(s_rep) for q in agent.critics], dim=0).min(0).values
        probs = a_dist.probs
        log_probs = torch.log_softmax(a_dist.logits, dim=1)
        if use_baseline:
            val_baseline = (probs * vals).sum(1, keepdim=True)
            # vals = A(s, a) = Q(s, a) - V(s)
            vals -= val_baseline
        actor_loss = (probs * (log_alpha.exp() * log_probs - vals)).sum(1).mean()
    else:
        a = a_dist.rsample()
        if not use_baseline:
            vals = (
                torch.stack([q(s_rep, a) for q in agent.critics], dim=0).min(0).values
            )
        else:
            vals = agent.adv_estimator(o, a)
        entropy_bonus = log_alpha.exp() * a_dist.log_prob(a).sum(
            -1, keepdim=True
        ).clamp(-1000.0, 1000.0)
        actor_loss = -(vals - entropy_bonus).mean()

    optimizer.zero_grad()
    actor_loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), clip)
    optimizer.step()

    logs["actor_online_loss"] = actor_loss.item()
    logs["actor_online_grad_norm"] = lu.get_grad_norm(agent.actor)
    return logs

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
):

    agent.train()

    logs = {}

    replay_dict = lu.sample_move_and_augment(
        buffer, batch_size, augmenter, aug_mix, per=per
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
    critic_loss *= 0.5 * backup_weights * replay_dict["imp_weights"]
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
            chain(*(critic.parameters() for critic in agent.critics)), critic_clip,
        )
    if encoder_clip:
        torch.nn.utils.clip_grad_norm_(agent.encoder.parameters(), encoder_clip)
    critic_optimizer.step()
    encoder_optimizer.step()

    logs["random_critic_grad_norm"] = lu.get_grad_norm(random.choice(agent.critics))
    logs["encoder_grad_norm"] = lu.get_grad_norm(agent.encoder)
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
    agent.train()
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

    lu.adjust_priorities(logs, replay_dict, agent, buffer)
    logs["actor_loss"] = loss.item()
    logs["actor_offline_grad_norm"] = lu.get_grad_norm(agent.actor)
    return logs


def alpha_update(
        buffer,
        agent,
        optimizer,
        batch_size,
        log_alpha,
        aug_mix,
        target_entropy,
    ):
    logs = {}
    replay_dict = lu.sample_move_and_augment(buffer=buffer, batch_size=batch_size, per=False, aug_mix=aug_mix)
    o, *_ = replay_dict["primary_batch"]
    with torch.no_grad():
        a_dist = agent.actor(o)
    breakpoint()
    alpha_loss = -log_alpha.exp() * (a_dist.entropy() + target_entropy).mean()
    optimizer.zero_grad()
    alpha_loss.backward()
    optimizer.step()
    logs["alpha_loss"] = alpha_loss.item()
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
        vals = torch.stack([q(s_rep) for q in agent.critics], dim=0).min(0).values - (log_alpha.exp() * a_dist.entropy())
        actor_loss = -(a_dist.probs * vals).sum(1, keepdim=True).mean()
    else:
        a = a_dist.rsample()
        vals = torch.stack([q(s_rep, a) for q in agent.critics], dim=0).min(0).values - (log_alpha.exp() * a_dist.entropy())
        actor_loss = -(vals).mean()

    optimizer.zero_grad()
    actor_loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), clip)
    optimizer.step()

    logs["actor_online_loss"] = actor_loss.item()
    logs["actor_online_grad_norm"] = lu.get_grad_norm(agent.actor)
    lu.adjust_priorities(logs, replay_dict, agent, buffer)
    return logs

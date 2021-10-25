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
    log_alphas,
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
    random_process,
    aug_mix=0.75,
    discrete=False,
    per=False,
    update_priorities=False,
):
    logs = {}

    critic_loss = 0.0
    replay_dicts = []
    for i in range(agent.ensemble_size):
        replay_dict = lu.sample_move_and_augment(
            buffer=buffer,
            batch_size=batch_size,
            augmenter=augmenter,
            aug_mix=aug_mix,
            per=per,
        )

        td_target = lu.compute_td_targets(
            logs=logs,
            replay_dict=replay_dict,
            agent=agent,
            target_agent=target_agent,
            log_alphas=log_alphas,
            ensemble_idx=i,
            ensemble_n=target_critic_ensemble_n,
            pop=pop,
            gamma=gamma,
            random_process=random_process,
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
        s_rep = agent.encoder(o)
        if discrete:
            q_preds = agent.critics[i](s_rep, subset=None, return_min=False)
        else:
            q_preds = agent.critics[i](s_rep, a, subset=None, return_min=False)
        for q_pred in q_preds:
            if discrete:
                q_pred = q_pred.gather(1, a.long())
            if agent.popart[i] and pop:
                q_pred = agent.popart[i](q_pred)
            td_error = td_target - q_pred
            critic_loss += (
                backup_weights * replay_dict["imp_weights"] * (td_error ** 2)
            ) / (0.5 * agent.num_critics)

        if update_priorities:
            lu.adjust_priorities(logs, replay_dict, agent, buffer)
        replay_dicts.append(replay_dict)

    critic_loss = (critic_loss).mean() / (agent.ensemble_size)

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

    logs["losses/last_member_critic_td_error"] = td_error.mean().item()
    logs["losses/critic_overall_loss"] = critic_loss
    logs["gradients/critic_random_grad"] = lu.get_grad_norm(
        random.choice(agent.critics)
    )
    logs["gradients/encoder_criticloss_grad_norm"] = lu.get_grad_norm(agent.encoder)

    return logs, replay_dicts


def offline_actor_update(
    buffer,
    agent,
    actor_optimizer,
    encoder_optimizer,
    batch_size,
    actor_clip,
    update_encoder,
    encoder_clip,
    augmenter,
    actor_lambda,
    aug_mix,
    premade_replay_dicts=None,
    per=True,
    discrete=False,
    filter_=True,
):
    logs = {}

    # build loss function
    loss = 0.0
    for ensemble_idx in range(agent.ensemble_size):
        if premade_replay_dicts is not None:
            replay_dict = premade_replay_dicts[ensemble_idx]
        else:
            replay_dict = lu.sample_move_and_augment(
                buffer=buffer,
                batch_size=batch_size,
                augmenter=augmenter,
                aug_mix=aug_mix,
                per=per,
            )
        loss += lu.filtered_bc_loss(
            logs=logs,
            replay_dict=replay_dict,
            agent=agent,
            ensemble_idx=ensemble_idx,
            filter_=filter_,
            discrete=discrete,
            need_critic_grad=update_encoder,
        )
        if actor_lambda:
            loss += actor_lambda * lu.action_invariance_constraint(
                logs=logs,
                replay_dict=replay_dict,
                agent=agent,
                ensemble_idx=ensemble_idx,
            )

        if per:
            lu.adjust_priorities(logs, replay_dict, agent, buffer)

    loss /= agent.ensemble_size

    actor_optimizer.zero_grad()
    encoder_optimizer.zero_grad()

    loss.backward()

    if actor_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(actor.parameters() for actor in agent.actors)), actor_clip
        )
    if encoder_clip:
        torch.nn.utils.clip_grad_norm_(agent.encoder.parameters(), encoder_clip)

    actor_optimizer.step()
    if update_encoder:
        encoder_optimizer.step()
    logs["losses/filtered_bc_overall_loss"] = loss.item()
    logs["gradients/actor_offline_grad_norm"] = lu.get_grad_norm(
        random.choice(agent.actors)
    )
    logs["gradients/encoder_offline_actorloss_grad_norm"] = lu.get_grad_norm(
        agent.encoder
    )
    return logs


def alpha_update(
    buffer,
    agent,
    optimizers,
    batch_size,
    log_alphas,
    augmenter,
    aug_mix,
    target_entropy,
    premade_replay_dicts,
    discrete,
):
    logs = {}
    for i in range(agent.ensemble_size):
        if premade_replay_dicts is not None:
            replay_dict = premade_replay_dicts[i]
        else:
            replay_dict = lu.sample_move_and_augment(
                buffer=buffer,
                batch_size=batch_size,
                augmenter=augmenter,
                per=False,
                aug_mix=aug_mix,
            )
        o, *_ = replay_dict["primary_batch"]

        with torch.no_grad():
            s_rep = agent.encoder(o)
            a_dist = agent.actors[i](s_rep)

        if discrete:
            logp_a = (a_dist.probs * torch.log_softmax(a_dist.logits, dim=1)).sum(-1)
        else:
            logp_a = a_dist.log_prob(a_dist.sample()).sum(-1, keepdim=True)
        alpha_loss = -(log_alphas[i].exp() * (logp_a + target_entropy).detach()).mean()
        optimizers[i].zero_grad()
        alpha_loss.backward()
        optimizers[i].step()
        logs[f"losses/alpha_loss_{i}"] = alpha_loss.item()
        logs[f"alphas/alpha_{i}"] = log_alphas[i].exp().item()
    return logs


def online_actor_update(
    buffer,
    agent,
    pop,
    actor_optimizer,
    log_alphas,
    batch_size,
    clip,
    random_process,
    augmenter,
    aug_mix,
    premade_replay_dicts=None,
    per=False,
    discrete=False,
    use_baseline=False,
):
    logs = {}

    actor_loss = 0.0
    for i, ((actor, critic), popart, log_alpha) in enumerate(
        zip(agent.ensemble, agent.popart, log_alphas)
    ):
        if premade_replay_dicts is not None:
            replay_dict = premade_replay_dicts[i]
        else:
            replay_dict = lu.sample_move_and_augment(
                buffer=buffer,
                batch_size=batch_size,
                augmenter=augmenter,
                aug_mix=aug_mix,
                per=per,
            )
        o, *_ = replay_dict["primary_batch"]
        with torch.no_grad():
            # actor gradients aren't used to train the encoder (pixel SAC trick)
            s_rep = agent.encoder(o)
        a_dist = actor(s_rep)
        if discrete:
            probs = a_dist.probs
            log_probs = torch.log_softmax(a_dist.logits, dim=1)
            with torch.no_grad():
                vals = critic(s_rep)
                if popart and pop:
                    vals = popart(vals)
            vals = (probs * vals).sum(1, keepdim=True)
            entropy_bonus = log_alpha.exp() * (probs * log_probs).sum(1, keepdim=True)
        else:
            # sample an action
            a = a_dist.rsample()
            # compute max ent term, if applicable
            if random_process is not None:
                a = random_process.sample(a, update_schedule=False)
                entropy_bonus = 0.0
            else:
                entropy_bonus = log_alpha.exp() * a_dist.log_prob(a).sum(
                    -1, keepdim=True
                )
            # get critic values for this action
            if not use_baseline:
                vals = critic(s_rep, a)
                if popart and pop:
                    vals = popart(vals)
            else:
                vals = agent.adv_estimator(o, a, ensemble_idx=i)
        actor_loss += -(vals - entropy_bonus).mean()
    actor_loss /= len(agent.actors)

    actor_optimizer.zero_grad()
    actor_loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(actor.parameters() for actor in agent.actors)), clip
        )
    actor_optimizer.step()

    logs["losses/actor_online_overall_loss"] = actor_loss.item()
    logs[f"gradients/random_actor_online_grad"] = lu.get_grad_norm(
        random.choice(agent.actors)
    )
    return logs

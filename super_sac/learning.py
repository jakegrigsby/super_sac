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
    noise_clip,
    aug_mix=0.75,
    discrete=False,
    per=False,
    update_priorities=False,
    dr3_coeff=0.0,
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

        td_target, (s1, a1) = lu.compute_td_targets(
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
            noise_clip=noise_clip,
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
        s_a_features = agent.critics[i].features

        for q_pred in q_preds:
            if discrete:
                q_pred = q_pred.gather(-1, a.long())
            if agent.popart[i] and pop:
                q_pred = agent.popart[i](q_pred)
            td_error = td_target - q_pred
            critic_loss += (
                backup_weights * replay_dict["imp_weights"] * (td_error**2)
            ).mean()

        if dr3_coeff > 0:
            if discrete:
                agent.critics[i](s1)
            else:
                agent.critics[i](s1, a1)
            s1_a1_features = agent.critics[i].features
            feature_co_adaptation = (s_a_features * s1_a1_features).sum(-1).mean()
            logs[f"dr3_dotproduct_{i}"] = feature_co_adaptation.item()
            critic_loss += dr3_coeff * feature_co_adaptation

        replay_dicts.append(replay_dict)

    critic_loss /= agent.ensemble_size * agent.num_critics

    if encoder_lambda:
        critic_loss += encoder_lambda * lu.encoder_invariance_constraint(
            logs, replay_dict, agent
        )

    encoder_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if critic_clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(critic.parameters() for critic in agent.critics)),
            critic_clip,
        )
    if encoder_clip:
        torch.nn.utils.clip_grad_norm_(agent.encoder.parameters(), encoder_clip)
    encoder_optimizer.step()
    critic_optimizer.step()

    logs["losses/last_member_critic_td_error"] = td_error.mean().item()
    logs["losses/critic_overall_loss"] = critic_loss
    logs["gradients/critic_random_grad"] = lu.get_grad_norm(
        random.choice(agent.critics)
    )
    logs["gradients/encoder_criticloss_grad_norm"] = lu.get_grad_norm(agent.encoder)

    if update_priorities:
        lu.adjust_priorities(logs, replay_dict, agent, buffer)
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
    if per:
        lu.adjust_priorities(logs, replay_dict, agent, buffer)
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
            logp_a = (a_dist.probs * torch.log_softmax(a_dist.logits, dim=-1)).sum(-1)
        else:
            logp_a = a_dist.log_prob(a_dist.sample()).sum(-1, keepdim=True)
        alpha_loss = -(log_alphas[i] * (logp_a + target_entropy).detach()).mean()
        # alpha_loss = -(log_alphas[i].exp() * (logp_a + target_entropy).detach()).mean()
        optimizers[i].zero_grad()
        alpha_loss.backward()
        optimizers[i].step()
        logs[f"losses/alpha_loss_{i}"] = alpha_loss.item()
        logs[f"alphas/alpha_{i}"] = log_alphas[i].exp().item()
    return logs


def markov_state_abstraction_update(
    buffer,
    agent,
    optimizer,
    batch_size,
    augmenter,
    aug_mix,
    discrete,
    inverse_coeff,
    contrastive_coeff,
    smoothness_coeff,
    smoothness_max_dist,
    grad_clip,
):
    """
    Self-Supervised state abstraction loss function from
    "Learning Markov State Abstractions for Deep Reinforcement Learning"
    (https://arxiv.org/abs/2106.04379).
    """
    logs = {}
    replay_dict = lu.sample_move_and_augment(
        buffer=buffer,
        batch_size=batch_size,
        augmenter=augmenter,
        per=False,
        aug_mix=aug_mix,
    )
    o, a, _, o1, d = replay_dict["primary_batch"]
    s_rep = agent.encoder(o)
    s1_rep = agent.encoder(o1)

    a_dist = agent.inverse_model(s_rep, s1_rep)
    inverse_loss = -a_dist.log_prob(a.squeeze(-1)).mean()

    s1_rep_neg = s1_rep[torch.randperm(batch_size)]
    pos_labels = torch.ones(*s_rep.shape[:-1], 1)
    neg_labels = torch.zeros(*s_rep.shape[:-1], 1)
    labels = torch.cat((pos_labels, neg_labels), dim=0).to(device)

    s_candidates = torch.cat((s_rep, s_rep), dim=0)
    s1_candidates = torch.cat((s1_rep, s1_rep_neg), dim=0)
    real_trans_preds = agent.contrastive_model(s_candidates, s1_candidates)
    contrastive_loss = F.binary_cross_entropy(real_trans_preds, labels)

    dist = torch.norm(s1_rep - s_rep, dim=-1, p=2) / math.sqrt(s_rep.shape[-1])
    smoothness_loss = (F.relu(dist - smoothness_max_dist)).square().mean()

    markov_loss = (
        inverse_coeff * inverse_loss
        + contrastive_coeff * contrastive_loss
        + smoothness_coeff * smoothness_loss
    )

    optimizer.zero_grad()
    markov_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            chain(
                agent.encoder.parameters(),
                agent.inverse_model.parameters(),
                agent.contrastive_model.parameters(),
            ),
            grad_clip,
        )
    optimizer.step()

    logs["gradients/contrastive_model_grad_norm"] = lu.get_grad_norm(
        agent.contrastive_model
    )
    logs["gradients/inverse_model_grad_norm"] = lu.get_grad_norm(agent.inverse_model)
    logs["gradients/encoder_markovloss_grad_norm"] = lu.get_grad_norm(agent.encoder)
    logs["losses/markov_loss"] = markov_loss.item()
    logs["losses/inverse_model_loss"] = inverse_loss.item()
    logs["losses/contrastive_model_loss"] = contrastive_loss.item()
    logs["losses/smoothness_loss"] = smoothness_loss.item()
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
    noise_clip,
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
            log_probs = torch.log_softmax(a_dist.logits, dim=-1)
            with torch.no_grad():
                vals = critic(s_rep)
                if popart and pop:
                    vals = popart(vals)
            vals = (probs * vals).sum(-1, keepdim=True)
            entropy_bonus = log_alpha.exp() * (probs * log_probs).sum(-1, keepdim=True)
        else:
            a = a_dist.rsample()
            if random_process is not None:
                a = random_process.sample(a, clip=noise_clip, update_schedule=False)
                entropy_bonus = torch.Tensor([0.0]).to(a.device).detach()
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
        actor_loss += (vals - entropy_bonus).mean()
    actor_loss = -actor_loss / len(agent.actors)

    actor_optimizer.zero_grad()
    actor_loss.backward()
    if clip:
        torch.nn.utils.clip_grad_norm_(
            chain(*(actor.parameters() for actor in agent.actors)), clip
        )
    actor_optimizer.step()
    logs[f"gradients/random_actor_online_grad"] = lu.get_grad_norm(
        random.choice(agent.actors)
    )
    logs["losses/actor_pg_loss"] = actor_loss.item()
    return logs

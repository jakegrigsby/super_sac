import argparse
import copy
import math
import os
from itertools import chain
import random
import time

import numpy as np
import tensorboardX
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

from . import device, learning, evaluation, replay, augmentations
from . import learning_utils as lu


def uafbc(
    # required args
    agent,
    buffer,
    train_env,
    test_env,
    # compute kwargs
    num_steps_offline=1_000_000,
    num_steps_online=100_000,
    offline_actor_updates_per_step=1,
    online_actor_updates_per_step=1,
    critic_updates_per_step=1,
    target_critic_ensemble_n=2,
    batch_size=512,
    # optimization kwargs
    actor_lr=3e-4,
    critic_lr=3e-4,
    encoder_lr=1e-4,
    alpha_lr=1e-4,
    actor_clip=40.0,
    critic_clip=40.0,
    encoder_clip=40.0,
    actor_l2=0.0,
    critic_l2=0.0,
    encoder_l2=0.0,
    pop=True,
    # rl kwargs
    use_exploration_process=False,
    exploration_param_init=1.0,
    exploration_param_final=0.1,
    exploration_param_anneal=1_000_000,
    init_alpha=0.1,
    target_entropy_mul=1.0,
    gamma=0.99,
    mlp_tau=0.005,
    encoder_tau=0.01,
    target_delay=2,
    use_pg_update_online=True,
    use_bc_update_online=True,
    weighted_bellman_temp=20.0,
    bc_warmup_steps=0,
    random_warmup_steps=0,
    weight_type="softmax",
    # env and eval kwargs
    transitions_per_online_step=1,
    infinite_bootstrap=True,
    max_episode_steps=1000,
    eval_episodes=10,
    eval_interval=10_000,
    render=False,
    # data augmentation
    augmenter=None,
    encoder_lambda=0.0,
    actor_lambda=0.0,
    aug_mix=0.9,
    # logging, misc
    name="afbc_run",
    log_to_disk=True,
    log_interval=5000,
    save_to_disk=True,
    save_interval=5000,
    verbosity=0,
):

    if save_to_disk or log_to_disk:
        save_dir = make_process_dirs(name)
    if log_to_disk:
        writer = tensorboardX.SummaryWriter(save_dir)
        # writer.add_hparams(locals(), {})

    if augmenter is None:
        augmenter = augmentations.AugmentationSequence(
            [augmentations.IdentityAug(batch_size)]
        )
    qprint = lambda x: print(x) if verbosity else None
    qprint(" ----- AFBC -----")
    qprint(f"\tART: {agent.popart is not False}")
    qprint(f"\tPOP: {pop}")
    qprint(f"\tBellman Backup Weight type: {weight_type}")
    qprint(f"\tBC Warmup Steps: {bc_warmup_steps}")
    qprint(f"\tCritic Ensemble Size: {len(agent.critics)}")
    qprint(f"\tCritic Updates per Step: {critic_updates_per_step}")
    qprint(f"\tActor Online Updates per Step: {online_actor_updates_per_step}")
    qprint(f"\tActor Offline Updates per Step: {offline_actor_updates_per_step}")
    qprint(f"\tEncoder Lambda: {encoder_lambda}")
    qprint(f"\tActor Lambda: {actor_lambda}")
    qprint(f"\tDiscrete Actions: {agent.discrete}")
    qprint(f"\tUse PG Update Online: {use_pg_update_online}")
    qprint(f"\tUse BC Update Online: {use_bc_update_online}")
    qprint(f"\tUse Random Exploration Noise: {use_exploration_process}")
    qprint(f"\tInit Alpha: {init_alpha}, Alpha LR: {alpha_lr}")
    qprint(
        f"\tUsing Beta Dist: {not agent.discrete and agent.actor.dist_impl == 'beta'}"
    )
    qprint(" -----      -----")

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()

    def _get_actors(env):
        _env = env
        while hasattr(_env, "env"):
            if hasattr(_env, "_PARALLEL_ACTORS"):
                return _env._PARALLEL_ACTORS
            else:
                _env = _env.env
        return 1

    actors = _get_actors(train_env)
    qprint(f"Detected {actors} training actors.")
    assert (
        _get_actors(test_env) == 1
    ), "Evaluation Envs are not compatible with parallel sampling."

    # create target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    for target_critic, agent_critic in zip(target_agent.critics, agent.critics):
        lu.hard_update(target_critic, agent_critic)
    lu.hard_update(target_agent.encoder, agent.encoder)
    target_agent.train()

    critic_optimizer = torch.optim.Adam(
        chain(*(critic.parameters() for critic in agent.critics)),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    offline_actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    online_actor_optimizer = torch.optim.Adam(
        agent.actor.parameters(),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    encoder_optimizer = torch.optim.Adam(
        agent.encoder.parameters(),
        lr=encoder_lr,
        weight_decay=encoder_l2,
        betas=(0.9, 0.999),
    )

    # max entropy
    init_alpha = max(init_alpha, 1e-11)
    log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
    log_alpha.requires_grad = True
    log_alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))
    if agent.discrete:
        target_entropy = -math.log(1.0 / train_env.action_space.n) * 0.98
    else:
        target_entropy = -train_env.action_space.shape[0]
    target_entropy *= target_entropy_mul

    if use_exploration_process:
        if agent.discrete:
            random_process = lu.EpsilonGreedyExplorationNoise(
                action_space=train_env.action_space,
                eps_start=exploration_param_init,
                eps_final=exploration_param_final,
                steps_annealed=exploration_param_anneal,
            )
        else:
            random_process = lu.GaussianExplorationNoise(
                action_space=train_env.action_space,
                start_scale=exploration_param_init,
                final_scale=exploration_param_final,
                steps_annealed=exploration_param_anneal,
            )
    ###################
    ## TRAINING LOOP ##
    ###################
    total_steps = num_steps_offline + num_steps_online
    progress_bar = lambda x: tqdm.tqdm(range(x)) if verbosity else range(x)

    if random_warmup_steps:
        lu.warmup_buffer(
            buffer, train_env, random_warmup_steps, max_episode_steps, actors=actors
        )

    # behavioral cloning
    for step in progress_bar(bc_warmup_steps):
        bc_logs = learning.offline_actor_update(
            buffer=buffer,
            agent=agent,
            optimizer=offline_actor_optimizer,
            batch_size=batch_size,
            clip=actor_clip,
            augmenter=augmenter,
            actor_lambda=actor_lambda,
            aug_mix=aug_mix,
            per=False,
            discrete=agent.discrete,
            filter_=False,
        )

        if (step % log_interval == 0) and log_to_disk:
            for key, val in bc_logs.items():
                writer.add_scalar(key, val, step)

        if (
            (step % eval_interval == 0) or (step == total_steps - 1)
        ) and eval_interval > 0:
            mean_return = evaluation.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step)
        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    qprint("\tRegular Training Begins...")
    done = True
    for step in progress_bar(total_steps):
        if step > num_steps_offline:
            # collect experience
            for _ in range(transitions_per_online_step):
                if done:
                    state = train_env.reset()
                    steps_this_ep = 0
                    done = False
                action, act_dist = agent.sample_action(
                    state, from_cpu=True, actors=actors, return_dist=True
                )
                if agent.discrete and step % 111 == 0:
                    np.set_printoptions(precision=2, suppress=True)
                    print(act_dist.probs.mean(0).cpu().numpy())
                if use_exploration_process:
                    action = random_process.sample(action)
                next_state, reward, done, info = train_env.step(action)
                if infinite_bootstrap and steps_this_ep + 1 == max_episode_steps:
                    # allow infinite bootstrapping
                    done = (
                        np.expand_dims(np.array([False for _ in range(actors)]), 1)
                        if actors > 1
                        else False
                    )
                buffer.push(state, action, reward, next_state, done)
                if actors > 1:
                    done = done.any()
                state = next_state
                steps_this_ep += 1
                if steps_this_ep >= max_episode_steps:
                    done = True

        for critic_update in range(critic_updates_per_step):
            critic_logs = learning.critic_update(
                buffer=buffer,
                agent=agent,
                target_agent=target_agent,
                critic_optimizer=critic_optimizer,
                encoder_optimizer=encoder_optimizer,
                log_alpha=log_alpha,
                batch_size=batch_size,
                gamma=gamma,
                critic_clip=critic_clip,
                encoder_clip=encoder_clip,
                target_critic_ensemble_n=target_critic_ensemble_n,
                weighted_bellman_temp=weighted_bellman_temp,
                weight_type=weight_type,
                pop=pop,
                augmenter=augmenter,
                encoder_lambda=encoder_lambda,
                aug_mix=aug_mix,
                discrete=agent.discrete,
                per=False,
                update_priorities=step < num_steps_offline or use_bc_update_online,
            )

            # move target model towards training model
            if critic_update % target_delay == 0:
                for agent_critic, target_critic in zip(
                    agent.critics, target_agent.critics
                ):
                    lu.soft_update(target_critic, agent_critic, mlp_tau)
                lu.soft_update(target_agent.encoder, agent.encoder, encoder_tau)

        # actor update
        actor_logs = {}
        for actor_update in range(offline_actor_updates_per_step):
            if step < num_steps_offline or use_bc_update_online:
                actor_logs.update(
                    learning.offline_actor_update(
                        buffer=buffer,
                        agent=agent,
                        optimizer=offline_actor_optimizer,
                        batch_size=batch_size,
                        clip=actor_clip,
                        augmenter=augmenter,
                        actor_lambda=actor_lambda,
                        aug_mix=aug_mix,
                        per=True,
                        discrete=agent.discrete,
                        filter_=True,
                    )
                )

        for actor_update in range(online_actor_updates_per_step):
            if step >= num_steps_offline and use_pg_update_online:
                actor_logs.update(
                    learning.online_actor_update(
                        buffer=buffer,
                        agent=agent,
                        optimizer=online_actor_optimizer,
                        log_alpha=log_alpha,
                        batch_size=batch_size,
                        aug_mix=aug_mix,
                        clip=actor_clip,
                        augmenter=augmenter,
                        per=False,
                        discrete=agent.discrete,
                        # use_baseline=True,
                        use_baseline=False,
                    )
                )

        if init_alpha > 0 and alpha_lr > 0:
            actor_logs.update(
                learning.alpha_update(
                    buffer=buffer,
                    agent=agent,
                    optimizer=log_alpha_optimizer,
                    batch_size=batch_size,
                    log_alpha=log_alpha,
                    augmenter=augmenter,
                    aug_mix=aug_mix,
                    target_entropy=target_entropy,
                    discrete=agent.discrete,
                )
            )

        #############
        ## LOGGING ##
        #############
        if (step % log_interval == 0) and log_to_disk:
            for key, val in critic_logs.items():
                writer.add_scalar(key, val, step)
            for key, val in actor_logs.items():
                writer.add_scalar(key, val, step)

        if (
            (step % eval_interval == 0) or (step == total_steps - 1)
        ) and eval_interval > 0:
            mean_return = evaluation.evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render
            )
            if log_to_disk:
                writer.add_scalar("return", mean_return, step)
                accepted_exp_pct = lu.compute_filter_stats(buffer, agent, 1024)
                writer.add_scalar("Accepted Exp (pct)", accepted_exp_pct, step)
        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)

    return agent


def make_process_dirs(run_name, base_path="saves"):
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir

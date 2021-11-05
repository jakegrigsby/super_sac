import argparse
import copy
import math
import os
from itertools import chain
import random
import time
from collections import deque

import numpy as np
import tensorboardX
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import gin

from . import device, learning, evaluation, replay, augmentations
from . import learning_utils as lu


@gin.configurable
def super_sac(
    # required args
    agent,
    buffer,
    train_env,
    test_env,
    # compute kwargs
    bc_warmup_steps=0,
    num_steps_offline=1_000_000,
    num_steps_online=100_000,
    afbc_actor_updates_per_step=1,
    pg_actor_updates_per_step=1,
    critic_updates_per_step=1,
    target_critic_ensemble_n=2,
    batch_size=512,
    reuse_replay_dicts=True,
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
    exploration_update_clip=0.3,
    init_alpha=0.1,
    target_entropy_mul=1.0,
    gamma=0.99,
    mlp_tau=0.005,
    encoder_tau=0.01,
    target_delay=2,
    n_step=1,
    use_pg_update_online=True,
    use_afbc_update_online=True,
    weighted_bellman_temp=20.0,
    random_warmup_steps=0,
    weight_type="softmax",
    afbc_per=True,
    # env and eval kwargs
    transitions_per_online_step=1,
    infinite_bootstrap=True,
    ignore_all_dones=False,
    max_episode_steps=1_000_000,
    eval_episodes=10,
    eval_interval=10_000,
    render=False,
    # data augmentation
    augmenter=None,
    encoder_lambda=0.0,
    actor_lambda=0.0,
    aug_mix=0.9,
    # logging, misc
    logging_method="tensorboard",
    wandb_entity=os.getenv("SSAC_WANDB_ACCOUNT"),
    wandb_project=os.getenv("SSAC_WANDB_PROJECT"),
    base_save_path=os.getenv("SSAC_SAVE"),
    name="afbc_run",
    log_to_disk=True,
    log_interval=5000,
    save_to_disk=True,
    save_interval=5000,
    verbosity=0,
):
    def _get_parallel_envs(env):
        _env = env
        while hasattr(_env, "env"):
            if hasattr(_env, "_PARALLEL_ACTORS"):
                return _env._PARALLEL_ACTORS
            else:
                _env = _env.env
        return 1

    num_envs = _get_parallel_envs(train_env)
    num_eval_envs = _get_parallel_envs(test_env)

    if save_to_disk or log_to_disk:
        save_dir = make_process_dirs(name, base_save_path)
    if log_to_disk:
        if logging_method == "tensorboard":
            writer = tensorboardX.SummaryWriter(save_dir)
        elif logging_method == "wandb":
            assert (
                wandb_project is not None and wandb_entity is not None
            ), "Set `wandb_entity` and `wandb_project` kwargs. Note that you can also \n\
                set the os environment variables `SSAC_WANDB_ACCOUNT` and `SSAC_WANDB_PROJECT`, \n\
                respectively. Super SAC will default to those values."
            import wandb

            wandb.init(
                project=wandb_project, entity=wandb_entity, dir=save_dir, reinit=True
            )
            wandb.run.name = name

    if augmenter is None:
        augmenter = augmentations.AugmentationSequence(
            [augmentations.IdentityAug(batch_size)]
        )

    qprint = lambda x: print(x) if verbosity else None
    qprint(" ----- Super SAC -----")
    qprint(f"\tART: {agent.popart[0] is not False}")
    qprint(f"\tPOP: {pop}")
    qprint(
        f"\tBellman Backup Weight Type: {weight_type}; Temperature: {weighted_bellman_temp}"
    )
    qprint(f"\tBC Warmup Steps: {bc_warmup_steps}")
    qprint(f"\tEnsemble Size: {agent.ensemble_size}")
    qprint(f"\tCritic Ensemble Size: {agent.num_critics}")
    if target_critic_ensemble_n > agent.num_critics:
        target_critic_ensemble_n = agent.num_critics
        qprint("\t\tWarning: too many redq target agent critics. Overriding.")
    qprint(f"\tTD Target Critic Ensemble Size: {target_critic_ensemble_n}")
    qprint(f"\tCritic Updates per Step: {critic_updates_per_step}")
    qprint(f"\tDiscrete Actions: {agent.discrete}")
    qprint(f"\tActor Updates per Online Step: {pg_actor_updates_per_step}")
    qprint(f"\tActor Updates per Offline Step: {afbc_actor_updates_per_step}")
    qprint(f"\tQ-Value Uncertainty Exploration Bonus: {agent.ucb_bonus}")
    qprint(f"\tEncoder Lambda: {encoder_lambda}")
    qprint(f"\tActor Lambda: {actor_lambda}")
    qprint(f"\tUse PG Update Online: {use_pg_update_online}")
    qprint(f"\tUse BC Update Online: {use_afbc_update_online}")
    qprint(f"\tUse Random Exploration Noise: {use_exploration_process}")
    qprint(f"\tInit Alpha: {init_alpha}, Alpha LR: {alpha_lr}")
    qprint(f"\tAugmenter: {augmenter}")
    qprint(f"\tAug Mix: {aug_mix}")
    qprint(
        f"\tUsing Beta Dist: {not agent.discrete and agent.actors[0].dist_impl == 'beta'}"
    )
    qprint(f"\tParallel Training Envs: {num_envs}")
    qprint(f"\tParallel Eval Envs: {num_eval_envs}")
    qprint(" -----           -----")

    ###########
    ## SETUP ##
    ###########
    agent.to(device)
    agent.train()

    # optimizers
    critic_optimizer = torch.optim.Adam(
        chain(*(critic.parameters() for critic in agent.critics)),
        lr=critic_lr,
        weight_decay=critic_l2,
        betas=(0.9, 0.999),
    )
    offline_actor_optimizer = torch.optim.Adam(
        chain(*(actor.parameters() for actor in agent.actors)),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    encoder_actorloss_optimizer = torch.optim.Adam(
        agent.encoder.parameters(),
        lr=encoder_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    online_actor_optimizer = torch.optim.Adam(
        chain(*(actor.parameters() for actor in agent.actors)),
        lr=actor_lr,
        weight_decay=actor_l2,
        betas=(0.9, 0.999),
    )
    encoder_criticloss_optimizer = torch.optim.Adam(
        agent.encoder.parameters(),
        lr=encoder_lr,
        weight_decay=encoder_l2,
        betas=(0.9, 0.999),
    )

    # max entropy, disabled with init_alpha = 0, alpha_lr = 0
    init_alpha = max(init_alpha, 1e-15)
    log_alphas = []
    log_alpha_optimizers = []
    for _ in range(agent.ensemble_size):
        log_alpha = torch.Tensor([math.log(init_alpha)]).to(device)
        log_alpha.requires_grad = True
        log_alphas.append(log_alpha)
        log_alpha_optimizers.append(
            torch.optim.Adam([log_alpha], lr=alpha_lr, betas=(0.5, 0.999))
        )
    if agent.discrete:
        target_entropy = -math.log(1.0 / train_env.action_space.n) * 0.98
    else:
        target_entropy = -train_env.action_space.shape[0]
    target_entropy *= target_entropy_mul

    # manual exploration, if applicable
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
    else:
        random_process = None

    ###################
    ## TRAINING LOOP ##
    ###################
    total_steps = bc_warmup_steps + num_steps_offline + num_steps_online
    progress_bar = lambda *x: tqdm.tqdm(range(*x)) if verbosity else range(*x)

    # warmup empty replay buffer
    if random_warmup_steps:
        lu.warmup_buffer(
            buffer=buffer,
            env=train_env,
            warmup_steps=random_warmup_steps,
            max_episode_steps=max_episode_steps,
            n_step=n_step,
            gamma=gamma,
            num_envs=num_envs,
        )

    done = True  # reset the env on first step
    exp_deque = deque([], maxlen=n_step)  # holds n-step transitions
    for step in progress_bar(total_steps):
        bc_logs, actor_logs, critic_logs = {}, {}, {}

        ###############################
        ## Behavioral Cloning Update ##
        ###############################

        if step < bc_warmup_steps:
            if step == 0:
                qprint("[Behavioral Cloning]")
            bc_logs.update(
                learning.offline_actor_update(
                    buffer=buffer,
                    agent=agent,
                    actor_optimizer=offline_actor_optimizer,
                    actor_clip=actor_clip,
                    encoder_optimizer=encoder_actorloss_optimizer,
                    encoder_clip=encoder_clip,
                    update_encoder=True,
                    batch_size=batch_size,
                    augmenter=augmenter,
                    actor_lambda=actor_lambda,
                    aug_mix=aug_mix,
                    per=False,
                    discrete=agent.discrete,
                    filter_=False,
                )
            )
            bc_logs.update({"schedule/bc_actor_update": 1.0})
        else:
            bc_logs.update({"schedule/bc_actor_update": 0.0})

        # create target networks
        if step == bc_warmup_steps:
            qprint("[Creating Target Networks]")
            # create target networks (after behavioral cloning)
            target_agent = copy.deepcopy(agent)
            target_agent.to(device)
            for target_critic, agent_critic in zip(target_agent.critics, agent.critics):
                lu.hard_update(target_critic, agent_critic)
            lu.hard_update(target_agent.encoder, agent.encoder)
            target_agent.train()

        if step > bc_warmup_steps + num_steps_offline:
            #############################
            ## Environment Interaction ##
            #############################

            if step == bc_warmup_steps + num_steps_offline + 1:
                qprint("[Collecting Experience For the First Time]")
            for _ in range(transitions_per_online_step):
                if done:
                    state = train_env.reset()
                    steps_this_ep = 0
                    done = False
                    exp_deque.clear()
                action = agent.sample_action(state, from_cpu=True, num_envs=num_envs)
                if use_exploration_process:
                    actor_logs["exploration_noise_param"] = random_process.current_scale
                    action = random_process.sample(action, update_schedule=True)
                next_state, reward, done, info = train_env.step(action)
                if ignore_all_dones or (
                    infinite_bootstrap and steps_this_ep + 1 == max_episode_steps
                ):
                    # override the replay buffer version of done to False
                    buffer_done = (
                        np.expand_dims(np.array([False for _ in range(num_envs)]), 1)
                        if num_envs > 1
                        else False
                    )
                else:
                    buffer_done = done
                # put this transition in our n-step queue
                exp_deque.append((state, action, reward, next_state, buffer_done))
                if len(exp_deque) == exp_deque.maxlen:
                    # enough transitions to compute n-step returns
                    s, a, r, s1, d = exp_deque.popleft()
                    for i, trans in enumerate(exp_deque):
                        *_, r_i, s1, d = trans
                        r += (gamma ** (i + 1)) * r_i
                    # buffer gets n-step transition
                    buffer.push(s, a, r, s1, d)
                if num_envs > 1:
                    done = done.any()
                state = next_state
                steps_this_ep += 1
                if steps_this_ep >= max_episode_steps:
                    done = True

        if step > bc_warmup_steps:
            ###################
            ## Critic Update ##
            ###################

            if step == bc_warmup_steps + 1:
                qprint("[First Critic Update]")
            for critic_update in range(critic_updates_per_step):
                critic_logs, premade_replay_dicts = learning.critic_update(
                    buffer=buffer,
                    agent=agent,
                    target_agent=target_agent,
                    critic_optimizer=critic_optimizer,
                    encoder_optimizer=encoder_criticloss_optimizer,
                    log_alphas=log_alphas,
                    batch_size=batch_size,
                    gamma=gamma ** n_step,
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
                    random_process=random_process,
                    noise_clip=exploration_update_clip,
                    per=False,
                    update_priorities=step < bc_warmup_steps + num_steps_offline
                    or use_afbc_update_online,
                )

                # move target model towards training model
                if (critic_update + step) % target_delay == 0:
                    for agent_critic, target_critic in zip(
                        agent.critics, target_agent.critics
                    ):
                        lu.soft_update(target_critic, agent_critic, mlp_tau)
                    lu.soft_update(target_agent.encoder, agent.encoder, encoder_tau)
            critic_logs.update({"schedule/critic_update": 1.0})
        else:
            critic_logs.update({"schedule/critic_update": 0.0})

        if (step > bc_warmup_steps and step < bc_warmup_steps + num_steps_offline) or (
            step >= bc_warmup_steps + 1 + num_steps_offline and use_afbc_update_online
        ):
            #######################
            ## AWAC Actor Update ##
            #######################

            if step == bc_warmup_steps + 1:
                qprint("[First Offline Actor Update]")
            if step == bc_warmup_steps + num_steps_offline:
                qprint("[First Online Filtered BC Update]")
            for actor_update in range(afbc_actor_updates_per_step):
                _use_past_dicts = (
                    not afbc_per and actor_update == 0
                ) and reuse_replay_dicts
                actor_logs.update(
                    learning.offline_actor_update(
                        buffer=buffer,
                        agent=agent,
                        actor_optimizer=offline_actor_optimizer,
                        # train encoder with critic grads only
                        encoder_optimizer=encoder_actorloss_optimizer,
                        encoder_clip=encoder_clip,
                        update_encoder=False,
                        batch_size=batch_size,
                        actor_clip=actor_clip,
                        augmenter=augmenter,
                        actor_lambda=actor_lambda,
                        aug_mix=aug_mix,
                        per=afbc_per,
                        premade_replay_dicts=premade_replay_dicts
                        if _use_past_dicts
                        else None,
                        discrete=agent.discrete,
                        filter_=True,
                    )
                )
            actor_logs.update({"schedule/afbc_actor_update": 1.0})
        else:
            actor_logs.update({"schedule/afbc_actor_update": 0.0})

        if step > bc_warmup_steps + num_steps_offline and use_pg_update_online:
            ######################
            ## DPG Actor Update ##
            ######################

            if step == bc_warmup_steps + num_steps_offline + 1:
                qprint("[First Online Actor Update]")
            for actor_update in range(pg_actor_updates_per_step):
                _use_past_dicts = (actor_update == 0) and reuse_replay_dicts
                actor_logs.update(
                    learning.online_actor_update(
                        buffer=buffer,
                        agent=agent,
                        pop=pop,
                        actor_optimizer=online_actor_optimizer,
                        log_alphas=log_alphas,
                        batch_size=batch_size,
                        aug_mix=aug_mix,
                        clip=actor_clip,
                        augmenter=augmenter,
                        per=False,
                        discrete=agent.discrete,
                        random_process=random_process,
                        noise_clip=exploration_update_clip,
                        premade_replay_dicts=premade_replay_dicts
                        if _use_past_dicts
                        else None,
                        use_baseline=False,
                    )
                )
            actor_logs.update({"schedule/pg_actor_update": 1.0})
        else:
            actor_logs.update({"schedule/pg_actor_update": 0.0})

        if (
            step > bc_warmup_steps + num_steps_offline
            and init_alpha > 0
            and alpha_lr > 0
        ):
            ##################
            ## Alpha Update ##
            ##################

            if step == bc_warmup_steps + num_steps_offline + 1:
                qprint("[First Alpha Update]")
            _use_past_dicts = reuse_replay_dicts
            actor_logs.update(
                learning.alpha_update(
                    buffer=buffer,
                    agent=agent,
                    optimizers=log_alpha_optimizers,
                    batch_size=batch_size,
                    log_alphas=log_alphas,
                    augmenter=augmenter,
                    aug_mix=aug_mix,
                    target_entropy=target_entropy,
                    premade_replay_dicts=premade_replay_dicts
                    if _use_past_dicts
                    else None,
                    discrete=agent.discrete,
                )
            )
            actor_logs.update({"schedule/alpha_update": 1.0})
        else:
            actor_logs.update({"schedule/alpha_update": 0.0})

        #############
        ## LOGGING ##
        #############
        if (step % log_interval == 0) and log_to_disk:
            performance_logs = {
                "replay_buffer_total_samples": buffer.total_sample_calls
            }
            if logging_method == "tensorboard":
                for key, val in critic_logs.items():
                    writer.add_scalar(key, val, step)
                for key, val in actor_logs.items():
                    writer.add_scalar(key, val, step)
                for key, val in bc_logs.items():
                    writer.add_scalar(key, val, step)
                for key, val in performance_logs.items():
                    writer.add_scalar(key, val, step)
            elif logging_method == "wandb":
                wandb.log(critic_logs, step=step)
                wandb.log(actor_logs, step=step)
                wandb.log(bc_logs, step=step)
                wandb.log(performance_logs, step=step)

        if (
            (step % eval_interval == 0) or (step == total_steps - 1)
        ) and eval_interval > 0:
            mean_return = evaluation.evaluate_agent(
                agent,
                test_env,
                eval_episodes,
                max_episode_steps,
                render,
                verbosity=verbosity,
                num_envs=num_eval_envs,
            )
            if log_to_disk:
                accepted_exp_pct = lu.compute_filter_stats(
                    buffer=buffer,
                    agent=agent,
                    augmenter=augmenter,
                    batch_size=batch_size,
                )
                if logging_method == "tensorboard":
                    writer.add_scalar("return", mean_return, step)
                    writer.add_scalar("Accepted Exp Pct", accepted_exp_pct, step)
                elif logging_method == "wandb":
                    wandb.log(
                        {"return": mean_return, "Accepted Exp Pct": accepted_exp_pct}
                    )
        if step % save_interval == 0 and save_to_disk:
            agent.save(save_dir)

    if save_to_disk:
        agent.save(save_dir)

    return agent


def make_process_dirs(run_name, base_path):
    if base_path is None:
        base_path == "./saves"
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir

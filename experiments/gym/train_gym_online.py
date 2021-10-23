import argparse
import os

import gym

import pybullet
import pybullet_envs

import super_sac
from super_sac.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    ScaleReward,
    DiscreteActionWrapper,
)


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


def train_gym_online(args):
    discrete = isinstance(gym.make(args.env).action_space, gym.spaces.Discrete)

    def make_env():
        env = gym.make(args.env)
        if discrete:
            env = DiscreteActionWrapper(env)
        else:
            env = NormActionSpace(env)
        return ScaleReward(env, args.r_scale)

    train_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_envs))
    test_env = SimpleGymWrapper(ParallelActors(make_env, args.parallel_eval_envs))
    if args.render:
        train_env.reset()
        test_env.reset()  # fix common gym render bug

    if discrete:
        actor_network_cls = super_sac.nets.mlps.DiscreteActor
        critic_network_cls = super_sac.nets.mlps.DiscreteCritic
        act_space_size = train_env.action_space.n
    else:
        actor_network_cls = super_sac.nets.mlps.ContinuousStochasticActor
        critic_network_cls = super_sac.nets.mlps.ContinuousCritic
        act_space_size = train_env.action_space.shape[0]

    # create agent
    agent = super_sac.Agent(
        act_space_size=act_space_size,
        encoder=IdentityEncoder(train_env.observation_space.shape[0]),
        actor_network_cls=actor_network_cls,
        critic_network_cls=critic_network_cls,
        discrete=discrete,
        ensemble_size=args.ensemble_size,
        num_critics=args.num_critics,
        hidden_size=args.hidden_size,
        ucb_bonus=args.ucb_bonus,
        auto_rescale_targets=args.popart,
    )

    buffer = super_sac.replay.PrioritizedReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        verbosity=1,
        name=args.name,
        use_pg_update_online=True,
        actor_lr=1e-4,
        critic_lr=1e-4,
        encoder_lr=1e-4,
        batch_size=args.batch_size,
        target_critic_ensemble_n=2,
        weighted_bellman_temp=None,
        weight_type=None,
        use_bc_update_online=False,
        bc_warmup_steps=0,
        num_steps_offline=0,
        num_steps_online=1_000_000,
        random_warmup_steps=10_000,
        max_episode_steps=args.max_episode_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        use_exploration_process=args.use_exploration_process,
        pop=args.popart,
        init_alpha=0.1,
        alpha_lr=1e-4,
        render=args.render,
        logging_method=args.logging_method,
        wandb_entity=os.getenv("SSAC_WANDB_ACCOUNT"),
        wandb_project=os.getenv("SSAC_WANDB_PROJECT"),
        base_save_path=os.getenv("SSAC_SAVE"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--name", type=str, default="super_sac_pendulum_run")
    parser.add_argument("--ucb_bonus", type=float, default=0.0)
    parser.add_argument("--r_scale", type=float, default=1.0)
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--num_critics", type=int, default=2)
    parser.add_argument("--parallel_envs", type=int, default=1)
    parser.add_argument("--parallel_eval_envs", type=int, default=1)
    parser.add_argument("--popart", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=5_000)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_exploration_process", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument(
        "--logging_method",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    args = parser.parse_args()
    train_gym_online(args)

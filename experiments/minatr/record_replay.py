import argparse
import tqdm
import pickle

import numpy as np
import gym
import torch

import uafbc
from uafbc.wrappers import SimpleGymWrapper, DiscreteActionWrapper
from minatar_utils import MinAtarEnv, MinAtarEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str)
    parser.add_argument("--save_experience", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--env", type=str, default="breakout")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of expert rollouts"
    )
    parser.add_argument("--ensemble_size", type=int, default=1)
    args = parser.parse_args()

    env = SimpleGymWrapper(DiscreteActionWrapper(MinAtarEnv(args.env)))

    # create agent
    agent = uafbc.Agent(
        act_space_size=6,
        encoder=MinAtarEncoder(channels=env.num_channels),
        actor_network_cls=uafbc.nets.mlps.DiscreteActor,
        critic_network_cls=uafbc.nets.mlps.DiscreteCritic,
        hidden_size=256,
        discrete=True,
        num_critics=2,
        ensemble_size=args.ensemble_size,
        ucb_bonus=0.0,
        auto_rescale_targets=True,
        beta_dist=False,
    )
    assert args.expert_policy_file
    agent.to(uafbc.device)
    agent.load(args.expert_policy_file)
    agent.eval()

    returns = []
    reward_histories = []
    states, actions, rewards, next_states, dones = [], [], [], [], []
    ep_lengths = []
    ep_sim_steps = []
    for i in tqdm.tqdm(range(args.episodes)):
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            with torch.no_grad():
                action = agent.sample_action(obs)
            next_obs, rew, done, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(rew)
            next_states.append(next_obs)
            dones.append(done)

            obs = next_obs
            if args.render:
                env.render()
            totalr += rew
            steps += 1

        ep_lengths.append(steps)
        returns.append(totalr)

    print("returns", returns)
    print("ep_lengths", ep_lengths)
    print("ep_sim_steps", ep_sim_steps)
    print("mean return", np.mean(returns))
    print("std of return", np.std(returns))

    with open(args.save_experience, "wb") as f:
        pickle.dump(
            dict(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones,
            ),
            f,
        )


if __name__ == "__main__":
    main()

import argparse
import tqdm
import pickle

import numpy as np
import dmc2gym
import torch
import gin

import super_sac
from super_sac.wrappers import Uint8Wrapper, FrameStack
from train_dmc import IdentityEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--save_experience", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config)

    env = dmc2gym.make(
            args.domain,
            args.task,
            visualize_reward=False,
            from_pixels=True,
            frame_skip=2,
        )
    env.reset()
    state_space_size = env.current_state.shape[0]
    env = Uint8Wrapper(FrameStack(env, 3))


    # create agent
    agent = super_sac.Agent(
        act_space_size=env.action_space.shape[0],
        encoder=IdentityEncoder(state_space_size),
    )
    agent.to(super_sac.device)
    agent.load(args.policy)
    agent.eval()

    returns = []
    reward_histories = []
    actions, rewards, dones = [], [], []
    state_keys = env.reset().keys()
    states = {k:[] for k in state_keys}
    next_states = {k:[] for k in state_keys}

    ep_lengths = []
    ep_sim_steps = []
    for i in tqdm.tqdm(range(args.episodes)):
        obs = env.reset()
        done = False
        totalr = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            internal_state = {"obs":env.current_state}
            with torch.no_grad():
                action = agent.sample_action(internal_state)
            next_obs, rew, done, _ = env.step(action)
            for key, val in obs.items():
                states[key].append(val)
            for key, val in next_obs.items():
                next_states[key].append(val)
            actions.append(action)
            rewards.append(rew)
            dones.append(done)

            obs = next_obs
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

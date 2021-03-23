import gym
import numpy as np
import torch


def run_env(agent, env, episodes, max_steps, render=False, verbosity=1):
    episode_return_history = []
    if render:
        env.render("rgb_array")
    for episode in range(episodes):
        episode_return = 0.0
        state = env.reset()
        done, info = False, {}
        for _ in range(max_steps):
            if done:
                break
            action = agent.forward(state)
            state, reward, done, info = env.step(action)
            if render:
                env.render("rgb_array")
            episode_return += reward
        if verbosity:
            print(f"Episode {episode}:: {episode_return}")
        episode_return_history.append(episode_return)
    return torch.tensor(episode_return_history)


def exploration_noise(action, random_process):
    return np.clip(action + random_process.sample(), -1.0, 1.0)


def evaluate_agent(
    agent, env, eval_episodes, max_episode_steps, render=False, verbosity=0
):
    agent.eval()
    returns = run_env(
        agent, env, eval_episodes, max_episode_steps, render, verbosity=verbosity
    )
    agent.train()
    mean_return = returns.mean()
    return mean_return

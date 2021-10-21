import random
from collections import deque

import gym
import numpy as np


def ParallelActors(make_env, actors):
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        raise ImportError(
            "Missing stable_baselines3 for parallel env. `pip install stable_baselines3`"
        )

    class _Compatible(gym.Wrapper):
        def __init__(self, env):
            if not hasattr(env, "reward_range"):
                env.reward_range = None
            self._PARALLEL_ACTORS = actors
            super().__init__(env)

        def step(self, action):
            s, r, d, i = self.env.step(action)
            return s, self.expand(r), self.expand(d), i

        def expand(self, x):
            return np.expand_dims(np.array(x), 1)

    if actors > 1:
        return _Compatible(SubprocVecEnv([make_env for _ in range(actors)]))
    else:
        return make_env()


class NormActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._true_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32,
        )

    def action(self, action):
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action


class ChannelsFirstWrapper(gym.ObservationWrapper):
    """
    Some pixel-based gym environments use a (Height, Width, Channel) image format.
    This wrapper rolls those axes to (Channel, Height, Width) to work with pytorch
    Conv2D layers.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space.shape = (
            env.observation_space.shape[-1],
        ) + env.observation_space.shape[:-1]

    def observation(self, frame):
        frame = np.transpose(frame, (2, 0, 1))
        return np.ascontiguousarray(frame)


class DiscreteActionWrapper(gym.ActionWrapper):
    def action(self, action):
        if len(action.shape) > 0:
            action = action[0]
        action = int(action)
        return action


class SimpleGymWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return {"obs": obs.astype(np.float32)}


class KeepKeysWrapper(gym.ObservationWrapper):
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys

    def observation(self, obs):
        return {key: obs[key] for key in self.keys}


def load_atari(
    game_id,
    seed=None,
    noop_max=30,
    frame_skip=1,
    screen_size=84,
    terminal_on_life_loss=False,
    rgb=False,
    normalize=False,
    frame_stack=4,
    clip_reward=True,
    **_,
):
    """
    Load a game from the Atari benchmark, with the usual settings
    Note that the simplest game ids (e.g. Boxing-v0) come with frame
    skipping by default, and you'll get an error if the frame_skp arg > 1.
    Use `BoxingNoFrameskip-v0` with frame_skip > 1.
    """
    env = gym.make(game_id)
    if seed is None:
        seed = random.randint(1, 100000)
    env.seed(seed)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=False,  # use GrayScale wrapper instead...
        scale_obs=normalize,
    )
    if not rgb:
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    if clip_reward:
        env = ClipReward(env)
    env = ChannelsFirstWrapper(env)
    env = FrameStack(env, num_stack=frame_stack)
    env = DiscreteActionWrapper(env)
    return env


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self._clip_low = low
        self._clip_high = high

    def reward(self, rew):
        return max(min(rew, self._clip_high), self._clip_low)


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, rew):
        return self.scale * rew


class DeltaReward(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._old_rew = 0

    def reward(self, rew):
        delta_rew = rew - self._old_rew
        self._old_rew = rew
        return delta_rew


class StateStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int, skip: int = 1):
        gym.Wrapper.__init__(self, env)
        self._k = num_stack
        self._frames = deque([], maxlen=num_stack * skip)
        shp = env.observation_space.shape[0]
        low = np.array([env.observation_space.low for _ in range(num_stack)]).flatten()
        high = np.array(
            [env.observation_space.high for _ in range(num_stack)]
        ).flatten()
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(shp * num_stack,),
            dtype=env.observation_space.dtype,
        )
        self._skip = skip

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k * self._skip):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k * self._skip
        obs = np.concatenate(list(self._frames)[:: -self._skip], axis=0)
        return obs


class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=0):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        tot_rew = 0
        for _ in range(self.skip + 1):
            next_state, rew, done, info = self.env.step(action)
            tot_rew += rew
            if done:
                break
        return next_state, tot_rew, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack):
        gym.Wrapper.__init__(self, env)
        self._k = num_stack
        self._frames = deque([], maxlen=num_stack)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * num_stack,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

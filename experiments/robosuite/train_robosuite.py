import argparse
import os

import robosuite
import gym
from robosuite import wrappers
import gin

import super_sac
from super_sac.wrappers import SimpleGymWrapper, NormActionSpace, Uint8Wrapper


class IdentityEncoder(super_sac.nets.Encoder):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


@gin.configurable
def create_robosuite_env(
    env_name="Lift",
    robots=["Panda"],
    controller="OSC_POSE",
    has_renderer=False,
    control_freq=20,
    use_camera_obs=False,
    use_object_obs=False,
    camera_names="agentview",
    env_configuration="single-arm-opposed",
    reward_shaping=True,
    horizon=500,
    camera_heights=84,
    camera_widths=84,
):
    controller_config = robosuite.controllers.load_controller_config(
        default_controller=controller
    )
    env = robosuite.make(
        env_name,
        robots=robots,
        gripper_types="default",
        controller_configs=controller_config,
        env_configuration=env_configuration,
        has_renderer=has_renderer,
        render_camera="frontview",
        has_offscreen_renderer=use_camera_obs,
        control_freq=control_freq,
        use_object_obs=use_object_obs,
        horizon=horizon,
        use_camera_obs=use_camera_obs,
        camera_heights=camera_heights,
        camera_widths=camera_widths,
        reward_shaping=reward_shaping,
    )
    env = wrappers.gym_wrapper.GymWrapper(env)
    env = NormActionSpace(env)

    if use_camera_obs:
        raise NotImplementedError
    else:
        env = SimpleGymWrapper(env)
    return env, use_camera_obs


def train_robosuite(args):
    train_env, from_pixels = create_robosuite_env(args.env)
    test_env, from_pixels = create_robosuite_env(args.env)

    if not from_pixels:
        encoder = IdentityEncoder(train_env.observation_space.shape[0])
    else:
        raise NotImplementedError

    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=encoder,
    )

    buffer = super_sac.replay.ReplayBuffer(size=1_000_000)

    # run training
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=args.name,
        logging_method=args.logging,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Door")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    for _ in range(args.trials):
        train_robosuite(args)

import argparse
import gin
import os

import dmc2gym

import super_sac
from super_sac import nets
from super_sac.augmentations import AugmentationSequence, Drqv2Aug, DrqAug
from super_sac.wrappers import Uint8Wrapper, FrameStack


class DMCPixelEncoder(nets.Encoder):
    def __init__(self, inp_shape, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv_block = nets.cnns.BigPixelEncoder(inp_shape, emb_dim)

    def forward(self, obs_dict):
        img = obs_dict["obs"]
        return self.conv_block(img)

    @property
    def embedding_dim(self):
        return self._dim


def train_dmc_from_pixels(args):
    gin.parse_config_file(args.config)

    train_env = FrameStack(
        dmc2gym.make(
            args.domain,
            args.task,
            visualize_reward=False,
            from_pixels=True,
            frame_skip=2,
        ),
        3,
    )
    test_env = FrameStack(
        dmc2gym.make(
            args.domain,
            args.task,
            visualize_reward=False,
            from_pixels=True,
            frame_skip=2,
        ),
        3,
    )
    train_env = Uint8Wrapper(train_env)
    test_env = Uint8Wrapper(test_env)

    # create agent
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=DMCPixelEncoder(train_env.observation_space.shape),
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
        augmenter=AugmentationSequence([Drqv2Aug(256)]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--task", type=str, default="walk")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--logging", type=str, choices=["tensorboard", "wandb"], default="tensorboard"
    )
    args = parser.parse_args()
    train_dmc_from_pixels(args)

import os
import argparse

from main.config import set_env
from main.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="envfile name", default="local", required=True)
    parser.add_argument('--gpu', type=str, help="use space between ids", default="0", required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    env = os.path.join("envs", args.env + ".yaml")

    cfg = set_env(env, args.gpu)
    trainer = Trainer(cfg)
    trainer.train()

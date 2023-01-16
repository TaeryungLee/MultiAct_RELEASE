import os
import torch
import argparse

from utils.logger import colorLogger as Logger
from main.config import set_env


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
    test_logger = Logger(cfg.log_dir, "test.log")
    from main.tester import Tester
    tester = Tester(cfg, test_logger=test_logger)
    tester.test()

import os
import pickle
import argparse

from main.config import set_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="gen_long, gen_short", default="gen_short")
    parser.add_argument('--env', type=str, help="envfile name", default="local", required=True)
    parser.add_argument('--gpu', type=str, help="use space between ids", default="0", required=True)


    # controller variables required

    # long: resume, resume_exp, resume_step, target_labels
    # short: nothing required

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    env = os.path.join("envs", args.env + ".yaml")
    cfg = set_env(env, args.gpu)
    from main.generator import Generator
    if args.mode == "gen_long":
        generator = Generator(cfg, args.mode)
        generator.generate_long_term()

    elif args.mode == "gen_short":
        generator = Generator(cfg, args.mode)
        generator.generate_short_term()

    
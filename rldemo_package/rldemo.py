
# Build-in modules
import argparse


# Third party modules 
import tensorflow as tf

# Custum modules
from rldemo_package.utils import argparser


def main():
    """Demonstration of reinforcement learning algorithms with selected OpenAI
    gym environments."""
    args = argparser.parse_args()

    if args.verbose:
        print(args.mode)
        print(args.algo)
        print(args.logging)
        print(args.checkpointdir)
        print(args.episodes)
        print(args.timesteps)
        print('tensorboard: ', args.tensorboard)
        


if __name__ == '__main__':
    main()
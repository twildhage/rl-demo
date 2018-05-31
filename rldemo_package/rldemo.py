
# Build-in modules
import argparse
import logging


# Third party modules 
import tensorflow as tf

# Custum modules
from utils import get_argparser, print_args
from utils import get_logger


def init():
    argparser = get_argparser()
    args = argparser.parse_args()
    log = get_logger(args.mode, args.logging)
    print_args(args, logger=log)

    return args, log

def main():
    """Demonstration of reinforcement learning algorithms with selected OpenAI
    gym environments."""
    args, log = init()


if __name__ == '__main__':
    main()
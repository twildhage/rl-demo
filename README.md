# RL Demo
***
Note: _THIS PROJECT IS WORK IN PROGRESS. PLEASE DON'T EXPECT SOMETHING FINISHED AND BUGFREE_
***
Demonstration of reinforcement learning algorithms with selected OpenAI gym environments.

The basic idea of the project is to have a simple command line client that can
run all OpenAI gym environments and allows for easy setup of custom reinforcement
algorithms.


# Installation
 Currently the installed program is still crashing due to failing module imports.
 Once these are fixed the installation can be done as follows:  
Simply run from within the rl-demo folder:

    $ python setup.py sdist
    $ pip install -e .

# Uninstallation

    $ pip uninstall rl_demo

# Usage

To use it:

    $ rldemo --help
If installation fails, the program can always be run via

    $ python rldemo.py [--options]

To see all available command line options type:

    $ python rldemo.py --help

'''This file originally used the typer module to facilitate CLI input. Now, I am using argparse and sys modules.'''

import argparse
import sys

# Absolute import:
# from clipkg import hello, another_function
# from clipkg.moduleA import hello_moduleA
# Relative import:
from . import hello, another_function
from . clipkg.moduleA import hello_moduleA, dummy_run_func

# Don't need these for now, use argparse module instead!
# opts = [opt for opt in sys.argv[1:] if opt.startswith('-')]
# args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='the argument value should =="run" ') # Test this arg by passing 'run' as the value of the arg
    parser.add_argument('--arg1', default='default')
    parser.add_argument('--arg2', default='default')
    args = parser.parse_args()
    print(f'args supplied: {args}')
    # args.__arg_name__ is a string variable:

    if args.run=='run':
        dummy_run_func()


if __name__ == '__main__':
    main()



'''This file originally used the typer module to facilitate CLI input. Now, I am using argparse and sys modules.'''

import argparse
import sys

# Absolute import:
from clipkg import hello, another_function
from clipkg.moduleA import hello_moduleA, dummy_run_func
from clipkg.src import dummy
# from clipkg.src import gluestick
# Relative import:
# from . import hello, another_function
# from . clipkg.moduleA import hello_moduleA, dummy_run_func

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run', action='store_true', help='the argument value should =="run" ')
    parser.add_argument('-tf', '--transform', action='store_true')
    parser.add_argument('-pr', '--predict', action='store_true')
    parser.add_argument('-dy', '--dummy', action='store_true')
    args = parser.parse_args()
    print(f'args supplied: {args}')
    # args.__arg_name__ is a string variable:

    if args.run:
        dummy_run_func()
    if args.dummy:
        dummy.src_dummy_func()
    # if args.transform:
    #     gluestick.mk_data()
    # if args.predict:
    #     gluestick.predict()

if __name__ == '__main__':
    main()
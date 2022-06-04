'''This is the package init file. I can declare project specific directories here.'''

from os import path

MODULE_PATH = path.normpath(path.join(path.dirname(path.realpath(__file__))))    # basically, where is this being executed from
PROJ_PATH = MODULE_PATH #path.join(MODULE_PATH, '..')
DATA_PATH = path.join(PROJ_PATH, 'data')    # optional, sometimes I declare this in another file
MODEL_OUT_PATH = path.join(PROJ_PATH, 'model')

def hello():
    print('hello from my_function')

def another_function():
    print('hello from another_function')
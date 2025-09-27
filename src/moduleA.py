def hello_moduleA():
    print(f'hello from file name: {__file__}')

def another_function():
    print('hello from another_function')

def dummy_run_func():
    print(f'this simulates running a func from a module at file {__file__}')


if __name__ == "__main__":
    """This runs when you execute '$ python3 mypackage/mymodule.py'"""
    hello_moduleA()
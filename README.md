A simple package to illustrate how to integrate CLI functionality!  
Packaging tutorial:  
'''https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/'''

# Helpful High-Level commands:
### Tip 1: To auto-generate requirements.txt:  
```$ pip freeze > requirements.txt```

### Tip 3: To create BOTH source and wheel distros, run:  
```$ python setup.py sdist bdist_wheel```

# Package Usage:
Within the root directory, you need to install this package! Do this by activating the virtual env, and then run:   
```$ pip install -e .```   
This will install the package in **editable** mode, meaning you don't have to re-install the package when you make new changes to your code.   
To actually run a command from the CLI from the package, run the following (which will run a dummy function and print out run-info when called)   
```python -m clipkg -r run```
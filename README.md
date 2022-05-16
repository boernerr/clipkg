A simple package to illustrate how to integrate CLI functionality!

to auto-generate requirements.txt:
$ pip freeze > requirements.txt

Once in the root project dir AND virtual env is activated, to install the package, run:  
`````$ pip install .`````

Packaging tutorial:
'''https://betterscientificsoftware.github.io/python-for-hpc/tutorials/python-pypi-packaging/'''

To create BOTH source and wheel distros, run:  
`````$ python setup.py sdist bdist_wheel`````

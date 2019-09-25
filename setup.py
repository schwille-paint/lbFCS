from setuptools import setup

setup(
   name='lbFCS',
   version='0.1',
   description='Package to analyze DNA-PAINT data via lbFCS',
   license="Max Planck Institute of Biochemistry",
   author='Stehr Florian',
   author_email='stehr@biochem.mpg.de',
   url="http://www.github.com/schwille-paint/lbFCS",
   packages=['lbfcs'],
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=['picasso'], #external packages as dependencies
)


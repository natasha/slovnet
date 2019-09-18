
from setuptools import setup, find_packages


with open('requirements/main.txt') as file:
    requirements = file.read().splitlines()


setup(
    name='slovnet',
    version='0.2.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'slov-ctl=slovnet.ctl:main'
        ],
    },
    install_requires=requirements
)


from setuptools import setup, find_packages


with open('README.md') as file:
    description = file.read()


with open('requirements/main.txt') as file:
    requirements = [_.strip() for _ in file]


setup(
    name='slovnet',
    version='0.3.0',

    description='Deep-learning based NLP modeling for Russian language',
    long_description=description,
    long_description_content_type='text/markdown',

    url='https://github.com/natasha/slovnet',
    author='Alexander Kukushkin',
    author_email='alex@alexkuk.ru',
    license='MIT',

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='nlp, deeplearning, russian',

    packages=find_packages(),
    install_requires=requirements
)

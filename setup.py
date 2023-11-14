from setuptools import setup
from setuptools import find_packages

setup(
    name='ensemble_learning_theory',
    version='1.0',
    description='',
    url='https://github.com/hitachi-nlp/ensemble-learning-thoery.git',
    author='Terufumi Morishita',
    author_email='terufumi.morishita.wp@hitachi.com',
    license='MIT',
    install_requires=[
        'numpy',
        'pydantic',
    ],
    packages=find_packages(),
    zip_safe=False,
)

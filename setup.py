from setuptools import setup
from setuptools import find_packages

setup(
    name='ensemble_metrics',
    version='1.0',
    description='',
    url='https://github.com/hitachi-nlp/ensemble-metrics.git',
    author='Terufumi Morishita',
    author_email='terufumi.morishita.wp@hitachi.com',
    license='MIT',
    install_requires=[
        'asyncio',
        'requests',
        'aiohttp',
        'tenacity',
        'sanic',
    ],
    packages=find_packages(),
    zip_safe=False,
)

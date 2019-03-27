import fastentrypoints
from setuptools import setup, find_packages

setup(
    name='hogwarts',
    version='0.1',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'pyyaml',
        'tensorboardX',
    ],
    entry_points={
        'console_scripts': [
            'hg-manage = hogwarts.command:manage',
            'hg-run = hogwarts.command:run',
            'hg-ls = hogwarts.command:ls',
        ],
    },
)

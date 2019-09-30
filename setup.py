import fastentrypoints
from setuptools import setup, find_packages

setup(
    name='hogwarts',
    version='0.1',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'pyyaml',
        'torch',
        'json5',
        'tensorboard',
    ],
    entry_points={
        'console_scripts': [
            'hcontrol = hogwarts.command:control',
            'hrun = hogwarts.command:run',
            'hls = hogwarts.command:ls',
        ],
    },
)

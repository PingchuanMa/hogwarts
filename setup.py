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
            'hog-manage = hogwarts.command:manage',
            'hog-mng = hogwarts.command:manage',
            'hog-run = hogwarts.command:run',
            'hog-ls = hogwarts.command:ls',
            'hog-list = hogwarts.command:ls',
        ],

    },
)

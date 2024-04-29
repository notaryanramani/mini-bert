from setuptools import setup, find_packages
import sys

def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    return requirements

def get_dependency_links():
    if sys.platform == 'win':
        link = ['https://download.pytorch.org/whl/cu121']
        return link
    elif sys.platform == 'darwin':
        link = ['https://download.pytorch.org/whl/nightly/cpu']
        return link
    else:
        return []

setup(
    name = 'mini-bert',
    version = '1.0.0',
    packages = find_packages(),
    install_requires = get_requirements(),
    dependency_links = get_dependency_links()
)

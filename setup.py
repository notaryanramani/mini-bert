from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name='mini-bert',
    version='1.0.0',
    packages=find_packages(),
    install_requires = get_requirements(),
    dependency_links = ['https://download.pytorch.org/whl/nightly/cpu']
)

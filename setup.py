from setuptools import setup, find_packages
import sys

def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
        requirements.append(get_torch_version())
    return requirements

def get_torch_version():
    python_version = 'cp' + str(sys.version_info.major) + str(sys.version_info.minor)
    if sys.platform == 'win32':
        torch_ = f'torch @https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-{python_version}-{python_version}-win_amd64.whl'
        return torch_
    
    elif sys.platform == 'darwin':
        torch_ = f'torch @https://download.pytorch.org/whl/cpu/torch-2.3.0%2Bcpu-{python_version}-{python_version}-macosx_10_9_x86_64.whl'
        return torch_
    
    elif sys.platform == 'linux':
        torch_ = f'torch @https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-{python_version}-{python_version}-linux_x86_64.whl'
        return torch_


setup(
    name = 'mini-bert',
    version = '1.0.0',
    packages = find_packages(),
    install_requires = get_requirements()
)

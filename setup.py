from setuptools import setup, find_packages
import sys

def get_requirements():
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()
        requirements.extend(get_torch_version())
    return requirements

def get_torch_version():
    python_version = 'cp' + str(sys.version_info.major) + str(sys.version_info.minor)
    if sys.platform == 'win32':
        torch_ = f'torch @https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-{python_version}-{python_version}-win_amd64.whl'
        torchaudio_ = f'torchvision @https://download.pytorch.org/whl/cu121/torchvision-0.10.0%2Bcu121-cp{python_version}-cp{python_version}-win_amd64.whl'
        torchvision_ = f'torchaudio @https://download.pytorch.org/whl/cu121/torchaudio-0.9.0-cp{python_version}-cp{python_version}-win_amd64.whl'
        return [torch_, torchaudio_, torchvision_]
    
    elif sys.platform == 'darwin':
        torch_ = f'torch @https://download.pytorch.org/whl/cpu/torch-2.3.0%2Bcpu-{python_version}-{python_version}-macosx_10_9_x86_64.whl'
        torchaudio_ = f'torchvision @https://download.pytorch.org/whl/cpu/torchvision-0.10.0%2Bcpu-cp{python_version}-cp{python_version}-macosx_10_9_x86_64.whl'
        torchvision_ = f'torchaudio @https://download.pytorch.org/whl/cpu/torchaudio-0.9.0-cp{python_version}-cp{python_version}-macosx_10_9_x86_64.whl'
        return [torch_, torchaudio_, torchvision_]
    
    elif sys.platform == 'linux':
        torch_ = f'torch @https://download.pytorch.org/whl/cu121/torch-2.3.0%2Bcu121-{python_version}-{python_version}-linux_x86_64.whl'
        torchaudio_ = f'torchvision @https://download.pytorch.org/whl/cu121/torchvision-0.10.0%2Bcu121-cp{python_version}-cp{python_version}-linux_x86_64.whl'
        torchvision_ = f'torchaudio @https://download.pytorch.org/whl/cu121/torchaudio-0.9.0-cp{python_version}-cp{python_version}-linux_x86_64.whl'
        return [torch_, torchaudio_, torchvision_]


setup(
    name = 'mini-bert',
    version = '1.0.0',
    packages = find_packages(),
    install_requires = get_requirements()
)

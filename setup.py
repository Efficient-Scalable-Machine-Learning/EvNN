# Copyright (c) 2023  Khaleelulla Khan Nazeer
# This file incorporates work covered by the following copyright:
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

from glob import glob
from platform import platform
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, CUDA_HOME


def get_gpu_arch_flags():
    try:
        major = torch.cuda.get_device_capability()[0]
        return [f"-gencode=arch=compute_{major}0,code=sm_{major}0"]
    except Exception as e:
        print(f"Error while detecting GPU architecture: {e}")
        return []


arch_flags = get_gpu_arch_flags()

VERSION = '0.1.0'
DESCRIPTION = 'EVNN: a torch extension for custom event based RNN models.'
AUTHOR = 'TUD and RUB'
AUTHOR_EMAIL = 'khaleelulla.khan_nazeer@tu-dresden.de'
URL = 'https://tu-dresden.de/ing/elektrotechnik/iee/hpsn'
LICENSE = 'Apache 2.0'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Software Development :: Libraries',
]


with open(f'frameworks/pytorch/_version.py', 'wt') as f:
    f.write(f'__version__ = "{VERSION}"')

base_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available() and CUDA_HOME is not None:

    extension = [CUDAExtension(
        'evnn_pytorch_lib',
        sources=glob('frameworks/pytorch/*.cc') + glob('lib/*.cu') + glob('lib/*.cc'),
        extra_compile_args={
                "cxx": ["-O2", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DWITH_CUDA", "-Wno-sign-compare"],
                "nvcc": ["-O2", "-std=c++17", 
                         "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-D_GLIBCXX_USE_CXX11_ABI=0", "-DWITH_CUDA",
                         "-Xcompiler", "-fPIC", "-lineinfo"]
                + arch_flags,
        },
        include_dirs=[os.path.join(base_path, 'lib'),
                      os.path.join(CUDA_HOME, 'include'),
                      os.path.join(CUDA_HOME, 'lib64')],
        libraries=['openblas', 'c10', 'cudart', 'cublas'],
        library_dirs=['.']),
    ]
else:
    extension = [CppExtension(
        'evnn_pytorch_lib',
        sources=glob('frameworks/pytorch/*.cc') + glob('lib/*.cc'),
        extra_compile_args={
                "cxx": ["-O2", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-Wno-sign-compare"],
        },
        include_dirs=[os.path.join(base_path, 'lib'),],
        libraries=['openblas'],
        library_dirs=['.', os.path.join('/usr/lib/x86_64-linux-gnu')])]


setup(name='evnn_pytorch',
      version=VERSION,
      description=DESCRIPTION,
      long_description=open('README.md', 'r', encoding='utf-8').read(),
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      license=LICENSE,
      keywords='pytorch machine learning rnn lstm gru custom op',
      packages=['evnn_pytorch'],
      package_dir={'evnn_pytorch': 'frameworks/pytorch'},
      install_requires=['torch'],
      ext_modules=extension,
      cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False),},
      classifiers=CLASSIFIERS)

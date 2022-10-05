# coding=utf-8
# Copyright 2022 The Ravens Authors.
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

"""Setup."""

from distutils import core
import os

from setuptools import find_packages


here = os.path.abspath(os.path.dirname(__file__))
try:
  README = open(os.path.join(here, 'README.md'), encoding='utf-8').read()
except IOError:
  README = ''


install_requires = [
    'absl-py>=0.7.0',
    'gym>=0.17.3',
    'numpy>=1.18.5',
    'pybullet>=3.0.4',
    'matplotlib>=3.1.1',
    'opencv-python>=4.1.2.30',
    'meshcat>=0.0.18',
    'scipy>=1.4.1',
    'scikit-image>=0.17.2',
    'tensorflow>=2.3.0',
    'tensorflow-addons>=0.11.2',
    'tensorflow_hub>=0.9.0',
    'transforms3d>=0.3.1',
]


core.setup(
    name='ravens',
    version='0.1',
    description='Ravens is a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation.',
    long_description='\n\n'.join([README]),
    long_description_content_type='text/markdown',
    author='Andy Zeng, Pete Florence, Daniel Seita, Jonathan Tompson, Ayzaan Wahid',
    author_email='ravens-team@google.com',
    url='https://github.com/google-research/ravens',
    packages=find_packages(),
    install_requires=install_requires,
)

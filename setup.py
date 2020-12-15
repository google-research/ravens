# coding=utf-8
# Copyright 2020 The Ravens Authors.
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
)

# coding=utf-8
# Copyright 2021 The Ravens Authors.
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

"""Integration tests for ravens tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from ravens import tasks
from ravens.environments import environment


ASSETS_PATH = 'ravens/environments/assets/'


class TaskTest(parameterized.TestCase):

  def _create_env(self):
    assets_root = ASSETS_PATH
    env = environment.Environment(assets_root)
    env.seed(0)
    return env

  def _run_oracle_in_env(self, env):
    agent = env.task.oracle(env)
    obs = env.reset()
    info = None
    done = False
    for _ in range(10):
      act = agent.act(obs, info)
      obs, _, done, info = env.step(act)
      if done:
        break

  @parameterized.named_parameters((
      'AlignBoxCorner',
      tasks.AlignBoxCorner(),
  ), (
      'AssemblingKits',
      tasks.AssemblingKits(),
  ), (
      'AssemblingKitsEasy',
      tasks.AssemblingKitsEasy(),
  ), (
      'BlockInsertion',
      tasks.BlockInsertion(),
  ), (
      'ManipulatingRope',
      tasks.ManipulatingRope(),
  ), (
      'PackingBoxes',
      tasks.PackingBoxes(),
  ), (
      'PalletizingBoxes',
      tasks.PalletizingBoxes(),
  ), (
      'PlaceRedInGreen',
      tasks.PlaceRedInGreen(),
  ), (
      'StackBlockPyramid',
      tasks.StackBlockPyramid(),
  ), (
      'SweepingPiles',
      tasks.SweepingPiles(),
  ), (
      'TowersOfHanoi',
      tasks.TowersOfHanoi(),
  ))
  def test_all_tasks(self, ravens_task):
    env = self._create_env()
    env.set_task(ravens_task)
    self._run_oracle_in_env(env)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for ravens.environments.environment."""

from absl.testing import absltest

from ravens import tasks
from ravens.environments import environment

ASSETS_PATH = 'ravens/environments/assets/'


class EnvironmentTest(absltest.TestCase):

  def test_environment_action(self):
    env = environment.Environment(ASSETS_PATH)
    task = tasks.BlockInsertion()
    env.set_task(task)
    env.seed(0)
    agent = task.oracle(env)
    obs = env.reset()
    info = None
    done = False
    for _ in range(10):
      act = agent.act(obs, info)
      self.assertTrue(env.action_space.contains(act))
      obs, _, done, info = env.step(act)
      if done:
        break

  def test_environment_action_continuous(self):
    env = environment.ContinuousEnvironment(ASSETS_PATH)
    task = tasks.BlockInsertion(continuous=True)
    env.set_task(task)
    env.seed(0)
    agent = task.oracle(env)
    obs = env.reset()
    info = None
    done = False
    for _ in range(100):
      act = agent.act(obs, info)
      self.assertTrue(env.action_space.contains(act))
      obs, _, done, info = env.step(act)
      if done:
        break


if __name__ == '__main__':
  absltest.main()

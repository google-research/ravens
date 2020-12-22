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

"""Ravens main training script."""

import os
import pickle

from absl import app
from absl import flags
import numpy as np
from ravens import agents
from ravens import dataset
from ravens import tasks
from ravens.environments.environment import Environment
import tensorflow as tf

flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('assets_root', './assets/', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_bool('shared_memory', False, '')
flags.DEFINE_string('task', 'hanoi', '')
flags.DEFINE_string('agent', 'transporter', '')
flags.DEFINE_integer('n_demos', 100, '')
flags.DEFINE_integer('n_steps', 40000, '')
flags.DEFINE_integer('n_runs', 1, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')

FLAGS = flags.FLAGS


def main(unused_argv):
  # Configure which GPU to use.
  cfg = tf.config.experimental
  gpus = cfg.list_physical_devices('GPU')
  if not gpus:
    print('No GPUs detected. Running with CPU.')
  else:
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')

  # Configure how much GPU to use (in Gigabytes).
  if FLAGS.gpu_limit is not None:
    mem_limit = 1024 * FLAGS.gpu_limit
    dev_cfg = [cfg.VirtualDeviceConfiguration(memory_limit=mem_limit)]
    cfg.set_virtual_device_configuration(gpus[0], dev_cfg)

  # Initialize environment and task.
  env = Environment(
      FLAGS.assets_root,
      disp=FLAGS.disp,
      shared_memory=FLAGS.shared_memory,
      hz=480)
  task = tasks.names[FLAGS.task]()
  task.mode = 'test'

  # Load test dataset.
  ds = dataset.Dataset(os.path.join(FLAGS.data_dir, f'{FLAGS.task}-test'))

  # Run testing for each training run.
  for train_run in range(FLAGS.n_runs):
    name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'

    # Initialize agent.
    np.random.seed(train_run)
    tf.random.set_seed(train_run)
    agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)

    # # Run testing every interval.
    # for train_step in range(0, FLAGS.n_steps + 1, FLAGS.interval):

    # Load trained agent.
    if FLAGS.n_steps > 0:
      agent.load(FLAGS.n_steps)

    # Run testing and save total rewards with last transition info.
    results = []
    for i in range(ds.n_episodes):
      print(f'Test: {i + 1}/{ds.n_episodes}')
      episode, seed = ds.load(i)
      goal = episode[-1]
      total_reward = 0
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      obs = env.reset()
      info = None
      reward = 0
      for _ in range(task.max_steps):
        act = agent.act(obs, info, goal)
        obs, reward, done, info = env.step(act)
        total_reward += reward
        print(f'Total Reward: {total_reward} Done: {done}')
        if done:
          break
      results.append((total_reward, info))

      # Save results.
      with tf.io.gfile.GFile(
          os.path.join(FLAGS.root_dir, f'{name}-{FLAGS.n_steps}.pkl'),
          'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
  app.run(main)

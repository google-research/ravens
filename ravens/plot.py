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

"""Script to plot training results."""

import os
import pickle

from absl import app
from absl import flags
import numpy as np
from ravens.utils import utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_bool('disp', True, 'Whether to display the environment.')
flags.DEFINE_string('task', 'insertion', 'The task to run.')
flags.DEFINE_string('agent', 'transporter', 'The agent to run.')
flags.DEFINE_integer('n_demos', 100, 'Number of demos to run.')


def main(unused_argv):
  name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-'
  print(name)

  # Load and print results to console.
  path = FLAGS.root_dir
  curve = []
  for fname in tf.io.gfile.listdir(path):
    fname = os.path.join(path, fname)
    if name in fname and '.pkl' in fname:
      n_steps = int(fname[(fname.rfind('-') + 1):-4])
      data = pickle.load(open(fname, 'rb'))
      rewards = []
      for reward, _ in data:
        rewards.append(reward)
      score = np.mean(rewards)
      std = np.std(rewards)
      curve.append((n_steps, score, std))
  curve.sort()
  for log in curve:
    print(f'  {log[0]} steps:\t{log[1]:.2f}%\t± {log[2]:.2f}%')

  # Plot results over training steps.
  title = f'{FLAGS.agent} on {FLAGS.task} w/ {FLAGS.n_demos} demos'
  ylabel = 'Testing Task Success (%)'
  xlabel = '# of Training Steps'
  if FLAGS.disp:
    logs = {}
    curve = np.array(curve)
    logs[name] = (curve[:, 0], curve[:, 1], curve[:, 2])
    fname = os.path.join(path, f'{name}-plot.png')
    utils.plot(fname, title, ylabel, xlabel, data=logs, ylim=[0, 1])
    print(f'Done. Plot image saved to: {fname}')

if __name__ == '__main__':
  app.run(main)

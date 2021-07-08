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

"""Planners for continuous environments."""

import abc

import matplotlib.pyplot as plt
import numpy as np
from ravens.utils import utils
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


class Planner(abc.ABC):
  """Spline-based planner base class.

  Subclasses must implement `_preprocess_poses` and `_postprocess_poses`
  methods.
  """

  def __init__(self, steps_per_seg=2, t_max=10.0, height=0.32):
    self.steps_per_seg = steps_per_seg
    self.t_max = t_max
    self.height = height

    self.poses = []
    self.rots = None
    self.rot_spline = None
    self.pos_spline = None

  @abc.abstractclassmethod
  def _preprocess_poses(cls, *poses):
    pass

  @abc.abstractclassmethod
  def _postprocess_poses(cls, poses, xnew):
    pass

  def _interpolate_position(self):
    xyzs = [p[0] for p in self.poses]
    self.pos_spline = [
        interp1d(self.times, [p[i]
                              for p in xyzs], kind="linear")
        for i in range(3)
    ]

  def _interpolate_rotation(self):
    quats = [p[-1] for p in self.poses]
    self.rots = Rotation.from_quat(quats)

    # Import here since older versions of scipy don't have RotationSpline.
    from scipy.spatial.transform import RotationSpline  # pylint: disable=g-import-not-at-top
    self.rot_spline = RotationSpline(self.times, self.rots)

  def _fit(self):
    self._interpolate_position()
    self._interpolate_rotation()

  def _generate_interpolants(self):
    xnew = []
    for i in range(len(self.times)):
      if i >= len(self.times) - 1:
        break
      linsp = np.linspace(
          self.times[i], self.times[i + 1], self.steps_per_seg, endpoint=False)
      xnew.extend(linsp)
    xnew.append(self.times[-1])
    return np.array(xnew)

  def __call__(self, *poses):
    self._preprocess_poses(*poses)
    assert self.poses is not None
    self.times = np.linspace(0, self.t_max, len(self.poses), endpoint=True)
    self._fit()
    xnew = self._generate_interpolants()
    xyz = np.vstack([self.pos_spline[i](xnew) for i in range(3)]).T
    quats = self.rot_spline(xnew).as_quat()
    poses = [(np.asarray(p, dtype=np.float32), np.asarray(q, dtype=np.float32))
             for p, q in zip(xyz, quats)]
    poses = self._postprocess_poses(poses, xnew)
    return poses

  # -------------------------------------------------------------------------
  # Plotting Functions
  # -------------------------------------------------------------------------

  def plot_xyz(self, xnew):
    """Plot xyz."""
    assert self.pos_spline is not None, "[!] You must first call plan()."
    names = ["x-coord", "y-coord", "z-coord"]
    xyzs = [p[0] for p in self.poses]
    _, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i in range(3):
      axes[i].plot(
          xnew,
          self.pos_spline[i](xnew),
          "--",
          self.times,
          [x[i] for x in xyzs],
          "o",
      )
      axes[i].set_title(names[i])
      axes[i].grid(linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

  def plot_quat(self, xnew):
    """Plot quaternion."""
    assert self.rot_spline is not None, "[!] You must first call plan()."
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xnew, self.rot_spline(xnew).as_euler("XYZ", degrees=True))
    axes[0].plot(self.times, self.rots.as_euler("XYZ", degrees=True), "x")
    axes[0].set_title("Euler angles")
    axes[1].plot(xnew, np.rad2deg(self.rot_spline(xnew, 1)))
    axes[1].plot(self.times, np.rad2deg(self.rot_spline(self.times, 1)), "x")
    axes[1].set_title("Angular rate")
    axes[0].grid(linestyle="--", linewidth=0.5)
    axes[1].grid(linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


class PickPlacePlanner(Planner):
  """A pick-and-place planner."""

  NUM_POSES = 7

  def _preprocess_poses(self, start_pose, pick_pose, place_pose):
    self.start_pose = start_pose

    # Slightly lower pick and place z-coords to make contact code register.
    mod = ((0, 0, -0.008), (0, 0, 0, 1))
    self.pick_pose = utils.multiply(pick_pose, mod)
    # self.place_pose = utils.multiply(place_pose, mod)
    self.place_pose = place_pose

    # Generate intermediate waypoints.
    prepick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
    self.prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
    postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
    self.postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
    preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
    self.preplace_pose = utils.multiply(place_pose, preplace_to_place)
    postplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
    self.postplace_pose = utils.multiply(place_pose, postplace_to_place)

    self.poses = [
        self.start_pose,
        self.prepick_pose,
        self.pick_pose,
        self.postpick_pose,
        self.preplace_pose,
        self.place_pose,
        self.postplace_pose,
    ]

  def _postprocess_poses(self, poses, xnew):
    """Add suction commands."""
    suction_idxs = [2, 5]
    suction_times = [self.times[i] for i in suction_idxs]
    suction_loc = [np.argwhere(xnew == s)[0][0] for s in suction_times]
    actions = []
    for i, pose in enumerate(poses):
      s = 0
      if i in suction_loc:
        s = 1
      actions.append({"move_cmd": pose, "suction_cmd": s})
    return actions


class PushPlanner(Planner):
  """A push planner."""

  NUM_POSES = 5

  def _preprocess_poses(self, start_pose, pushstart_pose, pushend_pose):
    # Adjust push start and end positions.
    pos0 = np.float32((pushstart_pose[0][0], pushstart_pose[0][1], 0.005))
    pos1 = np.float32((pushend_pose[0][0], pushend_pose[0][1], 0.005))
    vec = np.float32(pos1) - np.float32(pos0)
    length = np.linalg.norm(vec)
    vec = vec / length
    pos0 -= vec * 0.02
    pos1 -= vec * 0.05

    # Align spatula against push direction.
    theta = np.arctan2(vec[1], vec[0])
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

    over0 = (pos0[0], pos0[1], self.height)
    over1 = (pos1[0], pos1[1], self.height)

    self.poses = [
        start_pose,
        (over0, rot),
        (pos0, rot),
        (pos1, rot),
        (over1, rot),
    ]

  def _postprocess_poses(self, poses, xnew):
    """Add speed commands."""
    slowdown_idxs = [2, 3]
    slowdown_times = [self.times[i] for i in slowdown_idxs]
    slowdown_loc = [np.argwhere(xnew == s)[0][0] for s in slowdown_times]
    actions = []
    for i, pose in enumerate(poses):
      s = 0
      if i in slowdown_loc:
        s = 1
      actions.append({"move_cmd": pose, "slowdown_cmd": s})
    return actions

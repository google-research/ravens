import matplotlib.pyplot as plt
import numpy as np
from ravens.utils import utils
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, RotationSpline


class PathPlanner:
    def __init__(self, start_pose, pick_pose, place_pose, t_max=10.0, height=0.32):
        self.height = height
        self.start_pose = start_pose
        # Slightly lower pick_pose z-coord to make contact code register.
        mod = ((0, 0, -0.005), (0, 0, 0, 1))
        self.pick_pose = utils.multiply(pick_pose, mod)
        # Same for place_pose.
        self.place_pose = utils.multiply(place_pose, mod)

        # Generate intermediate waypoints.
        prepick_to_pick = ((0, 0, height), (0, 0, 0, 1))
        self.prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
        postpick_to_pick = ((0, 0, height), (0, 0, 0, 1))
        self.postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
        preplace_to_place = ((0, 0, height), (0, 0, 0, 1))
        self.preplace_pose = utils.multiply(place_pose, preplace_to_place)
        postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
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
        self.times = np.linspace(0, t_max, len(self.poses), endpoint=True)

        self.rots = None
        self.rot_spline = None
        self.pos_spline = None

    def _interpolate_position(self):
        xyzs = [p[0] for p in self.poses]
        self.pos_spline = [
            interp1d(self.times, [p[i] for p in xyzs], kind="linear") for i in range(3)
        ]

    def _interpolate_rotation(self):
        quats = [p[-1] for p in self.poses]
        self.rots = Rotation.from_quat(quats)
        self.rot_spline = RotationSpline(self.times, self.rots)

    def plan(self, times):
        self._interpolate_position()
        self._interpolate_rotation()
        xyz = np.vstack([self.pos_spline[i](times) for i in range(3)]).T
        quats = self.rot_spline(times).as_quat()
        poses = [(tuple(p), tuple(q)) for p, q in zip(xyz, quats)]
        return poses

    def plot_xyz(self, xnew):
        assert self.pos_spline is not None, "[!] You must first call plan()."
        names = ["x-coord", "y-coord", "z-coord"]
        xyzs = [p[0] for p in self.poses]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
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
        assert self.rot_spline is not None, "[!] You must first call plan()."
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
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

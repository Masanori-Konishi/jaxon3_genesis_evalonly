"""Keep private training metadata out of the public inference configuration."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from export_policy import make_config


class ExportConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "num_actions": 1, "dof_names": ["RLEG_JOINT4"],
            "default_joint_angles": {"RLEG_JOINT4": -0.21},
            "default_joint_armature": {"RLEG_JOINT4": 0.142},
            "default_kp": 200., "default_kd": 7., "ankle_kp": 40., "ankle_kd": 4.,
            "dt": 0.005, "substeps": 2, "base_init_pos": [0., 0., 1.05],
            "base_init_quat": [1., 0., 0., 0.], "action_scale": 0.5, "clip_actions": 100.,
            "termination_if_roll_greater_than": 20., "termination_if_pitch_greater_than": 45.,
            "episode_length_s": 40.,
            "reward_cfg": "PRIVATE_SENTINEL", "future_training_setting": "PRIVATE_SENTINEL",
            "model_dir": "/PRIVATE_SENTINEL", "terrain": {"secret": "PRIVATE_SENTINEL"},
        }
        self.obs = {"num_obs": 18, "obs_scales": {
            "lin_vel": 2., "ang_vel": 0.25, "dof_pos": 1., "dof_vel": 0.05,
            "unused": "PRIVATE_SENTINEL"}, "unused": "PRIVATE_SENTINEL"}

    def test_allowlist_drops_private_and_unknown_fields(self):
        result = make_config(self.env, self.obs, "robot.urdf")
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertEqual(result["kp"], [40.])
        self.assertEqual(result["kd"], [4.])
        self.assertEqual(result["armature"], [0.2 * 0.142])
        self.assertEqual(result["velocity_frame"], "body")
        self.assertEqual(result["urdf"], "models/urdf/robot.urdf")

    def test_reject_incompatible_observations(self):
        self.obs["num_obs"] = 19
        with self.assertRaises(ValueError):
            make_config(self.env, self.obs, "robot.urdf")

    def test_reject_incompatible_actions(self):
        self.env["num_actions"] = 2
        with self.assertRaises(ValueError):
            make_config(self.env, self.obs, "robot.urdf")


if __name__ == "__main__":
    unittest.main()

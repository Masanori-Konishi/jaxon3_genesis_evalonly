"""Flat-ground JAXON3 inference environment. No training API is required."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import genesis as gs
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat
import torch


class Jaxon3EvalEnv:
    def __init__(self, config, asset_root, num_envs=1, show_viewer=True,
                 command=(3.0, 0.0, 0.0), show_collision=False):
        self.cfg = config
        self.device = gs.device
        self.num_envs = num_envs
        self.dt = config["dt"]
        self.num_actions = len(config["joint_names"])
        self.num_obs = 15 + 3 * self.num_actions
        if config["num_obs"] != self.num_obs:
            raise ValueError("Unsupported observation layout: expected 15 + 3 * joint count")
        self.max_episode_steps = math.ceil(config["episode_length_s"] / self.dt)
        self.source = config["velocity_source"]
        self.frame = config["velocity_frame"]
        if self.source not in ("base", "head") or self.frame not in ("body", "heading"):
            raise ValueError("Unsupported velocity source/frame")
        cutoff = config["velocity_cutoff_hz"]
        delay = config["head_velocity_delay_s"]
        if cutoff < 0 or delay < 0 or self.dt <= 0:
            raise ValueError("Invalid velocity filter, delay or timestep")
        tau = 1 / (2 * 3.141592 * cutoff) if cutoff else 0.0
        self.alpha = self.dt / (self.dt + tau)
        self.delay_steps = int(round(delay / self.dt)) if self.source == "head" else 0
        urdf = Path(asset_root) / config["urdf"]
        limits = {j.get("name"): j.find("limit") for j in ET.parse(urdf).getroot().findall("joint")}
        self.effort = self.tensor([float(limits[n].get("effort")) for n in config["joint_names"]])
        self.kp = self.tensor(config["kp"])
        self.kd = self.tensor(config["kd"])
        self.default_pos = self.tensor(config["default_joint_positions"])
        self.base_init_pos = self.tensor(config["base_init_pos"])
        self.base_init_quat = self.tensor(config["base_init_quat"])
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=config["substeps"]),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True, tolerance=1e-5, max_collision_pairs=10),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.0, -5.0, 2.0), camera_lookat=(0.0, 0.0, 0.9),
                camera_fov=40, refresh_rate=int(1 / self.dt)),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        self.ground = self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf), pos=config["base_init_pos"], quat=config["base_init_quat"],
                links_to_keep=["lleg_end_coords", "rleg_end_coords", "larm_end_coords",
                               "rarm_end_coords", "head_end_coords"], convexify=True),
            surface=gs.surfaces.Default(vis_mode="collision" if show_collision else "visual"),
        )
        self.scene.build(n_envs=num_envs)
        self.ground.set_friction(1.0)
        self.robot.set_friction(1.0)
        self.dof_idx = torch.tensor([self.robot.get_joint(n).dof_start for n in config["joint_names"]],
                                    device=self.device, dtype=gs.tc_int)
        self.q_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6] +
                                  [self.robot.get_joint(n).q_start for n in config["joint_names"]],
                                  device=self.device, dtype=gs.tc_int)
        self.init_qpos = torch.cat((self.base_init_pos, self.base_init_quat, self.default_pos))
        self.robot.set_dofs_armature(self.tensor(config["armature"]), self.dof_idx)
        self.gravity = self.tensor([0.0, 0.0, -1.0]).repeat(num_envs, 1)
        scales = config["obs_scales"]
        self.command_scale = self.tensor([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]])
        self.target = self.tensor(command).repeat(num_envs, 1)
        self.commands = torch.zeros_like(self.target)
        self.velocity = torch.zeros_like(self.target)
        self.filtered_world = torch.zeros_like(self.target)
        self.history = torch.zeros((self.delay_steps + 1, num_envs, 3), device=self.device)
        self.history_index = 0
        self.actions = torch.zeros((num_envs, self.num_actions), device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reset()

    def tensor(self, value):
        return torch.tensor(value, dtype=gs.tc_float, device=self.device)

    def read_state(self):
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        inverse = inv_quat(self.base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inverse)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inverse)
        self.projected_gravity = transform_by_quat(self.gravity, inverse)
        self.dof_pos = self.robot.get_dofs_position(self.dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.dof_idx)

    def heading_velocity(self, velocity):
        w, x, y, z = self.base_quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx, vy, vz = velocity.unbind(-1)
        return torch.stack((c * vx + s * vy, -s * vx + c * vy, vz), dim=-1)

    def update_velocity(self):
        if self.source == "base" and self.frame == "body":
            self.velocity = self.alpha * self.base_lin_vel + (1 - self.alpha) * self.velocity
            return
        link = self.robot.get_link("head_end_coords") if self.source == "head" else self.robot.base_link
        self.filtered_world.mul_(1 - self.alpha).add_(link.get_vel(), alpha=self.alpha)
        velocity = self.filtered_world
        if self.source == "head":
            self.history[self.history_index].copy_(velocity)
            self.history_index = (self.history_index + 1) % len(self.history)
            velocity = self.history[self.history_index]
        self.velocity = (self.heading_velocity(velocity) if self.frame == "heading" else
                         transform_by_quat(velocity, inv_quat(self.base_quat)))

    def observations(self):
        s = self.cfg["obs_scales"]
        return torch.cat((
            self.base_ang_vel * s["ang_vel"], self.projected_gravity,
            self.target * self.command_scale, self.velocity * self.command_scale,
            self.commands * self.command_scale,
            (self.dof_pos - self.default_pos) * s["dof_pos"],
            self.dof_vel * s["dof_vel"], self.actions,
        ), dim=-1)

    def reset(self, mask=None):
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        count = int(mask.sum().item())
        if count:
            self.robot.set_qpos(self.init_qpos, qs_idx_local=self.q_idx,
                                envs_idx=mask, zero_velocity=False, skip_forward=False)
            self.robot.set_dofs_velocity(torch.zeros((count, self.robot.n_dofs), device=self.device),
                                        dofs_idx_local=slice(0, self.robot.n_dofs), envs_idx=mask)
            for buffer in (self.actions, self.last_actions, self.commands, self.velocity,
                           self.filtered_world, self.episode_steps):
                buffer[mask] = 0
            self.history[:, mask] = 0
            self.read_state()
        return self.observations()

    def action_to_torque(self, actions):
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError("Policy returned invalid actions")
        self.actions = actions.clamp(-self.cfg["clip_actions"], self.cfg["clip_actions"])
        # The original controller always executes the preceding step's action.
        offset = self.last_actions * self.cfg["action_scale"]
        return (-self.kp * (self.dof_pos - self.default_pos - offset) - self.kd * self.dof_vel).clamp(-self.effort, self.effort)

    def step(self, actions):
        self.robot.control_dofs_force(self.action_to_torque(actions), self.dof_idx)
        self.scene.step()
        self.episode_steps += 1
        self.read_state()
        angles = quat_to_xyz(transform_quat_by_quat(inv_quat(self.base_init_quat), self.base_quat),
                             rpy=True, degrees=True)
        done = ((angles[:, 0].abs() > self.cfg["max_roll_deg"]) |
                (angles[:, 1].abs() > self.cfg["max_pitch_deg"]) |
                (self.episode_steps > self.max_episode_steps))
        self.reset(done)
        max_delta = self.dt * self.tensor([3.0, 1.0, 1.0])
        self.commands += (self.target - self.commands).clamp(-max_delta, max_delta)
        self.update_velocity()
        obs = self.observations()
        self.last_actions = torch.where(done[:, None], self.last_actions, self.actions)
        if not torch.isfinite(obs).all():
            raise RuntimeError("Non-finite simulation observations")
        return obs, done

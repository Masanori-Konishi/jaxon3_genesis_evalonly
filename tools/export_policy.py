#!/usr/bin/env python3
"""Publisher-side conversion of a trusted local checkpoint into an inference bundle."""

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import shutil
import xml.etree.ElementTree as ET


def make_config(env, obs, urdf_name):
    """Build an explicit allowlist; never serialize the input configuration."""
    names = env["dof_names"]
    if len(names) != env["num_actions"] or len(set(names)) != len(names):
        raise ValueError("Joint/action count mismatch")
    if obs["num_obs"] != 15 + 3 * len(names):
        raise ValueError("Only the running observation layout is supported")
    kp, kd = [], []
    for name in names:
        group = None
        if name in ("LLEG_JOINT4", "LLEG_JOINT5", "RLEG_JOINT4", "RLEG_JOINT5"):
            group = "ankle"
        elif name in ("LLEG_JOINT3", "RLEG_JOINT3"):
            group = "knee"
        elif name in [f"{side}LEG_JOINT{i}" for side in ("L", "R") for i in range(3)]:
            group = "hip"
        kp.append(env.get(f"{group}_kp") or env["default_kp"])
        kd.append(env.get(f"{group}_kd") or env["default_kd"])
    return {
        "format_version": 1,
        "urdf": f"models/urdf/{urdf_name}",
        "dt": env["dt"], "substeps": env["substeps"],
        "joint_names": names,
        "default_joint_positions": [env["default_joint_angles"][n] for n in names],
        "kp": kp, "kd": kd,
        "armature": [0.2 * env["default_joint_armature"][n] for n in names],
        "base_init_pos": env["base_init_pos"], "base_init_quat": env["base_init_quat"],
        "action_scale": env["action_scale"], "clip_actions": env["clip_actions"],
        "num_obs": obs["num_obs"],
        "obs_scales": {k: obs["obs_scales"][k] for k in ("lin_vel", "ang_vel", "dof_pos", "dof_vel")},
        "velocity_source": env.get("velocity_observation_source", "base"),
        "velocity_frame": env.get("velocity_observation_frame", "body"),
        "velocity_cutoff_hz": env.get("velocity_observation_cutoff_hz", 4.0),
        "head_velocity_delay_s": env.get("head_velocity_delay_s", 0.0),
        "max_roll_deg": env["termination_if_roll_greater_than"],
        "max_pitch_deg": env["termination_if_pitch_greater_than"],
        "episode_length_s": env["episode_length_s"],
    }


def copy_assets(urdf, output):
    """Copy only the selected URDF and referenced mesh files, keeping relative paths."""
    target = output / "models" / "urdf" / urdf.name
    target.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(urdf)
    paths = []
    for elem in tree.getroot().iter("mesh"):
        reference = elem.get("filename")
        relative = Path(reference)
        if relative.is_absolute() or ":" in reference:
            raise ValueError(f"Only relative local mesh paths are supported: {reference}")
        source = (urdf.parent / relative).resolve()
        destination = (target.parent / relative).resolve()
        if not destination.is_relative_to((output / "models").resolve()):
            raise ValueError(f"Mesh path escapes the model directory: {reference}")
        if source.suffix.lower() == ".dae":
            mesh = ET.parse(source)
            if any(e.tag.endswith("}library_images") and len(e) for e in mesh.getroot()):
                raise ValueError(f"Textured DAE requires separate texture packaging: {source.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        paths.append(destination)
    # Drop XML comments, including inactive geometry, from the distributed URDF.
    tree.write(target, encoding="utf-8", xml_declaration=True)
    paths.append(target)
    return {str(p.relative_to(output.resolve())): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(p.resolve() for p in paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Trusted cfgs.pkl; defaults to checkpoint sibling")
    parser.add_argument("--output", type=Path, required=True, help="A new, non-existing bundle directory")
    parser.add_argument("--urdf-index", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, help="Override the local source URDF directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Choose a new directory to avoid mixing policy/config/assets.")
    if version("rsl-rl-lib") != "5.5.1":
        parser.error("Export requires rsl-rl-lib==5.5.1 in the publisher's environment")

    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel

    # Pickle is read only by the publisher from their own trusted training output.
    with (args.config or args.checkpoint.parent / "cfgs.pkl").open("rb") as f:
        env, obs, _, _, train = pickle.load(f)
    if env.get("task", "running") != "running":
        parser.error("Only running checkpoints are supported")
    actor_cfg = copy.deepcopy(train.get("actor", {}))
    if actor_cfg.pop("class_name", None) != "rsl_rl.models:MLPModel":
        parser.error("Expected a 5.5.1 MLP actor; old or recurrent checkpoints are unsupported")
    if train.get("obs_groups", {}).get("actor") != ["policy"]:
        parser.error("Only the single policy observation group is supported")
    dist = actor_cfg.get("distribution_cfg")
    if dist and dist.get("class_name") != "rsl_rl.modules:GaussianDistribution":
        parser.error("Unsupported policy distribution")
    names = env.get("model_names") or [env["model_name"]]
    if not 0 <= args.urdf_index < len(names):
        parser.error("URDF index out of range")
    urdf = (args.model_dir or Path(env["model_dir"])) / names[args.urdf_index]
    config = make_config(env, obs, urdf.name)
    example = TensorDict({"policy": torch.zeros(1, obs["num_obs"])}, batch_size=[1])
    actor = MLPModel(example, {"actor": ["policy"]}, "actor", env["num_actions"], **actor_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "actor_state_dict" not in checkpoint:
        parser.error("Expected a 5.5.1 actor checkpoint")
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    exported = torch.jit.script(actor.as_jit().eval())
    # Exercise normalization and dynamic batch sizes before writing any artifact.
    with torch.inference_mode():
        for batch in (1, 4):
            x = torch.randn(batch, obs["num_obs"])
            expected = actor(TensorDict({"policy": x}, batch_size=[batch]))
            torch.testing.assert_close(exported(x), expected, rtol=1e-6, atol=1e-6)
    args.output.mkdir(parents=True)
    config["asset_sha256"] = copy_assets(urdf, args.output)
    path = args.output / "policy.pt"
    exported.save(str(path))
    config["policy_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "eval_config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    print(f"Exported {args.output}: policy.pt, eval_config.json, models/")
    print("Actor output matches the original (batches 1 and 4).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run an exported JAXON3 policy on a flat plane."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent / "policy")
    parser.add_argument("--command", type=float, nargs=3, default=(3.0, 0.0, 0.0), metavar=("VX", "VY", "YAW"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0: run until Ctrl+C or viewer closure")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 0 or not all(map(math.isfinite, args.command)):
        parser.error("Invalid environment count, step count, or command")
    cfg_path, policy_path = args.bundle / "eval_config.json", args.bundle / "policy.pt"
    if not cfg_path.is_file() or not policy_path.is_file():
        parser.error(f"Place the matching policy.pt and eval_config.json in {args.bundle}")
    config = json.loads(cfg_path.read_text())
    if config.get("format_version") != 1:
        parser.error("Unsupported bundle version")
    if hashlib.sha256(policy_path.read_bytes()).hexdigest() != config["policy_sha256"]:
        parser.error("policy.pt does not match eval_config.json; use the exported pair")
    for name, expected in config["asset_sha256"].items():
        asset = args.bundle / name
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != expected:
            parser.error(f"Missing or modified model asset: {asset}")

    import genesis as gs
    import torch
    from jaxon3_env_evalonly import Jaxon3EvalEnv

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check the driver/PyTorch installation, or use --device cpu")
    gs.init(backend=gs.cuda if args.device == "cuda" else gs.cpu, seed=args.seed, logging_level="warning")
    try:
        policy = torch.jit.load(str(policy_path), map_location=gs.device).eval()
        with torch.inference_mode():
            probe = policy(torch.zeros(args.num_envs, config["num_obs"], device=gs.device))
            if probe.shape != (args.num_envs, len(config["joint_names"])):
                raise ValueError("Policy dimensions do not match the evaluation configuration")
            env = Jaxon3EvalEnv(config, args.bundle, args.num_envs, not args.headless,
                               args.command, args.show_collision)
            obs = env.observations()
            steps = resets = 0
            previous_pos = env.base_pos[0].cpu().numpy().copy()
            started = time.monotonic()
            while args.max_steps == 0 or steps < args.max_steps:
                if not args.headless and not env.scene.viewer.is_alive():
                    break
                tick = time.monotonic()
                obs, done = env.step(policy(obs))
                steps += 1
                resets += int(done.sum().item())
                if not args.headless:
                    pos = env.base_pos[0].cpu().numpy().copy()
                    if not args.fixed_camera:
                        camera = env.scene.viewer.camera_pose
                        # Keep the look-at point attached to the moving robot.
                        env.scene.viewer.set_camera_pose(pos=camera[:3, 3] + pos - previous_pos,
                                                         lookat=pos)
                    previous_pos = pos
                    time.sleep(max(0.0, env.dt - (time.monotonic() - tick)))
            print(f"Completed {steps} steps ({steps * env.dt:.3f} simulated seconds), "
                  f"{resets} resets, {time.monotonic() - started:.1f}s wall time.")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()

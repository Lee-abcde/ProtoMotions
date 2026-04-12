# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Standalone MuJoCo inference test for VQ-PAE ONNX policies."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deployment.motion_utils import MotionPlayer
from deployment.state_utils import (
    apply_heading_offset_np,
    compute_anchor_rot_np,
    compute_yaw_offset_np,
    _quat_conjugate_np,
    _quat_mul_np,
)
from deployment.test_tracker_mujoco import (
    _load_reset_noise_from_checkpoint,
    load_mujoco_model,
    read_robot_state,
    set_initial_pose,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


def _quat_rotate_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by xyzw quaternion(s)."""
    q = np.asarray(q_xyzw, dtype=np.float32)
    vec = np.asarray(v, dtype=np.float32)
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_vec, vec)
    return vec + q_w * t + np.cross(q_vec, t)


class HistoryBuffer:
    """Fixed-length history buffer for VQ-PAE deployment inputs."""

    def __init__(
        self,
        history_steps: int,
        num_dofs: int,
        init_state: dict,
        init_actions: np.ndarray | None = None,
    ):
        self.history_steps = history_steps
        self._dof_pos = deque(maxlen=history_steps)
        self._dof_vel = deque(maxlen=history_steps)
        self._root_local_ang_vel = deque(maxlen=history_steps)
        self._processed_actions = deque(maxlen=history_steps)

        actions = (
            np.zeros(num_dofs, dtype=np.float32)
            if init_actions is None
            else init_actions.astype(np.float32).copy()
        )
        for _ in range(history_steps):
            self._dof_pos.append(init_state["dof_pos"].astype(np.float32).copy())
            self._dof_vel.append(init_state["dof_vel"].astype(np.float32).copy())
            self._root_local_ang_vel.append(
                init_state["root_local_ang_vel"].astype(np.float32).copy()
            )
            self._processed_actions.append(actions.copy())

    def append(self, robot_state: dict, processed_actions: np.ndarray) -> None:
        self._dof_pos.append(robot_state["dof_pos"].astype(np.float32).copy())
        self._dof_vel.append(robot_state["dof_vel"].astype(np.float32).copy())
        self._root_local_ang_vel.append(
            robot_state["root_local_ang_vel"].astype(np.float32).copy()
        )
        self._processed_actions.append(processed_actions.astype(np.float32).copy())

    def export(self) -> dict[str, np.ndarray]:
        return {
            "historical.dof_pos": np.stack(list(self._dof_pos), axis=0)[None],
            "historical.dof_vel": np.stack(list(self._dof_vel), axis=0)[None],
            "historical.root_local_ang_vel": np.stack(
                list(self._root_local_ang_vel), axis=0
            )[None],
            "historical.processed_actions": np.stack(
                list(self._processed_actions), axis=0
            )[None],
        }


def build_onnx_inputs(
    robot_state: dict,
    history: HistoryBuffer,
    future_refs: dict,
    onnx_name_to_key: dict[str, str],
    anchor_body_index: int,
) -> dict[str, np.ndarray]:
    anchor_rot = compute_anchor_rot_np(robot_state["body_rot"], anchor_body_index)
    future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]
    future_anchor_ang_vel = future_refs["body_ang_vel"][:, anchor_body_index, :]

    key_to_array = {
        "current.anchor_rot": anchor_rot[None].astype(np.float32),
        "current.dof_pos": robot_state["dof_pos"][None].astype(np.float32),
        "current.dof_vel": robot_state["dof_vel"][None].astype(np.float32),
        "current.root_local_ang_vel": robot_state["root_local_ang_vel"][None].astype(
            np.float32
        ),
        "mimic.future_anchor_rot": future_anchor_rot[None].astype(np.float32),
        "mimic.future_anchor_ang_vel": future_anchor_ang_vel[None].astype(np.float32),
        "mimic.future_dof_pos": future_refs["dof_pos"][None].astype(np.float32),
        "mimic.future_dof_vel": future_refs["dof_vel"][None].astype(np.float32),
        **history.export(),
    }

    onnx_inputs: dict[str, np.ndarray] = {}
    for onnx_name, semantic_key in onnx_name_to_key.items():
        value = key_to_array.get(semantic_key)
        if value is None:
            log.warning(
                f"No value for ONNX input '{onnx_name}' (semantic key '{semantic_key}')"
            )
            continue
        onnx_inputs[onnx_name] = value
    return onnx_inputs


def run(
    onnx_path: str,
    motion_file: str,
    motion_index: int = 0,
    cache_motion: bool = False,
    num_loops: int = 1,
    render: bool = False,
    realtime: bool = True,
    action_ema_alpha: float | None = None,
    zero_init_vel: bool = True,
    use_checkpoint_reset_noise: bool = False,
) -> None:
    onnx_path = str(onnx_path)
    yaml_path = onnx_path.replace(".onnx", ".yaml")

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    robot_meta = meta["robot"]
    timing = meta["timing"]
    motion_meta = meta["motion"]
    control = meta["control"]
    runtime = meta["_runtime"]
    reset_noise_cfg = (
        _load_reset_noise_from_checkpoint(meta) if use_checkpoint_reset_noise else None
    )

    anchor_body_index = robot_meta["anchor_body_index"]
    root_body_index = robot_meta["root_body_index"]
    num_dofs = robot_meta["num_dofs"]
    mjcf_path = robot_meta["mjcf_path"]
    control_dt = timing["control_dt"]
    decimation = timing["decimation"]
    future_step_indices = motion_meta["future_step_indices"]
    stiffness = control["stiffness"]
    damping = control["damping"]
    pd_target_max_accel = control.get("pd_target_max_accel")
    if action_ema_alpha is None:
        action_ema_alpha = control.get("action_ema_alpha", 1.0)
    onnx_name_to_key = runtime["onnx_name_to_in_key"]

    history_steps = 1
    for policy_input in meta.get("policy_inputs", []):
        shape = policy_input.get("shape")
        key = policy_input.get("key")
        if key and key.startswith("historical.") and shape and len(shape) >= 3:
            history_steps = max(history_steps, int(shape[1]))

    log.info(f"ONNX: {onnx_path}")
    log.info(
        f"Robot: {robot_meta['num_dofs']} DOFs, anchor={robot_meta['anchor_body_name']}[{anchor_body_index}], "
        f"root={robot_meta['root_body_name']}[{root_body_index}]"
    )
    log.info(
        f"control_dt={control_dt}s, decimation={decimation}, history_steps={history_steps}"
    )
    log.info(f"Future steps: {future_step_indices}")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]
    log.info(f"ONNX inputs:  {actual_in_names}")
    log.info(f"ONNX outputs: {actual_out_names}")

    player = MotionPlayer(motion_file, motion_index=motion_index, control_dt=control_dt)
    if cache_motion:
        motion_p = Path(motion_file)
        cache_p = motion_p.parent / (motion_p.stem + ".50fps.pt")
        if not os.access(str(cache_p.parent), os.W_OK):
            cache_p = Path(onnx_path).parent / (motion_p.stem + ".50fps.pt")
        if not cache_p.exists():
            player.cache_to_file(str(cache_p))
        else:
            log.info(f"Cache already exists: {cache_p}")

    physics_dt = timing["physics_dt"]
    model, data = load_mujoco_model(mjcf_path, stiffness, damping, physics_dt)

    viewer = None
    if render:
        try:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            )
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 180.0
            viewer.cam.trackbodyid = 1
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            log.info("Viewer launched (tracking pelvis)")
        except Exception as exc:
            log.warning(f"Could not launch viewer: {exc}")

    use_ema = action_ema_alpha < 1.0
    total_steps = 0
    total_ort_ms = 0.0
    total_sim_ms = 0.0
    max_ref_err = 0.0

    prev_pd: np.ndarray | None = None
    prev_prev_pd: np.ndarray | None = None
    ema_prev_targets: np.ndarray | None = None

    loop_idx = 0
    while loop_idx < num_loops:
        log.info(f"\n--- Loop {loop_idx + 1}/{num_loops} ---")
        set_initial_pose(
            model,
            data,
            player,
            zero_init_vel=zero_init_vel,
            reset_noise_cfg=reset_noise_cfg,
        )
        initial_state = read_robot_state(data, anchor_body_index, root_body_index)
        history = HistoryBuffer(history_steps, num_dofs, initial_state)
        prev_pd = None
        prev_prev_pd = None
        ema_prev_targets = None
        heading_offset = None
        loop_wall_start = time.perf_counter()

        for frame_idx in range(player.total_frames):
            step_wall_start = time.perf_counter()
            robot_state = read_robot_state(data, anchor_body_index, root_body_index)

            if heading_offset is None:
                robot_anchor_rot = robot_state["body_rot"][anchor_body_index]
                motion_anchor_rot = player.get_state_at_frame(0)["body_rot"][
                    anchor_body_index
                ]
                heading_offset = compute_yaw_offset_np(
                    robot_anchor_rot, motion_anchor_rot
                )

            future_refs = player.get_future_references(frame_idx, future_step_indices)
            future_refs["body_rot"] = apply_heading_offset_np(
                heading_offset, future_refs["body_rot"]
            )
            future_refs["body_ang_vel"] = _quat_rotate_np(
                np.broadcast_to(heading_offset, future_refs["body_ang_vel"].shape[:-1] + (4,)),
                future_refs["body_ang_vel"],
            ).astype(np.float32)

            onnx_inputs = build_onnx_inputs(
                robot_state=robot_state,
                history=history,
                future_refs=future_refs,
                onnx_name_to_key=onnx_name_to_key,
                anchor_body_index=anchor_body_index,
            )

            t0 = time.perf_counter()
            ort_out = session.run(actual_out_names, onnx_inputs)
            total_ort_ms += (time.perf_counter() - t0) * 1000.0

            pd_targets = ort_out[1].squeeze().copy()
            if (
                pd_target_max_accel is not None
                and prev_pd is not None
                and prev_prev_pd is not None
            ):
                delta = pd_targets - prev_pd
                prev_delta = prev_pd - prev_prev_pd
                accel = delta - prev_delta
                pd_targets = prev_pd + prev_delta + np.clip(
                    accel, -pd_target_max_accel, pd_target_max_accel
                )

            prev_prev_pd = prev_pd
            prev_pd = pd_targets.copy()

            if use_ema:
                if ema_prev_targets is None:
                    ema_prev_targets = pd_targets.copy()
                pd_targets = (
                    action_ema_alpha * pd_targets
                    + (1.0 - action_ema_alpha) * ema_prev_targets
                )
                ema_prev_targets = pd_targets.copy()

            history.append(robot_state, pd_targets)
            data.ctrl[:] = pd_targets

            t0 = time.perf_counter()
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            total_sim_ms += (time.perf_counter() - t0) * 1000.0

            ref_dof_pos = player.get_state_at_frame(frame_idx)["dof_pos"]
            max_ref_err = max(max_ref_err, float(np.abs(data.qpos[7:] - ref_dof_pos).max()))

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()

            if realtime:
                elapsed = time.perf_counter() - step_wall_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_steps += 1
            if frame_idx % 100 == 0:
                wall_elapsed = time.perf_counter() - loop_wall_start
                sim_elapsed = (frame_idx + 1) * control_dt
                speed_ratio = sim_elapsed / max(wall_elapsed, 1e-6)
                log.info(
                    f"  step={total_steps:5d}  frame={frame_idx:4d}  "
                    f"root_h={float(data.qpos[2]):.3f}  max_ref_err={max_ref_err:.4f}  "
                    f"speed={speed_ratio:.2f}x"
                )

        loop_idx += 1
        if viewer is not None and not viewer.is_running():
            break

    log.info(
        f"\n=== Done: {total_steps} steps over {loop_idx} loop(s) ===\n"
        f"  avg ONNX inference : {total_ort_ms / max(total_steps, 1):.2f} ms/step\n"
        f"  avg physics        : {total_sim_ms / max(total_steps, 1):.2f} ms/step\n"
        f"  max joint ref error: {max_ref_err:.4f} rad"
    )

    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run VQ-PAE ONNX policy in MuJoCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx", required=True, help="Path to unified_pipeline.onnx")
    p.add_argument(
        "--motion",
        required=True,
        help="Path to motion .pt file (raw ProtoMotions or pre-cached)",
    )
    p.add_argument(
        "--motion-index",
        type=int,
        default=0,
        help="Motion index to load when --motion points to a packaged multi-motion .pt library.",
    )
    p.add_argument(
        "--cache-motion",
        action="store_true",
        default=False,
        help="Write a 50fps cache next to the motion after first load.",
    )
    p.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of times to loop the motion (default: infinite with --render, 1 otherwise)",
    )
    p.add_argument("--render", action="store_true", default=False, help="Open viewer")
    p.add_argument(
        "--no-realtime",
        action="store_true",
        default=False,
        help="Disable real-time pacing",
    )
    p.add_argument(
        "--action-ema-alpha",
        type=float,
        default=None,
        help="Override control.action_ema_alpha from YAML",
    )
    p.add_argument(
        "--stop-zero-velo-init",
        action="store_true",
        default=False,
        help="Use the first motion-frame velocities instead of the default zero qvel initialisation.",
    )
    p.add_argument(
        "--use-checkpoint-reset-noise",
        action="store_true",
        default=False,
        help="Apply reset noise from the source checkpoint's robot.reset_noise config.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    num_loops = args.loops if args.loops is not None else (10_000_000 if args.render else 1)
    run(
        onnx_path=args.onnx,
        motion_file=args.motion,
        motion_index=args.motion_index,
        cache_motion=args.cache_motion,
        num_loops=num_loops,
        render=args.render,
        realtime=not args.no_realtime,
        action_ema_alpha=args.action_ema_alpha,
        zero_init_vel=not args.stop_zero_velo_init,
        use_checkpoint_reset_noise=args.use_checkpoint_reset_noise,
    )
    os._exit(0)

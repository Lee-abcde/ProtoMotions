# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recording and rendering logic extracted from the Simulator base class.

This module contains the RecordingMixin class which provides video recording,
frame capture, marker management, and motion/object serialization functionality.
The mixin is designed to be used with the Simulator class, accessing simulator
state and methods via self.
"""

from collections import deque
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch

from protomotions.utils import rotations
from protomotions.simulator.base_simulator.config import MarkerState
from protomotions.simulator.base_simulator.utils import build_motion_data

log = logging.getLogger(__name__)


class RecordingMixin:
    """Mixin providing recording and rendering capabilities for simulators.

    This mixin expects the following attributes/methods to be provided by the
    host class (Simulator):
        - self.headless, self.config, self.scene_lib, self.num_envs
        - self._num_dof, self._proj_config
        - self._original_marker_configs
        - self.get_robot_state(), self.get_object_root_state()
        - self._get_projectile_positions_rotations()
        - self._write_viewport_to_file(file_name)
        - self._update_simulator_markers(markers_state)
    """

    # -------------------------
    # Initialization
    # -------------------------

    def _init_recording_state(self) -> None:
        """Initialize all recording-related attributes."""
        self._camera_target: Dict[str, int] = {"env": 0, "element": 0}
        self._show_markers: bool = True

        self._user_is_recording, self._user_recording_state_change = False, False
        self._user_recording_video_queue_size = 100000
        self._delete_user_viewer_recordings = False
        os.makedirs("output/renderings", exist_ok=True)
        self._user_recording_video_path = os.path.join(
            "output/renderings", f"{self.config.experiment_name}-%s"
        )
        self._recording_output_path_override: Optional[str] = None
        self._recording_text_overlay: Optional[str] = None
        self._recording_fps: int = 30
        self._overlay_warning_emitted: bool = False
        self._curr_user_recording_video_path: Optional[str] = None
        self._curr_user_recording_artifact_base: Optional[str] = None
        self._last_recorded_video_path: Optional[str] = None
        self._last_recorded_motion_path: Optional[str] = None

        # Last markers state for recording (set each step)
        self._last_markers_state: Optional[Dict[str, MarkerState]] = None

    # -------------------------
    # Recording state control
    # -------------------------

    def _toggle_video_record(self):
        self._user_is_recording = not self._user_is_recording
        self._user_recording_state_change = True

    def _cancel_video_record(self):
        self._user_is_recording = False
        self._user_recording_state_change = False
        self._delete_user_viewer_recordings = True

    def start_video_recording(
        self,
        output_path: Optional[str] = None,
        text_overlay: Optional[str] = None,
        fps: int = 30,
    ) -> None:
        """Start viewer recording from code instead of a keyboard event."""
        if self.headless:
            raise RuntimeError("Video recording requires a non-headless simulator.")
        if self._user_is_recording:
            raise RuntimeError("Video recording is already active.")

        self._recording_output_path_override = (
            str(Path(output_path)) if output_path is not None else None
        )
        self._recording_text_overlay = text_overlay
        self._recording_fps = max(1, int(fps))
        self._toggle_video_record()

    def stop_video_recording(self) -> Optional[str]:
        """Stop viewer recording and finalize the video immediately."""
        if not self._user_is_recording:
            return self._last_recorded_video_path

        self._toggle_video_record()
        self.render()
        return self._last_recorded_video_path

    def set_video_recording_text_overlay(
        self, text_overlay: Optional[str] = None
    ) -> None:
        """Update the text overlay used for subsequently recorded frames."""
        self._recording_text_overlay = text_overlay

    def _resolve_recording_paths(self) -> None:
        if self._recording_output_path_override is None:
            curr_date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
            recording_base = self._user_recording_video_path % curr_date_time
            self._curr_user_recording_name = recording_base
            self._curr_user_recording_video_path = f"{recording_base}.mp4"
            self._curr_user_recording_artifact_base = recording_base
            return

        output_path = Path(self._recording_output_path_override)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_path.parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_base = artifacts_dir / output_path.with_suffix("").name
        frames_dir = artifacts_dir / f".{artifact_base.name}_frames_{os.getpid()}"
        self._curr_user_recording_name = str(frames_dir)
        self._curr_user_recording_video_path = str(output_path)
        self._curr_user_recording_artifact_base = str(artifact_base)

    def _wrap_text_to_width(self, draw, text: str, font, max_width: int) -> List[str]:
        words = text.split()
        if not words:
            return []

        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _apply_text_overlay_to_image(self, file_name: str) -> None:
        if not self._recording_text_overlay:
            return

        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            if not self._overlay_warning_emitted:
                print(f"Warning: failed to import PIL for video text overlay: {exc}")
                self._overlay_warning_emitted = True
            return

        try:
            image = Image.open(file_name).convert("RGBA")
            width = image.size[0]
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            font_size = max(18, min(34, int(width * 0.035)))
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

            margin = max(14, int(width * 0.02))
            padding = max(8, int(width * 0.012))
            max_text_width = int(width * 0.82)
            lines = self._wrap_text_to_width(
                draw, str(self._recording_text_overlay), font, max_text_width
            )
            if not lines:
                return

            line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
            line_width = max(box[2] - box[0] for box in line_boxes)
            line_height = max(box[3] - box[1] for box in line_boxes)
            line_gap = max(4, int(font_size * 0.25))
            box_width = line_width + padding * 2
            box_height = (
                len(lines) * line_height
                + (len(lines) - 1) * line_gap
                + padding * 2
            )

            draw.rectangle(
                [margin, margin, margin + box_width, margin + box_height],
                fill=(0, 0, 0, 170),
            )
            y = margin + padding
            for line in lines:
                draw.text(
                    (margin + padding, y),
                    line,
                    fill=(255, 255, 255, 255),
                    font=font,
                )
                y += line_height + line_gap

            Image.alpha_composite(image, overlay).convert("RGB").save(file_name)
        except Exception as exc:
            if not self._overlay_warning_emitted:
                print(f"Warning: failed to apply video text overlay: {exc}")
                self._overlay_warning_emitted = True

    # -------------------------
    # Camera target
    # -------------------------

    def _toggle_camera_target(self) -> None:
        """
        Toggle the camera target between different environments and objects.

        The target cycles through all objects in the scene, with 0 referring to the environment.
        """
        if self.scene_lib.num_objects_per_scene > 0:
            self._camera_target["element"] = (self._camera_target["element"] + 1) % (
                self.scene_lib.num_objects_per_scene + 1
            )
            print("Updated camera target to element", self._camera_target["element"])

        if self._camera_target["element"] == 0:
            self._camera_target["env"] = (
                self._camera_target["env"] + 1
            ) % self.num_envs
            print("Updated camera target to env", self._camera_target["env"])

    # -------------------------
    # Marker management
    # -------------------------

    def _toggle_markers(self):
        self._show_markers = not self._show_markers
        print(f"Markers are now {'visible' if self._show_markers else 'hidden'}")

    def _update_markers(
        self, markers_state: Optional[Dict[str, MarkerState]] = None
    ) -> None:
        """
        Update visualization markers for the simulator.

        Converts marker orientations if necessary and delegates to the simulator-specific update.

        Args:
            markers_state (Dict[str, MarkerState]): Dictionary containing marker states.
        """

        if not markers_state or len(markers_state) == 0:
            return

        if not self.config.w_last:
            for key in markers_state.keys():
                markers_state[key].orientation = rotations.xyzw_to_wxyz(
                    markers_state[key].orientation
                )
        if not self._show_markers:
            for key in markers_state.keys():
                # Throw it out of view
                markers_state[key].translation = (
                    torch.zeros_like(markers_state[key].translation) - 1000000
                )
        self._update_simulator_markers(markers_state)

    def _build_markers_save_data(self) -> dict:
        """Build markers data dictionary for saving to .markers.pt file."""
        markers_data = {"fps": 30, "markers": {}}
        for name, frame_list in self._recorded_markers.items():
            translations = torch.stack([f[0] for f in frame_list], dim=0)
            orientations = torch.stack([f[1] for f in frame_list], dim=0)
            # Get marker config metadata from the original (pre-simulator)
            # configs, since simulator-specific init may wrap/replace them
            marker_config = self._original_marker_configs.get(name)
            marker_type = "sphere"
            color = (1.0, 0.0, 0.0)
            sizes = []
            if marker_config is not None:
                marker_type = marker_config.type
                color = marker_config.color
                sizes = [m.size for m in marker_config.markers]

            markers_data["markers"][name] = {
                "type": marker_type,
                "color": color,
                "sizes": sizes,
                "translation": translations,
                "orientation": orientations,
            }
        return markers_data

    # -------------------------
    # Object serialization
    # -------------------------

    def _build_terrain_save_data(self) -> Optional[dict]:
        """Build terrain data dictionary for saving to .terrain.pt file.

        Returns None if terrain is flat or not available.
        """
        terrain = getattr(self, "terrain", None)
        if terrain is None:
            return None

        # Skip saving for flat terrains — Blender's default ground plane suffices
        if terrain.is_flat():
            return None

        return {
            "height_field_raw": terrain.height_field_raw,
            "horizontal_scale": terrain.horizontal_scale,
            "vertical_scale": terrain.vertical_scale,
        }

    def _build_objects_save_data(self) -> dict:
        """Build objects data dictionary for saving to .objects.pt file."""
        objects_list = []

        # Scene objects
        if self._recorded_objects:
            from protomotions.components.scene_lib import (
                BoxSceneObject,
                SphereSceneObject,
                CylinderSceneObject,
                MeshSceneObject,
            )

            translations = torch.stack([f[0] for f in self._recorded_objects], dim=0)
            rotations = torch.stack([f[1] for f in self._recorded_objects], dim=0)

            eid = self._recording_env_id
            scene_idx = self.scene_lib._scene_to_original_scene_id[eid].item()
            scene = self.scene_lib._original_scenes[scene_idx]

            for obj_idx, obj in enumerate(scene.objects):
                obj_info = {
                    "name": f"object_{obj_idx}",
                    "translation": translations[:, obj_idx, :],
                    "rotation": rotations[:, obj_idx, :],
                }
                if isinstance(obj, BoxSceneObject):
                    obj_info["shape"] = "box"
                    obj_info["size"] = [obj.width, obj.depth, obj.height]
                elif isinstance(obj, SphereSceneObject):
                    obj_info["shape"] = "sphere"
                    obj_info["size"] = [obj.radius]
                elif isinstance(obj, CylinderSceneObject):
                    obj_info["shape"] = "cylinder"
                    obj_info["size"] = [obj.radius, obj.height]
                elif isinstance(obj, MeshSceneObject):
                    obj_info["shape"] = "mesh"
                    obj_info["size"] = []
                    obj_info["mesh_path"] = obj.object_path
                    obj_info["scale"] = list(obj.scale)
                else:
                    obj_info["shape"] = "box"
                    dims = obj.object_dims
                    if dims is not None:
                        obj_info["size"] = [
                            dims[1] - dims[0],
                            dims[3] - dims[2],
                            dims[5] - dims[4],
                        ]
                    else:
                        obj_info["size"] = [0.1, 0.1, 0.1]

                objects_list.append(obj_info)

        # Projectiles
        if self._recorded_projectiles:
            proj_pos = torch.stack(
                [f[0] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 3]
            proj_rot = torch.stack(
                [f[1] for f in self._recorded_projectiles], dim=0
            )  # [num_frames, num_proj, 4]

            half_sizes = self._proj_config.get_sizes()
            hide_z = self._proj_config.hide_z

            for p in range(proj_pos.shape[1]):
                # Only include projectiles that were visible at some point
                if (proj_pos[:, p, 2] > hide_z + 0.5).any():
                    hs = half_sizes[p]
                    full_size = hs * 2
                    objects_list.append(
                        {
                            "name": f"projectile_{p}",
                            "shape": "box",
                            "size": [full_size, full_size, full_size],
                            "translation": proj_pos[:, p, :],
                            "rotation": proj_rot[:, p, :],
                        }
                    )

        return {"fps": 30, "objects": objects_list}

    # -------------------------
    # Main render loop
    # -------------------------

    def render(self):
        """
        Render the current simulation state and handle video recording if enabled.

        This method manages:
        1. Video recording state transitions and initialization
        2. Frame capture and saving during recording
        3. Video compilation when recording ends
        4. Cleanup of temporary image files
        """
        if not self.headless:
            # Handle recording state transitions
            if self._user_recording_state_change:
                if self._user_is_recording:
                    # Initialize new recording
                    self._user_recording_video_queue = deque(
                        maxlen=self._user_recording_video_queue_size
                    )
                    self._resolve_recording_paths()
                    self._user_recording_frame = 0

                    self._recorded_motion = {
                        "gts": [],  # rigid_body_pos (global translations)
                        "grs": [],  # rigid_body_rot (global rotations)
                        "gvs": [],  # rigid_body_vel (global velocities)
                        "gavs": [],  # rigid_body_ang_vel (global angular velocities)
                        "dps": [],  # dof_pos
                        "dvs": [],  # dof_vel
                        "contacts": [],  # rigid_body_contacts
                    }
                    self._recorded_markers = {}
                    self._recorded_objects = []
                    self._recorded_projectiles = []
                    self._recording_env_id = self._camera_target["env"]

                    os.makedirs(self._curr_user_recording_name, exist_ok=False)
                    print(
                        f"Started recording to folder {self._curr_user_recording_name}"
                    )
                else:
                    # Finalize recording and create video
                    from moviepy import ImageSequenceClip

                    image_dir = self._curr_user_recording_name
                    images = sorted(
                        [
                            os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.endswith(".png")
                        ]
                    )

                    if not images:
                        raise RuntimeError("No rendered frames were captured for video.")

                    if self._recording_output_path_override is None:
                        # Preserve callers that still set only the legacy
                        # frame-directory field before finalization.
                        video_path = f"{image_dir}.mp4"
                        artifact_base = image_dir
                    else:
                        video_path = self._curr_user_recording_video_path
                        artifact_base = self._curr_user_recording_artifact_base
                    clip = ImageSequenceClip(images, fps=self._recording_fps)
                    clip.write_videofile(
                        video_path,
                        codec="libx264",
                        audio=False,
                        threads=32,
                        preset="veryfast",
                        ffmpeg_params=[
                            "-profile:v",
                            "main",
                            "-level",
                            "4.0",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            "-crf",
                            "23",
                            "-x264-params",
                            "keyint=60:min-keyint=30",
                        ],
                    )
                    self._delete_user_viewer_recordings = True
                    self._last_recorded_video_path = video_path
                    print(f"Video saved to {video_path}")

                    # Save the recorded motion as a .motion file
                    motion_data = build_motion_data(
                        self._recorded_motion,
                        fps=self._recording_fps,
                        num_dof=self._num_dof,
                    )
                    motion_file_path = f"{artifact_base}.motion"
                    torch.save(motion_data, motion_file_path)
                    self._last_recorded_motion_path = motion_file_path
                    print(f"Motion saved to {motion_file_path}")
                    self._recorded_motion = None

                    # Save markers and objects files
                    try:
                        if self._recorded_markers:
                            markers_data = self._build_markers_save_data()
                            markers_path = f"{artifact_base}.markers.pt"
                            torch.save(markers_data, markers_path)
                            print(f"Markers saved to {markers_path}")

                        if self._recorded_objects or self._recorded_projectiles:
                            objects_data = self._build_objects_save_data()
                            objects_path = f"{artifact_base}.objects.pt"
                            torch.save(objects_data, objects_path)
                            print(f"Objects saved to {objects_path}")
                        terrain_data = self._build_terrain_save_data()
                        if terrain_data is not None:
                            terrain_path = (
                                f"{self._curr_user_recording_name}.terrain.pt"
                            )
                            torch.save(terrain_data, terrain_path)
                            print(f"Terrain saved to {terrain_path}")

                    except Exception as e:
                        print(f"Warning: failed to save markers/objects/terrain: {e}")
                    self._recorded_markers = None
                    self._recorded_objects = None
                    self._recorded_projectiles = None

                self._user_recording_state_change = False

            # Capture frame if recording
            if self._user_is_recording:
                file_name = (
                    self._curr_user_recording_name
                    + "/%04d.png" % self._user_recording_frame
                )
                self._write_viewport_to_file(file_name)
                self._apply_text_overlay_to_image(file_name)
                self._user_recording_frame += 1

                eid = self._recording_env_id

                # Record motion (single env only)
                robot_state = self.get_robot_state()
                self._recorded_motion["gts"].append(
                    robot_state.rigid_body_pos[eid].cpu().clone()
                )
                self._recorded_motion["grs"].append(
                    robot_state.rigid_body_rot[eid].cpu().clone()
                )
                if robot_state.rigid_body_vel is not None:
                    self._recorded_motion["gvs"].append(
                        robot_state.rigid_body_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_ang_vel is not None:
                    self._recorded_motion["gavs"].append(
                        robot_state.rigid_body_ang_vel[eid].cpu().clone()
                    )
                if robot_state.dof_pos is not None:
                    self._recorded_motion["dps"].append(
                        robot_state.dof_pos[eid].cpu().clone()
                    )
                if robot_state.dof_vel is not None:
                    self._recorded_motion["dvs"].append(
                        robot_state.dof_vel[eid].cpu().clone()
                    )
                if robot_state.rigid_body_contacts is not None:
                    self._recorded_motion["contacts"].append(
                        robot_state.rigid_body_contacts[eid].cpu().clone()
                    )

                # Record markers (single env only, skip terrain markers)
                if self._last_markers_state:
                    for name, ms in self._last_markers_state.items():
                        if name == "terrain_markers":
                            continue
                        if name not in self._recorded_markers:
                            self._recorded_markers[name] = []
                        self._recorded_markers[name].append(
                            (
                                ms.translation[eid].cpu().clone(),
                                ms.orientation[eid].cpu().clone(),
                            )
                        )

                # Record objects (single env only)
                if (
                    self.scene_lib is not None
                    and self.scene_lib.num_objects_per_scene > 0
                ):
                    obj_state = self.get_object_root_state()
                    self._recorded_objects.append(
                        (
                            obj_state.root_pos[eid].cpu().clone(),
                            obj_state.root_rot[eid].cpu().clone(),
                        )
                    )

                # Record projectiles (single env only)
                if (
                    self._proj_config is not None
                    and self._proj_config.num_projectiles > 0
                ):
                    pos, rot = self._get_projectile_positions_rotations()
                    self._recorded_projectiles.append(
                        (
                            pos[eid].cpu().clone(),
                            rot[eid].cpu().clone(),
                        )
                    )

            # Clean up temporary files if needed
            if self._delete_user_viewer_recordings:
                images = [
                    img
                    for img in os.listdir(self._curr_user_recording_name)
                    if img.endswith(".png")
                ]
                for image in images:
                    os.remove(os.path.join(self._curr_user_recording_name, image))
                os.rmdir(self._curr_user_recording_name)
                self._delete_user_viewer_recordings = False
                self._recorded_motion = None
                self._recording_output_path_override = None
                self._recording_text_overlay = None
                self._recording_fps = 30

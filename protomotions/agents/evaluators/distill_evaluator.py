import math
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from protomotions.agents.evaluators.mimic_evaluator import (
    MimicEvaluator,
    MimicEpisodeContext,
)
from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.config import DistillEvaluatorConfig
from protomotions.agents.distill.model import DistillModel
from protomotions.agents.distill.vq_pae import DistillVQPAEModel


log = logging.getLogger(__name__)


class DistillEvaluator(MimicEvaluator):
    """Mimic evaluator variant that also evaluates privileged actions."""

    def __init__(self, agent: Any, fabric: Any, config: DistillEvaluatorConfig):
        super().__init__(agent, fabric, config)
        self._privileged_eval_state: Optional[Dict[str, Any]] = None
        self._vq_latent_capture: Optional[torch.Tensor] = None
        self._vq_manifold_index_capture: Optional[torch.Tensor] = None
        self._vq_phase_capture: Optional[torch.Tensor] = None
        self._vq_latent_loop_phase: Optional[torch.Tensor] = None
        self._vq_prior_phase_accum: Optional[torch.Tensor] = None
        self._vq_prior_phase_accum_valid: Optional[torch.Tensor] = None
        self._vq_posterior_phase_accum: Optional[torch.Tensor] = None
        self._vq_posterior_phase_accum_valid: Optional[torch.Tensor] = None
        self._vq_prior_frequency_accum: Optional[torch.Tensor] = None
        self._vq_prior_frequency_accum_valid: Optional[torch.Tensor] = None
        self._vq_prior_offset_accum: Optional[torch.Tensor] = None
        self._vq_prior_offset_accum_valid: Optional[torch.Tensor] = None
        self._vq_prior_state_accum: Optional[torch.Tensor] = None
        self._vq_prior_state_accum_valid: Optional[torch.Tensor] = None
        self._vq_posterior_frequency_accum: Optional[torch.Tensor] = None
        self._vq_posterior_frequency_accum_valid: Optional[torch.Tensor] = None
        self._vq_posterior_offset_accum: Optional[torch.Tensor] = None
        self._vq_posterior_offset_accum_valid: Optional[torch.Tensor] = None
        self._vq_posterior_state_accum: Optional[torch.Tensor] = None
        self._vq_posterior_state_accum_valid: Optional[torch.Tensor] = None

    def _reset_vq_latent_loop(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset latent-loop playback phase for selected environments."""
        if self._vq_latent_loop_phase is None:
            return
        if env_ids is None:
            self._vq_latent_loop_phase.zero_()
            return
        if env_ids.numel() > 0:
            self._vq_latent_loop_phase[env_ids] = 0.0

    def _sample_vq_loop_tensor(self, clip: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Interpolate a captured [frames, envs, dim] clip at the current loop phase."""
        if clip is None or self._vq_latent_loop_phase is None:
            return None
        num_frames = clip.shape[0]
        if num_frames == 0:
            return None
        if num_frames == 1:
            return clip[0]

        phases = torch.remainder(self._vq_latent_loop_phase, float(num_frames))
        idx0 = torch.floor(phases).long()
        idx1 = (idx0 + 1) % num_frames
        alpha = (phases - idx0.float()).unsqueeze(-1)
        env_ids = torch.arange(
            clip.shape[1], device=clip.device
        )
        clip0 = clip[idx0, env_ids]
        clip1 = clip[idx1, env_ids]
        sampled = clip0 + alpha * (clip1 - clip0)
        return sampled

    def _sample_vq_loop_latent(self) -> Optional[torch.Tensor]:
        """Interpolate from the captured actor-latent clip using looped phase."""
        return self._sample_vq_loop_tensor(self._vq_latent_capture)

    def _advance_vq_loop_phase(self, speed_scale: float) -> None:
        """Advance loop playback phase after all captures for the step are sampled."""
        if self._vq_latent_loop_phase is None:
            return
        if self._vq_latent_capture is None:
            return
        num_frames = self._vq_latent_capture.shape[0]
        if num_frames <= 1:
            return
        self._vq_latent_loop_phase = torch.remainder(
            self._vq_latent_loop_phase + float(speed_scale),
            float(num_frames),
        )

    def _ensure_vq_accumulator(
        self,
        attr_name: str,
        valid_attr_name: str,
        batch_size: int,
        dim: int,
        device: torch.device,
    ) -> None:
        accum = getattr(self, attr_name)
        if (
            accum is not None
            and accum.shape == (batch_size, dim)
            and accum.device == device
        ):
            return

        setattr(
            self,
            attr_name,
            torch.zeros(
                batch_size,
                dim,
                device=device,
                dtype=torch.float32,
            ),
        )
        setattr(
            self,
            valid_attr_name,
            torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool,
            ),
        )

    def _reset_vq_accumulator(
        self,
        attr_name: str,
        valid_attr_name: str,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        accum_valid = getattr(self, valid_attr_name)
        if accum_valid is None:
            return
        accum = getattr(self, attr_name)
        if env_ids is None:
            accum_valid.zero_()
            if accum is not None:
                accum.zero_()
            return
        if env_ids.numel() > 0:
            accum_valid[env_ids] = False
            if accum is not None:
                accum[env_ids] = 0.0

    def _decode_vq_codebook_manifold(
        self,
        model_module: DistillVQPAEModel,
        num_phase_samples: int,
    ) -> Optional[torch.Tensor]:
        """Decode every VQ state over one phase cycle."""
        codebook = getattr(model_module.quantizer, "_codebook", None)
        if codebook is None or codebook.numel() == 0:
            return None

        codebook = codebook.detach()
        num_phase_samples = max(int(num_phase_samples), 2)
        phase_grid = torch.linspace(
            -0.5,
            0.5,
            num_phase_samples + 1,
            device=codebook.device,
            dtype=codebook.dtype,
        )
        angles = (
            model_module.two_pi.to(device=codebook.device, dtype=codebook.dtype)
            * phase_grid.view(1, 1, -1)
        )
        angles = angles.expand(
            codebook.shape[0],
            model_module.config.n_timing_phases,
            -1,
        )
        manifold, _ = model_module.get_phase_manifold(codebook, angles)
        return manifold.permute(0, 2, 1).contiguous()

    def _project_latent_manifold_to_2d(
        self,
        background_curves: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project high-dimensional latent manifold points with torch PCA."""
        latent_dim = background_curves.shape[-1]
        if latent_dim == 1:
            background_2d = torch.cat(
                [background_curves, torch.zeros_like(background_curves)],
                dim=-1,
            )
            trajectory_2d = torch.cat(
                [trajectory, torch.zeros_like(trajectory)],
                dim=-1,
            )
            return background_2d, trajectory_2d

        background_flat = background_curves.reshape(-1, latent_dim)
        all_points = torch.cat([background_flat, trajectory], dim=0).float()
        mean = all_points.mean(dim=0, keepdim=True)
        centered = all_points - mean
        denom = max(centered.shape[0] - 1, 1)
        covariance = centered.t().matmul(centered) / denom
        _, eigenvectors = torch.linalg.eigh(covariance)
        components = eigenvectors[:, [-1, -2]]

        background_2d = (background_flat - mean).matmul(components)
        trajectory_2d = (trajectory.float() - mean).matmul(components)
        return (
            background_2d.reshape(*background_curves.shape[:-1], 2),
            trajectory_2d,
        )

    def _decode_vq_phase_points(
        self,
        model_module: DistillVQPAEModel,
        indices: torch.Tensor,
        phases: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Decode points selected by recorded VQ code indices and phases."""
        codebook = getattr(model_module.quantizer, "_codebook", None)
        if codebook is None or codebook.numel() == 0:
            return None

        indices = indices.to(device=codebook.device, dtype=torch.long)
        if indices.numel() == 0:
            return None
        valid = (0 <= indices) & (indices < codebook.shape[0])
        if not valid.all():
            return None

        phases = phases.to(device=codebook.device, dtype=codebook.dtype)
        if not torch.isfinite(phases).all():
            return None
        if phases.ndim == 1:
            phases = phases.unsqueeze(-1)
        angles = (
            model_module.two_pi.to(device=codebook.device, dtype=codebook.dtype)
            * phases.unsqueeze(-1)
        )
        selected_states = codebook[indices]
        manifold, _ = model_module.get_phase_manifold(selected_states, angles)
        return manifold[:, :, 0]

    def _get_vq_latent_manifold_plot_path(
        self,
        action_key: str,
        env_idx: int,
    ) -> Path:
        root_dir = Path(self.root_dir) if self.root_dir is not None else Path(".")
        configured_path = getattr(self, "vq_latent_manifold_plot_path", None)
        if configured_path:
            plot_path = Path(configured_path)
            if plot_path.suffix:
                return plot_path if plot_path.is_absolute() else root_dir / plot_path
            plot_dir = plot_path if plot_path.is_absolute() else root_dir / plot_path
            return plot_dir / f"vq_latent_manifold_{action_key}_env{env_idx}.png"

        return root_dir / f"vq_latent_manifold_{action_key}_env{env_idx}.png"

    def _save_vq_latent_manifold_gif(
        self,
        plot_path: Path,
        background_np,
        trajectory_np,
        phase_indices,
        frame_count: int,
        plot_mode: str,
        action_key: str,
        env_idx: int,
        latent_key: Optional[str],
        plt,
        np,
        LineCollection,
        Normalize,
    ) -> None:
        gif_fps = int(getattr(self, "vq_latent_manifold_gif_fps", 20))
        if gif_fps <= 0 or frame_count == 0:
            return
        gif_dpi = int(getattr(self, "vq_latent_manifold_gif_dpi", 120))

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            print("[vq-latent-loop] GIF writer unavailable; skipping manifold GIF")
            return

        gif_path = plot_path.with_suffix(".gif")
        fig, ax = plt.subplots(figsize=(8, 8), dpi=gif_dpi)
        background_lines = LineCollection(
            background_np,
            colors=[(0.45, 0.45, 0.45, 0.14)],
            linewidths=0.6,
            zorder=1,
        )
        ax.add_collection(background_lines)
        ax.scatter(
            trajectory_np[:, 0],
            trajectory_np[:, 1],
            color=(0.05, 0.05, 0.05, 0.16),
            s=14,
            edgecolors="none",
            zorder=2,
        )

        norm = Normalize(vmin=0, vmax=max(frame_count - 1, 1))
        cmap = plt.get_cmap("viridis")
        draw_segment = np.ones(max(frame_count - 1, 0), dtype=bool)
        if frame_count > 1:
            if phase_indices is not None:
                draw_segment = phase_indices[:-1].cpu().numpy() == phase_indices[
                    1:
                ].cpu().numpy()
            colorbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                pad=0.02,
            )
            colorbar.set_label("Capture frame")
        else:
            ax.scatter(
                trajectory_np[:, 0],
                trajectory_np[:, 1],
                color="tab:blue",
                s=28,
                edgecolors="none",
                zorder=4,
            )
        ax.scatter(
            trajectory_np[0, 0],
            trajectory_np[0, 1],
            color="limegreen",
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            s=90,
            label="start",
            zorder=5,
        )
        ax.scatter(
            trajectory_np[-1, 0],
            trajectory_np[-1, 1],
            color="crimson",
            edgecolors="black",
            linewidths=0.8,
            marker="X",
            s=110,
            label="end",
            zorder=5,
        )
        ax.set_title(
            f"VQ-PAE manifold {plot_mode} ({action_key}, env {env_idx}, {latent_key})"
        )
        ax.set_xlabel("Latent PC 1")
        ax.set_ylabel("Latent PC 2")
        ax.grid(True, alpha=0.2)
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        fig.tight_layout()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        try:
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            base_rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
            base_image = Image.fromarray(base_rgba, mode="RGBA")
            pixel_positions = ax.transData.transform(trajectory_np)
            pixel_positions[:, 1] = height - pixel_positions[:, 1]
            label_xy = ax.transAxes.transform((0.02, 0.98))
            label_x = int(round(label_xy[0]))
            label_y = int(round(height - label_xy[1]))
            marker_radius = max(5, int(round(math.sqrt(130) * gif_dpi / 144)))
            outline_width = max(1, int(round(gif_dpi / 120)))
            label_padding = max(3, int(round(gif_dpi / 40)))
            duration_ms = max(1, int(round(1000 / gif_fps)))
            path_width = max(2, int(round(2.8 * gif_dpi / 72)))
            path_point_radius = max(2, int(round(math.sqrt(24) * gif_dpi / 144)))
            path_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
            path_draw = ImageDraw.Draw(path_layer, "RGBA")
            frames = []
            for frame_idx in range(frame_count):
                x, y = pixel_positions[frame_idx]
                x = int(round(x))
                y = int(round(y))
                path_color = tuple(
                    int(round(channel * 255))
                    for channel in cmap(norm(frame_idx))[:3]
                )
                if frame_idx > 0 and draw_segment[frame_idx - 1]:
                    prev_x, prev_y = pixel_positions[frame_idx - 1]
                    path_draw.line(
                        (
                            int(round(prev_x)),
                            int(round(prev_y)),
                            x,
                            y,
                        ),
                        fill=(*path_color, 255),
                        width=path_width,
                    )
                path_draw.ellipse(
                    (
                        x - path_point_radius,
                        y - path_point_radius,
                        x + path_point_radius,
                        y + path_point_radius,
                    ),
                    fill=(*path_color, 255),
                )
                frame = Image.alpha_composite(base_image, path_layer)
                draw = ImageDraw.Draw(frame, "RGBA")
                draw.ellipse(
                    (
                        x - marker_radius,
                        y - marker_radius,
                        x + marker_radius,
                        y + marker_radius,
                    ),
                    fill=(*path_color, 255),
                    outline=(0, 0, 0, 255),
                    width=outline_width,
                )
                label = f"frame {frame_idx + 1}/{frame_count}"
                text_bbox = draw.textbbox((label_x, label_y), label)
                draw.rectangle(
                    (
                        text_bbox[0] - label_padding,
                        text_bbox[1] - label_padding,
                        text_bbox[2] + label_padding,
                        text_bbox[3] + label_padding,
                    ),
                    fill=(255, 255, 255, 200),
                )
                draw.text((label_x, label_y), label, fill=(0, 0, 0, 255))
                frames.append(frame)

            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            print(f"[vq-latent-loop] latent manifold GIF saved to: {gif_path}")
        except Exception as exc:
            print(f"[vq-latent-loop] failed to save manifold GIF: {exc}")
        finally:
            plt.close(fig)

    def _save_vq_latent_manifold_plot(
        self,
        model_module: DistillVQPAEModel,
        captured_latents: torch.Tensor,
        action_key: str,
        latent_key: Optional[str],
        captured_indices: Optional[torch.Tensor] = None,
        captured_phases: Optional[torch.Tensor] = None,
        env_idx: int = 0,
    ) -> None:
        """Save a 2D figure of the learned VQ manifold and recorded trajectory."""
        if captured_latents.ndim != 3 or captured_latents.shape[0] == 0:
            return
        env_idx = min(max(int(env_idx), 0), captured_latents.shape[1] - 1)
        trajectory = captured_latents[:, env_idx].detach()
        phase_indices = None
        if captured_indices is not None and captured_phases is not None:
            captured_phase_indices = captured_indices[:, env_idx].detach()
            phase_values = captured_phases[:, env_idx].detach()
            phase_trajectory = self._decode_vq_phase_points(
                model_module,
                captured_phase_indices,
                phase_values,
            )
            if phase_trajectory is not None:
                trajectory = phase_trajectory.detach()
                phase_indices = captured_phase_indices
        if trajectory.shape[0] == 0:
            return
        if not torch.isfinite(trajectory).all():
            print(
                "[vq-latent-loop] skipping manifold plot; captured latents "
                "contain NaN/Inf"
            )
            return

        num_phase_samples = int(getattr(self, "vq_latent_manifold_phase_samples", 512))
        background_curves = self._decode_vq_codebook_manifold(
            model_module,
            num_phase_samples=num_phase_samples,
        )
        if background_curves is None:
            print(
                "[vq-latent-loop] skipping manifold plot; VQ codebook was "
                "unavailable"
            )
            return
        finite_curves = torch.isfinite(background_curves).all(dim=(1, 2))
        background_curves = background_curves[finite_curves]
        if background_curves.shape[0] == 0:
            print(
                "[vq-latent-loop] skipping manifold plot; decoded manifold "
                "was invalid"
            )
            return
        if background_curves.shape[-1] != trajectory.shape[-1]:
            print(
                "[vq-latent-loop] skipping manifold plot; latent dimensions differ "
                f"(manifold={background_curves.shape[-1]}, "
                f"capture={trajectory.shape[-1]})"
            )
            return

        try:
            import os

            matplotlib_config_dir = "/tmp/protomotions_matplotlib"
            Path(matplotlib_config_dir).mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = matplotlib_config_dir
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
        except ImportError:
            print("[vq-latent-loop] matplotlib not available; skipping manifold plot")
            return

        background_2d, trajectory_2d = self._project_latent_manifold_to_2d(
            background_curves.detach().cpu(),
            trajectory.detach().cpu(),
        )
        background_np = background_2d.numpy()
        trajectory_np = trajectory_2d.numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        background_lines = LineCollection(
            background_np,
            colors=[(0.45, 0.45, 0.45, 0.18)],
            linewidths=0.6,
            zorder=1,
        )
        ax.add_collection(background_lines)

        frame_count = trajectory_np.shape[0]
        if frame_count > 1:
            segments = np.stack([trajectory_np[:-1], trajectory_np[1:]], axis=1)
            if phase_indices is not None:
                same_code = phase_indices[:-1].cpu().numpy() == phase_indices[
                    1:
                ].cpu().numpy()
                segments = segments[same_code]
                segment_frames = np.arange(frame_count - 1)[same_code]
            else:
                segment_frames = np.arange(frame_count - 1)
            trajectory_lines = LineCollection(
                segments,
                cmap="viridis",
                norm=Normalize(vmin=0, vmax=max(frame_count - 1, 1)),
                linewidths=2.5,
                zorder=3,
            )
            trajectory_lines.set_array(segment_frames)
            if len(segments) > 0:
                ax.add_collection(trajectory_lines)

        point_plot = ax.scatter(
            trajectory_np[:, 0],
            trajectory_np[:, 1],
            c=np.arange(frame_count),
            cmap="viridis",
            norm=Normalize(vmin=0, vmax=max(frame_count - 1, 1)),
            s=24,
            edgecolors="none",
            zorder=4,
        )
        if frame_count > 1:
            colorbar = fig.colorbar(point_plot, ax=ax, pad=0.02)
            colorbar.set_label("Capture frame")
        else:
            point_plot.set_color("tab:blue")

        ax.scatter(
            trajectory_np[0, 0],
            trajectory_np[0, 1],
            color="limegreen",
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            s=90,
            label="start",
            zorder=4,
        )
        ax.scatter(
            trajectory_np[-1, 0],
            trajectory_np[-1, 1],
            color="crimson",
            edgecolors="black",
            linewidths=0.8,
            marker="X",
            s=110,
            label="end",
            zorder=5,
        )
        plot_mode = "phase points" if phase_indices is not None else "latent PCA"
        ax.set_title(
            f"VQ-PAE manifold {plot_mode} ({action_key}, env {env_idx}, {latent_key})"
        )
        ax.set_xlabel("Latent PC 1")
        ax.set_ylabel("Latent PC 2")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        ax.autoscale()
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()

        plot_path = self._get_vq_latent_manifold_plot_path(action_key, env_idx)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[vq-latent-loop] latent manifold plot saved to: {plot_path}")
        self._save_vq_latent_manifold_gif(
            plot_path=plot_path,
            background_np=background_np,
            trajectory_np=trajectory_np,
            phase_indices=phase_indices,
            frame_count=frame_count,
            plot_mode=plot_mode,
            action_key=action_key,
            env_idx=env_idx,
            latent_key=latent_key,
            plt=plt,
            np=np,
            LineCollection=LineCollection,
            Normalize=Normalize,
        )

    def interactive_edit_text_prompt(self) -> None:
        """Pause interactive inference and switch the live text-conditioning prompt."""
        motion_lib = self.motion_lib
        available_count = len(motion_lib.get_available_text_embeddings())
        current_label = motion_lib.get_text_embedding_override_label()
        prompt_state = {"text_prompt": current_label or ""}

        print("\n[text-debug] Entering live prompt editor.")
        print(
            "[text-debug] In ipdb/pdb set prompt_state['text_prompt'] to a packaged prompt, then continue."
        )
        print(
            "[text-debug] Example: prompt_state['text_prompt'] = 'walk'"
        )
        print(
            "[text-debug] Use motion_lib.search_text_embeddings('walk') to search packaged prompts."
        )
        print(
            "[text-debug] Set prompt_state['text_prompt'] = '' to clear the fixed override."
        )
        print(
            f"[text-debug] current_override={current_label!r} available_packaged_prompts={available_count}"
        )

        try:
            import ipdb as debugger
        except ImportError:
            import pdb as debugger

        debugger.set_trace()

        requested_prompt = str(prompt_state.get("text_prompt", "")).strip()
        if not requested_prompt:
            motion_lib.clear_text_embedding_override()
            print("[text-debug] Cleared live text override; using motion-timed text.")
            return

        try:
            motion_lib.set_text_embedding_override_by_text(requested_prompt)
            print(
                "[text-debug] Applied live text override: "
                f"{motion_lib.get_text_embedding_override_label()!r}"
            )
        except Exception as exc:
            print(f"[text-debug] Failed to apply live text override: {exc}")
            log.exception("Failed to apply interactive text prompt override")

    def _supports_privileged_action(self) -> bool:
        """Check whether the model exposes a privileged action output."""
        out_keys = getattr(self.agent.model, "out_keys", None)
        if out_keys is None and hasattr(self.agent.model, "module"):
            out_keys = getattr(self.agent.model.module, "out_keys", None)
        return out_keys is not None and "privileged_action" in out_keys

    def _supports_prior_action(self) -> bool:
        """Check whether the model exposes a distinct prior-action output."""
        out_keys = getattr(self.agent.model, "out_keys", None)
        if out_keys is None and hasattr(self.agent.model, "module"):
            out_keys = getattr(self.agent.model.module, "out_keys", None)
        if out_keys is not None and "prior_action" in out_keys:
            return True

        model_module = (
            self.agent.model.module if hasattr(self.agent.model, "module") else self.agent.model
        )
        return isinstance(model_module, DistillVQPAEModel)

    def _get_interaction_action_key(self) -> str:
        """Action head used for the interactive inference loop."""
        if self.config.use_privileged_action_for_interaction:
            if not self._supports_privileged_action():
                raise RuntimeError(
                    "Distill evaluator requested privileged_action for interaction, "
                    "but the model does not expose that output."
                )
            return "privileged_action"
        if self._supports_prior_action():
            return "prior_action"
        raise RuntimeError(
            "Distill evaluator requires an explicit prior_action output for "
            "non-privileged interaction, but the model does not expose it."
        )

    def _select_actions(self, model_outs: Dict[str, Tensor], action_key: str) -> Tensor:
        """Select the requested explicit action head."""
        if action_key not in model_outs:
            raise KeyError(
                f"Requested action key '{action_key}' not found in model outputs. "
                f"Available keys: {sorted(model_outs.keys())}"
            )
        return model_outs[action_key]

    def _create_eval_state(
        self,
        num_motions: int,
        motion_num_frames: Tensor,
        max_eval_steps: int,
    ) -> Dict[str, Any]:
        """Create an isolated evaluator buffer set for one action mode."""
        return {
            "metrics": self._create_metrics(num_motions, motion_num_frames, max_eval_steps),
            "motion_failed": torch.zeros(num_motions, dtype=torch.bool, device=self.device),
            "per_component_failures": {
                name: torch.zeros(num_motions, dtype=torch.bool, device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_value_sum": {
                name: torch.zeros(num_motions, device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_value_min": {
                name: torch.full((num_motions,), float("inf"), device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_value_max": {
                name: torch.full((num_motions,), float("-inf"), device=self.device)
                for name in self.config.evaluation_components.keys()
            },
            "component_step_count": {
                name: torch.zeros(num_motions, dtype=torch.long, device=self.device)
                for name in self.config.evaluation_components.keys()
            },
        }

    def _capture_active_eval_state(self) -> Dict[str, Any]:
        """Snapshot the currently active evaluator buffers."""
        return {
            "metrics": self._metrics,
            "motion_failed": self._motion_failed,
            "per_component_failures": self._per_component_failures,
            "component_value_sum": self._component_value_sum,
            "component_value_min": self._component_value_min,
            "component_value_max": self._component_value_max,
            "component_step_count": self._component_step_count,
        }

    def _restore_active_eval_state(self, state: Dict[str, Any]) -> None:
        """Swap the evaluator to a previously captured buffer set."""
        self._metrics = state["metrics"]
        self._motion_failed = state["motion_failed"]
        self._per_component_failures = state["per_component_failures"]
        self._component_value_sum = state["component_value_sum"]
        self._component_value_min = state["component_value_min"]
        self._component_value_max = state["component_value_max"]
        self._component_step_count = state["component_step_count"]

    def _summarize_eval_state(
        self,
        state: Dict[str, Any],
        *,
        prefix: str,
        success_rate_key: str,
    ) -> Tuple[Dict[str, float], Optional[float]]:
        """Summarize one evaluation state without mutating the active one."""
        to_log: Dict[str, float] = {}
        motion_failed = state["motion_failed"]
        success_rate = None

        if motion_failed is not None:
            success_rate = 1.0 - motion_failed.float().mean().item()
            to_log[success_rate_key] = success_rate

            for name, component in self.config.evaluation_components.items():
                threshold = component.static_params.get("threshold", None)
                if threshold is not None:
                    failure_rate = state["per_component_failures"][name].float().mean().item()
                    to_log[f"{prefix}/{name}/failure_rate"] = failure_rate

            for name in state["component_value_sum"].keys():
                step_count = state["component_step_count"][name].float()
                valid = step_count > 0
                if valid.any():
                    mean_per_motion = state["component_value_sum"][name] / step_count.clamp(min=1)
                    to_log[f"{prefix}/{name}/mean"] = mean_per_motion[valid].mean().item()
                    to_log[f"{prefix}/{name}/max"] = state["component_value_max"][name][valid].max().item()
                    to_log[f"{prefix}/{name}/min"] = state["component_value_min"][name][valid].min().item()

        additional_metrics = self._compute_additional_metrics(state["metrics"])
        for key, value in additional_metrics.items():
            if key.startswith("eval/"):
                to_log[f"{prefix}/{key[len('eval/') :]}"] = value
            else:
                to_log[f"{prefix}/{key}"] = value

        return to_log, success_rate

    def _save_privileged_failed_motions(self, failed_motions: list, epoch: int) -> None:
        """Save failed motions from the privileged-action pass."""
        filename = (
            f"failed_motions_epoch_{epoch}_rank_{self.fabric.global_rank}.txt"
        )
        self._save_list_to_file(
            failed_motions,
            filename,
            subdirectory="privileged_failed_motions",
        )

    def _update_motion_sampling_weights(self) -> None:
        """Optionally update sampling weights from privileged-action evaluation failures."""
        if (
            self.config.use_privileged_success_for_motion_weights
            and self._privileged_eval_state is not None
        ):
            motion_failed = self._privileged_eval_state["motion_failed"]
            if motion_failed is None:
                return

            failed_motions = torch.nonzero(motion_failed).flatten().tolist()
            success_motions = torch.nonzero(~motion_failed).flatten().tolist()

            self._save_privileged_failed_motions(
                failed_motions, self.agent.current_epoch
            )

            success_discount = math.pow(
                self.config.motion_weights_rules.motion_weights_update_success_discount,
                self.config.eval_metrics_every,
            )
            failure_discount = math.pow(
                self.config.motion_weights_rules.motion_weights_update_failure_discount,
                self.config.eval_metrics_every,
            )
            new_weights = self.env.motion_manager.motion_weights.clone()
            new_weights[success_motions] *= success_discount
            if failure_discount != 0:
                new_weights[failed_motions] /= failure_discount
            else:
                new_weights[failed_motions] = 1.0
            self.env.motion_manager.update_sampling_weights(new_weights)
            return

        super()._update_motion_sampling_weights()

    def initialize_eval(self) -> Dict[str, MotionMetrics]:
        """Initialize normal and privileged evaluation state."""
        metrics = super().initialize_eval()
        self._privileged_eval_state = None
        if self._supports_privileged_action():
            num_motions = self.motion_lib.num_motions()
            motion_lengths = self.motion_lib.get_motion_length(None)
            motion_num_frames = (motion_lengths / self.env.dt).floor().long()
            motion_num_frames = motion_num_frames.clamp(max=self.config.max_eval_steps)
            self._privileged_eval_state = self._create_eval_state(
                num_motions, motion_num_frames, self.config.max_eval_steps
            )
        return metrics

    def evaluate_episode(
        self,
        env_ids: torch.Tensor,
        max_steps: int,
        action_key: str = "prior_action",
    ) -> None:
        """Run one evaluation episode using the requested action output."""
        ema_alpha = self.config.eval_action_ema_alpha

        self._on_episode_start(env_ids)

        obs, _ = self.env.reset(env_ids, **self._get_reset_kwargs())
        obs = self.agent.add_agent_info_to_obs(obs)
        obs_td = self.agent.obs_dict_to_tensordict(obs)

        prev_actions = None

        for step_idx in range(max_steps):
            model_outs = self.agent.model(obs_td)
            actions = self._select_actions(model_outs, action_key)

            if ema_alpha is not None:
                if prev_actions is None:
                    prev_actions = actions.clone()
                actions = ema_alpha * actions + (1.0 - ema_alpha) * prev_actions
                prev_actions = actions.clone()

            obs, rewards, dones, terminated, extras = self.env.step(actions)
            obs = self.agent.add_agent_info_to_obs(obs)
            obs_td = self.agent.obs_dict_to_tensordict(obs)

            self._check_eval_components(env_ids, step_idx)
            self._on_episode_step(env_ids, extras, actions)

    def run_evaluation(self) -> None:
        """Run normal and optional privileged evaluation across motion batches."""
        if not self._supports_prior_action():
            raise RuntimeError(
                "Distill evaluator requires prior_action for the primary evaluation pass, "
                "but the model does not expose it."
            )
        primary_action_key = "prior_action"
        for env_ids, motion_ids in self._build_eval_batches():
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            max_len = min(
                (motion_lengths.max() / self.env.dt).floor().long().item(),
                self.config.max_eval_steps,
            )
            self._episode_ctx = MimicEpisodeContext(
                motion_ids=motion_ids,
                frame_limits=(motion_lengths / self.env.dt).floor().long().clamp(
                    max=self.config.max_eval_steps
                ),
            )
            self.evaluate_episode(env_ids, max_len, action_key=primary_action_key)
            if self._privileged_eval_state is not None:
                normal_state = self._capture_active_eval_state()
                self._restore_active_eval_state(self._privileged_eval_state)
                self.evaluate_episode(env_ids, max_len, action_key="privileged_action")
                self._privileged_eval_state = self._capture_active_eval_state()
                self._restore_active_eval_state(normal_state)

    def process_eval_results(self) -> Tuple[Dict, Optional[float]]:
        """Process normal metrics and append privileged-action metrics."""
        to_log, success_rate = super().process_eval_results()

        if self._privileged_eval_state is not None:
            privileged_failed_motions = (
                torch.nonzero(self._privileged_eval_state["motion_failed"])
                .flatten()
                .tolist()
            )
            self._save_privileged_failed_motions(
                privileged_failed_motions,
                self.agent.current_epoch,
            )
            privileged_log, privileged_success_rate = self._summarize_eval_state(
                self._privileged_eval_state,
                prefix="privileged_eval",
                success_rate_key="eval/privileged_success_rate",
            )
            to_log.update(privileged_log)
            if success_rate is not None and privileged_success_rate is not None:
                to_log["eval/privileged_prior_gap"] = privileged_success_rate - success_rate

        return to_log, success_rate

    def _get_vq_accumulator_alpha(
        self,
        model_module: DistillVQPAEModel,
        branch: str,
        component: str,
    ) -> Optional[float]:
        runtime_value = getattr(
            self,
            f"vq_{branch}_{component}_accumulator_alpha",
            None,
        )
        if runtime_value is not None:
            return runtime_value
        return getattr(
            model_module.config,
            f"{branch}_{component}_accumulator_alpha",
            None,
        )

    def _store_vq_accumulator_next(
        self,
        attr_name: str,
        valid_attr_name: str,
        next_accum: torch.Tensor,
    ) -> None:
        next_accum = next_accum.detach()
        accum = getattr(self, attr_name)
        if accum is None:
            setattr(self, attr_name, next_accum.clone())
        else:
            accum.copy_(next_accum)
        setattr(
            self,
            valid_attr_name,
            torch.ones(
                next_accum.shape[0],
                device=next_accum.device,
                dtype=torch.bool,
            ),
        )

    def simple_test_policy(self, collect_metrics: bool = False) -> None:
        """Interactive policy loop using the configured main action head."""
        self.agent.eval()
        done_indices = None
        step = 0
        action_key = self._get_interaction_action_key()
        model_module = self.agent.model.module if hasattr(self.agent.model, "module") else self.agent.model
        is_vq_pae_model = isinstance(model_module, DistillVQPAEModel)
        is_distill_vae_model = isinstance(model_module, DistillModel)
        latent_key = None
        actor_external_key = None
        privileged_external_key = None
        if is_vq_pae_model:
            latent_key = (
                "vq_pae_privileged_latent"
                if action_key == "privileged_action"
                else "vq_pae_actor_latent"
            )
            actor_external_key = "vq_external_vae_latent"
            privileged_external_key = "vq_external_privileged_vae_latent"
        elif is_distill_vae_model:
            latent_key = (
                "distill_privileged_latent"
                if action_key == "privileged_action"
                else "distill_actor_latent"
            )
            actor_external_key = "distill_external_vae_latent"
            privileged_external_key = "distill_external_privileged_vae_latent"
        motion_manager = getattr(self.env, "motion_manager", None)
        motion_lib = getattr(self, "motion_lib", None)
        record_frames_nums = int(
            getattr(self, "record_frames_nums", getattr(self, "vq_latent_loop_frames", 0))
        )
        self._vq_latent_capture = None
        self._vq_manifold_index_capture = None
        self._vq_phase_capture = None
        self._vq_latent_loop_phase = None
        self._vq_prior_phase_accum = None
        self._vq_prior_phase_accum_valid = None
        self._vq_posterior_phase_accum = None
        self._vq_posterior_phase_accum_valid = None
        self._vq_prior_frequency_accum = None
        self._vq_prior_frequency_accum_valid = None
        self._vq_prior_offset_accum = None
        self._vq_prior_offset_accum_valid = None
        self._vq_prior_state_accum = None
        self._vq_prior_state_accum_valid = None
        self._vq_posterior_frequency_accum = None
        self._vq_posterior_frequency_accum_valid = None
        self._vq_posterior_offset_accum = None
        self._vq_posterior_offset_accum_valid = None
        self._vq_posterior_state_accum = None
        self._vq_posterior_state_accum_valid = None
        capture_step = 0

        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        original_motion_speed_scale = None
        if motion_manager is not None:
            original_motion_speed_scale = float(
                getattr(motion_manager, "speed_scale", motion_manager.config.speed_scale)
            )

        print("Evaluating policy... (Ctrl+C to stop)")
        if is_distill_vae_model and action_key == "privileged_action":
            print(
                "[distill-eval] using environment-provided interpolated targets "
                "for privileged tracking"
            )
        try:
            while True:
                obs, _ = self.env.reset(done_indices)
                obs = self.agent.add_agent_info_to_obs(obs)
                obs_td = self.agent.obs_dict_to_tensordict(obs)
                configured_motion_speed_scale = getattr(
                    self, "vq_motion_speed_scale", 1.0
                )
                configured_prior_frequency_scale = getattr(
                    self, "vq_prior_frequency_scale", 1.0
                )
                configured_prior_frequency_override = getattr(
                    self, "vq_prior_frequency_override", None
                )
                vq_accumulator_branches = ("prior", "posterior")
                vq_accumulator_components = (
                    "phase",
                    "frequency",
                    "offset",
                    "state",
                )
                configured_vq_accum_alphas = {
                    branch: {
                        component: self._get_vq_accumulator_alpha(
                            model_module,
                            branch,
                            component,
                        )
                        for component in vq_accumulator_components
                    }
                    for branch in vq_accumulator_branches
                }
                # Decide if record/replay Mode
                is_loop_playback_active = self._vq_latent_loop_phase is not None
                is_recording_vq_latent = (
                    is_vq_pae_model
                    and record_frames_nums > 0
                    and not is_loop_playback_active
                    and capture_step < record_frames_nums
                )
                # Decide whether to speed up the motion
                should_scale_motion_now = (
                    is_vq_pae_model
                    or (is_distill_vae_model and action_key == "privileged_action")
                ) and not is_recording_vq_latent
                active_motion_speed_scale = (
                    configured_motion_speed_scale if should_scale_motion_now else 1.0
                )
                if motion_manager is not None:
                    motion_manager.speed_scale = float(active_motion_speed_scale)
                use_stored_vq_playback = (
                    is_vq_pae_model
                    and is_loop_playback_active
                )
                playback_latent = (
                    self._sample_vq_loop_latent() if use_stored_vq_playback else None
                )
                if is_loop_playback_active:
                    self._advance_vq_loop_phase(configured_motion_speed_scale)
                if playback_latent is not None and actor_external_key is not None:
                    if action_key == "privileged_action":
                        obs_td[privileged_external_key] = playback_latent
                    else:
                        obs_td[actor_external_key] = playback_latent
                model_frequency_scale = 1.0
                if is_vq_pae_model and action_key == "prior_action":
                    model_frequency_scale = float(configured_prior_frequency_scale)
                if (
                    model_frequency_scale != 1.0
                    and is_vq_pae_model
                    and action_key == "prior_action"
                ):
                    obs_td["vq_speed_scale"] = torch.full(
                        (obs_td.batch_size[0],),
                        float(model_frequency_scale),
                        device=self.device,
                    )
                if (
                    configured_prior_frequency_override is not None
                    and is_vq_pae_model
                    and action_key == "prior_action"
                ):
                    obs_td["vq_prior_frequency_override"] = torch.full(
                        (obs_td.batch_size[0],),
                        float(configured_prior_frequency_override),
                        device=self.device,
                    )
                vq_accumulator_action_keys = {
                    "prior": "prior_action",
                    "posterior": "privileged_action",
                }
                use_vq_accumulator = {
                    branch: {
                        component: (
                            is_vq_pae_model
                            and action_key == vq_accumulator_action_keys[branch]
                            and alpha is not None
                        )
                        for component, alpha in branch_accum_alphas.items()
                    }
                    for branch, branch_accum_alphas in (
                        configured_vq_accum_alphas.items()
                    )
                }
                for branch in vq_accumulator_branches:
                    for component in vq_accumulator_components:
                        if not use_vq_accumulator[branch][component]:
                            continue
                        accum_alpha = configured_vq_accum_alphas[branch][component]
                        if component == "phase":
                            phase_accum_alpha = float(accum_alpha)
                            if not 0.0 <= phase_accum_alpha <= 1.0:
                                raise ValueError(
                                    f"vq_{branch}_phase_accumulator_alpha must be "
                                    f"in [0, 1], got {phase_accum_alpha}"
                                )

                        attr_name = f"_vq_{branch}_{component}_accum"
                        valid_attr_name = f"{attr_name}_valid"
                        self._ensure_vq_accumulator(
                            attr_name=attr_name,
                            valid_attr_name=valid_attr_name,
                            batch_size=obs_td.batch_size[0],
                            dim=(
                                model_module.config.phase_state_dim
                                if component == "state"
                                else model_module.config.n_timing_phases
                            ),
                            device=self.device,
                        )
                        obs_td[f"vq_{branch}_{component}_accum"] = getattr(
                            self,
                            attr_name,
                        ).clone()
                        obs_td[f"vq_{branch}_{component}_accum_valid"] = getattr(
                            self,
                            valid_attr_name,
                        ).clone()
                        if branch == "prior" and component == "phase":
                            obs_td["vq_prior_phase_blend_alpha"] = torch.full(
                                (obs_td.batch_size[0],),
                                phase_accum_alpha,
                                device=self.device,
                            )

                model_outs = self.agent.model(obs_td)
                for branch in vq_accumulator_branches:
                    for component in vq_accumulator_components:
                        accum_next_key = f"vq_pae_{branch}_{component}_accum_next"
                        if (
                            use_vq_accumulator[branch][component]
                            and accum_next_key in model_outs
                        ):
                            self._store_vq_accumulator_next(
                                attr_name=f"_vq_{branch}_{component}_accum",
                                valid_attr_name=(
                                    f"_vq_{branch}_{component}_accum_valid"
                                ),
                                next_accum=model_outs[accum_next_key],
                            )

                selected_indices = None
                selected_phase = None
                selected_phase_used = None
                if "vq_pae_indices" in model_outs:
                    env_idx = 0
                    indices_key = (
                        "vq_pae_posterior_indices"
                        if action_key == "privileged_action"
                        else "vq_pae_prior_indices"
                    )
                    selected_indices = model_outs.get(
                        indices_key, model_outs["vq_pae_indices"]
                    )
                    manifold_idx = int(selected_indices[env_idx].item())
                    phase_key = (
                        "vq_pae_posterior_phase"
                        if action_key == "privileged_action"
                        else "vq_pae_prior_phase"
                    )
                    phase_used_key = (
                        "vq_pae_posterior_phase_used"
                        if action_key == "privileged_action"
                        else "vq_pae_prior_phase_used"
                    )
                    frequency_key = (
                        "vq_pae_posterior_frequency"
                        if action_key == "privileged_action"
                        else "vq_pae_prior_frequency"
                    )
                    offset_key = (
                        "vq_pae_posterior_offset"
                        if action_key == "privileged_action"
                        else "vq_pae_prior_offset"
                    )
                    selected_phase = model_outs.get(
                        phase_key, model_outs.get("vq_pae_phase", None)
                    )
                    frequency = model_outs.get(
                        frequency_key, model_outs.get("vq_pae_frequency", None)
                    )
                    offset = model_outs.get(
                        offset_key, model_outs.get("vq_pae_offset", None)
                    )
                    selected_phase_used = model_outs.get(phase_used_key, None)
                    phase_str = (
                        f"{selected_phase[env_idx].detach().cpu().tolist()}"
                        if selected_phase is not None
                        else "N/A"
                    )
                    used_phase_str = (
                        f"{selected_phase_used[env_idx].detach().cpu().tolist()}"
                        if selected_phase_used is not None
                        else "N/A"
                    )
                    frequency_str = (
                        f"{frequency[env_idx].detach().cpu().tolist()}"
                        if frequency is not None
                        else "N/A"
                    )
                    offset_str = (
                        f"{offset[env_idx].detach().cpu().tolist()}"
                        if offset is not None
                        else "N/A"
                    )
                    next_phase_wrapped_str = "N/A"
                    accum_next = model_outs.get("vq_pae_prior_phase_accum_next", None)
                    if action_key == "prior_action" and accum_next is not None:
                        next_phase_wrapped_str = (
                            f"{accum_next[env_idx].detach().cpu().tolist()}"
                        )
                    elif selected_phase is not None and frequency is not None:
                        dt = float(model_module.config.time_step)
                        phase_source = (
                            selected_phase_used
                            if selected_phase_used is not None
                            else selected_phase
                        )
                        phase_env = phase_source[env_idx].detach()
                        frequency_env = frequency[env_idx].detach()
                        next_phase = (
                            phase_env
                            + frequency_env * float(model_frequency_scale) * dt
                        )
                        next_phase_wrapped = torch.remainder(next_phase + 0.5, 1.0) - 0.5
                        next_phase_wrapped_str = f"{next_phase_wrapped.cpu().tolist()}"
                    accum_alpha_debug = " ".join(
                        f"{branch}_{component}_accum_alpha="
                        f"{configured_vq_accum_alphas[branch][component]}"
                        for branch in vq_accumulator_branches
                        for component in vq_accumulator_components
                    )
                    print(
                        "[vq-pae-debug] "
                        f"step={step} env=0 manifold_idx={manifold_idx} "
                        f"action_key={action_key} "
                        f"phase={phase_str} used_phase={used_phase_str} "
                        f"frequency={frequency_str} "
                        f"offset={offset_str} "
                        f"next_phase_wrapped={next_phase_wrapped_str} "
                        f"prior_frequency_scale={model_frequency_scale:.3f} "
                        f"{accum_alpha_debug} "
                        f"motion_speed_scale={active_motion_speed_scale:.3f}"
                    )
                actor_latent = model_outs.get(latent_key, None) if latent_key is not None else None
                capture_latent = (
                    is_recording_vq_latent
                    and actor_latent is not None
                )
                if capture_latent:
                    if capture_step == 0:
                        latent_dim = actor_latent.shape[-1]
                        self._vq_latent_capture = torch.empty(
                            record_frames_nums,
                            actor_latent.shape[0],
                            latent_dim,
                            device=actor_latent.device,
                            dtype=actor_latent.dtype,
                        )
                        print(
                            "[vq-latent-loop] capturing "
                            f"{record_frames_nums} frames "
                            f"(action_key={action_key}, latent_key={latent_key})"
                        )
                    if self._vq_latent_capture is not None:
                        self._vq_latent_capture[capture_step].copy_(actor_latent.detach())
                    phase_for_capture = (
                        selected_phase_used
                        if selected_phase_used is not None
                        else selected_phase
                    )
                    if selected_indices is not None and phase_for_capture is not None:
                        phase_for_capture = phase_for_capture.reshape(
                            phase_for_capture.shape[0],
                            -1,
                        )
                        if self._vq_manifold_index_capture is None:
                            self._vq_manifold_index_capture = torch.full(
                                (
                                    record_frames_nums,
                                    selected_indices.shape[0],
                                ),
                                -1,
                                device=selected_indices.device,
                                dtype=torch.long,
                            )
                            self._vq_phase_capture = torch.full(
                                (
                                    record_frames_nums,
                                    phase_for_capture.shape[0],
                                    phase_for_capture.shape[1],
                                ),
                                float("nan"),
                                device=phase_for_capture.device,
                                dtype=phase_for_capture.dtype,
                            )
                        self._vq_manifold_index_capture[capture_step].copy_(
                            selected_indices.detach().long()
                        )
                        self._vq_phase_capture[capture_step].copy_(
                            phase_for_capture.detach()
                        )
                    capture_step += 1
                    if capture_step >= record_frames_nums:
                        if self._vq_latent_capture is None:
                            raise RuntimeError(
                                "Latent playback was enabled but no latent replay "
                                "buffer was captured."
                            )
                        num_envs = self._vq_latent_capture.shape[1]
                        device = self._vq_latent_capture.device
                        dtype = self._vq_latent_capture.dtype
                        self._vq_latent_loop_phase = torch.zeros(
                            num_envs,
                            device=device,
                            dtype=dtype,
                        )
                        print(
                            "[vq-latent-loop] capture complete; loop playback enabled "
                            f"(action_key={action_key}, frames={record_frames_nums}, "
                            f"speed_scale={configured_motion_speed_scale:.3f})"
                        )
                        self._save_vq_latent_manifold_plot(
                            model_module=model_module,
                            captured_latents=self._vq_latent_capture,
                            action_key=action_key,
                            latent_key=latent_key,
                            captured_indices=self._vq_manifold_index_capture,
                            captured_phases=self._vq_phase_capture,
                        )
                actions = self._select_actions(model_outs, action_key)

                obs, rewards, dones, terminated, extras = self.env.step(actions)
                obs = self.agent.add_agent_info_to_obs(obs)

                if collect_metrics and "eval_values" in extras:
                    for k, v in extras["eval_values"].items():
                        val = v.mean().item()
                        metric_sums[k] = metric_sums.get(k, 0.0) + val
                        metric_counts[k] = metric_counts.get(k, 0) + 1

                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.numel() > 0:
                    self._reset_vq_latent_loop(done_indices)
                    for branch in vq_accumulator_branches:
                        for component in vq_accumulator_components:
                            attr_name = f"_vq_{branch}_{component}_accum"
                            self._reset_vq_accumulator(
                                attr_name=attr_name,
                                valid_attr_name=f"{attr_name}_valid",
                                env_ids=done_indices,
                            )
                    terminated_indices = terminated.nonzero(as_tuple=False).squeeze(-1)
                    print(f"\n[reset-debug] step={step}")
                    print(f"  done_envs={done_indices.tolist()}")
                    print(
                        f"  terminated_envs={terminated_indices.tolist() if terminated_indices.numel() > 0 else []}"
                    )

                    if motion_manager is not None and motion_lib is not None:
                        done_motion_ids = motion_manager.motion_ids[done_indices]
                        done_motion_times = motion_manager.motion_times[done_indices]
                        done_motion_lengths = motion_lib.get_motion_length(done_motion_ids)
                        done_clip = motion_manager.get_done_tracks(done_indices)

                        for i in range(done_indices.shape[0]):
                            env_id = int(done_indices[i].item())
                            motion_id = int(done_motion_ids[i].item())
                            motion_time = float(done_motion_times[i].item())
                            motion_length = float(done_motion_lengths[i].item())
                            clip_end = bool(done_clip[i].item())
                            terminated_flag = bool(terminated[done_indices[i]].item())
                            reason = (
                                "clip_end"
                                if clip_end and not terminated_flag
                                else "termination_or_failure"
                            )
                            print(
                                "  "
                                f"env={env_id} motion_id={motion_id} "
                                f"motion_time={motion_time:.4f} motion_length={motion_length:.4f} "
                                f"clip_end={clip_end} terminated={terminated_flag} reason={reason}"
                            )
                step += 1
        except KeyboardInterrupt:
            print(f"\nStopped after {step} steps.")
            if collect_metrics and metric_counts:
                print("Average metrics:")
                for k in sorted(metric_counts.keys()):
                    avg = metric_sums[k] / metric_counts[k]
                    print(f"  {k}: {avg:.4f}")
        finally:
            if motion_manager is not None and original_motion_speed_scale is not None:
                motion_manager.speed_scale = original_motion_speed_scale

    def cleanup_after_evaluation(self) -> None:
        """Clear privileged evaluator state after cleanup."""
        self._privileged_eval_state = None
        super().cleanup_after_evaluation()

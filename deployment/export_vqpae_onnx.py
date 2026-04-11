"""ONNX export for VQ-PAE tracker policies.

Exports a ProtoMotions VQ-PAE tracker model to a unified ONNX model without
running a simulator. The export includes:

- observation construction from raw context inputs
- full VQ-PAE policy forward pass
- PD action processing

Compared to ``export_bm_tracker_onnx.py``, this exporter loads the full model
instead of only the actor because the VQ-PAE action path depends on the model's
internal latent branch.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


class _MockState:
    """Mock for CurrentStateView."""

    def __init__(self, num_envs: int, num_dofs: int, num_bodies: int, anchor_idx: int):
        import torch
        import torch.nn.functional as F

        self.dof_pos = torch.randn(num_envs, num_dofs)
        self.dof_vel = torch.randn(num_envs, num_dofs)
        self.anchor_rot = F.normalize(torch.randn(num_envs, 4), dim=-1)
        self.anchor_pos = torch.randn(num_envs, 3)
        self.root_local_ang_vel = torch.randn(num_envs, 3)
        self.rigid_body_pos = torch.randn(num_envs, num_bodies, 3)
        self.rigid_body_rot = F.normalize(torch.randn(num_envs, num_bodies, 4), dim=-1)
        self.rigid_body_vel = torch.randn(num_envs, num_bodies, 3)
        self.rigid_body_ang_vel = torch.randn(num_envs, num_bodies, 3)


class _MockMimic:
    """Mock for MimicContext."""

    def __init__(
        self,
        num_envs: int,
        num_future_steps: int,
        num_dofs: int,
        num_bodies: int,
    ):
        import torch
        import torch.nn.functional as F

        self.future_rot = F.normalize(
            torch.randn(num_envs, num_future_steps, num_bodies, 4), dim=-1
        )
        self.future_pos = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_vel = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_ang_vel = torch.randn(num_envs, num_future_steps, num_bodies, 3)
        self.future_dof_pos = torch.randn(num_envs, num_future_steps, num_dofs)
        self.future_dof_vel = torch.randn(num_envs, num_future_steps, num_dofs)
        self.future_anchor_rot = F.normalize(
            torch.randn(num_envs, num_future_steps, 4), dim=-1
        )
        self.future_anchor_pos = torch.randn(num_envs, num_future_steps, 3)
        self.future_anchor_vel = torch.randn(num_envs, num_future_steps, 3)
        self.future_anchor_ang_vel = torch.randn(num_envs, num_future_steps, 3)
        self.ref_anchor_pos = torch.randn(num_envs, 3)


class _MockHistorical:
    """Mock for HistoricalContext used by VQ-PAE export."""

    def __init__(
        self,
        num_envs: int,
        history_steps: int,
        num_dofs: int,
        num_bodies: int,
    ):
        import torch
        import torch.nn.functional as F

        self.rigid_body_pos = torch.randn(num_envs, history_steps, num_bodies, 3)
        self.rigid_body_rot = F.normalize(
            torch.randn(num_envs, history_steps, num_bodies, 4), dim=-1
        )
        self.rigid_body_vel = torch.randn(num_envs, history_steps, num_bodies, 3)
        self.rigid_body_ang_vel = torch.randn(num_envs, history_steps, num_bodies, 3)
        self.dof_pos = torch.randn(num_envs, history_steps, num_dofs)
        self.dof_vel = torch.randn(num_envs, history_steps, num_dofs)
        self.root_pos = self.rigid_body_pos[:, :, 0, :]
        self.root_rot = self.rigid_body_rot[:, :, 0, :]
        self.root_ang_vel = self.rigid_body_ang_vel[:, :, 0, :]
        self.root_local_ang_vel = torch.randn(num_envs, history_steps, 3)
        self.anchor_pos = self.rigid_body_pos[:, :, 0, :]
        self.anchor_rot = F.normalize(
            torch.randn(num_envs, history_steps, 4), dim=-1
        )
        self.anchor_vel = torch.randn(num_envs, history_steps, 3)
        self.anchor_ang_vel = torch.randn(num_envs, history_steps, 3)
        self.ground_heights = torch.zeros(num_envs, history_steps)
        self.body_contacts = torch.zeros(num_envs, history_steps, 2, dtype=torch.bool)
        self.actions = torch.randn(num_envs, history_steps, num_dofs)
        self.processed_actions = torch.randn(num_envs, history_steps, num_dofs)


class MockContext:
    """Minimal stand-in for EnvContext used only during VQ-PAE ONNX export tracing."""

    def __init__(
        self,
        num_envs: int,
        num_dofs: int,
        num_bodies: int,
        num_future_steps: int,
        anchor_idx: int,
        history_steps: int = 1,
    ):
        import torch

        self.current = _MockState(num_envs, num_dofs, num_bodies, anchor_idx)
        self.mimic = _MockMimic(num_envs, num_future_steps, num_dofs, num_bodies)
        self.historical = _MockHistorical(
            num_envs, history_steps, num_dofs, num_bodies
        )
        self.body_contacts = torch.zeros(num_envs, num_bodies, dtype=torch.bool)
        self.ground_heights = torch.zeros(num_envs)


def export_vqpae(
    checkpoint: str,
    output_dir: str,
    validate: bool = True,
) -> Path:
    import torch
    from tensordict import TensorDict

    from protomotions.utils.export_utils import (
        ObservationExportModule,
        ActionExportModule,
        UnifiedPipelineModule,
    )
    from protomotions.utils.hydra_replacement import get_class
    try:
        from deployment.export_bm_tracker_onnx import _resolve_attr_path, _build_yaml
    except ModuleNotFoundError:
        from export_bm_tracker_onnx import _resolve_attr_path, _build_yaml  # type: ignore

    checkpoint_path = Path(checkpoint)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load resolved configs (no simulator import required)
    # ------------------------------------------------------------------
    resolved_path = checkpoint_path.parent / "resolved_configs_inference.pt"
    if not resolved_path.exists():
        log.warning(
            "resolved_configs_inference.pt not found, falling back to "
            "resolved_configs.pt. Domain randomization may still be active!"
        )
        resolved_path = checkpoint_path.parent / "resolved_configs.pt"
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find resolved_configs*.pt in {checkpoint_path.parent}"
        )

    log.info(f"Loading configs from {resolved_path}")
    resolved = torch.load(resolved_path, map_location="cpu", weights_only=False)

    robot_config = resolved["robot"]
    env_config = resolved["env"]
    agent_config = resolved["agent"]
    simulator_config = resolved.get("simulator")

    # ------------------------------------------------------------------
    # 2. Auto-detect actor obs keys from agent config
    # ------------------------------------------------------------------
    ModelClass = get_class(agent_config.model._target_)
    model = ModelClass(agent_config.model)
    model.eval()

    preprocessor_in_keys = list(getattr(model._preprocessor, "in_keys", []))
    preprocessor_out_keys = set(getattr(model._preprocessor, "out_keys", []))
    trunk_in_keys = [
        key for key in getattr(model._trunk, "in_keys", []) if key != "vae_latent"
    ]
    external_policy_keys = list(preprocessor_in_keys)
    for key in trunk_in_keys:
        if key not in preprocessor_out_keys and key not in external_policy_keys:
            external_policy_keys.append(key)

    log.info(f"Derived external VQ-PAE input keys: {external_policy_keys}")

    # ------------------------------------------------------------------
    # 3. Extract dimensions from configs
    # ------------------------------------------------------------------
    num_dofs = robot_config.kinematic_info.num_dofs
    num_bodies = len(robot_config.kinematic_info.body_names)
    body_names = list(robot_config.kinematic_info.body_names)
    joint_names = list(robot_config.kinematic_info.dof_names)
    anchor_body_name = robot_config.anchor_body_name
    anchor_body_index = robot_config.anchor_body_index
    root_body_index = 0

    mimic_ctrl_cfg = env_config.control_components.get("mimic")
    if mimic_ctrl_cfg is None:
        raise ValueError("env_config.control_components must contain 'mimic'")
    raw_future_steps = mimic_ctrl_cfg.future_steps
    if isinstance(raw_future_steps, int):
        future_step_indices = list(range(1, raw_future_steps + 1))
    else:
        future_step_indices = list(raw_future_steps)
    num_future_steps = max(future_step_indices) if future_step_indices else 1

    history_steps = max(1, int(getattr(env_config, "num_state_history_steps", 1) or 1))

    # Resolve MuJoCo-specific timing.
    control_dt = 0.02
    physics_dt = 0.001
    decimation = 20
    pd_target_max_accel = None
    if simulator_config is not None:
        try:
            from protomotions.simulator.factory import update_simulator_config_for_test

            mj_sim_cfg = update_simulator_config_for_test(
                current_simulator_config=simulator_config,
                new_simulator="mujoco",
                robot_config=robot_config,
            )
            physics_dt = 1.0 / mj_sim_cfg.sim.fps
            decimation = mj_sim_cfg.sim.decimation
            control_dt = physics_dt * decimation
        except Exception as exc:
            log.warning(f"Could not apply sim2sim conversion: {exc}")
            sim_cfg = getattr(simulator_config, "sim", None)
            if sim_cfg is not None:
                fps = getattr(sim_cfg, "fps", None)
                dec = getattr(sim_cfg, "decimation", None)
                if fps and dec:
                    physics_dt = 1.0 / fps
                    decimation = dec
                    control_dt = physics_dt * decimation
        accel = getattr(simulator_config, "pd_target_max_accel", None)
        if accel is not None:
            pd_target_max_accel = float(accel)

    log.info(
        f"Robot: {num_dofs} DOFs, {num_bodies} bodies, "
        f"anchor={anchor_body_name}(idx={anchor_body_index})"
    )
    log.info(
        f"Timing: control_dt={control_dt}s physics_dt={physics_dt}s "
        f"decimation={decimation}"
    )
    log.info(f"Future steps: {future_step_indices} ({num_future_steps} total)")

    # ------------------------------------------------------------------
    # 4. Build MockContext for ONNX tracing shape inference
    # ------------------------------------------------------------------
    mock = MockContext(
        num_envs=1,
        num_dofs=num_dofs,
        num_bodies=num_bodies,
        num_future_steps=num_future_steps,
        anchor_idx=anchor_body_index,
        history_steps=history_steps,
    )

    # ------------------------------------------------------------------
    # 5. Build ObservationExportModule
    # ------------------------------------------------------------------
    obs_configs = {
        k: v
        for k, v in env_config.observation_components.items()
        if k in set(external_policy_keys)
    }
    missing = set(external_policy_keys) - set(env_config.observation_components.keys())
    if missing:
        raise ValueError(
            f"Model requires obs keys missing from env_config.observation_components: {missing}"
        )

    log.info(f"Observation components for export: {list(obs_configs.keys())}")
    obs_module = ObservationExportModule(obs_configs, mock, device="cpu")
    obs_module.eval()

    obs_input_keys = obs_module.get_input_keys()
    obs_output_keys = obs_module.get_output_keys()
    log.info(f"  Context input keys: {obs_input_keys}")
    log.info(f"  Observation outputs: {obs_output_keys}")

    # ------------------------------------------------------------------
    # 6. Reconstruct actor-only and load weights
    # ------------------------------------------------------------------
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    sample_inputs = [_resolve_attr_path(k, mock) for k in obs_input_keys]
    with torch.no_grad():
        mock_obs_out = obs_module(*sample_inputs)
        mock_obs_td = TensorDict(
            {k: v for k, v in zip(obs_output_keys, mock_obs_out)},
            batch_size=[1],
        )
        model(mock_obs_td)  # materialises LazyLinear layers; For VQ_PAE, it has no Lazy layers;

    model.load_state_dict(ckpt["model"])
    model.eval()

    # ------------------------------------------------------------------
    # 7. Build ActionExportModule
    # ------------------------------------------------------------------
    action_module = ActionExportModule(env_config.action_config, device="cpu")
    action_module.eval()

    # ------------------------------------------------------------------
    # 8. Compose UnifiedPipelineModule
    # ------------------------------------------------------------------
    unified = UnifiedPipelineModule(
        observation_module=obs_module,
        policy_module=model,
        action_module=action_module,
        policy_in_keys=external_policy_keys,
        policy_action_key="privileged_action",
    )
    unified.cpu().eval()
    # ------------------------------------------------------------------
    # 9. Collect sample inputs and verify forward pass
    # ---------------------------------------------------------------
    input_shapes = {k: list(v.shape) for k, v in zip(obs_input_keys, sample_inputs)}

    with torch.no_grad():
        actions, pd_targets, stiffness_t, damping_t = unified(*sample_inputs)
    log.info(
        f"Forward pass OK: actions={list(actions.shape)}, "
        f"pd_targets={list(pd_targets.shape)}"
    )

    # ------------------------------------------------------------------
    # 10. Export to ONNX
    # ------------------------------------------------------------------
    def _sanitize(name: str) -> str:
        return name.replace(".", "_").replace("[", "_").replace("]", "_")

    onnx_input_names = [_sanitize(k) for k in obs_input_keys]
    onnx_output_names = [
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    ]

    onnx_path = output_path / "unified_pipeline.onnx"
    log.info(f"Exporting ONNX to {onnx_path} ...")
    torch.onnx.export(
        unified,
        tuple(sample_inputs),
        str(onnx_path),
        input_names=onnx_input_names,
        output_names=onnx_output_names,
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{name: {0: "batch_size"} for name in onnx_input_names},
            **{name: {0: "batch_size"} for name in onnx_output_names},
        },
        dynamo=False,
    )
    log.info(f"ONNX exported -> {onnx_path}")

    # ------------------------------------------------------------------
    # 11. Read back actual ONNX names (ONNX may rename inputs)
    # ------------------------------------------------------------------
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_in_names = [inp.name for inp in session.get_inputs()]
    actual_out_names = [out.name for out in session.get_outputs()]

    sanitized_to_key = {_sanitize(k): k for k in obs_input_keys}
    onnx_name_to_key: dict[str, str] = {}
    for onnx_name in actual_in_names:
        base = onnx_name
        for suffix in (".1", ".2", ".3", "_1", "_2", "_3"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[base]
        elif onnx_name in sanitized_to_key:
            onnx_name_to_key[onnx_name] = sanitized_to_key[onnx_name]
        else:
            log.warning(f"Cannot map ONNX input '{onnx_name}' to a semantic key")

    if validate:
        import numpy as np

        log.info("Validating with onnxruntime ...")
        key_to_tensor = {k: t for k, t in zip(obs_input_keys, sample_inputs)}
        ort_inputs = {
            name: key_to_tensor[onnx_name_to_key[name]].detach().numpy()
            for name in actual_in_names
            if name in onnx_name_to_key
        }
        ort_outputs = session.run(actual_out_names, ort_inputs)
        pytorch_outputs = [
            actions.detach().numpy(),
            pd_targets.detach().numpy(),
            stiffness_t.detach().numpy(),
            damping_t.detach().numpy(),
        ]
        for i, (name, pt_out) in enumerate(zip(onnx_output_names, pytorch_outputs)):
            diff = np.abs(ort_outputs[i] - pt_out).max()
            status = "OK" if diff < 1e-4 else "WARN"
            log.info(f"  {status}  {name}: max_diff = {diff:.2e}")
        log.info("Validation complete")

    stiffness_vals = [
        float(robot_config.control.control_info[j].stiffness) for j in joint_names
    ]
    damping_vals = [
        float(robot_config.control.control_info[j].damping) for j in joint_names
    ]

    effort_limits = None
    try:
        effort_limits = [
            float(robot_config.control.control_info[j].effort) for j in joint_names
        ]
    except (AttributeError, KeyError):
        pass

    mjcf_path = robot_config.asset.asset_file_name
    control_type = "BUILT_IN_PD"
    action_cfg = env_config.action_config
    if hasattr(action_cfg, "_target_"):
        control_type = action_cfg._target_.rsplit(".", 1)[-1]

    yaml_content = _build_yaml(
        onnx_in_names=actual_in_names,
        onnx_out_names=actual_out_names,
        onnx_name_to_key=onnx_name_to_key,
        input_shapes=input_shapes,
        obs_input_keys=obs_input_keys,
        actor_obs_configs=obs_configs,
        joint_names=joint_names,
        body_names=body_names,
        stiffness=stiffness_vals,
        damping=damping_vals,
        effort_limits=effort_limits,
        pd_target_max_accel=pd_target_max_accel,
        anchor_body_name=anchor_body_name,
        anchor_body_index=anchor_body_index,
        root_body_index=root_body_index,
        num_bodies=num_bodies,
        num_dofs=num_dofs,
        mjcf_path=mjcf_path,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        future_step_indices=future_step_indices,
        checkpoint=str(checkpoint_path),
        control_type=control_type,
    )
    yaml_content["type"] = "vqpae_unified_pipeline"
    yaml_content["metadata"]["policy_module_target"] = agent_config.model._target_

    yaml_path = output_path / "unified_pipeline.yaml"
    import yaml

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=None, sort_keys=False)
    log.info(f"YAML metadata -> {yaml_path}")

    return onnx_path


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export a VQ-PAE tracker policy to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <checkpoint_dir>/compiled_models/)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip onnxruntime validation after export",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = (
        str(Path(args.checkpoint).parent / "compiled_models")
        if args.output is None
        else args.output
    )
    onnx_path = export_vqpae(
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        validate=not args.no_validate,
    )
    log.info("Done!")
    log.info(f"Model exported to: {onnx_path}")
    log.info(f"YAML sidecar: {onnx_path.with_suffix('.yaml')}")

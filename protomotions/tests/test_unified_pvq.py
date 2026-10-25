# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified posterior regression tests without simulator or external datasets."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from protomotions.agents.multi_source_distill import model as model_module
from protomotions.agents.multi_source_distill.data import complete_observations
from protomotions.agents.multi_source_distill.model import (
    JointModelConfig,
    JointPVQModel,
    joint_model_config_from_dict,
)
from protomotions.agents.multi_source_distill.unified import (
    ENCODER_MODES,
    NON_FINGER_IDS,
    FINGER_IDS,
    FUTURE_STEPS,
    KEY_IDS,
    configure_student_observations,
    pack_unified_target,
    unified_target_mask,
)
from protomotions.envs.obs.target_poses import build_max_coords_target_poses
from protomotions.train_multi_source_distill import resolve_encoder_mode


def config(mode="single_unified"):
    return JointModelConfig(
        obs_dims=dict(
            max_coords_obs=830,
            previous_actions=153,
            mimic_target_poses=1248,
            intermimic_target_obs=1420,
            intermimic_object_obs=171,
        ),
        num_actions=153,
        encoder_mode=mode,
        object_obs_has_contacts=False,
        encoder_widths=(16,),
        decoder_widths=(16,),
        latent_dim=8,
        num_quantizers=2,
        num_embeddings=8,
    )


def observations(task, batch=3):
    obs = {k: torch.randn(batch, d) for k, d in config().obs_dims.items()}
    obs["unified_human_target"] = torch.randn(batch, 2496)
    return obs


def inputs(task, mode="single_unified", batch=3):
    mask = torch.ones(batch, 1, dtype=torch.bool) if task == "hoi" else None
    return complete_observations(observations(task, batch), task, config(mode), mask)


def teacher_config(task):
    component = lambda params: SimpleNamespace(static_params=params)
    observations = {
        "mimic_target_poses": component(dict(with_relative=True, with_velocities=True)),
        "intermimic_object_obs": component(dict(include_object_contacts=False)),
        "intermimic_target_obs": component(
            dict(
                key_body_ids=KEY_IDS,
                non_finger_body_ids=NON_FINGER_IDS,
                include_object_contact_features=False,
            )
        ),
    }
    return {
        "robot": SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=list(range(52)))
        ),
        "env": SimpleNamespace(
            observation_components=observations,
            control_components={
                "mimic": SimpleNamespace(future_steps=1),
                "intermimic": SimpleNamespace(future_steps=[1, 16]),
            },
        ),
    }


def test_target_field_order_and_contact_preservation():
    # Encode field/body/horizon independently so an accidental reshape is visible.
    body_features = torch.arange(2 * 52 * 24).reshape(1, 2, 52, 24).float()
    fields = body_features.split((3, 3, 6, 6, 3, 3), -1)
    raw = torch.cat([f.flatten(2) for f in fields], -1).flatten(1)
    native = torch.arange(1420).reshape(1, 1420).float()
    obs = dict(unified_human_target=raw, intermimic_target_obs=native)
    for task in ("hoi", "locomotion"):
        values, flags = pack_unified_target(obs, task, torch.tensor([[task == "hoi"]]))
        mask = unified_target_mask(flags)
        values, mask = values.view(1, 2, -1), mask.view(1, 2, -1)
        assert values.shape == (1, 2, 1439)
        torch.testing.assert_close(
            values[..., :528], body_features[:, :, NON_FINGER_IDS].flatten(2)
        )
        if task == "hoi":
            assert not values[..., 528:1248].any()
            assert not mask[..., 528:1248].any()
            torch.testing.assert_close(
                values[..., 1248:], native.view(1, 2, 710)[..., 519:]
            )
            assert mask[..., -104:].all()  # All body contacts, including fingers.
            assert flags.tolist() == [[[True, False], [True, False]]]
        else:
            torch.testing.assert_close(
                values[..., 528:1248], body_features[:, :, FINGER_IDS].flatten(2)
            )
            assert not values[..., 1248:].any()
            assert flags.tolist() == [[[False, True], [False, True]]]


@pytest.mark.parametrize("mode", ENCODER_MODES)
@pytest.mark.parametrize("task", ("hoi", "locomotion"))
def test_modes_backward_and_checkpoint_roundtrip(mode, task, tmp_path):
    model = JointPVQModel(config(mode))
    obs = inputs(task, mode)
    model.update_normalizers(obs, task)
    output = model(obs, task)
    assert output["action"].shape == (3, 153)
    (output["action"].square().mean() + output["vq_loss"].mean()).backward()
    active = (
        model.encoder
        if mode == "single_unified"
        else getattr(model, "hoi_encoder" if task == "hoi" else "loco_encoder")
    )
    assert active[0].weight.grad is not None
    if mode != "dual_native":
        assert active[0].in_features == 4036
    path = tmp_path / "model.pt"
    torch.save(dict(model_config=asdict(model.config), model=model.state_dict()), path)
    saved = torch.load(path, weights_only=False)
    restored = JointPVQModel(joint_model_config_from_dict(saved["model_config"]))
    restored.load_state_dict(saved["model"], strict=True)
    torch.testing.assert_close(restored(obs, task)["action"], output["action"])


def test_unified_encoders_have_identical_inputs_and_statistics():
    single, dual = JointPVQModel(config()), JointPVQModel(config("dual_unified"))
    # Copy common weights, then give both independent encoders the shared weights.
    common = {
        k: v for k, v in single.state_dict().items() if not k.startswith("encoder.")
    }
    dual.load_state_dict(common, strict=False)
    dual.loco_encoder.load_state_dict(single.encoder.state_dict())
    dual.hoi_encoder.load_state_dict(single.encoder.state_dict())
    for task in ("hoi", "locomotion"):
        obs = inputs(task)
        single.update_normalizers(obs, task)
        dual.update_normalizers(obs, task)
        torch.testing.assert_close(
            single(obs, task)["action"], dual(obs, task)["action"]
        )


@pytest.mark.parametrize("task", ("hoi", "locomotion"))
def test_invalid_features_never_affect_statistics_or_actions(task):
    model = JointPVQModel(config())
    obs = inputs(task)
    changed = deepcopy(obs)
    changed["unified_target"][~unified_target_mask(changed["unified_target_flags"])] = (
        float("nan")
    )
    other = deepcopy(model)
    model.update_normalizers(obs, task)
    other.update_normalizers(changed, task)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key])
    torch.testing.assert_close(
        model(obs, task)["action"], other(changed, task)["action"]
    )


def test_missing_object_masks_all_interaction_but_not_body():
    obs = observations("hoi")
    result = complete_observations(
        obs, "hoi", config(), torch.zeros(3, 1, dtype=torch.bool)
    )
    mask = unified_target_mask(result["unified_target_flags"]).view(3, 2, -1)
    assert mask[..., :528].all()
    assert not mask[..., 528:].any()
    assert not result["object_feature_mask"].any()


@pytest.mark.parametrize("mode", ENCODER_MODES)
@pytest.mark.parametrize("task", ("hoi", "locomotion"))
def test_rollout_payload_omits_only_unused_unified_fields(mode, task):
    raw = observations(task)
    before = {key: value.clone() for key, value in raw.items()}
    valid = torch.ones(3, 1, dtype=torch.bool) if task == "hoi" else None
    result = complete_observations(raw, task, config(mode), valid)
    for key in raw:
        torch.testing.assert_close(raw[key], before[key], rtol=0, atol=0)
    targets = {"mimic_target_poses", "intermimic_target_obs"}
    if mode == "dual_native":
        assert targets <= result.keys()
        assert "unified_target" not in result
    else:
        assert not targets.intersection(result)
        assert "unified_target_mask" not in result
        assert result["unified_target_flags"].dtype == torch.bool
        assert result["unified_target_flags"].numel() == 3 * 4
        # Saved per-sample bytes: two float32 native targets + the dense bool mask.
        removed_bytes = (1248 + 1420) * 4 + 2878
        assert removed_bytes == 13550
        assert sum(t.numel() * t.element_size() for t in result.values()) == 3 * 16304


@pytest.mark.parametrize("contacts", (False, True))
def test_multi_object_native_masks_follow_actual_contact_layout(contacts):
    cfg = config("dual_native")
    # Two objects: object 0 absent, object 1 present; geometry remains valid.
    width = 16 if contacts else 15
    cfg = replace(
        cfg,
        num_objects=2,
        object_obs_has_contacts=contacts,
        obs_dims={**cfg.obs_dims, "intermimic_object_obs": 2 * width + 156},
    )
    obs = observations("hoi", 1)
    obs["intermimic_object_obs"] = torch.randn(1, 2 * width + 156)
    result = complete_observations(obs, "hoi", cfg, torch.tensor([[False, True]]))
    offset = 0
    for field_width in (3, 6, 3, 3, 1) if contacts else (3, 6, 3, 3):
        field = result["object_feature_mask"][:, offset : offset + 2 * field_width]
        assert not field[:, :field_width].any()
        assert field[:, field_width:].all()
        offset += 2 * field_width
    assert result["object_feature_mask"][:, offset:].all()


def test_old_checkpoint_without_new_fields_loads_strictly():
    cfg = config("dual_native")
    stored = asdict(cfg)
    for key in (
        "encoder_mode",
        "unified_layout_version",
        "unified_non_finger_ids",
        "unified_finger_ids",
        "unified_future_steps",
        "object_obs_has_contacts",
    ):
        stored.pop(key)
    old = JointPVQModel(joint_model_config_from_dict(stored))
    old.load_state_dict(JointPVQModel(cfg).state_dict(), strict=True)
    assert old.config.encoder_mode == "dual_native"


def test_layout_change_keeps_native_resume_and_saved_unified_layout(monkeypatch):
    native, unified = asdict(config("dual_native")), asdict(config())
    assert native["unified_future_steps"] is None
    assert unified["unified_future_steps"] == FUTURE_STEPS
    # A later layout version applies only to new unified configs.
    monkeypatch.setattr(model_module, "FUTURE_STEPS", (1, 8))
    assert asdict(joint_model_config_from_dict(native)) == asdict(config("dual_native"))
    assert joint_model_config_from_dict(unified).unified_future_steps == FUTURE_STEPS
    assert config().unified_future_steps == (1, 8)


def test_resume_restores_mode_and_rejects_cross_mode(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(model_config=asdict(config())), path)
    manifest = SimpleNamespace(
        sources=[SimpleNamespace(task=t) for t in ("locomotion", "hoi")]
    )
    configs = [teacher_config(s.task) for s in manifest.sources]
    args = SimpleNamespace(encoder_mode=None, resume=str(path), warm_start=None)
    resolve_encoder_mode(args, configs, manifest)
    assert args.encoder_mode == "single_unified"
    args.encoder_mode = "dual_unified"
    with pytest.raises(ValueError, match="differs from checkpoint"):
        resolve_encoder_mode(args, configs, manifest)


def test_teacher_reference_stays_one_step_and_actions_unchanged():
    original = teacher_config("locomotion")
    changed = deepcopy(original)
    configure_student_observations(changed, "locomotion", "single_unified")
    assert original["env"].control_components["mimic"].future_steps == 1
    assert changed["env"].control_components["mimic"].future_steps == [1, 16]
    static = changed["env"].observation_components["mimic_target_poses"].static_params
    torch.manual_seed(4)
    pos = torch.randn(2, 52, 3)
    rot = torch.randn(2, 52, 4)
    rot /= rot.norm(dim=-1, keepdim=True)
    future_rot = torch.randn(2, 2, 52, 4)
    future_rot /= future_rot.norm(dim=-1, keepdim=True)
    state = dict(
        current_state_body_pos=pos,
        current_state_body_rot=rot,
        current_state_body_vel=pos * 0.2,
        current_state_body_ang_vel=pos * 0.1,
    )
    future = dict(
        mimic_ref_pos=torch.randn(2, 2, 52, 3),
        mimic_ref_rot=future_rot,
        mimic_ref_vel=torch.randn(2, 2, 52, 3),
        mimic_ref_ang_vel=torch.randn(2, 2, 52, 3),
    )
    old = build_max_coords_target_poses(
        **state,
        **{k: v[:, :1] for k, v in future.items()},
        with_relative=True,
        with_velocities=True,
        w_last=True,
    )
    new = build_max_coords_target_poses(**state, **future, **static, w_last=True)
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    teacher = torch.nn.Linear(1248, 153)
    torch.testing.assert_close(teacher(old), teacher(new), rtol=0, atol=0)
    full = build_max_coords_target_poses(
        **state, **future, with_relative=True, with_velocities=True, w_last=True
    )
    torch.testing.assert_close(full[:, :1248], old)
    assert not torch.equal(full[:, 1248:], old)


def test_neutral_fingers_remain_valid_and_rotation_uses_current_inverse():
    # Current heading +90 degrees, neutral world-space reference rotation.
    current_rot = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]).repeat(1, 52, 1)
    ref_rot = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(1, 2, 52, 1)
    zero = torch.zeros(1, 52, 3)
    future_zero = torch.zeros(1, 2, 52, 3)
    ref_pos = future_zero.clone()
    ref_pos[..., 0] = 1.0
    raw = build_max_coords_target_poses(
        zero,
        current_rot,
        zero,
        zero,
        ref_pos,
        ref_rot,
        future_zero,
        future_zero,
        with_velocities=True,
        with_relative=True,
        w_last=True,
    )
    target, flags = pack_unified_target(
        {"unified_human_target": raw}, "locomotion", torch.zeros(1, 1, dtype=torch.bool)
    )
    mask = unified_target_mask(flags)
    fingers = target.view(1, 2, -1)[..., 528:1248].reshape(1, 2, 30, 24)
    expected_pos = torch.tensor([0.0, -1.0, 0.0])
    expected_rot = torch.tensor([0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(fingers[0, 0, 0, :3], expected_pos, atol=1e-6, rtol=0)
    torch.testing.assert_close(fingers[0, 0, 0, 12:18], expected_rot, atol=1e-6, rtol=0)
    assert mask.view(1, 2, -1)[..., 528:1248].all()
    assert flags[..., 1].all()


def test_student_horizons_clamp_to_motion_end():
    from protomotions.envs.control.mimic_control import MimicControl

    class CapturedQuery(Exception):
        pass

    recorded = []

    def capture(ids, times):
        recorded.append(times.clone())
        raise CapturedQuery

    control = MimicControl.__new__(MimicControl)
    control.config = SimpleNamespace(future_steps=[1, 16])
    control._future_time_offsets_cache_key = None
    control.env = SimpleNamespace(
        device=torch.device("cpu"),
        dt=1 / 30,
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor([0, 1]), motion_times=torch.tensor([0.0, 0.99])
        ),
        motion_lib=SimpleNamespace(
            get_motion_length=lambda ids: torch.ones(2), get_motion_state=capture
        ),
    )
    with pytest.raises(CapturedQuery):
        control.populate_context(SimpleNamespace(env_ids=None))
    torch.testing.assert_close(
        recorded[0], torch.tensor([0.0, 0.99, 1 / 30, 16 / 30, 1.0, 1.0])
    )


def test_hoi_native_target_configuration_is_preserved():
    original = teacher_config("hoi")
    changed = deepcopy(original)
    configure_student_observations(changed, "hoi", "single_unified")
    for key, observation in original["env"].observation_components.items():
        assert (
            changed["env"].observation_components[key].static_params
            == observation.static_params
        )
    assert original["env"].control_components["intermimic"].future_steps == [1, 16]
    assert "unified_human_target" not in original["env"].observation_components


def test_unsupported_layout_fails_explicitly():
    with pytest.raises(ValueError, match="Unsupported unified"):
        JointPVQModel(replace(config(), unified_future_steps=(1, 8)))
    old_teacher = teacher_config("hoi")
    old_teacher["env"].observation_components["intermimic_target_obs"].static_params[
        "include_object_contact_features"
    ] = True
    with pytest.raises(ValueError, match="v9_3 target"):
        configure_student_observations(old_teacher, "hoi", "single_unified")


def distributed_worker(rank, rendezvous, mode):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        torch.manual_seed(5)
        model = JointPVQModel(config(mode))
        ddp = DistributedDataParallel(model, find_unused_parameters=True)
        task = "hoi" if rank else "locomotion"
        obs = inputs(task, mode, batch=2)
        model.update_normalizers(obs, task)
        counts = model.normalizers["unified_target"].count.view(2, -1)
        assert (counts[:, :528] == 4).all()
        assert (counts[:, 528:] == 2).all()
        opt = torch.optim.Adam(model.parameters())
        for _ in range(2):
            opt.zero_grad()
            out = ddp(obs, task)
            (out["action"].square().mean() + out["vq_loss"].mean()).backward()
            opt.step()
        for value in model.parameters():
            reference = value.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(reference, value)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ("single_unified", "dual_unified"))
def test_two_task_distributed_update(mode, tmp_path):
    mp.spawn(
        distributed_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", mode),
        nprocs=2,
        join=True,
    )

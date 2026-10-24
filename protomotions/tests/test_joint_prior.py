# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regression tests; no simulator, GPU, or motion assets required."""

import errno
import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator
from protomotions.agents.joint_prior.conditioning import (
    PriorConditioning,
    expand_feature_mask,
    local_history,
    reference_residual,
)
from protomotions.agents.joint_prior.config import JointPriorConfig
from protomotions.agents.joint_prior.model import JointPriorModel
from protomotions.agents.joint_prior.runtime import (
    LivePriorPolicy,
    load_prior_checkpoint,
    save_prior_checkpoint,
)
from protomotions.agents.multi_source_distill.config import (
    DistillationSource,
    SourceManifest,
)
from protomotions.agents.multi_source_distill.data import complete_observations
from protomotions.agents.multi_source_distill.model import (
    JointModelConfig,
    JointPVQModel,
    task_balanced_source_weights,
    weighted_sample_loss,
)
from protomotions.agents.multi_source_distill.runtime import EvaluationAgent
from protomotions.train_joint_prior import (
    evaluation_report,
    parser,
    posterior_contract,
    publish_report,
    resolve_sampling,
    rollout_probability,
    training_contract,
    validate_resume,
)
from protomotions.train_multi_source_distill import WandbRun


def motor(num_quantizers=2):
    return JointPVQModel(
        JointModelConfig(
            obs_dims={
                "max_coords_obs": 6,
                "previous_actions": 3,
                "mimic_target_poses": 5,
                "intermimic_target_obs": 7,
                "intermimic_object_obs": 22,
            },
            num_actions=3,
            latent_dim=8,
            num_quantizers=num_quantizers,
            num_embeddings=8,
            encoder_widths=(16,),
            decoder_widths=(16,),
        )
    )


def config(num_layers=1, **kwargs):
    return JointPriorConfig(
        num_bodies=2,
        token_dim=16,
        num_layers=num_layers,
        num_heads=2,
        feedforward_dim=32,
        **kwargs,
    )


def raw_state(n, bodies):
    pos = torch.zeros(n, bodies, 3)
    rot = torch.zeros(n, bodies, 4)
    rot[..., 3] = 1
    return SimpleNamespace(
        rigid_body_pos=pos,
        rigid_body_rot=rot,
        rigid_body_vel=pos.clone(),
        rigid_body_ang_vel=pos.clone(),
        rigid_body_object_contacts=torch.zeros(n, bodies, dtype=torch.bool),
    )


class FakeMotionLib:
    def get_motion_length(self, ids):
        return torch.full(ids.shape, 10.0)

    def get_motion_state(self, ids, times):
        state = raw_state(len(ids), 2)
        state.rigid_body_pos[..., 0] = times[:, None]
        state.rigid_body_vel[..., 0] = 1
        state.rigid_body_ang_vel[..., 2] = 2
        return state


class FakeSceneLib:
    def get_scene_pose(self, ids, times, **kwargs):
        state = raw_state(len(ids), 1)
        state.rigid_body_pos[..., 0] = 2 * times[:, None] + 10
        return SimpleNamespace(
            root_pos=state.rigid_body_pos,
            root_rot=state.rigid_body_rot,
            root_vel=state.rigid_body_vel,
            root_ang_vel=state.rigid_body_ang_vel,
        )


def environment():
    robot, objects = raw_state(2, 2), raw_state(2, 1)
    objects.rigid_body_pos[..., 0] = 3
    scene = SimpleNamespace(
        object_pos=objects.rigid_body_pos,
        object_rot=objects.rigid_body_rot,
        object_vel=objects.rigid_body_vel,
        object_ang_vel=objects.rigid_body_ang_vel,
        object_valid_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    return SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.zeros(2, dtype=torch.long),
        config=SimpleNamespace(ref_object_respawn_offset=0),
        context=SimpleNamespace(current=robot, scene=scene),
        object_valid_mask=lambda: scene.object_valid_mask,
        motion_manager=SimpleNamespace(
            motion_ids=torch.zeros(2, dtype=torch.long),
            motion_times=torch.zeros(2),
            config=SimpleNamespace(speed_scale=1.0),
        ),
        motion_lib=FakeMotionLib(),
        scene_lib=FakeSceneLib(),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda pos,
        **kwargs: (torch.zeros_like(pos)),
    )


def inputs(model, task):
    raw = {k: torch.randn(2, d) for k, d in model.config.obs_dims.items()}
    return complete_observations(
        raw, task, model.config, torch.ones(2, 1, dtype=torch.bool)
    )


@pytest.mark.parametrize("task", ["locomotion", "hoi"])
def test_code_lookup_matches_original_decoder_and_does_not_need_reference(task):
    posterior = motor().eval()
    obs = inputs(posterior, task)
    out = posterior(obs, task)
    obs.pop("mimic_target_poses")
    obs.pop("intermimic_target_obs")
    torch.testing.assert_close(
        posterior.decode_indices(obs, out["indices"]), out["action"]
    )


@pytest.mark.parametrize("task", ["locomotion", "hoi"])
@pytest.mark.parametrize("autoregressive", [False, True])
def test_masked_values_cannot_leak_and_only_prior_trains(task, autoregressive):
    torch.manual_seed(3)
    model = JointPriorModel(
        motor(),
        config(
            human_visible_prob=0.0,
            object_visible_prob=0.0,
            codebook_autoregressive=autoregressive,
        ),
    )
    obs = inputs(model.posterior, task)
    conditions = PriorConditioning(environment(), task, model.config, 1).observe(obs)
    assert conditions["human_reference"].shape == (2, 5, 24)
    assert conditions["object_history"].shape == (2, 5, 15)
    frozen = {k: v.clone() for k, v in model.posterior.state_dict().items()}
    model.train()
    assert not model.posterior.training
    model.update_normalizers(conditions)
    labels = model.posterior(obs, task)["indices"]
    before = model(conditions, labels).detach()
    dirty = {k: v.clone() for k, v in conditions.items()}
    for key in model.encoders:
        dirty[key][~expand_feature_mask(dirty[key], dirty[key + "_valid"])] = float(
            "nan"
        )
    torch.testing.assert_close(model(dirty, labels), before)
    assert before.shape == (2, 2, 8)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    loss = model.categorical_loss(model(conditions, labels), labels).mean()
    loss.backward()
    assert torch.isfinite(loss)
    heads = model.code_heads if autoregressive else [model.head]
    for head in heads:
        assert head.weight.grad.abs().sum() > 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
    assert all(p.grad is None for p in model.posterior.parameters())
    optimizer.step()
    for key, value in frozen.items():
        torch.testing.assert_close(
            model.posterior.state_dict()[key], value, rtol=0, atol=0
        )
    if task == "locomotion":
        for key in ("object_state", "object_history", "object_reference"):
            assert not conditions[key + "_valid"].any()
            assert not model.normalizers[key].count.any()


def test_reference_body_mask_covers_all_twelve_features():
    env, posterior = environment(), motor()
    observer = PriorConditioning(env, "hoi", config(), 1)
    observer.human_visible[:] = False
    observer.human_visible[:, 0, 1] = True
    observer.object_visible[:] = True
    obs = observer.observe(inputs(posterior, "hoi"))
    assert obs["human_reference_valid"].shape == (2, 5, 2)
    mask = expand_feature_mask(
        obs["human_reference"], obs["human_reference_valid"]
    ).view(2, 5, 2, 12)
    assert mask[:, 0, 1].all()
    assert not mask[:, :, 0].any()
    assert not mask[:, 1:].any()
    # Visible linear and angular velocity residuals are present, not pose-only.
    body = obs["human_reference"].view(2, 5, 2, 12)[:, 0, 1]
    torch.testing.assert_close(body[:, 6], torch.ones(2))
    torch.testing.assert_close(body[:, 11], torch.full((2,), 2.0))
    torch.testing.assert_close(
        obs["human_reference_time"][:, :4, 0],
        torch.tensor([[0.1, 0.2, 0.4, 1.6]]).expand(2, -1),
    )
    torch.testing.assert_close(
        obs["object_reference_time"], obs["human_reference_time"]
    )


def test_actual_history_and_reset_do_not_use_reference_or_cross_episodes():
    env, posterior = environment(), motor()
    observer = PriorConditioning(env, "hoi", config(mask_resample_prob=0.0), 1)
    obs = inputs(posterior, "hoi")
    observer.observe(obs)
    target_time = observer.long_target_time.clone()
    env.context.scene.object_pos[..., 0] = 4
    env.motion_manager.motion_times += 0.1
    observer.advance()
    after = observer.observe(obs)
    # The stored actual object was at x=3, not reference x=10.
    torch.testing.assert_close(after["object_history"][:, -1, 0], torch.full((2,), 3.0))
    assert after["object_history_valid"][:, -1].all()
    assert not after["object_history_valid"][:, :-1].any()
    unreached = target_time > 0.100001
    torch.testing.assert_close(
        observer.long_target_time[unreached], target_time[unreached]
    )
    observer.reset(torch.tensor([0]))
    after = observer.observe(obs)
    assert not after["object_history_valid"][0].any()
    assert after["object_history_valid"][1, -1].all()


@pytest.mark.parametrize("speed_scale", [0.5, 1.0, 2.0])
def test_snapshot_refreshes_on_scheduled_step_despite_float32_drift(speed_scale):
    env, posterior = environment(), motor()
    env.dt = 1 / 30
    env.motion_manager.speed_scale = speed_scale
    env.motion_manager.motion_times.fill_(60.0)
    observer = PriorConditioning(env, "hoi", config(mask_resample_prob=0.0), 1)
    observer.observe(inputs(posterior, "hoi"))
    motion_dt = env.dt * speed_scale
    target = env.motion_manager.motion_times + 128 * motion_dt
    observer.long_target_time.copy_(target)
    for _ in range(127):
        env.motion_manager.motion_times += motion_dt
        observer.advance()
        # No early refresh on the preceding control step.
        torch.testing.assert_close(observer.long_target_time, target, rtol=0, atol=0)
    env.motion_manager.motion_times += motion_dt
    # Reproduce the old predicate missing the intended arrival step.
    assert (target > env.motion_manager.motion_times + 1e-6).all()
    observer.advance()
    assert (observer.long_target_time > target).all()


@pytest.mark.parametrize("speed_scale", [0.5, 1.0, 2.0])
def test_snapshot_half_step_tolerance_scales_with_motion_clock(speed_scale):
    env, posterior = environment(), motor()
    env.dt = 1 / 30
    env.motion_manager.speed_scale = speed_scale
    observer = PriorConditioning(
        env,
        "hoi",
        config(mask_resample_prob=0.0, human_visible_prob=1.0, object_visible_prob=1.0),
        1,
    )
    observer.observe(inputs(posterior, "hoi"))
    motion_dt = env.dt * speed_scale
    env.motion_manager.motion_times.fill_(1.0)
    target = env.motion_manager.motion_times + torch.tensor([0.49, 0.51]) * motion_dt
    observer.long_target_time.copy_(target)
    observer.human_visible.fill_(False)
    observer.object_visible.fill_(False)
    observer.advance()
    assert observer.long_target_time[0] > target[0]
    assert observer.long_target_time[1] == target[1]
    # Arrival refreshes human/object snapshot masks together, not short references.
    for mask in (observer.human_visible, observer.object_visible):
        assert mask[0, -1].all()
        assert not mask[0, :-1].any()
        assert not mask[1].any()


def test_horizon_mask_hides_whole_human_horizons():
    env, posterior = environment(), motor()
    observer = PriorConditioning(
        env, "hoi", config(human_visible_prob=1.0, human_horizon_visible_prob=1.0), 1
    )
    assert observer.observe(inputs(posterior, "hoi"))["human_reference_valid"].all()
    observer = PriorConditioning(
        env,
        "hoi",
        config(
            human_visible_prob=1.0,
            object_visible_prob=1.0,
            human_horizon_visible_prob=0.0,
        ),
        1,
    )
    out = observer.observe(inputs(posterior, "hoi"))
    # Every human horizon is hidden while object targets stay independent.
    assert not out["human_reference_valid"].any()
    assert out["object_reference_valid"].all()
    assert observer.human_visible.all()
    observer.advance()
    assert not observer.observe(inputs(posterior, "hoi"))["human_reference_valid"].any()
    # Viewer body selection overrides the horizon gate.
    viewer = PriorConditioning(
        env,
        "hoi",
        config(human_horizon_visible_prob=0.0),
        1,
        reference_body_ids=(1,),
    )
    valid = viewer.observe(inputs(posterior, "hoi"))["human_reference_valid"]
    assert valid[:, :-1, 1].all() and not valid[:, :, 0].any()


def test_horizon_gate_follows_the_per_body_refresh_schedule():
    class Draws:
        gate = 0.1  # Stream 12 draws the horizon gate; < 0.5 shows the horizon.

        def __call__(self, shape, env_ids, stream):
            return torch.full(shape, self.gate if stream == 12 else 0.5)

    env, posterior = environment(), motor()
    draws = Draws()
    observer = PriorConditioning(
        env,
        "hoi",
        config(human_horizon_visible_prob=0.5, mask_resample_prob=0.0),
        1,
        random_source=draws,
    )
    assert observer.human_horizon_visible.shape == (2, 5, 1)
    assert observer.human_horizon_visible.all()
    draws.gate = 0.9
    observer.observe(inputs(posterior, "hoi"))
    observer.long_target_time.fill_(1e9)
    observer.advance()
    assert observer.human_horizon_visible.all()  # No refresh without arrival.
    observer.observe(inputs(posterior, "hoi"))
    observer.long_target_time[0] = env.motion_manager.motion_times[0]
    observer.advance()
    # Only the arrived env's long horizon is resampled, like the per-body masks.
    gate = observer.human_horizon_visible[..., 0]
    assert not gate[0, -1] and gate[0, :-1].all() and gate[1].all()
    observer.reset(torch.tensor([1]))
    assert not observer.human_horizon_visible[1].any()
    assert gate[0, :-1].all()


def test_reference_time_clamping_and_spawn_alignment():
    env, posterior = environment(), motor()
    env.motion_manager.motion_times[:] = 9.95
    env.get_spawn_to_ref_pose_offset_with_terrain_height_correction = (
        lambda pos, **kw: (torch.ones_like(pos))
    )
    observer = PriorConditioning(
        env, "hoi", config(human_visible_prob=1, object_visible_prob=1), 1
    )
    out = observer.observe(inputs(posterior, "hoi"))
    torch.testing.assert_close(out["human_reference_time"], torch.full((2, 5, 1), 0.05))
    # Human reference x=10+1, object reference x=2*10+10+1 minus actual x=3.
    torch.testing.assert_close(
        out["human_reference"][:, :, 0], torch.full((2, 5), 11.0)
    )
    torch.testing.assert_close(
        out["object_reference"][:, :, 0], torch.full((2, 5), 28.0)
    )


def test_heading_frame_and_rotation_residual():
    states = torch.zeros(1, 1, 1, 13)
    states[..., 6] = 1
    states[..., 1] = 1
    heading = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]])
    local = local_history(states, torch.zeros(1, 3), heading)
    torch.testing.assert_close(
        local[..., :3], torch.tensor([[[[1.0, 0.0, 0.0]]]]), atol=1e-6, rtol=0
    )
    current = states[:, 0].clone()
    target = states.expand(-1, 5, -1, -1).clone()
    target[..., 3:7] = heading
    residual = reference_residual(target, current, heading)
    torch.testing.assert_close(residual[..., 5], torch.full((1, 5, 1), torch.pi / 2))


def test_live_adapter_preserves_history_on_duplicate_calls_and_clears_on_reset(
    tmp_path,
):
    env = environment()
    model = JointPriorModel(motor(), config()).eval()
    policy = LivePriorPolicy(model, env, "hoi")
    agent = EvaluationAgent(
        policy, env, "hoi", tmp_path, torch.zeros(2, dtype=torch.long)
    )
    obs = inputs(model.posterior, "hoi")
    agent.pre_collect_step(0)
    policy(obs, "hoi")
    policy(obs, "hoi")
    assert not policy.conditions.history_count.any()
    # History follows explicit hooks even if all clocks/IDs happen to be equal.
    agent.pre_collect_step(1)
    policy(obs, "hoi")
    assert policy.conditions.history_count.tolist() == [1, 1]
    agent.on_env_reset(torch.tensor([0]))
    agent.pre_collect_step(2)
    policy(obs, "hoi")
    assert policy.conditions.history_count[0] == 0
    assert policy.conditions.history_count[1] == 2
    # A new evaluation wave starts with step zero even at the same motion/time.
    agent.pre_collect_step(0)
    policy(obs, "hoi")
    assert not policy.conditions.history_count.any()


@pytest.mark.parametrize("task", ["locomotion", "hoi"])
def test_live_adapter_first_forward_resamples_long_target_after_env_reset(task):
    env = environment()
    env.motion_manager.motion_times[:] = 8.0
    model = JointPriorModel(motor(), config(max_long_horizon=1)).eval()
    policy = LivePriorPolicy(model, env, task)
    torch.testing.assert_close(
        policy.conditions.long_target_time, torch.full((2,), 8.1)
    )
    # Native evaluation resets after constructing the adapter, before any action.
    env.motion_manager.motion_times[:] = torch.tensor([0.0, 2.0])
    with patch.object(
        policy.conditions, "reset", wraps=policy.conditions.reset
    ) as reset:
        obs = inputs(model.posterior, task)
        policy.pre_collect_step(0)
        policy(obs, task)
        reset.assert_called_once()
        assert reset.call_args.args[0].tolist() == [0, 1]
        torch.testing.assert_close(
            policy.conditions.long_target_time, torch.tensor([0.1, 2.1])
        )
        assert not policy.conditions.history_count.any()
        policy(obs, task)
        reset.assert_called_once()


@pytest.mark.parametrize("evaluator_class", [BaseEvaluator, MimicEvaluator])
def test_viewer_hooks_clear_only_explicit_reset_envs(tmp_path, evaluator_class):
    env = environment()
    model = JointPriorModel(motor(), config()).eval()
    policy = LivePriorPolicy(model, env, "hoi")
    agent = EvaluationAgent(
        policy, env, "hoi", tmp_path, torch.zeros(2, dtype=torch.long)
    )
    obs = inputs(model.posterior, "hoi")
    resets, counts = [], []

    def reset(ids=None, **kwargs):
        resets.append((ids, kwargs))
        return obs, {}

    def step(actions):
        counts.append(policy.conditions.history_count.tolist())
        if len(counts) == 3:
            raise KeyboardInterrupt
        return obs, None, torch.tensor([True, False]), None, {}

    env.reset, env.step = reset, step
    evaluator = object.__new__(evaluator_class)
    evaluator.agent = agent
    evaluator.config = SimpleNamespace(evaluation_action_key="action")
    # Initial reset -> failure reset of env 0 -> F9 switch of env 1.
    evaluator._consume_interactive_motion_id_request = Mock(
        side_effect=[None, None, torch.tensor([1])]
    )
    evaluator.simple_test_policy()
    assert counts == [[0, 0], [0, 1], [1, 0]]
    assert resets[0][0] is None
    assert resets[1][0].tolist() == [0]
    assert resets[2][0].tolist() == [1]
    assert resets[2][1] == {"disable_motion_resample": True}


@pytest.mark.parametrize("evaluator_class", [BaseEvaluator, MimicEvaluator])
def test_viewer_reset_key_inside_step_clears_history_and_snapshot(
    tmp_path, evaluator_class
):
    env = environment()
    model = JointPriorModel(motor(), config(max_long_horizon=1)).eval()
    policy = LivePriorPolicy(model, env, "hoi")
    agent = EvaluationAgent(
        policy, env, "hoi", tmp_path, torch.zeros(2, dtype=torch.long)
    )
    obs = inputs(model.posterior, "hoi")
    seen = []

    def step(actions):
        conditions = policy.conditions
        seen.append(
            (conditions.history_count.tolist(), conditions.long_target_time.tolist())
        )
        dones = torch.zeros(2, dtype=torch.bool)
        if len(seen) == 1:
            env.motion_manager.motion_times[:] = 5.0
        elif len(seen) == 2:
            # The viewer reset key: BaseEnv.user_reset marks every env done.
            dones[:] = True
        else:
            raise KeyboardInterrupt
        return obs, None, dones, None, {}

    def reset(ids=None, **kwargs):
        if ids is not None:
            # BaseEnv replays the R-key motion from t=0.
            env.motion_manager.motion_times[ids] = 0.0
        return obs, {}

    env.reset = reset
    env.step = step
    evaluator = object.__new__(evaluator_class)
    evaluator.agent = agent
    evaluator.config = SimpleNamespace(evaluation_action_key="action")
    evaluator._consume_interactive_motion_id_request = Mock(return_value=None)
    evaluator.simple_test_policy()
    counts = [count for count, _ in seen]
    targets = [target for _, target in seen]
    assert counts == [[0, 0], [1, 1], [0, 0]]
    torch.testing.assert_close(torch.tensor(targets[1]), torch.full((2,), 5.1))
    # The snapshot follows the reset clock instead of the old timeline.
    torch.testing.assert_close(torch.tensor(targets[2]), torch.full((2,), 0.1))


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("autoregressive", [False, True])
def test_prior_checkpoint_roundtrip_and_resume_contract(
    tmp_path, legacy, autoregressive
):
    model = JointPriorModel(motor(), config(codebook_autoregressive=autoregressive))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    manifest = SourceManifest(
        (
            DistillationSource("loco", "locomotion", "a", "b"),
            DistillationSource("hoi", "hoi", "c", "d", "e"),
        ),
        locomotion_num_ranks=1,
    )
    args = parser().parse_args(
        ["--checkpoint", "motor.ckpt", "--distillation-sources", "sources.yaml"]
    )
    resolve_sampling(args, None)
    training = training_contract(
        args,
        model.config,
        2,
        None,
        model.posterior.config,
        model.posterior.state_dict(),
    )
    saved_training = deepcopy(training)
    if legacy:
        saved_training.pop("posterior")
    path = tmp_path / "last.ckpt"
    save_prior_checkpoint(
        path, model, optimizer, 12, manifest, [], saved_training, [{}, {}]
    )
    restored, saved = load_prior_checkpoint(path)
    validate_resume(saved, manifest, [], training)
    assert restored.config.codebook_autoregressive == autoregressive
    obs = inputs(model.posterior, "locomotion")
    conditions = PriorConditioning(
        environment(), "locomotion", model.config, 1
    ).observe(obs)
    torch.testing.assert_close(
        restored.generate_indices(conditions, temperature=0),
        model.generate_indices(conditions, temperature=0),
    )
    changed_mode = deepcopy(training)
    changed_mode["prior_config"]["codebook_autoregressive"] = not autoregressive
    with pytest.raises(ValueError, match="settings"):
        validate_resume(saved, manifest, [], changed_mode)
    if not autoregressive:
        # Pre-AR checkpoints have neither the flag nor any new model parameters.
        old = deepcopy(saved)
        old["prior_config"].pop("codebook_autoregressive")
        old["training_config"]["prior_config"].pop("codebook_autoregressive")
        legacy_path = tmp_path / "legacy.ckpt"
        torch.save(old, legacy_path)
        old_model, old_saved = load_prior_checkpoint(legacy_path)
        assert not old_model.config.codebook_autoregressive
        validate_resume(old_saved, manifest, [], training)
        with pytest.raises(ValueError, match="settings"):
            validate_resume(old_saved, manifest, [], changed_mode)
        torch.testing.assert_close(
            old_model.generate_indices(conditions, temperature=0),
            model.generate_indices(conditions, temperature=0),
        )
    assert saved["iteration"] == 12
    assert saved["training_config"]["temperature"] == 1.0
    changed_temperature = deepcopy(training)
    changed_temperature["temperature"] = 0.5
    with pytest.raises(ValueError, match="settings"):
        validate_resume(saved, manifest, [], changed_temperature)
    old_sampling = deepcopy(saved)
    old_sampling["training_config"].pop("temperature")
    with pytest.raises(ValueError, match="settings"):
        validate_resume(old_sampling, manifest, [], training)
    greedy_training = deepcopy(training)
    greedy_training["temperature"] = 0.0
    validate_resume(old_sampling, manifest, [], greedy_training)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    changed = deepcopy(training)
    changed["world_size"] = 3
    with pytest.raises(ValueError, match="settings"):
        validate_resume(saved, manifest, [], changed)
    # An unchanged motor can move paths (e.g. local -> Euler).
    args.checkpoint = "/new/location/motor.ckpt"
    moved = training_contract(
        args,
        model.config,
        2,
        None,
        model.posterior.config,
        model.posterior.state_dict(),
    )
    validate_resume(saved, manifest, [], moved)
    # Same architecture and dimensions but different frozen weights must fail.
    other_motor = motor()
    changed = training_contract(
        args, model.config, 2, None, other_motor.config, other_motor.state_dict()
    )
    with pytest.raises(ValueError, match="frozen posterior"):
        validate_resume(saved, manifest, [], changed)
    changed = deepcopy(training)
    changed["posterior"]["model_config"]["num_objects"] += 1
    with pytest.raises(ValueError, match="frozen posterior"):
        validate_resume(saved, manifest, [], changed)
    # A mask field newer than the checkpoint takes its default when compared.
    old = deepcopy(saved)
    old["training_config"]["prior_config"].pop("human_horizon_visible_prob")
    validate_resume(old, manifest, [], training)
    masked = deepcopy(training)
    masked["prior_config"]["human_horizon_visible_prob"] = 0.5
    with pytest.raises(ValueError, match="settings"):
        validate_resume(old, manifest, [], masked)
    # Also verify the embedded weights, rather than trusting contract metadata.
    key = next(
        k
        for k in saved["model"]
        if k.startswith("posterior.") and saved["model"][k].is_floating_point()
    )
    saved["model"][key] = saved["model"][key] + 1
    with pytest.raises(ValueError, match="frozen posterior"):
        validate_resume(saved, manifest, [], training)


def test_posterior_digest_includes_buffers_and_is_order_independent():
    state = {"weight": torch.ones(2, 3, dtype=torch.bfloat16), "count": torch.tensor(7)}
    identity = posterior_contract({}, state)
    assert identity == posterior_contract({}, dict(reversed(list(state.items()))))
    assert identity != posterior_contract({}, {**state, "count": torch.tensor(8)})
    assert identity != posterior_contract(
        {}, {**state, "weight": state["weight"].float()}
    )
    assert identity != posterior_contract(
        {}, {**state, "weight": state["weight"].reshape(3, 2)}
    )


def evaluation_records():
    return [
        {
            "source_id": source,
            "task": task,
            "shard_index": shard,
            "local_motion_id": motion,
            "evaluated": True,
            "success": success,
            "failure_components": [] if success else ["tracking"],
            "component_means": {"error": 0.0 if success else 1.0},
        }
        for source, task, shard, motion, success in (
            ("amassx", "locomotion", 0, 0, True),
            ("omomo_sub2", "hoi", None, 0, False),
            ("omomo_sub2", "hoi", None, 1, True),
            ("omomo_sub3", "hoi", None, 0, True),
        )
    ]


@pytest.mark.parametrize("mode", ["none", "empty", "exclude", "all"])
def test_evaluation_filter_and_wandb_metrics(tmp_path, monkeypatch, mode):
    records = evaluation_records()
    document = None if mode == "none" else {"sources": {}}
    if mode in ("exclude", "all"):
        document["sources"] = {
            source: {
                "motions": [
                    {
                        "local_motion_id": motion,
                        "excluded": mode == "all" or motion == 0,
                    }
                    for motion in ids
                ]
            }
            for source, ids in (
                ("omomo_sub2", [0, 1]),
                ("omomo_sub3", [0]) if mode == "all" else ("unused", []),
            )
        }
    report = evaluation_report(12, records, document)
    assert report["motions"] == records
    assert report["summary"]["hoi_all"]["success_rate"] == pytest.approx(2 / 3)
    assert (
        report["filtered_summary"]["locomotion_all"]
        == report["summary"]["locomotion_all"]
    )
    assert (
        report["filtered_record_count"]
        == {"none": 0, "empty": 0, "exclude": 1, "all": 3}[mode]
    )
    if mode == "exclude":
        assert report["filtered_summary"]["hoi_all"]["num_motions"] == 2
        assert report["filtered_summary"]["hoi_all"]["success_rate"] == 1.0
    elif mode == "all":
        assert "hoi_all" not in report["filtered_summary"]
    else:
        assert report["filtered_summary"] == report["summary"]
    # Exercise the actual shared W&B flattening without starting an external run.
    wandb_run = object.__new__(WandbRun)
    wandb_run.run = Mock()
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    path = tmp_path / "eval_prior_12.json"
    publish_report(path, report, wandb_run, "evaluation")
    assert json.loads(path.read_text()) == report
    wandb_run.run.log.assert_called_once()
    metrics = wandb_run.run.log.call_args.args[0]
    assert wandb_run.run.log.call_args.kwargs == {"step": 12}
    assert metrics["eval_prior/hoi_all/success_rate"] == pytest.approx(2 / 3)
    if mode == "exclude":
        assert metrics["eval_prior/filtered/hoi_all/success_rate"] == 1.0


def _distributed_report_worker(rank, rendezvous, directory):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for filename, kind, append in (
            ("run_config.json", None, False),
            ("metrics.jsonl", "training", True),
            ("eval_prior_12.json", "evaluation", False),
        ):
            path = Path(directory) / filename
            logger = Mock()
            publish_report(path, {"iteration": 12}, logger, kind, append)
            if rank == 0 and kind is not None:
                getattr(logger, f"log_{kind}").assert_called_once_with(
                    {"iteration": 12}
                )
            else:
                assert not logger.mock_calls
            with patch(
                "protomotions.train_joint_prior.write_report",
                side_effect=OSError(errno.ENOSPC, "disk full"),
            ) as write:
                with pytest.raises(ValueError, match="rank 0: OSError.*disk full"):
                    publish_report(path, {"iteration": 13}, logger, kind, append)
                assert write.call_count == (1 if rank == 0 else 0)
            # Both ranks reached here after a rank-zero-only I/O exception.
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires CPU Gloo")
def test_report_io_failure_reaches_every_rank(tmp_path):
    mp.spawn(
        _distributed_report_worker,
        args=(f"file://{tmp_path}/rendezvous", str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for filename in ("run_config.json", "metrics.jsonl", "eval_prior_12.json"):
        assert json.loads((tmp_path / filename).read_text()) == {"iteration": 12}


@pytest.mark.parametrize("task", ["locomotion", "hoi"])
@pytest.mark.parametrize("autoregressive", [False, True])
def test_compact_masks_match_dense_outputs_moments_and_gradients(task, autoregressive):
    model = JointPriorModel(motor(), config(codebook_autoregressive=autoregressive))
    dense_model = deepcopy(model)
    obs = inputs(model.posterior, task)
    compact = PriorConditioning(environment(), task, model.config, 1).observe(obs)
    dense = dict(compact)
    for key in model.encoders:
        # Build the legacy feature layout independently of the expansion helper.
        valid = compact[key + "_valid"]
        dense[key + "_valid"] = valid.repeat_interleave(
            compact[key].shape[-1] // valid.shape[-1], -1
        )
    assert compact["human_history_valid"].shape == (2, 5, 1)
    assert compact["human_reference_valid"].shape == (2, 5, 2)
    assert compact["object_history_valid"].shape == (2, 5, 1)
    assert compact["object_reference_valid"].shape == (2, 5, 1)
    assert (
        compact["human_reference_valid"].numel() * 12
        == dense["human_reference_valid"].numel()
    )
    model.update_normalizers(compact)
    dense_model.update_normalizers(dense)
    for key, value in model.normalizers.state_dict().items():
        torch.testing.assert_close(
            value, dense_model.normalizers.state_dict()[key], rtol=0, atol=0
        )
    labels = model.posterior(obs, task)["indices"]
    logits = model(compact, labels)
    expected = dense_model(dense, labels)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    logits.sum().backward()
    expected.sum().backward()
    for param, other in zip(model.parameters(), dense_model.parameters()):
        if param.requires_grad:
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)


@pytest.mark.parametrize("task", ["locomotion", "hoi"])
@pytest.mark.parametrize("selected", [[False, False], [True, False], [True, True]])
@pytest.mark.parametrize("autoregressive", [False, True])
def test_rollout_only_runs_prior_for_selected_environments(
    task, selected, autoregressive
):
    model = JointPriorModel(
        motor(), config(codebook_autoregressive=autoregressive)
    ).eval()
    obs = inputs(model.posterior, task)
    conditions = PriorConditioning(environment(), task, model.config, 1).observe(obs)
    posterior_actions = model.posterior(obs, task)["action"]
    original = posterior_actions.clone()
    mask = torch.tensor(selected)
    expected = torch.where(
        mask[:, None], model.act(obs, conditions, temperature=0), posterior_actions
    )
    with patch.object(model, "act", wraps=model.act) as act:
        actual = model.rollout_actions(
            obs, conditions, posterior_actions, mask, temperature=0
        )
        if any(selected):
            act.assert_called_once()
            sliced_inputs, sliced_conditions = act.call_args.args
            assert all(
                value.shape[0] == sum(selected) for value in sliced_inputs.values()
            )
            assert all(
                value.shape[0] == sum(selected) for value in sliced_conditions.values()
            )
        else:
            act.assert_not_called()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(posterior_actions, original, rtol=0, atol=0)


def test_rollout_schedule():
    assert rollout_probability(500, 500, 9500, 0.95) == 0
    assert rollout_probability(5250, 500, 9500, 0.95) == pytest.approx(0.475)
    assert rollout_probability(10000, 500, 9500, 0.95) == 0.95
    assert rollout_probability(501, 500, 0, 0.95) == 0.95


def _distributed_prior_worker(rank, rendezvous, directory, autoregressive):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    try:
        torch.manual_seed(7)
        model = JointPriorModel(
            motor(),
            config(
                human_visible_prob=1,
                object_visible_prob=1,
                codebook_autoregressive=autoregressive,
            ),
        )
        ddp = DistributedDataParallel(model, broadcast_buffers=False)
        task = "locomotion" if rank == 0 else "hoi"
        obs = inputs(model.posterior, task)
        conditions = PriorConditioning(environment(), task, model.config, 1).observe(
            obs
        )
        model.update_normalizers(conditions)
        labels = model.posterior(obs, task)["indices"]
        counts = torch.tensor([2.0, 2.0])
        weights = task_balanced_source_weights(counts, torch.tensor([False, True]), 0.5)
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        # Two backward passes catch DDP's unused-parameter errors on the next iteration.
        for _ in range(2):
            logits = ddp(conditions, target_indices=labels)
            losses = model.categorical_loss(logits, labels)
            loss = weighted_sample_loss(
                losses, torch.full((2,), rank), counts, weights, 2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), f"{directory}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires CPU Gloo")
@pytest.mark.parametrize("autoregressive", [False, True])
def test_joint_prior_distributed_modalities_stay_synchronized(tmp_path, autoregressive):
    mp.spawn(
        _distributed_prior_worker,
        args=(f"file://{tmp_path}/rendezvous", str(tmp_path), autoregressive),
        nprocs=2,
        join=True,
    )
    left = torch.load(tmp_path / "rank0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert left["normalizers.object_state.count"].min() > 0

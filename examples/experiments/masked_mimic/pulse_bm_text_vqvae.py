# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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

"""Text-conditioned PULSE BM config using a VQ-VAE latent model."""

import argparse

from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.config import (
    SimulatorConfig,
    ActionNoiseDomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    RobotNoiseConfig,
    PushDomainRandomizationConfig,
    DomainRandomizationConfig,
)
from protomotions.components.terrains.config import (
    TerrainConfig,
    TerrainSimConfig,
    CombineMode,
)
from protomotions.envs.base_env.config import EnvConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.agents.distill.config import (
    DistillAgentConfig,
    SoftCodeTargetConfig,
    VQDistillLossConfig,
    VQDistillModelConfig,
)


VQ_LATENT_DIM = 64
NUM_EMBEDDINGS = 512


def additional_experiment_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--expert-model-path",
        type=str,
        default=None,
        help="Path to expert model checkpoint for distillation training",
    )
    parser.add_argument(
        "--vq-latent-dim",
        type=int,
        default=VQ_LATENT_DIM,
        help="Dimension of the VQ latent/codebook embedding.",
    )
    parser.add_argument(
        "--vq-prior-history-steps",
        type=int,
        default=0,
        help=(
            "Number of previous VQ code tokens to feed to the categorical prior. "
            "Uses teacher-forced posterior tokens during training and sampled "
            "prior tokens during inference."
        ),
    )
    parser.add_argument(
        "--vq-prior-future-steps",
        type=int,
        default=0,
        help=(
            "Number of next posterior VQ tokens to predict with an auxiliary "
            "categorical prior CE loss. For example, 4 predicts t+1..t+4."
        ),
    )
    parser.add_argument(
        "--vq-prior-future-loss-weight",
        type=float,
        default=0.25,
        help="Weight for the auxiliary future posterior VQ token CE loss.",
    )
    parser.add_argument(
        "--rollout-action-key",
        type=str,
        default="privileged_action",
        choices=["privileged_action", "prior_action"],
        help=(
            "Model output key used to step the environment during training "
            "rollout."
        ),
    )
    parser.add_argument(
        "--reduced-target-anchor-rotation-mode",
        type=str,
        default="current_to_ref",
        choices=["current_to_ref", "ref_delta"],
        help=(
            "Anchor rotation target encoding for non-expert reduced target "
            "poses. Expert observations keep their checkpoint encoding."
        ),
    )
    parser.add_argument(
        "--reduced-target-ref-delta-prob",
        type=float,
        default=None,
        help=(
            "Optional fixed probability of using ref_delta anchor rotation "
            "targets in non-expert reduced target poses. Expert observations "
            "keep their checkpoint encoding."
        ),
    )
    parser.add_argument(
        "--use-categorical-prior-film",
        action="store_true",
        help="Use text FiLM modulation in the categorical VQ prior.",
    )
    parser.add_argument(
        "--use-categorical-prior-transformer",
        "--use-transformer-prior",
        dest="use_categorical_prior_transformer",
        action="store_true",
        help="Use a causal Transformer categorical VQ prior over recent observations.",
    )
    parser.add_argument(
        "--use-categorical-prior-moe",
        action="store_true",
        help="Use a top-k MoE MLP for the categorical VQ prior.",
    )
    parser.add_argument(
        "--categorical-prior-moe-num-experts",
        type=int,
        default=4,
        help="Number of experts in the categorical-prior MoE.",
    )
    parser.add_argument(
        "--categorical-prior-moe-top-k",
        type=int,
        default=2,
        help="Number of experts selected per sample in the categorical-prior MoE.",
    )
    parser.add_argument(
        "--categorical-prior-moe-gate-input",
        type=str,
        default="text",
        choices=["text", "full"],
        help=(
            "Input used by the MoE router. 'text' routes only from the text "
            "embedding for smoother expert selection; 'full' routes from the "
            "same state/context/text input used by the experts."
        ),
    )
    parser.add_argument(
        "--categorical-prior-moe-balance-weight",
        type=float,
        default=1e-2,
        help="Weight for the categorical-prior MoE load-balancing loss.",
    )
    parser.add_argument(
        "--vq-prior-transformer-context-steps",
        type=int,
        default=16,
        help="Number of chronological obs steps consumed by the Transformer prior.",
    )
    parser.add_argument(
        "--vq-prior-transformer-d-model",
        type=int,
        default=512,
        help="Hidden size for the Transformer categorical prior.",
    )
    parser.add_argument(
        "--vq-prior-transformer-num-layers",
        type=int,
        default=2,
        help="Number of Transformer encoder layers in the categorical prior.",
    )
    parser.add_argument(
        "--vq-prior-transformer-num-heads",
        type=int,
        default=4,
        help="Number of attention heads in the Transformer categorical prior.",
    )
    parser.add_argument(
        "--vq-prior-transformer-ff-size",
        type=int,
        default=1024,
        help="Feed-forward hidden size in the Transformer categorical prior.",
    )
    parser.add_argument(
        "--vq-prior-transformer-dropout",
        type=float,
        default=0.1,
        help="Dropout probability in the Transformer categorical prior.",
    )
    parser.add_argument(
        "--categorical-prior-film-hidden-dim",
        type=int,
        default=1024,
        help="Hidden feature dimension modulated by text FiLM.",
    )
    parser.add_argument(
        "--categorical-prior-film-scale",
        type=float,
        default=1.0,
        help="Scale applied to categorical-prior FiLM gamma/beta outputs.",
    )
    parser.add_argument(
        "--use-prior-oracle-motion-command",
        action="store_true",
        help=(
            "Append reference-derived local target root vx/vy and yaw rate "
            "to the categorical VQ prior input for oracle ablation."
        ),
    )
    parser.add_argument(
        "--use-prior-state-history",
        action="store_true",
        help=(
            "Append noisy historical reduced-coords state observations to the "
            "categorical VQ prior input."
        ),
    )
    parser.add_argument(
        "--prior-state-history-steps",
        type=int,
        default=4,
        help="Number of previous noisy reduced-coords state frames for the prior.",
    )
    parser.add_argument(
        "--use-decoder-film",
        action="store_true",
        help="Use text FiLM modulation inside the VQ decoder/action trunk.",
    )
    parser.add_argument(
        "--decoder-film-hidden-dim",
        type=int,
        default=1024,
        help="Hidden feature dimension modulated by decoder text FiLM.",
    )
    parser.add_argument(
        "--decoder-film-scale",
        type=float,
        default=1.0,
        help="Scale applied to decoder FiLM gamma/beta outputs.",
    )


def terrain_config(args: argparse.Namespace):
    return TerrainConfig(
        sim_config=TerrainSimConfig(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.0,
            combine_mode=CombineMode.MULTIPLY,
        )
    )


def scene_lib_config(args: argparse.Namespace):
    scene_file = args.scenes_file if hasattr(args, "scenes_file") else None
    return SceneLibConfig(scene_file=scene_file)


def motion_lib_config(args: argparse.Namespace):
    return MotionLibConfig(motion_file=args.motion_file)


def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
    from protomotions.envs.control.mimic_control import MimicControlConfig
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        historical_reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
        max_coords_obs_factory,
        mimic_target_poses_max_coords_factory,
        previous_actions_factory,
        target_root_velocity_yaw_command_factory,
        action_smoothness_factory,
        global_anchor_ori_rew_factory,
        relative_body_pos_rew_factory,
        relative_body_ori_rew_factory,
        global_body_lin_vel_rew_factory,
        global_body_ang_vel_rew_factory,
        anchor_height_error_term_factory,
    )
    from protomotions.envs.rewards import compute_soft_pos_limit_rew
    from protomotions.envs.action import make_bm_pd_action_config
    from protomotions.envs.obs.vq_pae_bm import passthrough_text_embedding

    reduced_target_anchor_rotation_mode = getattr(
        args, "reduced_target_anchor_rotation_mode", "current_to_ref"
    )
    reduced_target_ref_delta_prob = getattr(
        args, "reduced_target_ref_delta_prob", None
    )
    control_components = {
        "mimic": MimicControlConfig(
            bootstrap_on_episode_end=True,
            future_steps=[1, 2, 4, 8],
        )
    }

    observation_components = {
        "noisy_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=True,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "noisy_mimic_reduced_coords_target_poses": (
            mimic_target_poses_reduced_coords_factory(
                use_noisy=True,
                include_dof_vel=True,
                include_xy_offset=False,
                anchor_rotation_mode=reduced_target_anchor_rotation_mode,
                ref_delta_prob=reduced_target_ref_delta_prob,
            )
        ),
        "clean_reduced_coords_obs": reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        ),
        "clean_mimic_reduced_coords_target_poses": (
            mimic_target_poses_reduced_coords_factory(
                use_noisy=False,
                include_dof_vel=True,
                include_xy_offset=False,
                anchor_rotation_mode=reduced_target_anchor_rotation_mode,
                ref_delta_prob=reduced_target_ref_delta_prob,
            )
        ),
        "max_coords_obs": max_coords_obs_factory(
            use_noisy=False,
            local_obs=True,
            root_height_obs=True,
            observe_contacts=False,
        ),
        "mimic_max_coords_target_poses": mimic_target_poses_max_coords_factory(
            use_noisy=False,
            with_velocities=True,
            with_relative=True,
        ),
        "historical_previous_processed_actions": previous_actions_factory(
            history_steps=1, processed=True
        ),
        "text_embedding_obs": MdpComponent(
            compute_func=passthrough_text_embedding,
            dynamic_vars={
                "text_embedding": EnvContext.mimic.text_embedding,
            },
        ),
    }
    if bool(getattr(args, "use_prior_oracle_motion_command", False)):
        observation_components["target_root_velocity_yaw_command"] = (
            target_root_velocity_yaw_command_factory(use_noisy=True)
        )
    if bool(getattr(args, "use_prior_state_history", False)):
        observation_components["noisy_historical_reduced_coords_obs"] = (
            historical_reduced_coords_obs_factory(
                use_noisy=True,
                history_steps=int(getattr(args, "prior_state_history_steps", 4)),
            )
        )

    expert_model_path = getattr(args, "expert_model_path", None)
    if expert_model_path:
        from protomotions.agents.distill.utils import (
            get_expert_observation_components,
            load_expert_configs,
        )

        expert_configs = load_expert_configs(expert_model_path)
        expert_obs_components = get_expert_observation_components(
            expert_configs["env"],
            expert_configs["agent"],
            existing_obs_keys=list(observation_components.keys()),
        )
        observation_components.update(expert_obs_components)

    reward_components = {
        "global_anchor_ori": global_anchor_ori_rew_factory(weight=0.5, sigma=0.4),
        "relative_body_pos": relative_body_pos_rew_factory(
            weight=1.0,
            sigma=0.3,
            use_region_weights=True,
        ),
        "relative_body_ori": relative_body_ori_rew_factory(
            weight=1.0,
            sigma=0.4,
            use_region_weights=True,
        ),
        "body_lin_vel": global_body_lin_vel_rew_factory(
            weight=1.0,
            sigma=1.0,
            use_region_weights=True,
        ),
        "body_ang_vel": global_body_ang_vel_rew_factory(
            weight=1.0,
            sigma=3.14,
            use_region_weights=True,
        ),
        "action_rate": action_smoothness_factory(weight=-0.1),
        "limits_dof_pos": MdpComponent(
            compute_func=compute_soft_pos_limit_rew,
            dynamic_vars={"dof_pos": EnvContext.current.dof_pos},
            static_params={
                "weight": -10.0,
                "dof_limits_lower": robot_cfg.kinematic_info.dof_limits_lower,
                "dof_limits_upper": robot_cfg.kinematic_info.dof_limits_upper,
            },
        ),
    }

    return EnvConfig(
        ref_contact_smooth_window=7,
        max_episode_length=1000,
        num_state_history_steps=max(
            1,
            (
                int(getattr(args, "prior_state_history_steps", 4))
                if bool(getattr(args, "use_prior_state_history", False))
                else 1
            ),
        ),
        control_components=control_components,
        observation_components=observation_components,
        termination_components={
            "fall": anchor_height_error_term_factory(threshold=0.25),
        },
        reward_components=reward_components,
        action_config=make_bm_pd_action_config(robot_cfg),
        motion_manager=MimicMotionManagerConfig(
            init_start_prob=0.2,
            resample_on_reset=True,
            realign_motion_with_humanoid_on_each_step=False,
        ),
    )


def agent_config(
    robot_config: RobotConfig, env_config: EnvConfig, args: argparse.Namespace
) -> DistillAgentConfig:
    from protomotions.agents.base_agent.config import OptimizerConfig
    from protomotions.agents.common.config import (
        CausalTransformerCategoricalPriorConfig,
        FiLMConfig,
        MLPWithConcatConfig,
        MLPLayerConfig,
        MoEMLPWithConcatConfig,
        ModuleContainerConfig,
        ModuleOperationForwardConfig,
        ObsProcessorConfig,
    )
    from protomotions.agents.evaluators.config import (
        DistillEvaluatorConfig,
        MotionWeightsRulesConfig,
    )
    from protomotions.envs.component_factories import (
        anchor_ori_metric_factory,
        relative_body_pos_metric_factory,
        anchor_height_error_metric_factory,
        gt_error_factory,
        gr_error_factory,
        max_joint_error_factory,
    )

    vq_latent_dim = int(getattr(args, "vq_latent_dim", VQ_LATENT_DIM))

    encoder_config = ModuleContainerConfig(
        in_keys=[
            "noisy_reduced_coords_obs",
            "noisy_mimic_reduced_coords_target_poses",
            "historical_previous_processed_actions",
        ],
        out_keys=["encoder_latent"],
        models=[
            ObsProcessorConfig(
                in_keys=[
                    "noisy_reduced_coords_obs",
                    "noisy_mimic_reduced_coords_target_poses",
                    "historical_previous_processed_actions",
                ],
                out_keys=["encoder_motion_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "encoder_motion_obs_norm",
                ],
                out_keys=["encoder_trunk_out"],
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["encoder_trunk_out"],
                out_keys=["encoder_latent"],
                num_out=vq_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    prior_config = ModuleContainerConfig(
        in_keys=[
            "noisy_reduced_coords_obs",
            "historical_previous_processed_actions",
        ],
        out_keys=["prior_latent"],
        models=[
            ObsProcessorConfig(
                in_keys=[
                    "noisy_reduced_coords_obs",
                    "historical_previous_processed_actions",
                ],
                out_keys=["prior_motion_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=[
                    "prior_motion_obs_norm",
                ],
                out_keys=["prior_trunk_out"],
                num_out=512,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(4)],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["prior_trunk_out"],
                out_keys=["prior_latent"],
                num_out=vq_latent_dim,
                layers=[
                    MLPLayerConfig(units=256, activation="relu"),
                    MLPLayerConfig(units=128, activation="relu"),
                ],
            ),
        ],
    )

    vq_prior_history_steps = int(getattr(args, "vq_prior_history_steps", 0))
    vq_prior_future_steps = int(getattr(args, "vq_prior_future_steps", 0))
    vq_prior_future_loss_weight = float(
        getattr(args, "vq_prior_future_loss_weight", 1.0)
    )
    categorical_prior_num_out = NUM_EMBEDDINGS * (1 + vq_prior_future_steps)
    use_categorical_prior_film = bool(
        getattr(args, "use_categorical_prior_film", False)
    )
    use_categorical_prior_transformer = bool(
        getattr(args, "use_categorical_prior_transformer", False)
    )
    use_categorical_prior_moe = bool(
        getattr(args, "use_categorical_prior_moe", False)
    )
    if use_categorical_prior_transformer and use_categorical_prior_film:
        raise ValueError(
            "--use-categorical-prior-transformer and "
            "--use-categorical-prior-film are separate prior architectures; "
            "enable only one for this ablation."
        )
    if use_categorical_prior_moe and use_categorical_prior_transformer:
        raise ValueError(
            "--use-categorical-prior-moe and "
            "--use-categorical-prior-transformer are separate prior architectures; "
            "enable only one for this ablation."
        )
    if use_categorical_prior_moe and use_categorical_prior_film:
        raise ValueError(
            "--use-categorical-prior-moe and --use-categorical-prior-film are "
            "separate prior architectures; enable only one for this ablation."
        )
    if use_categorical_prior_transformer and vq_prior_history_steps > 0:
        raise ValueError(
            "--vq-prior-history-steps is not wired into the Transformer prior "
            "yet; keep it at 0 for this ablation."
        )
    categorical_prior_film_hidden_dim = int(
        getattr(args, "categorical_prior_film_hidden_dim", 1024)
    )
    categorical_prior_film_scale = float(
        getattr(args, "categorical_prior_film_scale", 1.0)
    )
    transformer_sequence_key = "categorical_prior_transformer_obs_seq"
    transformer_text_sequence_key = "categorical_prior_transformer_text_seq"
    transformer_mask_key = "categorical_prior_transformer_obs_seq_mask"
    vq_code_history_feature_key = "_vq_code_history_obs"
    categorical_prior_container_in_keys = [
        "noisy_reduced_coords_obs",
        "historical_previous_processed_actions",
        "text_embedding_obs",
    ]
    categorical_prior_motion_obs_in_keys = [
        "noisy_reduced_coords_obs",
        "historical_previous_processed_actions",
    ]
    categorical_prior_context_in_keys = ["categorical_prior_motion_obs_norm"]
    if bool(getattr(args, "use_prior_oracle_motion_command", False)):
        categorical_prior_container_in_keys.append("target_root_velocity_yaw_command")
        categorical_prior_motion_obs_in_keys.append("target_root_velocity_yaw_command")
    if bool(getattr(args, "use_prior_state_history", False)):
        categorical_prior_container_in_keys.append(
            "noisy_historical_reduced_coords_obs"
        )
        categorical_prior_motion_obs_in_keys.append(
            "noisy_historical_reduced_coords_obs"
        )
    if vq_prior_history_steps > 0:
        categorical_prior_container_in_keys.append(vq_code_history_feature_key)
        categorical_prior_context_in_keys.append(vq_code_history_feature_key)

    if use_categorical_prior_transformer:
        categorical_prior_container_in_keys = [
            transformer_sequence_key,
            transformer_text_sequence_key,
            transformer_mask_key,
        ]
        categorical_prior_models = [
            ObsProcessorConfig(
                in_keys=[transformer_sequence_key],
                out_keys=["categorical_prior_transformer_obs_seq_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            CausalTransformerCategoricalPriorConfig(
                in_keys=[
                    "categorical_prior_transformer_obs_seq_norm",
                    transformer_text_sequence_key,
                    transformer_mask_key,
                ],
                out_keys=["prior_code_logits"],
                num_out=categorical_prior_num_out,
                context_steps=int(
                    getattr(args, "vq_prior_transformer_context_steps", 16)
                ),
                d_model=int(getattr(args, "vq_prior_transformer_d_model", 512)),
                num_heads=int(
                    getattr(args, "vq_prior_transformer_num_heads", 4)
                ),
                ff_size=int(getattr(args, "vq_prior_transformer_ff_size", 1024)),
                num_layers=int(
                    getattr(args, "vq_prior_transformer_num_layers", 2)
                ),
                dropout=float(getattr(args, "vq_prior_transformer_dropout", 0.1)),
            ),
        ]
    else:
        categorical_prior_models = [
            ObsProcessorConfig(
                in_keys=categorical_prior_motion_obs_in_keys,
                out_keys=["categorical_prior_motion_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
        ]

    if use_categorical_prior_film:
        categorical_prior_models += [
            MLPWithConcatConfig(
                in_keys=categorical_prior_context_in_keys,
                out_keys=["categorical_prior_context_feature"],
                num_out=categorical_prior_film_hidden_dim,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                ],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["text_embedding_obs"],
                out_keys=["categorical_prior_text_film_params"],
                num_out=categorical_prior_film_hidden_dim * 2,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            ),
            FiLMConfig(
                in_keys=[
                    "categorical_prior_context_feature",
                    "categorical_prior_text_film_params",
                ],
                out_keys=["categorical_prior_film_feature"],
                scale=categorical_prior_film_scale,
            ),
            MLPWithConcatConfig(
                in_keys=["categorical_prior_film_feature"],
                out_keys=["prior_code_logits"],
                num_out=categorical_prior_num_out,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                ],
            ),
        ]
    elif use_categorical_prior_moe:
        categorical_prior_mlp_in_keys = categorical_prior_context_in_keys + [
            "text_embedding_obs"
        ]
        categorical_prior_moe_gate_input = getattr(
            args, "categorical_prior_moe_gate_input", "text"
        )
        if categorical_prior_moe_gate_input == "text":
            categorical_prior_moe_gate_in_keys = ["text_embedding_obs"]
        else:
            categorical_prior_moe_gate_in_keys = categorical_prior_mlp_in_keys
        categorical_prior_models.append(
            MoEMLPWithConcatConfig(
                in_keys=categorical_prior_mlp_in_keys,
                gate_in_keys=categorical_prior_moe_gate_in_keys,
                out_keys=["prior_code_logits"],
                num_out=categorical_prior_num_out,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(4)
                ],
                num_experts=int(
                    getattr(args, "categorical_prior_moe_num_experts", 4)
                ),
                top_k=int(getattr(args, "categorical_prior_moe_top_k", 2)),
                balance_loss_key="categorical_prior_moe_balance_loss",
                gate_probs_key="categorical_prior_moe_gate_probs",
                topk_indices_key="categorical_prior_moe_topk_indices",
                expert_load_key="categorical_prior_moe_expert_load",
            )
        )
    elif not use_categorical_prior_transformer:
        categorical_prior_mlp_in_keys = categorical_prior_context_in_keys + [
            "text_embedding_obs"
        ]
        categorical_prior_models.append(
            MLPWithConcatConfig(
                in_keys=categorical_prior_mlp_in_keys,
                out_keys=["prior_code_logits"],
                num_out=categorical_prior_num_out,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(4)
                ],
            )
        )

    categorical_prior_config = ModuleContainerConfig(
        in_keys=categorical_prior_container_in_keys,
        out_keys=["prior_code_logits"],
        models=categorical_prior_models,
    )

    decoder_trunk_in_keys = [
        "noisy_reduced_coords_obs",
        "historical_previous_processed_actions",
        "vae_latent",
    ]
    use_decoder_film = bool(getattr(args, "use_decoder_film", False))
    decoder_film_hidden_dim = int(getattr(args, "decoder_film_hidden_dim", 1024))
    decoder_film_scale = float(getattr(args, "decoder_film_scale", 1.0))
    if use_decoder_film:
        decoder_trunk_in_keys = decoder_trunk_in_keys + ["text_embedding_obs"]
        decoder_trunk_models = [
            ObsProcessorConfig(
                in_keys=[
                    "noisy_reduced_coords_obs",
                    "historical_previous_processed_actions",
                    "vae_latent",
                ],
                out_keys=["decoder_motion_obs_norm"],
                normalize_obs=True,
                norm_clamp_value=5,
                module_operations=[ModuleOperationForwardConfig()],
            ),
            MLPWithConcatConfig(
                in_keys=["decoder_motion_obs_norm"],
                out_keys=["decoder_context_feature"],
                num_out=decoder_film_hidden_dim,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu") for _ in range(4)
                ],
                output_activation="relu",
            ),
            MLPWithConcatConfig(
                in_keys=["text_embedding_obs"],
                out_keys=["decoder_text_film_params"],
                num_out=decoder_film_hidden_dim * 2,
                layers=[
                    MLPLayerConfig(units=512, activation="relu"),
                    MLPLayerConfig(units=512, activation="relu"),
                ],
            ),
            FiLMConfig(
                in_keys=["decoder_context_feature", "decoder_text_film_params"],
                out_keys=["decoder_film_feature"],
                scale=decoder_film_scale,
            ),
            MLPWithConcatConfig(
                in_keys=["decoder_film_feature"],
                out_keys=["actor_trunk_out"],
                num_out=robot_config.number_of_actions,
                layers=[
                    MLPLayerConfig(units=1024, activation="relu"),
                    MLPLayerConfig(units=1024, activation="relu"),
                ],
                output_activation="tanh",
            ),
        ]
    else:
        decoder_trunk_models = [
            MLPWithConcatConfig(
                in_keys=[
                    "noisy_reduced_coords_obs",
                    "historical_previous_processed_actions",
                    "vae_latent",
                ],
                out_keys=["actor_trunk_out"],
                normalize_obs=True,
                norm_clamp_value=5,
                num_out=robot_config.number_of_actions,
                layers=[MLPLayerConfig(units=1024, activation="relu") for _ in range(6)],
                output_activation="tanh",
            ),
        ]

    trunk_config = ModuleContainerConfig(
        in_keys=decoder_trunk_in_keys,
        out_keys=["actor_trunk_out"],
        models=decoder_trunk_models,
    )

    model_config = VQDistillModelConfig(
        encoder=encoder_config,
        prior=prior_config,
        trunk=trunk_config,
        categorical_prior=categorical_prior_config,
        reconstruction=None,
        latent_dim=vq_latent_dim,
        num_embeddings=NUM_EMBEDDINGS,
        commitment_cost=0.25,
        codebook_update_mode="gradient",
        ema_decay=0.99,
        dead_code_threshold=2,
        dead_code_revive_every=100,
        use_categorical_prior=True,
        categorical_prior_moe_balance_weight=float(
            getattr(args, "categorical_prior_moe_balance_weight", 1e-2)
        ),
        categorical_prior_history_steps=vq_prior_history_steps,
        categorical_prior_future_steps=vq_prior_future_steps,
        use_categorical_prior_transformer=use_categorical_prior_transformer,
        categorical_prior_transformer_context_steps=int(
            getattr(args, "vq_prior_transformer_context_steps", 16)
        ),
        categorical_prior_transformer_input_keys=(
            categorical_prior_motion_obs_in_keys
            if use_categorical_prior_transformer
            else []
        ),
        categorical_prior_transformer_sequence_key=transformer_sequence_key,
        categorical_prior_transformer_text_sequence_key=(
            transformer_text_sequence_key
        ),
        categorical_prior_transformer_mask_key=transformer_mask_key,
        losses=VQDistillLossConfig(
            commitment_weight=1.0,
            prior_commitment_weight=0.25,
            prior_alignment_weight=1.0,
            prior_bc_weight=0.0,
            reconstruction_weight=0.0,
            future_prior_categorical_weight=vq_prior_future_loss_weight,
        ),
        soft_code_target=SoftCodeTargetConfig(
            enabled=False,
            tau=0.1,
            lambda_soft=1.0,
            lambda_hard_ce=0.2,
            use_no_grad_decoder_eval=True,
            full_codebook=False,
            topk_eval=64,
        ),
        use_text_conditioning=True,
        text_obs_key="text_embedding_obs",
        text_obs_dim=512,
        text_conditioning_scale=0.25,
        optimizer=OptimizerConfig(_target_="torch.optim.Adam", lr=2e-5),
    )

    return DistillAgentConfig(
        model=model_config,
        batch_size=args.batch_size,
        training_max_steps=args.training_max_steps,
        rollout_action_key=getattr(args, "rollout_action_key", "privileged_action"),
        gradient_clip_val=50.0,
        num_mini_epochs=6,
        evaluator=DistillEvaluatorConfig(
            use_privileged_success_for_motion_weights=True,
            evaluation_components={
                "anchor_ori": anchor_ori_metric_factory(),
                "relative_body_pos": relative_body_pos_metric_factory(threshold=0.20),
                "anchor_height_error": anchor_height_error_metric_factory(
                    threshold=0.25
                ),
                "gt_error": gt_error_factory(),
                "gr_error": gr_error_factory(),
                "max_joint_error": max_joint_error_factory(),
            },
            motion_weights_rules=MotionWeightsRulesConfig(
                motion_weights_update_success_discount=0.999,
                motion_weights_update_failure_discount=0,
            ),
        ),
        expert_model_path=getattr(args, "expert_model_path", None),
    )


def configure_robot_and_simulator(
    robot_cfg: RobotConfig, simulator_cfg: SimulatorConfig, args: argparse.Namespace
):
    robot_cfg.update_fields(
        contact_bodies=["all_left_foot_bodies", "all_right_foot_bodies"]
    )

    robot_cfg.reset_noise = RobotNoiseConfig(
        dof_pos_noise=0.1,
        root_pos_noise=[0.05, 0.05, 0.01],
        root_rot_noise=[0.1, 0.1, 0.2],
        root_vel_noise=[0.1, 0.1, 0.05],
        root_ang_vel_noise=[0.1, 0.1, 0.1],
    )

    simulator_cfg.domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(-0.025, 0.025), dof_names=[".*"], dof_indices=None
        ),
        friction=FrictionDomainRandomizationConfig(
            num_buckets=64,
            static_friction_range=(0.3, 1.6),
            dynamic_friction_range=(0.3, 1.2),
            restitution_range=(0.0, 0.5),
            body_names=[".*"],
            body_indices=None,
        ),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            body_names=robot_cfg.common_naming_to_robot_body_names["torso_body_name"],
            body_indices=None,
        ),
        observation_noise=RobotNoiseConfig(
            dof_pos_noise=0.01,
            dof_vel_noise=0.5,
            anchor_ang_vel_noise=0.2,
            anchor_rot_noise=0.05,
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(1.0, 3.0),
            max_linear_velocity=(0.5, 0.5, 0.2),
            max_angular_velocity=(0.52, 0.52, 0.78),
        ),
    )


def apply_inference_overrides(
    robot_cfg: RobotConfig,
    simulator_cfg: SimulatorConfig,
    env_cfg: EnvConfig,
    agent_cfg: DistillAgentConfig,
    terrain_cfg: TerrainConfig,
    motion_lib_cfg: MotionLibConfig,
    scene_lib_cfg: SceneLibConfig,
    args: argparse.Namespace,
):
    from protomotions.envs.component_factories import (
        reduced_coords_obs_factory,
        mimic_target_poses_reduced_coords_factory,
    )

    if hasattr(env_cfg, "termination_components") and env_cfg.termination_components:
        env_cfg.termination_components = {}

    env_cfg.max_episode_length = 1000000
    env_cfg.motion_manager.resample_on_reset = True
    env_cfg.motion_manager.init_start_prob = 1.0

    if agent_cfg is not None and hasattr(agent_cfg, "expert_model_path"):
        agent_cfg.expert_model_path = None

    terrain_cfg.sim_config = TerrainSimConfig(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        combine_mode=CombineMode.AVERAGE,
    )
    simulator_cfg.domain_randomization = None

    env_cfg.observation_components["noisy_reduced_coords_obs"] = (
        reduced_coords_obs_factory(
            use_noisy=False,
            root_height_obs=False,
            root_vel_obs=False,
        )
    )
    env_cfg.observation_components["noisy_mimic_reduced_coords_target_poses"] = (
        mimic_target_poses_reduced_coords_factory(
            use_noisy=False,
            include_dof_vel=True,
            include_xy_offset=False,
            anchor_rotation_mode=getattr(
                args, "reduced_target_anchor_rotation_mode", "current_to_ref"
            ),
            ref_delta_prob=None,
        )
    )

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control manager for BaseEnv.

Manages control_components lifecycle and orchestrates their interactions.
Control components define task behaviors and provide context for observations/rewards.
"""

from typing import Dict, Any, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.simulator.base_simulator.config import VisualizationMarkerConfig, MarkerState
    from protomotions.envs.control.base import ControlComponent
    from protomotions.envs.context_views import EnvContext


class ControlManager:
    """Manages control_components lifecycle and orchestration.
    
    Control components are stateful task managers that:
    - Define task objectives and behaviors
    - Provide context variables for observations/rewards
    - Can define custom reset and termination logic
    - Can create visualization markers
    
    Note: Unlike other managers, control components DO receive env reference
    because they need deep integration with environment state (motion_manager,
    simulator, etc.). This manager orchestrates their lifecycle.
    
    Attributes:
        components: Dict mapping component names to ControlComponent instances
        device: Device for tensor operations
        num_envs: Number of parallel environments
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        env: "BaseEnv",
    ):
        """Initialize control manager.
        
        Args:
            config: Dictionary mapping component names to component configurations
            env: Parent environment instance (control components need env access)
        """
        self.device = env.device
        self.num_envs = env.num_envs
        
        # Initialize control components
        # Components are enabled by being present in the dict
        self.components: Dict[str, "ControlComponent"] = {}
        for name, comp_config in config.items():
            comp_class = get_class(comp_config._target_)
            self.components[name] = comp_class(comp_config, env)
    
    def step(self):
        """Update all control components after each physics step.
        
        Called during post_physics_step(). Components update their
        time-dependent state or check for task updates.
        """
        for component in self.components.values():
            component.step()

    def before_render(self) -> None:
        """Let components restore state after physics and before rendering."""
        for component in self.components.values():
            component.before_render()
    
    def reset(self, env_ids: Tensor):
        """Reset all control components for specified environments.
        
        Called when environments are reset. Components reinitialize
        stateful buffers and sample new task parameters.
        
        Args:
            env_ids: Indices of environments to reset [num_reset_envs]
        """
        for component in self.components.values():
            component.reset(env_ids)

    def before_reset(self, env_ids: Tensor) -> None:
        """Let components process outgoing episodes before reset."""
        for component in self.components.values():
            component.before_reset(env_ids)

    def post_reward(self, rewards: Tensor) -> None:
        """Let components observe rewards computed for the current step."""
        for component in self.components.values():
            component.post_reward(rewards)

    def modify_ref_reset_state(
        self,
        env_ids: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
        robot_state,
        object_state,
    ):
        """Apply component-specific reference-state initialization in order."""
        for component in self.components.values():
            robot_state, object_state = component.modify_ref_reset_state(
                env_ids,
                motion_ids,
                motion_times,
                robot_state,
                object_state,
            )
        return robot_state, object_state

    def get_state_dict(self) -> Dict[str, Dict]:
        """Collect persistent state from stateful control components."""
        return {
            name: component_state
            for name, component in self.components.items()
            if (component_state := component.get_state_dict())
        }

    def load_state_dict(self, state_dict: Dict[str, Dict]) -> None:
        """Restore persistent state for matching control components."""
        for name, component_state in state_dict.items():
            component = self.components.get(name)
            if component is not None:
                component.load_state_dict(component_state)

    def set_evaluation_mode(self, enabled: bool) -> None:
        """Notify stateful controls when evaluation starts or finishes."""
        for component in self.components.values():
            component.set_evaluation_mode(enabled)
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check control component-specific reset and termination conditions.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors [num_envs]
        """
        reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for component in self.components.values():
            comp_reset, comp_terminate = component.check_resets_and_terminations()
            reset_buf = reset_buf | comp_reset
            terminate_buf = terminate_buf | comp_terminate
        
        return reset_buf, terminate_buf
    
    def populate_context(self, ctx: "EnvContext") -> None:
        """Populate control-specific views in the EnvContext.
        
        Each control component populates its own view (e.g., ctx.mimic, ctx.steering)
        with task-specific data.
        
        Args:
            ctx: The EnvContext to populate with control-specific views.
        """
        for component in self.components.values():
            component.populate_context(ctx)
    
    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """Create visualization marker configurations from all components.
        
        Called during environment initialization. Collects marker configs
        from all components for visualizing task state (e.g., target poses,
        waypoints).
        
        Args:
            headless: If True, should return empty dict (no visualization)
        
        Returns:
            Dictionary mapping marker names to VisualizationMarkerConfig
        """
        if headless:
            return {}
        
        markers = {}
        for component in self.components.values():
            comp_markers = component.create_visualization_markers(headless)
            markers.update(comp_markers)
        return markers
    
    def get_markers_state(self) -> Dict[str, "MarkerState"]:
        """Compute current marker positions and orientations from all components.
        
        Called each frame to update visualization markers. Collects marker
        states from all components.
        
        Returns:
            Dictionary mapping marker names to MarkerState with positions/orientations
        """
        markers_state = {}
        for component in self.components.values():
            comp_markers_state = component.get_markers_state()
            markers_state.update(comp_markers_state)
        return markers_state

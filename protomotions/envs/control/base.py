# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for control components.

Control components are stateful task managers that define the objectives and
behaviors of the environment. They manage task-specific state, provide context
for observations and rewards, and can define custom termination conditions.

Examples:
    - MimicControlComponent: Manages motion tracking tasks
    - SteeringControlComponent: Manages heading and speed targets
    - PathFollowingControlComponent: Manages path generation and following
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from dataclasses import dataclass

if TYPE_CHECKING:
    from protomotions.simulator.base_simulator.config import VisualizationMarkerConfig, MarkerState
    from protomotions.simulator.base_simulator.simulator_state import (
        ObjectState,
        ResetState,
    )
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.envs.context_views import EnvContext


@dataclass
class ControlComponentConfig:
    """Base configuration for control components.
    
    Note: Components are enabled by being present in the control_components dict.
    To disable a component, remove it from the dict.
    """
    pass


class ControlComponent(ABC):
    """Base class for control components.
    
    Control components are stateful modules that define task behavior. They:
    - Maintain task-specific state across timesteps
    - Provide context variables for observations, rewards, and terminations
    - Can define custom reset and termination logic
    - Can create visualization markers
    
    Attributes:
        config: Component configuration.
        env: Parent environment instance.
    """
    
    def __init__(self, config: ControlComponentConfig, env: "BaseEnv"):
        """Initialize the control component.
        
        Args:
            config: Component configuration.
            env: Parent environment instance.
        """
        self.config = config
        self.env = env
    
    def reset(self, env_ids: Tensor):
        """Reset component state for the given environments.
        
        Called when environments are reset. Should reinitialize any stateful
        buffers and sample new task parameters.
        
        Args:
            env_ids: Indices of environments to reset [num_reset_envs].
        """
        pass

    def before_reset(self, env_ids: Tensor) -> None:
        """Handle the outgoing episode before the simulator state is replaced."""
        pass

    def post_reward(self, rewards: Tensor) -> None:
        """Observe rewards after they are computed for the current step."""
        pass

    def modify_ref_reset_state(
        self,
        env_ids: Tensor,
        motion_ids: Tensor,
        motion_times: Tensor,
        robot_state: "ResetState",
        object_state: "ObjectState",
    ) -> Tuple["ResetState", "ObjectState"]:
        """Optionally replace reference reset states before simulator reset."""
        return robot_state, object_state

    def get_state_dict(self) -> Dict:
        """Return persistent component state for environment checkpoints."""
        return {}

    def load_state_dict(self, state_dict: Dict) -> None:
        """Restore persistent component state from an environment checkpoint."""
        pass

    def set_evaluation_mode(self, enabled: bool) -> None:
        """Enter or leave evaluation without corrupting training-time state."""
        pass
    
    @abstractmethod
    def step(self):
        """Update component state after each physics step.
        
        Called during post_physics_step(). Should update any time-dependent
        state or check for task updates.
        """
        pass
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check for component-specific reset and termination conditions.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors [num_envs].
            Default implementation returns all False.
        """
        device = self.env.device
        num_envs = self.env.num_envs
        return (
            torch.zeros(num_envs, dtype=torch.bool, device=device),
            torch.zeros(num_envs, dtype=torch.bool, device=device),
        )
    
    @abstractmethod
    def populate_context(self, ctx: "EnvContext") -> None:
        """Populate control-specific views in the EnvContext.
        
        Each control component populates its own view (e.g., ctx.mimic, ctx.steering)
        with task-specific data for use in observation, reward, and termination functions.
        
        Args:
            ctx: The EnvContext to populate with control-specific views.
            
        Example:
            >>> ctx.mimic = MimicContext(
            ...     ref_state=self.get_reference_state(),
            ...     future_pos=...,
            ... )
        """
        pass
    
    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """Create visualization marker configurations.
        
        Called during environment initialization. Should return marker configs
        for visualizing the task (e.g., target poses, waypoints).
        
        Args:
            headless: If True, should return empty dict (no visualization).
            
        Returns:
            Dictionary mapping marker names to VisualizationMarkerConfig.
            Default implementation returns empty dict.
        """
        return {}
    
    def get_markers_state(self) -> Dict[str, "MarkerState"]:
        """Compute current marker positions and orientations.
        
        Called each frame to update visualization markers. Should return
        marker states corresponding to the configs from create_visualization_markers().
        
        Returns:
            Dictionary mapping marker names to MarkerState with positions/orientations.
            Default implementation returns empty dict.
        """
        return {}

    def before_render(self) -> None:
        """Apply any state that must be visible in the upcoming render.

        This hook runs after the simulator physics step and before marker
        updates/rendering. Most controls do not need it. Kinematic replay uses
        it to restore the exact reference pose after physics has perturbed the
        articulation.
        """
        pass

    def close(self) -> None:
        """Release resources owned by this component."""
        pass

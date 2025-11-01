"""Base environment interface for goal-conditioned reinforcement learning."""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GoalConditionedEnv(ABC):
    """Abstract base class for goal-conditioned environments."""
    
    def __init__(self, **kwargs):
        """Initialize the environment."""
        self.observation_space: Optional[spaces.Space] = None
        self.action_space: Optional[spaces.Space] = None
        self.goal_space: Optional[spaces.Space] = None
        
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return initial observation and info."""
        pass
        
    @abstractmethod
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        pass
        
    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> float:
        """Compute reward based on achieved and desired goals."""
        pass
        
    @abstractmethod
    def get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """Extract achieved goal from current state."""
        pass
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        return None
        
    def close(self) -> None:
        """Close the environment and clean up resources."""
        pass

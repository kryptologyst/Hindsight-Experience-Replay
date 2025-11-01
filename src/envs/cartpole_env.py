"""CartPole environment adapted for goal-conditioned reinforcement learning."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from .base_env import GoalConditionedEnv


class GoalConditionedCartPole(GoalConditionedEnv):
    """
    CartPole environment adapted for goal-conditioned RL.
    
    The goal is to reach a specific position and velocity state.
    """
    
    def __init__(self, goal_position: float = 0.0, goal_velocity: float = 0.0, 
                 goal_angle: float = 0.0, goal_angular_velocity: float = 0.0,
                 tolerance: float = 0.1):
        """
        Initialize the goal-conditioned CartPole environment.
        
        Args:
            goal_position: Target cart position
            goal_velocity: Target cart velocity
            goal_angle: Target pole angle
            goal_angular_velocity: Target pole angular velocity
            tolerance: Tolerance for goal achievement
        """
        super().__init__()
        
        # Create underlying CartPole environment
        self.env = gym.make("CartPole-v1")
        
        # Define goal
        self.goal = np.array([goal_position, goal_velocity, goal_angle, goal_angular_velocity])
        self.tolerance = tolerance
        
        # Define spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.goal_space = spaces.Box(
            low=np.array([-4.8, -np.inf, -0.42, -np.inf]),
            high=np.array([4.8, np.inf, 0.42, np.inf]),
            dtype=np.float32
        )
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, _ = self.env.reset()
        info = {
            "goal": self.goal.copy(),
            "achieved_goal": self.get_achieved_goal(obs),
            "is_success": False
        }
        return obs, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        obs, _, terminated, truncated, _ = self.env.step(action)
        
        achieved_goal = self.get_achieved_goal(obs)
        reward = self.compute_reward(achieved_goal, self.goal, {})
        
        # Check if goal is achieved
        is_success = np.allclose(achieved_goal, self.goal, atol=self.tolerance)
        
        info = {
            "goal": self.goal.copy(),
            "achieved_goal": achieved_goal,
            "is_success": is_success
        }
        
        return obs, reward, terminated, truncated, info
        
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> float:
        """Compute reward based on distance to goal."""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 1.0 if distance < self.tolerance else -distance
        
    def get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """Extract achieved goal from state."""
        return state.copy()
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render(mode)
        
    def close(self) -> None:
        """Close the environment."""
        self.env.close()

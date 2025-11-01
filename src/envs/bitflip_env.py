"""BitFlip environment for goal-conditioned reinforcement learning."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from .base_env import GoalConditionedEnv


class BitFlipEnv(GoalConditionedEnv):
    """
    BitFlip environment where the agent must transform a bit vector into a target vector.
    
    This is a classic goal-conditioned environment used to test HER algorithms.
    The agent receives a random bit vector and must flip bits to match a target vector.
    """
    
    def __init__(self, n_bits: int = 5, max_steps: int = None):
        """
        Initialize the BitFlip environment.
        
        Args:
            n_bits: Number of bits in the vector
            max_steps: Maximum number of steps per episode (default: n_bits * 2)
        """
        super().__init__()
        self.n_bits = n_bits
        self.max_steps = max_steps or n_bits * 2
        self.current_step = 0
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_bits,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(n_bits)
        self.goal_space = spaces.Box(
            low=0, high=1, shape=(n_bits,), dtype=np.int32
        )
        
        # Initialize state
        self.state: np.ndarray = np.zeros(n_bits, dtype=np.int32)
        self.goal: np.ndarray = np.zeros(n_bits, dtype=np.int32)
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation and info dictionary
        """
        self.state = np.random.randint(0, 2, self.n_bits, dtype=np.int32)
        self.goal = np.random.randint(0, 2, self.n_bits, dtype=np.int32)
        self.current_step = 0
        
        info = {
            "goal": self.goal.copy(),
            "achieved_goal": self.get_achieved_goal(self.state),
            "is_success": False
        }
        
        return self.state.copy(), info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of bit to flip
            
        Returns:
            Next state, reward, terminated, truncated, info
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not valid")
            
        # Flip the specified bit
        self.state[action] = 1 - self.state[action]
        self.current_step += 1
        
        # Check if goal is achieved
        achieved_goal = self.get_achieved_goal(self.state)
        reward = self.compute_reward(achieved_goal, self.goal, {})
        
        # Check termination conditions
        terminated = np.array_equal(self.state, self.goal)
        truncated = self.current_step >= self.max_steps
        
        info = {
            "goal": self.goal.copy(),
            "achieved_goal": achieved_goal,
            "is_success": terminated,
            "episode_length": self.current_step
        }
        
        return self.state.copy(), reward, terminated, truncated, info
        
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> float:
        """
        Compute reward based on achieved and desired goals.
        
        Args:
            achieved_goal: The goal that was achieved
            desired_goal: The goal that was desired
            info: Additional information
            
        Returns:
            Reward value
        """
        return 1.0 if np.array_equal(achieved_goal, desired_goal) else 0.0
        
    def get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Extract achieved goal from current state.
        
        Args:
            state: Current state
            
        Returns:
            Achieved goal (same as state for BitFlip)
        """
        return state.copy()
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered image if mode is 'rgb_array', None otherwise
        """
        if mode == "human":
            print(f"State: {self.state}")
            print(f"Goal:  {self.goal}")
            print(f"Match: {np.array_equal(self.state, self.goal)}")
            print("-" * 20)
        elif mode == "rgb_array":
            # Create a simple visualization
            img = np.zeros((3, self.n_bits * 2, 3), dtype=np.uint8)
            for i in range(self.n_bits):
                # State bits (top row)
                color = [255, 255, 255] if self.state[i] else [0, 0, 0]
                img[0, i*2:(i+1)*2] = color
                # Goal bits (bottom row)
                color = [255, 255, 255] if self.goal[i] else [0, 0, 0]
                img[2, i*2:(i+1)*2] = color
            return img
        return None
        
    def close(self) -> None:
        """Close the environment."""
        pass

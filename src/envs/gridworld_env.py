"""GridWorld environment for goal-conditioned reinforcement learning."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
from .base_env import GoalConditionedEnv


class GridWorldEnv(GoalConditionedEnv):
    """
    GridWorld environment where the agent must navigate to a goal position.
    
    The agent starts at a random position and must reach a goal position
    by taking actions (up, down, left, right).
    """
    
    def __init__(self, width: int = 5, height: int = 5, max_steps: int = 50):
        """
        Initialize the GridWorld environment.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            max_steps: Maximum steps per episode
        """
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=0, high=max(width, height), shape=(2,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.goal_space = spaces.Box(
            low=0, high=max(width, height), shape=(2,), dtype=np.int32
        )
        
        # Initialize state
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([0, 0], dtype=np.int32)
        
        # Action mappings
        self.actions = {
            0: np.array([-1, 0]),  # up
            1: np.array([1, 0]),   # down
            2: np.array([0, -1]),  # left
            3: np.array([0, 1])    # right
        }
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        # Random agent position
        self.agent_pos = np.array([
            np.random.randint(0, self.height),
            np.random.randint(0, self.width)
        ], dtype=np.int32)
        
        # Random goal position (different from agent)
        while True:
            self.goal_pos = np.array([
                np.random.randint(0, self.height),
                np.random.randint(0, self.width)
            ], dtype=np.int32)
            if not np.array_equal(self.agent_pos, self.goal_pos):
                break
                
        self.current_step = 0
        
        info = {
            "goal": self.goal_pos.copy(),
            "achieved_goal": self.get_achieved_goal(self.agent_pos),
            "is_success": False
        }
        
        return self.agent_pos.copy(), info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not valid")
            
        # Move agent
        movement = self.actions[action]
        new_pos = self.agent_pos + movement
        
        # Check bounds
        if (0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width):
            self.agent_pos = new_pos
            
        self.current_step += 1
        
        # Check termination
        achieved_goal = self.get_achieved_goal(self.agent_pos)
        reward = self.compute_reward(achieved_goal, self.goal_pos, {})
        
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.current_step >= self.max_steps
        
        info = {
            "goal": self.goal_pos.copy(),
            "achieved_goal": achieved_goal,
            "is_success": terminated,
            "episode_length": self.current_step
        }
        
        return self.agent_pos.copy(), reward, terminated, truncated, info
        
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any]) -> float:
        """Compute reward based on distance to goal."""
        if np.array_equal(achieved_goal, desired_goal):
            return 1.0
        else:
            # Negative distance as penalty
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -distance / (self.width + self.height)  # Normalize
            
    def get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """Extract achieved goal from state."""
        return state.copy()
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            grid = np.zeros((self.height, self.width), dtype=str)
            grid.fill('.')
            
            # Mark agent position
            grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            
            # Mark goal position
            grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
            
            print("Grid World:")
            for row in grid:
                print(' '.join(row))
            print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")
            print("-" * 20)
            
        elif mode == "rgb_array":
            # Create RGB visualization
            img = np.zeros((self.height * 20, self.width * 20, 3), dtype=np.uint8)
            
            # Draw grid
            img[::20, :] = [128, 128, 128]  # Horizontal lines
            img[:, ::20] = [128, 128, 128]  # Vertical lines
            
            # Draw agent (blue)
            agent_y, agent_x = self.agent_pos * 20 + 10
            img[agent_y-5:agent_y+5, agent_x-5:agent_x+5] = [0, 0, 255]
            
            # Draw goal (red)
            goal_y, goal_x = self.goal_pos * 20 + 10
            img[goal_y-5:goal_y+5, goal_x-5:goal_x+5] = [255, 0, 0]
            
            return img
            
        return None
        
    def close(self) -> None:
        """Close the environment."""
        pass

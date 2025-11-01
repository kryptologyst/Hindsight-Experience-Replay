"""Hindsight Experience Replay (HER) buffer implementation."""

import random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from collections import deque


class HERBuffer:
    """
    Hindsight Experience Replay buffer with support for different HER strategies.
    
    HER allows learning from failed episodes by treating the achieved state
    as a virtual goal, making every trajectory useful for learning.
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        her_strategy: str = "future",
        k_future: int = 4,
        reward_function: str = "sparse"
    ):
        """
        Initialize the HER buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            her_strategy: HER strategy ('future', 'final', 'episode')
            k_future: Number of future goals to sample for 'future' strategy
            reward_function: Type of reward function ('sparse', 'dense')
        """
        self.capacity = capacity
        self.her_strategy = her_strategy
        self.k_future = k_future
        self.reward_function = reward_function
        
        # Buffer storage
        self.buffer: deque = deque(maxlen=capacity)
        self.episode_buffer: List[Tuple] = []
        
    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """
        Add a single transition to the episode buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            goal: Goal state
            done: Whether episode is done
            info: Additional information
        """
        transition = (state, action, reward, next_state, goal, done, info)
        self.episode_buffer.append(transition)
        
    def finish_episode(self) -> None:
        """Finish the current episode and add transitions to buffer with HER."""
        if not self.episode_buffer:
            return
            
        # Add original transitions
        for transition in self.episode_buffer:
            self.buffer.append(transition)
            
        # Add HER transitions
        her_transitions = self._generate_her_transitions()
        for transition in her_transitions:
            self.buffer.append(transition)
            
        # Clear episode buffer
        self.episode_buffer.clear()
        
    def _generate_her_transitions(self) -> List[Tuple]:
        """Generate hindsight experience replay transitions."""
        if not self.episode_buffer:
            return []
            
        her_transitions = []
        episode_length = len(self.episode_buffer)
        
        for i, (state, action, _, next_state, _, done, info) in enumerate(self.episode_buffer):
            # Sample new goal based on strategy
            if self.her_strategy == "future":
                # Sample from future states in the episode
                future_indices = list(range(i + 1, episode_length))
                if future_indices:
                    future_idx = random.choice(future_indices)
                    new_goal = self.episode_buffer[future_idx][3]  # next_state becomes goal
                else:
                    continue
                    
            elif self.her_strategy == "final":
                # Use final state as goal
                new_goal = self.episode_buffer[-1][3]  # final next_state
                
            elif self.her_strategy == "episode":
                # Sample from any state in the episode
                random_idx = random.randint(0, episode_length - 1)
                new_goal = self.episode_buffer[random_idx][3]  # random next_state
                
            else:
                raise ValueError(f"Unknown HER strategy: {self.her_strategy}")
                
            # Compute new reward
            achieved_goal = self._extract_achieved_goal(next_state, info)
            new_reward = self._compute_reward(achieved_goal, new_goal)
            
            # Create HER transition
            her_transition = (state, action, new_reward, next_state, new_goal, done, info)
            her_transitions.append(her_transition)
            
        return her_transitions
        
    def _extract_achieved_goal(self, state: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Extract achieved goal from state and info."""
        if "achieved_goal" in info:
            return info["achieved_goal"]
        else:
            # Fallback: assume state is the achieved goal
            return state.copy()
            
    def _compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> float:
        """Compute reward based on achieved and desired goals."""
        if self.reward_function == "sparse":
            return 1.0 if np.array_equal(achieved_goal, desired_goal) else 0.0
        elif self.reward_function == "dense":
            # Negative distance as dense reward
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return -distance
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
            
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, goals, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, goals, dones, infos = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(goals)),
            torch.BoolTensor(np.array(dones))
        )
        
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
        
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.episode_buffer.clear()

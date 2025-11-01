"""Hindsight Experience Replay (HER) agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import random
from collections import deque

from ..envs.base_env import GoalConditionedEnv
from ..utils.her_buffer import HERBuffer
from .models import GoalConditionedQNetwork, DuelingQNetwork


class HERAgent:
    """
    Hindsight Experience Replay agent for goal-conditioned RL.
    
    Implements HER with DQN for discrete action spaces.
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.98,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_frequency: int = 10,
        her_strategy: str = "future",
        k_future: int = 4,
        use_dueling: bool = False,
        device: str = "auto"
    ):
        """
        Initialize the HER agent.
        
        Args:
            state_dim: Dimension of state space
            goal_dim: Dimension of goal space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Final epsilon for exploration
            epsilon_decay: Epsilon decay rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_frequency: Frequency of target network updates
            her_strategy: HER strategy ('future', 'final', 'episode')
            k_future: Number of future goals for 'future' strategy
            use_dueling: Whether to use dueling DQN architecture
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize networks
        if use_dueling:
            self.q_network = DuelingQNetwork(state_dim, goal_dim, action_dim).to(self.device)
            self.target_network = DuelingQNetwork(state_dim, goal_dim, action_dim).to(self.device)
        else:
            self.q_network = GoalConditionedQNetwork(state_dim, goal_dim, action_dim).to(self.device)
            self.target_network = GoalConditionedQNetwork(state_dim, goal_dim, action_dim).to(self.device)
            
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.buffer = HERBuffer(
            capacity=buffer_size,
            her_strategy=her_strategy,
            k_future=k_future
        )
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_successes: deque = deque(maxlen=100)
        
    def select_action(self, state: np.ndarray, goal: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            goal: Current goal
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor, goal_tensor)
                action = q_values.argmax().item()
                
            return action
            
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """Store transition in the HER buffer."""
        self.buffer.add_transition(state, action, reward, next_state, goal, done, info)
        
    def finish_episode(self) -> None:
        """Finish the current episode and generate HER transitions."""
        self.buffer.finish_episode()
        
    def train(self) -> Optional[float]:
        """
        Train the agent on a batch of transitions.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if len(self.buffer) < self.batch_size:
            return None
            
        # Sample batch from buffer
        states, actions, rewards, next_states, goals, dones = self.buffer.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        goals = goals.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states, goals).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states, goals).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
        
    def update_statistics(self, episode_reward: float, episode_success: bool) -> None:
        """Update training statistics."""
        self.episode_rewards.append(episode_reward)
        self.episode_successes.append(episode_success)
        
    def get_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        if not self.episode_rewards:
            return {}
            
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "mean_success_rate": np.mean(self.episode_successes),
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "training_step": self.training_step
        }
        
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "episode_rewards": list(self.episode_rewards),
            "episode_successes": list(self.episode_successes)
        }
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.episode_rewards = deque(checkpoint["episode_rewards"], maxlen=100)
        self.episode_successes = deque(checkpoint["episode_successes"], maxlen=100)

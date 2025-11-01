"""Neural network models for goal-conditioned reinforcement learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GoalConditionedQNetwork(nn.Module):
    """
    Goal-conditioned Q-network for HER algorithms.
    
    Takes state and goal as input and outputs Q-values for each action.
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu"
    ):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of state space
            goal_dim: Dimension of goal space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Input layer (state + goal concatenated)
        input_dim = state_dim + goal_dim
        layers = [nn.Linear(input_dim, hidden_dim), self.activation()]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation()])
            
        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            goal: Goal tensor of shape (batch_size, goal_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        # Concatenate state and goal
        x = torch.cat([state, goal], dim=-1)
        return self.network(x)
        
    def get_action(self, state: torch.Tensor, goal: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: State tensor
            goal: Goal tensor
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.forward(state, goal)
                return q_values.argmax().item()


class GoalConditionedPolicyNetwork(nn.Module):
    """
    Goal-conditioned policy network for continuous control.
    
    Outputs action probabilities for discrete actions or mean/std for continuous actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        action_type: str = "discrete"
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            goal_dim: Dimension of goal space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation: Activation function
            action_type: Type of actions ('discrete', 'continuous')
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.action_type = action_type
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU
        elif activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "elu":
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Input layer
        input_dim = state_dim + goal_dim
        layers = [nn.Linear(input_dim, hidden_dim), self.activation()]
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation()])
            
        # Output layer
        if action_type == "discrete":
            layers.append(nn.Linear(hidden_dim, action_dim))
        else:  # continuous
            layers.append(nn.Linear(hidden_dim, action_dim * 2))  # mean and log_std
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor
            goal: Goal tensor
            
        Returns:
            Action logits (discrete) or mean/std (continuous)
        """
        x = torch.cat([state, goal], dim=-1)
        return self.network(x)
        
    def get_action(self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy.
        
        Args:
            state: State tensor
            goal: Goal tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_probability)
        """
        if self.action_type == "discrete":
            logits = self.forward(state, goal)
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            # Continuous actions
            output = self.forward(state, goal)
            mean, log_std = output.chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical issues
            
            if deterministic:
                action = torch.tanh(mean)
                log_prob = torch.zeros_like(action[..., 0])
            else:
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action_raw = dist.sample()
                action = torch.tanh(action_raw)
                
                # Compute log probability with tanh transformation
                log_prob = dist.log_prob(action_raw) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1)
                
            return action, log_prob


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture for improved value estimation.
    
    Separates value and advantage streams for better learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu"
    ):
        """Initialize the dueling Q-network."""
        super().__init__()
        
        input_dim = state_dim + goal_dim
        
        # Shared feature extraction
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        shared_layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(num_layers - 1):
            shared_layers.extend([nn.Linear(hidden_dim, hidden_dim), act_fn()])
            
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Linear(hidden_dim, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network."""
        x = torch.cat([state, goal], dim=-1)
        features = self.shared_network(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values

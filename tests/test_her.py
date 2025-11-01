"""Unit tests for HER components."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from src.envs.bitflip_env import BitFlipEnv
from src.envs.gridworld_env import GridWorldEnv
from src.envs import EnvironmentFactory
from src.agents.her_agent import HERAgent
from src.agents.models import GoalConditionedQNetwork, DuelingQNetwork
from src.utils.her_buffer import HERBuffer


class TestBitFlipEnv:
    """Test cases for BitFlip environment."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = BitFlipEnv(n_bits=5)
        assert env.n_bits == 5
        assert env.observation_space.shape == (5,)
        assert env.action_space.n == 5
        assert env.goal_space.shape == (5,)
    
    def test_reset(self):
        """Test environment reset."""
        env = BitFlipEnv(n_bits=3)
        state, info = env.reset()
        
        assert len(state) == 3
        assert len(info["goal"]) == 3
        assert state.dtype == np.int32
        assert info["goal"].dtype == np.int32
        assert "achieved_goal" in info
        assert "is_success" in info
    
    def test_step(self):
        """Test environment step."""
        env = BitFlipEnv(n_bits=3)
        state, info = env.reset()
        goal = info["goal"]
        
        # Test valid action
        action = 0
        next_state, reward, terminated, truncated, info = env.step(action)
        
        assert len(next_state) == 3
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "goal" in info
        assert "achieved_goal" in info
    
    def test_goal_achievement(self):
        """Test goal achievement detection."""
        env = BitFlipEnv(n_bits=2)
        
        # Set specific state and goal
        env.state = np.array([1, 0], dtype=np.int32)
        env.goal = np.array([1, 0], dtype=np.int32)
        
        # Any action should not change the reward since goal is already achieved
        _, reward, terminated, _, _ = env.step(0)
        assert reward == 1.0
        assert terminated == True
    
    def test_render(self):
        """Test environment rendering."""
        env = BitFlipEnv(n_bits=3)
        state, info = env.reset()
        
        # Test human mode (should not raise exception)
        env.render(mode="human")
        
        # Test rgb_array mode
        img = env.render(mode="rgb_array")
        assert img is not None
        assert img.shape == (3, 6, 3)  # height, width, channels


class TestGridWorldEnv:
    """Test cases for GridWorld environment."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = GridWorldEnv(width=5, height=5)
        assert env.width == 5
        assert env.height == 5
        assert env.observation_space.shape == (2,)
        assert env.action_space.n == 4
    
    def test_reset(self):
        """Test environment reset."""
        env = GridWorldEnv(width=3, height=3)
        state, info = env.reset()
        
        assert len(state) == 2
        assert 0 <= state[0] < 3
        assert 0 <= state[1] < 3
        assert not np.array_equal(state, info["goal"])
    
    def test_step(self):
        """Test environment step."""
        env = GridWorldEnv(width=3, height=3)
        state, info = env.reset()
        
        # Test valid action
        action = 0  # up
        next_state, reward, terminated, truncated, info = env.step(action)
        
        assert len(next_state) == 2
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_boundary_handling(self):
        """Test boundary handling."""
        env = GridWorldEnv(width=2, height=2)
        state, info = env.reset()
        
        # Set agent at top-left corner
        env.agent_pos = np.array([0, 0])
        
        # Try to move up (should stay in place)
        _, reward, _, _, _ = env.step(0)  # up
        assert np.array_equal(env.agent_pos, np.array([0, 0]))


class TestEnvironmentFactory:
    """Test cases for EnvironmentFactory."""
    
    def test_create_bitflip(self):
        """Test creating BitFlip environment."""
        env = EnvironmentFactory.create("bitflip", n_bits=4)
        assert isinstance(env, BitFlipEnv)
        assert env.n_bits == 4
    
    def test_create_gridworld(self):
        """Test creating GridWorld environment."""
        env = EnvironmentFactory.create("gridworld", width=4, height=4)
        assert isinstance(env, GridWorldEnv)
        assert env.width == 4
        assert env.height == 4
    
    def test_invalid_environment(self):
        """Test creating invalid environment."""
        with pytest.raises(ValueError):
            EnvironmentFactory.create("invalid_env")
    
    def test_list_environments(self):
        """Test listing available environments."""
        envs = EnvironmentFactory.list_environments()
        assert "bitflip" in envs
        assert "gridworld" in envs


class TestHERBuffer:
    """Test cases for HER buffer."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = HERBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert len(buffer) == 0
    
    def test_add_transition(self):
        """Test adding transitions."""
        buffer = HERBuffer()
        
        state = np.array([1, 0, 1])
        action = 0
        reward = 1.0
        next_state = np.array([0, 0, 1])
        goal = np.array([0, 0, 1])
        done = True
        info = {"achieved_goal": next_state}
        
        buffer.add_transition(state, action, reward, next_state, goal, done, info)
        assert len(buffer.episode_buffer) == 1
    
    def test_finish_episode(self):
        """Test finishing episode."""
        buffer = HERBuffer()
        
        # Add some transitions
        for i in range(3):
            state = np.array([i, i+1, i+2])
            action = i
            reward = float(i)
            next_state = np.array([i+1, i+2, i+3])
            goal = np.array([1, 1, 1])
            done = i == 2
            info = {"achieved_goal": next_state}
            
            buffer.add_transition(state, action, reward, next_state, goal, done, info)
        
        buffer.finish_episode()
        assert len(buffer) > 0  # Should have original + HER transitions
    
    def test_sample(self):
        """Test sampling from buffer."""
        buffer = HERBuffer()
        
        # Add transitions and finish episode
        for i in range(5):
            state = np.array([i])
            action = i % 2
            reward = float(i)
            next_state = np.array([i+1])
            goal = np.array([5])
            done = i == 4
            info = {"achieved_goal": next_state}
            
            buffer.add_transition(state, action, reward, next_state, goal, done, info)
        
        buffer.finish_episode()
        
        # Sample batch
        batch = buffer.sample(3)
        assert len(batch) == 6  # states, actions, rewards, next_states, goals, dones
        assert batch[0].shape[0] == 3  # batch size


class TestGoalConditionedQNetwork:
    """Test cases for Q-network."""
    
    def test_initialization(self):
        """Test network initialization."""
        net = GoalConditionedQNetwork(state_dim=4, goal_dim=4, action_dim=2)
        assert net.state_dim == 4
        assert net.goal_dim == 4
        assert net.action_dim == 2
    
    def test_forward(self):
        """Test forward pass."""
        net = GoalConditionedQNetwork(state_dim=3, goal_dim=3, action_dim=2)
        
        state = torch.randn(2, 3)  # batch_size=2
        goal = torch.randn(2, 3)
        
        q_values = net(state, goal)
        assert q_values.shape == (2, 2)  # batch_size, action_dim
    
    def test_get_action(self):
        """Test action selection."""
        net = GoalConditionedQNetwork(state_dim=2, goal_dim=2, action_dim=3)
        
        state = torch.randn(1, 2)
        goal = torch.randn(1, 2)
        
        action = net.get_action(state, goal, epsilon=0.0)
        assert 0 <= action < 3


class TestHERAgent:
    """Test cases for HER agent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = HERAgent(state_dim=4, goal_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.goal_dim == 4
        assert agent.action_dim == 2
        assert agent.epsilon > 0
    
    def test_select_action(self):
        """Test action selection."""
        agent = HERAgent(state_dim=3, goal_dim=3, action_dim=2)
        
        state = np.array([1, 0, 1])
        goal = np.array([0, 1, 0])
        
        action = agent.select_action(state, goal, training=True)
        assert 0 <= action < 2
    
    def test_store_transition(self):
        """Test storing transitions."""
        agent = HERAgent(state_dim=2, goal_dim=2, action_dim=2)
        
        state = np.array([1, 0])
        action = 0
        reward = 1.0
        next_state = np.array([0, 1])
        goal = np.array([0, 1])
        done = True
        info = {"achieved_goal": next_state}
        
        agent.store_transition(state, action, reward, next_state, goal, done, info)
        assert len(agent.buffer.episode_buffer) == 1
    
    def test_train(self):
        """Test training."""
        agent = HERAgent(state_dim=2, goal_dim=2, action_dim=2, batch_size=4)
        
        # Add some transitions
        for i in range(10):
            state = np.array([i % 2, (i+1) % 2])
            action = i % 2
            reward = float(i % 2)
            next_state = np.array([(i+1) % 2, (i+2) % 2])
            goal = np.array([1, 1])
            done = i == 9
            info = {"achieved_goal": next_state}
            
            agent.store_transition(state, action, reward, next_state, goal, done, info)
        
        agent.finish_episode()
        
        # Train
        loss = agent.train()
        assert loss is not None
        assert isinstance(loss, float)


if __name__ == "__main__":
    pytest.main([__file__])

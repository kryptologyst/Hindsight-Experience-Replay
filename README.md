# Hindsight Experience Replay

A production-ready implementation of Hindsight Experience Replay (HER) for goal-conditioned reinforcement learning, featuring multiple environments, state-of-the-art techniques, and comprehensive tooling.

## Features

- **Modern HER Implementation**: Complete HER algorithm with multiple strategies (future, final, episode)
- **Multiple Environments**: BitFlip, CartPole, and GridWorld environments
- **Advanced Neural Networks**: Goal-conditioned Q-networks, Dueling DQN, and policy networks
- **Comprehensive Tooling**: Streamlit UI, TensorBoard logging, checkpoint saving
- **Production Ready**: Type hints, comprehensive tests, configuration management
- **Extensible Design**: Easy to add new environments and algorithms

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Environments](#environments)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for faster training)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Hindsight-Experience-Replay.git
cd Hindsight-Experience-Replay

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium version: {gymnasium.__version__}')"
```

## Quick Start

### 1. Train HER Agent on BitFlip Environment

```bash
# Basic training
python train.py --env bitflip --episodes 1000

# With custom configuration
python train.py --config configs/config.yaml --env bitflip --episodes 2000 --render
```

### 2. Launch Streamlit Dashboard

```bash
streamlit run app.py
```

### 3. Run Tests

```bash
pytest tests/ -v
```

## Usage

### Command Line Training

```bash
# Train on different environments
python train.py --env bitflip --episodes 1000
python train.py --env cartpole --episodes 2000
python train.py --env gridworld --episodes 1500

# Custom configuration
python train.py --config configs/config.yaml --env bitflip --episodes 5000 --save-dir logs/experiment_1
```

### Python API

```python
from src.envs import EnvironmentFactory
from src.agents.her_agent import HERAgent
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment
env = EnvironmentFactory.create("bitflip", n_bits=5)

# Create agent
agent = HERAgent(
    state_dim=env.observation_space.shape[0],
    goal_dim=env.goal_space.shape[0],
    action_dim=env.action_space.n,
    **config['training']
)

# Training loop
for episode in range(1000):
    state, info = env.reset()
    goal = info['goal']
    
    for step in range(20):
        action = agent.select_action(state, goal)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, goal, terminated, info)
        state = next_state
        
        if terminated or truncated:
            break
    
    agent.finish_episode()
    agent.train()
```

### Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app.py
```

Features:
- Real-time training visualization
- Environment preview
- Configuration management
- Training progress monitoring
- Interactive plots

## Configuration

Configuration is managed through YAML files. See `configs/config.yaml` for all available options:

```yaml
# Training Configuration
training:
  episodes: 1000
  max_steps_per_episode: 20
  batch_size: 64
  learning_rate: 0.001
  gamma: 0.98
  epsilon_start: 0.9
  epsilon_end: 0.05
  epsilon_decay: 0.995

# HER Configuration
her:
  strategy: "future"  # Options: future, final, episode
  k_future: 4
  reward_function: "sparse"  # Options: sparse, dense

# Environment Configuration
environment:
  name: "bitflip"
  n_bits: 5
  render_mode: null

# Model Configuration
model:
  hidden_size: 64
  num_layers: 2
  activation: "relu"
  device: "auto"
```

## Environments

### BitFlip Environment

A classic goal-conditioned environment where the agent must transform a bit vector into a target vector.

```python
from src.envs import EnvironmentFactory

env = EnvironmentFactory.create("bitflip", n_bits=5)
state, info = env.reset()
print(f"State: {state}")
print(f"Goal: {info['goal']}")
```

### CartPole Environment

Adapted CartPole environment for goal-conditioned RL.

```python
env = EnvironmentFactory.create("cartpole", 
                               goal_position=0.0, 
                               goal_velocity=0.0,
                               tolerance=0.1)
```

### GridWorld Environment

Navigation task where the agent must reach a goal position.

```python
env = EnvironmentFactory.create("gridworld", width=5, height=5)
```

## Algorithms

### HER Agent

The main HER implementation with support for:

- **Future Strategy**: Sample goals from future states in the episode
- **Final Strategy**: Use the final state as a virtual goal
- **Episode Strategy**: Sample goals from any state in the episode

### Neural Networks

- **Goal-Conditioned Q-Network**: Standard Q-network with goal conditioning
- **Dueling DQN**: Separates value and advantage streams
- **Policy Networks**: For continuous action spaces

## Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard

# View training metrics
# Navigate to http://localhost:6006
```

### Checkpoint Saving

```python
# Save checkpoint
agent.save_checkpoint("checkpoints/agent_episode_1000.pt")

# Load checkpoint
agent.load_checkpoint("checkpoints/agent_episode_1000.pt")
```

### Training Statistics

```python
stats = agent.get_statistics()
print(f"Mean Reward: {stats['mean_reward']:.2f}")
print(f"Success Rate: {stats['mean_success_rate']:.2f}")
print(f"Epsilon: {stats['epsilon']:.3f}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_her.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📁 Project Structure

```
0264_Hindsight_experience_replay/
├── src/
│   ├── agents/
│   │   ├── her_agent.py      # HER agent implementation
│   │   └── models.py         # Neural network models
│   ├── envs/
│   │   ├── base_env.py       # Base environment interface
│   │   ├── bitflip_env.py    # BitFlip environment
│   │   ├── cartpole_env.py   # CartPole environment
│   │   ├── gridworld_env.py  # GridWorld environment
│   │   └── __init__.py       # Environment factory
│   └── utils/
│       ├── her_buffer.py     # HER replay buffer
│       └── logger.py         # Logging utilities
├── configs/
│   └── config.yaml           # Configuration file
├── tests/
│   └── test_her.py           # Unit tests
├── notebooks/                # Jupyter notebooks
├── logs/                     # Training logs and checkpoints
├── train.py                  # Training script
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Examples

### Basic HER Training

```python
# examples/basic_training.py
from src.envs import EnvironmentFactory
from src.agents.her_agent import HERAgent

# Create environment
env = EnvironmentFactory.create("bitflip", n_bits=4)

# Create agent
agent = HERAgent(
    state_dim=env.observation_space.shape[0],
    goal_dim=env.goal_space.shape[0],
    action_dim=env.action_space.n,
    learning_rate=0.001,
    gamma=0.98
)

# Training loop
for episode in range(500):
    state, info = env.reset()
    goal = info['goal']
    
    for step in range(10):
        action = agent.select_action(state, goal)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        agent.store_transition(state, action, reward, next_state, goal, terminated, info)
        state = next_state
        
        if terminated or truncated:
            break
    
    agent.finish_episode()
    loss = agent.train()
    
    if episode % 50 == 0:
        stats = agent.get_statistics()
        print(f"Episode {episode}: Success Rate = {stats['mean_success_rate']:.2f}")
```

### Custom Environment

```python
# examples/custom_env.py
from src.envs.base_env import GoalConditionedEnv
import numpy as np

class CustomEnv(GoalConditionedEnv):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.observation_space = spaces.Box(low=0, high=size, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.goal_space = spaces.Box(low=0, high=size, shape=(2,), dtype=np.int32)
    
    def reset(self):
        self.pos = np.random.randint(0, self.size, 2)
        self.goal = np.random.randint(0, self.size, 2)
        return self.pos.copy(), {"goal": self.goal.copy()}
    
    def step(self, action):
        # Implement your environment logic
        pass
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return 1.0 if np.array_equal(achieved_goal, desired_goal) else 0.0
    
    def get_achieved_goal(self, state):
        return state.copy()

# Register custom environment
from src.envs import EnvironmentFactory
EnvironmentFactory.register("custom", CustomEnv)
```

## 🔧 Development

### Code Style

This project follows PEP 8 style guidelines. Use the provided tools:

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Environments

1. Inherit from `GoalConditionedEnv`
2. Implement required methods
3. Register with `EnvironmentFactory`

### Adding New Algorithms

1. Create new agent class
2. Implement required methods
3. Add configuration options
4. Write tests

## Performance Tips

1. **Use GPU**: Set `device: "cuda"` in configuration
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Buffer Size**: Larger buffers improve sample diversity
4. **HER Strategy**: "future" strategy often works best
5. **Learning Rate**: Start with 1e-3, adjust based on performance

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or buffer size
2. **Slow Training**: Check if GPU is being used
3. **Poor Performance**: Try different HER strategies or hyperparameters
4. **Import Errors**: Ensure all dependencies are installed

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original HER paper: [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
- OpenAI Gym/Gymnasium for environment interfaces
- PyTorch team for the deep learning framework
- Streamlit team for the web interface

## References

1. Andrychowicz, M., et al. "Hindsight experience replay." NIPS 2017.
2. Schaul, T., et al. "Prioritized experience replay." ICLR 2016.
3. Wang, Z., et al. "Dueling network architectures for deep reinforcement learning." ICML 2016.

 
# Hindsight-Experience-Replay

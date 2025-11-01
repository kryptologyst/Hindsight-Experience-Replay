"""Training script for HER agent."""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from src.envs import EnvironmentFactory
from src.agents.her_agent import HERAgent
from src.utils.logger import setup_logger
from src.utils.config import Config


def train_her_agent(
    config: Dict[str, Any],
    env_name: str = "bitflip",
    save_dir: str = "logs",
    render: bool = False
) -> Dict[str, Any]:
    """
    Train HER agent on specified environment.
    
    Args:
        config: Training configuration
        env_name: Name of environment to train on
        save_dir: Directory to save logs and checkpoints
        render: Whether to render environment
        
    Returns:
        Training results dictionary
    """
    # Setup logging
    logger = setup_logger("HER_Training", level=config["logging"]["log_level"])
    logger.info(f"Starting HER training on {env_name} environment")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env_config = config["environment"].copy()
    env_config["name"] = env_name
    env = EnvironmentFactory.create(env_name, **env_config)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    goal_dim = env.goal_space.shape[0]
    action_dim = env.action_space.n
    
    logger.info(f"Environment: {env_name}")
    logger.info(f"State dim: {state_dim}, Goal dim: {goal_dim}, Action dim: {action_dim}")
    
    # Initialize agent
    agent_config = config["training"].copy()
    agent_config.update(config["her"])
    agent = HERAgent(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **agent_config
    )
    
    # Training loop
    episodes = config["training"]["episodes"]
    max_steps = config["training"]["max_steps_per_episode"]
    log_frequency = config["logging"]["log_frequency"]
    checkpoint_frequency = config["logging"]["checkpoint_frequency"]
    
    episode_rewards = []
    episode_successes = []
    training_losses = []
    
    logger.info(f"Starting training for {episodes} episodes")
    
    for episode in range(episodes):
        # Reset environment
        state, info = env.reset()
        goal = info["goal"]
        episode_reward = 0
        episode_success = False
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, goal, training=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(
                state, action, reward, next_state, goal, 
                terminated or truncated, info
            )
            
            episode_reward += reward
            state = next_state
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                training_losses.append(loss)
            
            # Check termination
            if terminated or truncated:
                episode_success = terminated
                break
                
        # Finish episode and update statistics
        agent.finish_episode()
        agent.update_statistics(episode_reward, episode_success)
        
        episode_rewards.append(episode_reward)
        episode_successes.append(episode_success)
        
        # Logging
        if episode % log_frequency == 0:
            stats = agent.get_statistics()
            logger.info(
                f"Episode {episode}: "
                f"Reward={episode_reward:.2f}, "
                f"Success={episode_success}, "
                f"Mean Reward={stats.get('mean_reward', 0):.2f}, "
                f"Success Rate={stats.get('mean_success_rate', 0):.2f}, "
                f"Epsilon={stats.get('epsilon', 0):.3f}"
            )
            
        # Save checkpoint
        if episode % checkpoint_frequency == 0 and episode > 0:
            checkpoint_path = save_path / f"checkpoint_episode_{episode}.pt"
            agent.save_checkpoint(str(checkpoint_path))
            logger.info(f"Saved checkpoint at episode {episode}")
            
        # Render if requested
        if render and episode % 100 == 0:
            env.render()
            
    # Final checkpoint
    final_checkpoint_path = save_path / "final_checkpoint.pt"
    agent.save_checkpoint(str(final_checkpoint_path))
    
    # Save training results
    results = {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "training_losses": training_losses,
        "final_stats": agent.get_statistics()
    }
    
    # Plot training curves
    plot_training_curves(results, save_path)
    
    logger.info("Training completed successfully")
    return results


def plot_training_curves(results: Dict[str, Any], save_path: Path) -> None:
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(results["episode_rewards"])
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    
    # Success rate (smoothed)
    window_size = 50
    if len(results["episode_successes"]) >= window_size:
        success_rate = np.convolve(
            results["episode_successes"], 
            np.ones(window_size)/window_size, 
            mode='valid'
        )
        axes[0, 1].plot(success_rate)
        axes[0, 1].set_title(f"Success Rate (Smoothed, window={window_size})")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Success Rate")
        axes[0, 1].grid(True)
    
    # Training loss
    if results["training_losses"]:
        axes[1, 0].plot(results["training_losses"])
        axes[1, 0].set_title("Training Loss")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)
    
    # Reward distribution
    axes[1, 1].hist(results["episode_rewards"], bins=50, alpha=0.7)
    axes[1, 1].set_title("Reward Distribution")
    axes[1, 1].set_xlabel("Reward")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train HER agent")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--env", type=str, default="bitflip",
                       choices=["bitflip", "cartpole", "gridworld"],
                       help="Environment to train on")
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of training episodes (overrides config)")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during training")
    parser.add_argument("--save-dir", type=str, default="logs",
                       help="Directory to save logs and checkpoints")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.episodes is not None:
        config["training"]["episodes"] = args.episodes
    
    # Train agent
    results = train_her_agent(
        config=config,
        env_name=args.env,
        save_dir=args.save_dir,
        render=args.render
    )
    
    print(f"Training completed. Final success rate: {results['final_stats'].get('mean_success_rate', 0):.2f}")


if __name__ == "__main__":
    main()

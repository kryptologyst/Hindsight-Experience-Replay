#!/usr/bin/env python3
"""Command-line interface for HER training."""

import argparse
import sys
import yaml
from pathlib import Path

from src.envs import EnvironmentFactory
from src.agents.her_agent import HERAgent
from src.utils.logger import setup_logger


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Train HER agent on goal-conditioned environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  her-train --env bitflip --episodes 1000
  her-train --env cartpole --episodes 2000 --render
  her-train --config configs/custom.yaml --env gridworld
        """
    )
    
    # Environment selection
    parser.add_argument(
        "--env", 
        choices=["bitflip", "cartpole", "gridworld"],
        default="bitflip",
        help="Environment to train on (default: bitflip)"
    )
    
    # Training parameters
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=1000,
        help="Number of training episodes (default: 1000)"
    )
    
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=None,
        help="Maximum steps per episode (default: environment specific)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size (default: 64)"
    )
    
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=0.98,
        help="Discount factor (default: 0.98)"
    )
    
    # HER parameters
    parser.add_argument(
        "--her-strategy", 
        choices=["future", "final", "episode"],
        default="future",
        help="HER strategy (default: future)"
    )
    
    parser.add_argument(
        "--k-future", 
        type=int, 
        default=4,
        help="Number of future goals for 'future' strategy (default: 4)"
    )
    
    # Output options
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="logs",
        help="Directory to save logs and checkpoints (default: logs)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render environment during training"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger("HER_CLI", level=log_level)
    
    try:
        # Load configuration
        config = {}
        if Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Configuration file not found: {args.config}")
        
        # Override config with command line arguments
        if "training" not in config:
            config["training"] = {}
        
        config["training"].update({
            "episodes": args.episodes,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
        })
        
        if args.max_steps:
            config["training"]["max_steps_per_episode"] = args.max_steps
        
        if "her" not in config:
            config["her"] = {}
        
        config["her"].update({
            "strategy": args.her_strategy,
            "k_future": args.k_future,
        })
        
        # Create environment
        logger.info(f"Creating {args.env} environment")
        env_config = config.get("environment", {})
        env_config["name"] = args.env
        
        env = EnvironmentFactory.create(args.env, **env_config)
        
        # Get environment dimensions
        state_dim = env.observation_space.shape[0]
        goal_dim = env.goal_space.shape[0]
        action_dim = env.action_space.n
        
        logger.info(f"Environment: {args.env}")
        logger.info(f"State dim: {state_dim}, Goal dim: {goal_dim}, Action dim: {action_dim}")
        
        # Create agent
        logger.info("Creating HER agent")
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
        max_steps = config["training"].get("max_steps_per_episode", 20)
        
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
                
                # Check termination
                if terminated or truncated:
                    episode_success = terminated
                    break
            
            # Finish episode and update statistics
            agent.finish_episode()
            agent.update_statistics(episode_reward, episode_success)
            
            # Logging
            if episode % 100 == 0:
                stats = agent.get_statistics()
                logger.info(
                    f"Episode {episode}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Success={episode_success}, "
                    f"Mean Reward={stats.get('mean_reward', 0):.2f}, "
                    f"Success Rate={stats.get('mean_success_rate', 0):.2f}"
                )
            
            # Render if requested
            if args.render and episode % 100 == 0:
                env.render()
        
        # Final statistics
        final_stats = agent.get_statistics()
        logger.info("Training completed!")
        logger.info(f"Final success rate: {final_stats.get('mean_success_rate', 0):.2%}")
        logger.info(f"Final mean reward: {final_stats.get('mean_reward', 0):.2f}")
        
        # Save final checkpoint
        save_path = Path(args.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_path / "final_checkpoint.pt"
        agent.save_checkpoint(str(checkpoint_path))
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    main()

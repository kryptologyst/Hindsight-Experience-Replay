"""Streamlit UI for HER training visualization and control."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import yaml
from pathlib import Path
import torch
import subprocess
import threading
import time
import json

from src.envs import EnvironmentFactory
from src.agents.her_agent import HERAgent
from src.utils.logger import setup_logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}


def save_config(config: dict, config_path: str = "configs/config.yaml") -> None:
    """Save configuration to YAML file."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training(config: dict, env_name: str, episodes: int) -> None:
    """Run training in background thread."""
    try:
        # Update config with UI parameters
        config["training"]["episodes"] = episodes
        
        # Save updated config
        save_config(config)
        
        # Run training script
        cmd = [
            "python", "train.py",
            "--config", "configs/config.yaml",
            "--env", env_name,
            "--episodes", str(episodes),
            "--save-dir", "logs"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Training completed successfully!")
        else:
            st.error(f"Training failed: {result.stderr}")
            
    except Exception as e:
        st.error(f"Error running training: {str(e)}")


def load_training_results(log_dir: str = "logs") -> dict:
    """Load training results from log directory."""
    log_path = Path(log_dir)
    
    if not log_path.exists():
        return {}
    
    # Look for checkpoint files to extract training info
    checkpoints = list(log_path.glob("checkpoint_*.pt"))
    
    if not checkpoints:
        return {}
    
    # Load the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
    
    try:
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Extract training statistics
        results = {
            "episode_rewards": checkpoint.get("episode_rewards", []),
            "episode_successes": checkpoint.get("episode_successes", []),
            "training_step": checkpoint.get("training_step", 0),
            "epsilon": checkpoint.get("epsilon", 0.0)
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error loading checkpoint: {str(e)}")
        return {}


def create_training_plots(results: dict) -> None:
    """Create and display training plots."""
    if not results or not results.get("episode_rewards"):
        st.warning("No training results available. Please run training first.")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Episode Rewards", "Success Rate", "Reward Distribution", "Training Progress"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(
            y=results["episode_rewards"],
            mode='lines',
            name='Episode Rewards',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Success rate (smoothed)
    if results.get("episode_successes"):
        window_size = min(50, len(results["episode_successes"]) // 10)
        if window_size > 1:
            success_rate = np.convolve(
                results["episode_successes"], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            fig.add_trace(
                go.Scatter(
                    y=success_rate,
                    mode='lines',
                    name='Success Rate',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
    
    # Reward distribution
    fig.add_trace(
        go.Histogram(
            x=results["episode_rewards"],
            name='Reward Distribution',
            nbinsx=30
        ),
        row=2, col=1
    )
    
    # Training progress (cumulative success rate)
    if results.get("episode_successes"):
        cumulative_success = np.cumsum(results["episode_successes"]) / np.arange(1, len(results["episode_successes"]) + 1)
        fig.add_trace(
            go.Scatter(
                y=cumulative_success,
                mode='lines',
                name='Cumulative Success Rate',
                line=dict(color='red')
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="HER Training Results",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="HER Training Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Hindsight Experience Replay (HER) Training Dashboard")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Load default config
    config = load_config()
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["bitflip", "cartpole", "gridworld"],
        index=0
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    episodes = st.sidebar.number_input("Episodes", min_value=100, max_value=10000, value=1000)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.0e")
    batch_size = st.sidebar.number_input("Batch Size", min_value=16, max_value=256, value=64)
    gamma = st.sidebar.slider("Gamma (Discount Factor)", 0.9, 0.99, 0.98)
    
    # HER parameters
    st.sidebar.subheader("HER Parameters")
    her_strategy = st.sidebar.selectbox("HER Strategy", ["future", "final", "episode"])
    k_future = st.sidebar.number_input("K Future", min_value=1, max_value=10, value=4)
    
    # Update config
    if config:
        config["training"]["episodes"] = episodes
        config["training"]["learning_rate"] = learning_rate
        config["training"]["batch_size"] = batch_size
        config["training"]["gamma"] = gamma
        config["her"]["strategy"] = her_strategy
        config["her"]["k_future"] = k_future
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Control")
        
        # Training buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("🚀 Start Training", type="primary"):
                if config:
                    with st.spinner("Training in progress..."):
                        run_training(config, env_name, episodes)
                else:
                    st.error("Configuration not loaded. Please check config file.")
        
        with col_stop:
            if st.button("⏹️ Stop Training"):
                st.warning("Training stop functionality not implemented yet.")
        
        # Environment visualization
        st.header("Environment Preview")
        
        if st.button("🎮 Preview Environment"):
            try:
                env = EnvironmentFactory.create(env_name)
                state, info = env.reset()
                
                col_env1, col_env2 = st.columns(2)
                
                with col_env1:
                    st.subheader("Initial State")
                    st.write(f"State: {state}")
                    st.write(f"Goal: {info['goal']}")
                
                with col_env2:
                    st.subheader("Environment Info")
                    st.write(f"State Space: {env.observation_space}")
                    st.write(f"Action Space: {env.action_space}")
                    st.write(f"Goal Space: {env.goal_space}")
                
                # Render environment
                if hasattr(env, 'render'):
                    render_output = env.render(mode="rgb_array")
                    if render_output is not None:
                        st.image(render_output, caption=f"{env_name} Environment", use_column_width=True)
                
                env.close()
                
            except Exception as e:
                st.error(f"Error previewing environment: {str(e)}")
    
    with col2:
        st.header("Training Statistics")
        
        # Load and display results
        results = load_training_results()
        
        if results:
            st.metric("Training Steps", results.get("training_step", 0))
            st.metric("Current Epsilon", f"{results.get('epsilon', 0):.3f}")
            
            if results.get("episode_rewards"):
                avg_reward = np.mean(results["episode_rewards"][-100:])  # Last 100 episodes
                st.metric("Avg Reward (Last 100)", f"{avg_reward:.2f}")
            
            if results.get("episode_successes"):
                recent_success = np.mean(results["episode_successes"][-100:])  # Last 100 episodes
                st.metric("Success Rate (Last 100)", f"{recent_success:.2%}")
        else:
            st.info("No training results available. Start training to see statistics.")
    
    # Training plots
    st.header("Training Progress")
    create_training_plots(results)
    
    # Configuration display
    with st.expander("📋 Current Configuration"):
        if config:
            st.json(config)
        else:
            st.error("Configuration not loaded")
    
    # Footer
    st.markdown("---")
    st.markdown("**HER Training Dashboard** - Built with Streamlit and PyTorch")


if __name__ == "__main__":
    main()

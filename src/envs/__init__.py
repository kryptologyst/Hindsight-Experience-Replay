"""Environment factory for creating different goal-conditioned environments."""

from typing import Dict, Type, Any
from .base_env import GoalConditionedEnv
from .bitflip_env import BitFlipEnv
from .cartpole_env import GoalConditionedCartPole
from .gridworld_env import GridWorldEnv


class EnvironmentFactory:
    """Factory class for creating goal-conditioned environments."""
    
    _environments: Dict[str, Type[GoalConditionedEnv]] = {
        "bitflip": BitFlipEnv,
        "cartpole": GoalConditionedCartPole,
        "gridworld": GridWorldEnv,
    }
    
    @classmethod
    def create(cls, env_name: str, **kwargs) -> GoalConditionedEnv:
        """
        Create an environment instance.
        
        Args:
            env_name: Name of the environment to create
            **kwargs: Additional arguments for environment initialization
            
        Returns:
            Environment instance
            
        Raises:
            ValueError: If environment name is not supported
        """
        if env_name not in cls._environments:
            available = ", ".join(cls._environments.keys())
            raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")
            
        return cls._environments[env_name](**kwargs)
    
    @classmethod
    def list_environments(cls) -> list:
        """List all available environments."""
        return list(cls._environments.keys())
    
    @classmethod
    def register(cls, name: str, env_class: Type[GoalConditionedEnv]) -> None:
        """
        Register a new environment class.
        
        Args:
            name: Name to register the environment under
            env_class: Environment class to register
        """
        cls._environments[name] = env_class

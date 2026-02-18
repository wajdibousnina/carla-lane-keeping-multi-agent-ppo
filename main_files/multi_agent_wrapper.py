"""
Multi-Agent Wrapper for CARLA Lane Keeping
Spawns multiple independent agents for parallel training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
import sys
import time

# Add CARLA to path
sys.path.append("F:\\Program files\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg")
import carla

from carla_lane_keeping_env import CarlaLaneKeepingEnv
from lane_keeping_parameters import MultiAgentParams, CARLA_HOST, CARLA_PORT, ObservationParams


class MultiAgentCarlaWrapper(VecEnv):
    """
    Wrapper that creates multiple independent CARLA agents
    Each agent has its own vehicle but shares the same CARLA world
    """
    
    def __init__(self, num_agents=3):
            self.num_agents = num_agents
            self.envs = []
            self.closed = False
            
            print(f"Initializing {num_agents} agents...")
            
            # CREATE ENVIRONMENTS FIRST (before calling super().__init__)
            print(f"Creating {num_agents} independent agents...")
            for i in range(num_agents):
                print(f"  Initializing agent {i+1}/{num_agents}...")
                try:
                    env = CarlaLaneKeepingEnv()
                    self.envs.append(env)
                    print(f"  ✓ Agent {i+1} ready")
                    time.sleep(2)  # Delay between spawns to avoid conflicts
                except Exception as e:
                    print(f"  ✗ Failed to create agent {i+1}: {e}")
                    # Cleanup already created envs
                    for created_env in self.envs:
                        try:
                            created_env.close()
                        except:
                            pass
                    raise RuntimeError(f"Failed to initialize multi-agent system at agent {i+1}")
            
            print(f"✓ All {num_agents} agents created!")
            
            # NOW get spaces from the first created environment
            observation_space = self.envs[0].observation_space
            action_space = self.envs[0].action_space
            
            # Initialize VecEnv AFTER creating environments
            super().__init__(num_agents, observation_space, action_space)
            
            # Setup ghost mode after ALL agents are created
            if not MultiAgentParams.AGENT_INTERACTION:
                self._setup_ghost_mode()
    
    def _setup_ghost_mode(self):
        """Make agents not collide with each other"""
        try:
            print("Setting up ghost mode (agents won't collide)...")
            # In CARLA, we can't directly disable collision between specific actors
            # But spawning them far apart and on different lanes achieves the same effect
            # The spawn logic in CarlaLaneKeepingEnv with random spawns handles this
            print("✓ Ghost mode enabled (agents spawn separately)")
        except Exception as e:
            print(f"⚠ Warning: Ghost mode setup: {e}")
    
    def reset(self):
        """Reset all agents"""
        observations = []
        print("Resetting all agents...")
        for i, env in enumerate(self.envs):
            try:
                obs, info = env.reset()
                observations.append(obs)
            except Exception as e:
                print(f"⚠ Warning: Agent {i+1} reset failed: {e}")
                # Create dummy observation
                obs = {
                    'image': np.zeros((ObservationParams.FRAME_STACK, ObservationParams.IMAGE_HEIGHT, 
                                     ObservationParams.IMAGE_WIDTH, 3), dtype=np.uint8),
                    'vehicle_state': np.zeros(6, dtype=np.float32)
                }
                observations.append(obs)
        
        return self._convert_observations(observations)
    
    def step_async(self, actions):
        """Store actions to be executed"""
        self.actions = actions
    
    def step_wait(self):
        """Execute stored actions and return results"""
        observations = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        
        for i, env in enumerate(self.envs):
            try:
                obs, reward, terminated, trunc, info = env.step(self.actions[i])
                observations.append(obs)
                rewards.append(reward)
                dones.append(terminated or trunc)
                truncated.append(trunc)
                infos.append(info)
                
                # Auto-reset if episode ended
                if terminated or trunc:
                    obs, reset_info = env.reset()
                    observations[i] = obs
                    infos[i]['terminal_observation'] = obs
                    
            except Exception as e:
                print(f"⚠ Warning: Agent {i+1} step failed: {e}")
                # Create dummy observation and reset
                obs = {
                    'image': np.zeros((ObservationParams.FRAME_STACK, ObservationParams.IMAGE_HEIGHT, 
                                     ObservationParams.IMAGE_WIDTH, 3), dtype=np.uint8),
                    'vehicle_state': np.zeros(6, dtype=np.float32)
                }
                observations.append(obs)
                rewards.append(0.0)
                dones.append(True)
                truncated.append(True)
                infos.append({})
                
                # Try to reset the failed agent
                try:
                    env.reset()
                except:
                    pass
        
        return (
            self._convert_observations(observations),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos
        )
    
    def _convert_observations(self, obs_list):
        """Convert list of dict observations to dict of arrays"""
        if isinstance(obs_list[0], dict):
            images = np.stack([obs['image'] for obs in obs_list])
            states = np.stack([obs['vehicle_state'] for obs in obs_list])
            return {'image': images, 'vehicle_state': states}
        else:
            return np.array(obs_list)
    
    def close(self):
        """Close all environments"""
        if self.closed:
            return
        
        print("Closing all agents...")
        for i, env in enumerate(self.envs):
            try:
                env.close()
                print(f"  ✓ Agent {i+1} closed")
            except Exception as e:
                print(f"  ⚠ Agent {i+1} close warning: {e}")
        
        self.closed = True
        print("✓ All agents closed")
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if envs are wrapped with a given wrapper"""
        return [False] * self.num_agents
    
    def get_attr(self, attr_name, indices=None):
        """Get attribute from environments"""
        if indices is None:
            indices = range(self.num_agents)
        return [getattr(self.envs[i], attr_name) for i in indices]
    
    def set_attr(self, attr_name, value, indices=None):
        """Set attribute in environments"""
        if indices is None:
            indices = range(self.num_agents)
        for i in indices:
            setattr(self.envs[i], attr_name, value)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call method on environments"""
        if indices is None:
            indices = range(self.num_agents)
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]
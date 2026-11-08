# Linux Training: Gymnasium Environment (cs16_gym_env.py)

```python
"""
Gymnasium environment for CS 1.6 RL agent.
Connects to Windows bridge server via ZeroMQ.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import zmq
import msgpack
import time
import logging
from typing import Dict, Tuple, Optional, Any
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class CS16GymEnv(gym.Env):
    """
    Gymnasium environment for Counter-Strike 1.6.
    
    Observation Space (Dict):
    - self: health, armor, ammo, origin, view_angles, velocity
    - enemy: relative_angles, distance, visible, health
    - lidar: front, left, right distances
    
    Action Space (Box, continuous):
    - move_forward: [-1, 1]
    - move_strafe: [-1, 1]
    - delta_yaw: [-1, 1]
    - delta_pitch: [-1, 1]
    - fire: [0, 1]
    """
    
    metadata = {"render_modes": [], "render_fps": 30}
    
    def __init__(
        self,
        host: str = "192.168.1.100",  # Windows machine IP
        port: int = 5555,
        config_path: Optional[str] = None,
        max_episode_steps: int = 10000,
        reward_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.host = host
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        
        # Load config
        self.config = self._load_config(config_path)
        if reward_config:
            self.config['reward'].update(reward_config)
        
        # ZeroMQ connection
        self.context = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self._connect()
        
        # Define spaces
        self._define_spaces()
        
        # Episode tracking
        self.total_reward = 0.0
        self.episode_count = 0
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        default_config = {
            'reward': {
                'aim_weight': 0.1,
                'approach_weight': 0.001,
                'damage_weight': 5.0,
                'kill_weight': 30.0,
                'miss_penalty': 0.02,
                'wall_penalty': 0.1,
                'health_penalty': 0.05,
            },
            'observation': {
                'normalize_origin': True,
                'map_bounds': [[-4096, -4096, -4096], [4096, 4096, 4096]],
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _connect(self):
        """Connect to Windows bridge server."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        logger.info(f"Connected to bridge server at {self.host}:{self.port}")
        
        # Test connection
        self._send_request({"type": "ping"})
    
    def _define_spaces(self):
        """Define observation and action spaces."""
        
        # Observation space - Dict of Box spaces
        self.observation_space = spaces.Dict({
            'self': spaces.Dict({
                'health': spaces.Box(0, 1, shape=(), dtype=np.float32),
                'armor': spaces.Box(0, 1, shape=(), dtype=np.float32),
                'ammo': spaces.Box(0, 1, shape=(), dtype=np.float32),
                'origin': spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                'view_angles': spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32),
                'velocity': spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            }),
            'enemy': spaces.Dict({
                'relative_angles': spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32),
                'distance': spaces.Box(0, 1, shape=(), dtype=np.float32),
                'visible': spaces.Box(0, 1, shape=(), dtype=np.float32),
                'health': spaces.Box(0, 1, shape=(), dtype=np.float32),
            }),
            'lidar': spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        })
        
        # Action space - 5 continuous values
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
    
    def _send_request(self, request: Dict) -> Dict:
        """Send request to bridge server and get response."""
        try:
            self.socket.send(msgpack.packb(request, use_bin_type=True))
            response_bytes = self.socket.recv()
            return msgpack.unpackb(response_bytes, raw=False)
        except zmq.Again:
            logger.error("Timeout waiting for bridge server response")
            raise ConnectionError("Bridge server timeout")
        except Exception as e:
            logger.error(f"Communication error: {e}")
            raise
    
    def _dict_to_obs_array(self, obs_dict: Dict) -> Dict:
        """Convert observation dict to numpy arrays matching space definition."""
        # Ensure all values are numpy arrays with correct dtype
        def to_array(val, dtype=np.float32):
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=dtype)
            return np.array([val], dtype=dtype)[0] if np.isscalar(val) else np.array(val, dtype=dtype)
        
        return {
            'self': {
                'health': to_array(obs_dict['self']['health']),
                'armor': to_array(obs_dict['self']['armor']),
                'ammo': to_array(obs_dict['self']['ammo']),
                'origin': to_array(obs_dict['self']['origin']),
                'view_angles': to_array(obs_dict['self']['view_angles']),
                'velocity': to_array(obs_dict['self']['velocity']),
            },
            'enemy': {
                'relative_angles': to_array(obs_dict['enemy']['relative_angles']),
                'distance': to_array(obs_dict['enemy']['distance']),
                'visible': to_array(obs_dict['enemy']['visible']),
                'health': to_array(obs_dict['enemy']['health']),
            },
            'lidar': to_array(obs_dict['lidar']),
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.episode_step = 0
        self.total_reward = 0.0
        self.episode_count += 1
        
        # Request reset from bridge
        response = self._send_request({"type": "reset"})
        
        if "error" in response:
            raise RuntimeError(f"Reset failed: {response['error']}")
        
        obs = self._dict_to_obs_array(response["observation"])
        info = response.get("info", {})
        
        logger.info(f"Episode {self.episode_count} started")
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Send step request
        response = self._send_request({
            "type": "step",
            "action": action.tolist()
        })
        
        if "error" in response:
            raise RuntimeError(f"Step failed: {response['error']}")
        
        obs = self._dict_to_obs_array(response["observation"])
        reward = float(response["reward"])
        terminated = bool(response["terminated"])
        truncated = bool(response["truncated"])
        info = response.get("info", {})
        
        self.episode_step += 1
        self.total_reward += reward
        
        # Add episode info
        info["episode_step"] = self.episode_step
        info["episode_reward"] = self.total_reward
        
        if terminated or truncated:
            logger.info(f"Episode {self.episode_count} ended: steps={self.episode_step}, reward={self.total_reward:.2f}")
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render not implemented - game runs on Windows."""
        pass
    
    def close(self):
        """Close connection to bridge server."""
        if self.socket:
            self.socket.close()
        self.context.term()
        logger.info("Environment closed")
    
    def __del__(self):
        self.close()


# Convenience function for creating env
def make_cs16_env(
    host: str = "192.168.1.100",
    port: int = 5555,
    config_path: Optional[str] = None,
    **kwargs
) -> CS16GymEnv:
    """Factory function to create CS16GymEnv."""
    return CS16GymEnv(host=host, port=port, config_path=config_path, **kwargs)


# Test script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test connection (update IP to your Windows machine)
    env = make_cs16_env(host="192.168.1.100", port=5555)
    
    try:
        print("Testing environment...")
        obs, info = env.reset()
        print(f"Observation keys: {obs.keys()}")
        print(f"Self health: {obs['self']['health']}")
        print(f"Enemy distance: {obs['enemy']['distance']}")
        
        # Random actions for 10 steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: reward={reward:.4f}, terminated={terminated}")
            if terminated or truncated:
                break
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        env.close()
```

## Configuration File (configs/training_config.yaml)

```yaml
reward:
  aim_weight: 0.1
  approach_weight: 0.001
  damage_weight: 5.0
  kill_weight: 30.0
  miss_penalty: 0.02
  wall_penalty: 0.1
  health_penalty: 0.05

observation:
  normalize_origin: true
  map_bounds: [[-4096, -4096, -4096], [4096, 4096, 4096]]

training:
  total_timesteps: 1000000
  n_envs: 1
  eval_freq: 50000
  save_freq: 100000
```

## Usage with Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from cs16_gym_env import make_cs16_env

# Create environment
env = make_cs16_env(host="192.168.1.100", config_path="configs/training_config.yaml")
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Callbacks
eval_callback = EvalCallback(
    env, 
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=50000,
    deterministic=True
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./logs/checkpoints",
    name_prefix="ppo_cs16"
)

# Train
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/tensorboard",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

model.learn(
    total_timesteps=1000000,
    callback=[eval_callback, checkpoint_callback],
    tb_log_name="PPO_CS16"
)

model.save("ppo_cs16_final")
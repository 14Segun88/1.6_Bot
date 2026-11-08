# Linux Training: PPO Training Script (train_ppo.py)

```python
"""
PPO training script for CS 1.6 RL agent.
Uses Stable-Baselines3 with custom callbacks for curriculum learning.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cs16_gym_env import make_cs16_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """
    Curriculum learning callback - adjusts environment difficulty based on performance.
    """
    def __init__(self, eval_env, phase_thresholds: dict, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.phase_thresholds = phase_thresholds
        self.current_phase = 1
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Check if we should advance curriculum phase
        if len(self.model.ep_info_buffer) > 10:
            mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
            
            if self.current_phase == 1 and mean_reward > self.phase_thresholds.get('phase1_to_2', 50):
                self._advance_phase(2)
            elif self.current_phase == 2 and mean_reward > self.phase_thresholds.get('phase2_to_3', 150):
                self._advance_phase(3)
                
        return True
    
    def _advance_phase(self, new_phase: int):
        self.current_phase = new_phase
        logger.info(f"=== CURRICULUM: Advancing to Phase {new_phase} ===")
        
        # Send phase change to environment (would need custom method)
        # self.eval_env.env_method("set_phase", new_phase)
        
        # For now, just log
        if new_phase == 2:
            logger.info("Phase 2: Navigation & Search - Multiple dummies on map")
        elif new_phase == 3:
            logger.info("Phase 3: Active 1v1 Combat - ZBots with weapons")


class TensorboardCallback(BaseCallback):
    """Custom tensorboard logging."""
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep['l'] for ep in self.model.ep_info_buffer]
            
            self.logger.record("rollout/ep_rew_mean", np.mean(ep_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(ep_lengths))
            self.logger.record("rollout/ep_rew_std", np.std(ep_rewards))
        
        return True


def make_env(host: str, port: int, config_path: str, rank: int, seed: int = 0):
    """Factory for creating vectorized environments."""
    def _init():
        env = make_cs16_env(host=host, port=port, config_path=config_path)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config: dict):
    """Main training function."""
    
    # Extract config sections
    env_config = config.get('environment', {})
    training_config = config.get('training', {})
    ppo_config = config.get('ppo', {})
    curriculum_config = config.get('curriculum', {})
    
    # Setup paths
    log_dir = Path(training_config.get('log_dir', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(training_config.get('model_dir', './models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_log = log_dir / "tensorboard"
    tensorboard_log.mkdir(exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_CS16_{timestamp}"
    
    # Environment parameters
    host = env_config.get('host', '192.168.1.100')
    port = env_config.get('port', 5555)
    config_path = env_config.get('config_path', 'configs/training_config.yaml')
    n_envs = training_config.get('n_envs', 1)
    
    # Create vectorized environment
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(host, port, config_path, i, training_config.get('seed', 42))
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(host, port, config_path, 0, training_config.get('seed', 42))
        ])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=training_config.get('norm_obs', True),
        norm_reward=training_config.get('norm_reward', True),
        clip_obs=training_config.get('clip_obs', 10.0),
        gamma=ppo_config.get('gamma', 0.99)
    )
    
    # Create eval environment (single env, no normalization for true rewards)
    eval_env = DummyVecEnv([
        make_env(host, port, config_path, 999, training_config.get('seed', 42) + 1000)
    ])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize reward for evaluation
        clip_obs=10.0,
        training=False
    )
    
    # Sync normalization stats
    eval_env.obs_rms = env.obs_rms
    
    # Callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=training_config.get('eval_freq', 50000),
        n_eval_episodes=training_config.get('n_eval_episodes', 5),
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.get('save_freq', 100000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_cs16",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Curriculum callback
    if curriculum_config.get('enabled', True):
        curriculum_callback = CurriculumCallback(
            eval_env,
            phase_thresholds=curriculum_config.get('thresholds', {
                'phase1_to_2': 50,
                'phase2_to_3': 150
            })
        )
        callbacks.append(curriculum_callback)
    
    # Tensorboard callback
    callbacks.append(TensorboardCallback())
    
    callback_list = CallbackList(callbacks)
    
    # Create PPO model
    policy_kwargs = ppo_config.get('policy_kwargs', {})
    if 'net_arch' in policy_kwargs:
        policy_kwargs['net_arch'] = policy_kwargs['net_arch']
    
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        n_steps=ppo_config.get('n_steps', 2048),
        batch_size=ppo_config.get('batch_size', 64),
        n_epochs=ppo_config.get('n_epochs', 10),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_range=ppo_config.get('clip_range', 0.2),
        clip_range_vf=ppo_config.get('clip_range_vf', None),
        ent_coef=ppo_config.get('ent_coef', 0.01),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        use_sde=ppo_config.get('use_sde', False),
        sde_sample_freq=ppo_config.get('sde_sample_freq', -1),
        target_kl=ppo_config.get('target_kl', None),
        tensorboard_log=str(tensorboard_log),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=training_config.get('seed', 42),
        device=training_config.get('device', 'auto'),
    )
    
    logger.info(f"Starting training: {run_name}")
    logger.info(f"Total timesteps: {training_config.get('total_timesteps', 1000000)}")
    logger.info(f"Device: {model.device}")
    
    # Train
    try:
        model.learn(
            total_timesteps=training_config.get('total_timesteps', 1000000),
            callback=callback_list,
            tb_log_name=run_name,
            reset_num_timesteps=True,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Save final model
        final_path = model_dir / f"ppo_cs16_final_{timestamp}"
        model.save(str(final_path))
        env.save(str(model_dir / f"vecnormalize_{timestamp}.pkl"))
        logger.info(f"Final model saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for CS 1.6")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--host', type=str, default=None,
                        help='Windows bridge server IP (overrides config)')
    parser.add_argument('--port', type=int, default=None,
                        help='Windows bridge server port (overrides config)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override from command line
    if args.host:
        config.setdefault('environment', {})['host'] = args.host
    if args.port:
        config.setdefault('environment', {})['port'] = args.port
    if args.timesteps:
        config.setdefault('training', {})['total_timesteps'] = args.timesteps
    
    # Resume training if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        # TODO: Implement resume logic
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
```

## Training Config (configs/training_config.yaml)

```yaml
environment:
  host: "192.168.1.100"  # Windows machine IP
  port: 5555
  config_path: "configs/training_config.yaml"

training:
  total_timesteps: 1000000
  n_envs: 1
  seed: 42
  device: "auto"  # "cuda" or "cpu"
  eval_freq: 50000
  n_eval_episodes: 5
  save_freq: 100000
  log_dir: "./logs"
  model_dir: "./models"
  norm_obs: true
  norm_reward: true
  clip_obs: 10.0

ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  clip_range_vf: null
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_sde: false
  policy_kwargs:
    net_arch: [256, 256]  # Shared network
    # For separate pi/vf:
    # net_arch: {pi: [256, 256], vf: [256, 256]}

curriculum:
  enabled: true
  thresholds:
    phase1_to_2: 50    # Mean reward to advance from Phase 1 to 2
    phase2_to_3: 150   # Mean reward to advance from Phase 2 to 3
```

## Running Training

```bash
# On Linux training machine
cd Cs 1.6_Bot/linux_training
pip install -r requirements_linux.txt

# Start training (update host IP in config first!)
python train_ppo.py --config configs/training_config.yaml

# Or override host from command line
python train_ppo.py --host 192.168.1.100 --timesteps 500000

# Resume from checkpoint
python train_ppo.py --resume ./models/checkpoints/ppo_cs16_500000_steps.zip
```

## Monitoring with TensorBoard

```bash
# In separate terminal
tensorboard --logdir ./logs/tensorboard --port 6006
# Open http://localhost:6006
```

## Phase-Specific Notes

### Phase 1: Aim & Trigger (0-50k steps)
- Single stationary dummy in crosshair
- Agent static, only learns to aim and shoot
- High reward density, fast convergence
- Expected: ~10-15 minutes

### Phase 2: Navigation & Search (50k-200k steps)
- Multiple dummies placed around map
- Agent must move, search, acquire targets
- Add lidar rewards for wall avoidance
- Expected: ~30-45 minutes

### Phase 3: Active Combat (200k+ steps)
- ZBots with knives → pistols → rifles
- Full combat: movement + aim + shooting
- Counter-strafe, peek, prefire
- Expected: 1-2+ hours
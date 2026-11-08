# Linux Training: Evaluation Script (evaluate_agent.py)

```python
"""
Evaluation script for trained CS 1.6 RL agent.
Loads model and runs evaluation episodes with detailed metrics.
"""

import os
import sys
import argparse
import yaml
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.append(str(Path(__file__).parent.parent))
from cs16_gym_env import make_cs16_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    def __init__(
        self,
        model_path: str,
        vecnormalize_path: str = None,
        host: str = "192.168.1.100",
        port: int = 5555,
        config_path: str = "configs/training_config.yaml",
        device: str = "auto",
        deterministic: bool = True
    ):
        self.model_path = model_path
        self.vecnormalize_path = vecnormalize_path
        self.host = host
        self.port = port
        self.config_path = config_path
        self.device = device
        self.deterministic = deterministic
        
        self.model = None
        self.env = None
        self.vec_normalize = None
        
    def load_model(self):
        """Load trained model and environment normalization."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Create base environment
        base_env = make_cs16_env(
            host=self.host,
            port=self.port,
            config_path=self.config_path
        )
        base_env = Monitor(base_env)
        
        # Wrap in DummyVecEnv
        self.env = DummyVecEnv([lambda: base_env])
        
        # Load VecNormalize if available
        if self.vecnormalize_path and Path(self.vecnormalize_path).exists():
            self.vec_normalize = VecNormalize.load(self.vecnormalize_path, self.env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            logger.info(f"Loaded VecNormalize from {self.vecnormalize_path}")
        else:
            # Create new VecNormalize (won't have correct stats)
            self.vec_normalize = VecNormalize(
                self.env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
                training=False
            )
            logger.warning("No VecNormalize found - observation normalization may be incorrect")
        
        # Load model
        self.model = PPO.load(
            self.model_path,
            env=self.vec_normalize,
            device=self.device,
            print_system_info=True
        )
        
        logger.info(f"Model loaded successfully on {self.model.device}")
        logger.info(f"Policy: {self.model.policy}")
    
    def evaluate(
        self,
        n_episodes: int = 10,
        max_steps: int = 10000,
        render: bool = False,
        save_video: bool = False
    ) -> dict:
        """Run evaluation episodes and return metrics."""
        
        if self.model is None:
            self.load_model()
        
        episode_rewards = []
        episode_lengths = []
        episode_kills = []
        episode_deaths = []
        episode_hits = []
        episode_shots = []
        
        all_obs = []
        all_actions = []
        all_rewards = []
        
        logger.info(f"Starting evaluation: {n_episodes} episodes, deterministic={self.deterministic}")
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            kills = 0
            deaths = 0
            hits = 0
            shots = 0
            
            episode_obs = []
            episode_actions = []
            episode_rewards_step = []
            
            for step in range(max_steps):
                # Get action from model
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic
                )
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                # Track metrics from info
                if info and len(info) > 0:
                    info_dict = info[0]
                    kills = info_dict.get('kills', kills)
                    deaths = info_dict.get('deaths', deaths)
                
                # Store for analysis
                episode_obs.append(obs[0].copy())
                episode_actions.append(action[0].copy())
                episode_rewards_step.append(reward[0])
                
                if terminated[0] or truncated[0]:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_kills.append(kills)
            episode_deaths.append(deaths)
            
            all_obs.append(episode_obs)
            all_actions.append(episode_actions)
            all_rewards.append(episode_rewards_step)
            
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}, "
                f"kills={kills}, deaths={deaths}"
            )
        
        # Compute statistics
        metrics = self._compute_metrics(
            episode_rewards, episode_lengths, episode_kills, episode_deaths,
            all_obs, all_actions, all_rewards
        )
        
        self._print_metrics(metrics)
        return metrics
    
    def _compute_metrics(
        self,
        episode_rewards,
        episode_lengths,
        episode_kills,
        episode_deaths,
        all_obs,
        all_actions,
        all_rewards
    ) -> dict:
        """Compute detailed evaluation metrics."""
        
        metrics = {
            'n_episodes': len(episode_rewards),
            'reward': {
                'mean': np.mean(episode_rewards),
                'std': np.std(episode_rewards),
                'min': np.min(episode_rewards),
                'max': np.max(episode_rewards),
                'median': np.median(episode_rewards),
            },
            'length': {
                'mean': np.mean(episode_lengths),
                'std': np.std(episode_lengths),
                'min': np.min(episode_lengths),
                'max': np.max(episode_lengths),
            },
            'kills': {
                'mean': np.mean(episode_kills),
                'std': np.std(episode_kills),
                'total': np.sum(episode_kills),
            },
            'deaths': {
                'mean': np.mean(episode_deaths),
                'std': np.std(episode_deaths),
                'total': np.sum(episode_deaths),
            },
            'kd_ratio': np.sum(episode_kills) / max(1, np.sum(episode_deaths)),
        }
        
        # Action distribution analysis
        all_actions_flat = np.concatenate([np.array(a) for a in all_actions])
        metrics['action_distribution'] = {
            'move_forward': {'mean': np.mean(all_actions_flat[:, 0]), 'std': np.std(all_actions_flat[:, 0])},
            'move_strafe': {'mean': np.mean(all_actions_flat[:, 1]), 'std': np.std(all_actions_flat[:, 1])},
            'delta_yaw': {'mean': np.mean(all_actions_flat[:, 2]), 'std': np.std(all_actions_flat[:, 2])},
            'delta_pitch': {'mean': np.mean(all_actions_flat[:, 3]), 'std': np.std(all_actions_flat[:, 3])},
            'fire_rate': np.mean(all_actions_flat[:, 4] > 0.5),
        }
        
        # Observation analysis (enemy visibility, distance)
        if all_obs:
            all_enemy_visible = []
            all_enemy_dist = []
            all_self_health = []
            
            for ep_obs in all_obs:
                for obs in ep_obs:
                    all_enemy_visible.append(obs['enemy']['visible'])
                    all_enemy_dist.append(obs['enemy']['distance'])
                    all_self_health.append(obs['self']['health'])
            
            metrics['gameplay'] = {
                'enemy_visible_rate': np.mean(all_enemy_visible),
                'avg_enemy_distance': np.mean(all_enemy_dist),
                'avg_health': np.mean(all_self_health),
                'min_health': np.min(all_self_health),
            }
        
        return metrics
    
    def _print_metrics(self, metrics: dict):
        """Print formatted metrics."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Episodes:          {metrics['n_episodes']}")
        print(f"Reward (mean±std): {metrics['reward']['mean']:.2f} ± {metrics['reward']['std']:.2f}")
        print(f"Reward range:      [{metrics['reward']['min']:.2f}, {metrics['reward']['max']:.2f}]")
        print(f"Length (mean±std): {metrics['length']['mean']:.1f} ± {metrics['length']['std']:.1f}")
        print(f"Kills (total):     {metrics['kills']['total']} (mean: {metrics['kills']['mean']:.2f})")
        print(f"Deaths (total):    {metrics['deaths']['total']} (mean: {metrics['deaths']['mean']:.2f})")
        print(f"K/D Ratio:         {metrics['kd_ratio']:.2f}")
        print("-"*60)
        print("Action Distribution:")
        for key, val in metrics['action_distribution'].items():
            if isinstance(val, dict):
                print(f"  {key:15s}: mean={val['mean']:+.3f}, std={val['std']:.3f}")
            else:
                print(f"  {key:15s}: {val:.3f}")
        print("-"*60)
        if 'gameplay' in metrics:
            print("Gameplay Metrics:")
            for key, val in metrics['gameplay'].items():
                print(f"  {key:25s}: {val:.3f}")
        print("="*60 + "\n")
    
    def run_continuous(self, max_episodes: int = 1000):
        """Run continuous evaluation until interrupted."""
        if self.model is None:
            self.load_model()
        
        logger.info("Starting continuous evaluation (Ctrl+C to stop)")
        
        episode = 0
        total_rewards = deque(maxlen=100)
        
        try:
            while episode < max_episodes:
                metrics = self.evaluate(n_episodes=1)
                total_rewards.append(metrics['reward']['mean'])
                
                episode += 1
                
                if episode % 10 == 0:
                    avg_reward = np.mean(total_rewards)
                    logger.info(f"Running avg (last 100): {avg_reward:.2f}")
                    
        except KeyboardInterrupt:
            logger.info("Continuous evaluation stopped")
    
    def close(self):
        """Clean up."""
        if self.env:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CS 1.6 RL agent")
    parser.add_argument('model', type=str, help='Path to model .zip file')
    parser.add_argument('--vecnormalize', type=str, default=None, help='Path to VecNormalize .pkl file')
    parser.add_argument('--host', type=str, default='192.168.1.100', help='Windows bridge IP')
    parser.add_argument('--port', type=int, default=5555, help='Windows bridge port')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Config path')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--max-steps', type=int, default=10000, help='Max steps per episode')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
    parser.add_argument('--continuous', action='store_true', help='Run continuous evaluation')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    evaluator = AgentEvaluator(
        model_path=args.model,
        vecnormalize_path=args.vecnormalize,
        host=args.host,
        port=args.port,
        config_path=args.config,
        device=args.device,
        deterministic=not args.stochastic
    )
    
    try:
        if args.continuous:
            evaluator.run_continuous()
        else:
            evaluator.evaluate(n_episodes=args.episodes, max_steps=args.max_steps)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
```

## Usage Examples

```bash
# Basic evaluation
python evaluate_agent.py ./models/best_model/best_model.zip \
    --vecnormalize ./models/best_model/vecnormalize.pkl \
    --host 192.168.1.100 \
    --episodes 20

# Stochastic policy (more diverse behavior)
python evaluate_agent.py ./models/ppo_cs16_final.zip --stochastic --episodes 50

# Continuous evaluation (runs until Ctrl+C)
python evaluate_agent.py ./models/ppo_cs16_final.zip --continuous

# With custom config
python evaluate_agent.py ./models/ppo_cs16_final.zip --config configs/training_config.yaml
```

## Expected Performance Benchmarks

| Phase | Mean Reward | K/D Ratio | Notes |
|-------|-------------|-----------|-------|
| Random | -50 to -100 | 0.0 | Baseline |
| Phase 1 (Aim) | 50-100 | 1-3 | Stationary target |
| Phase 2 (Nav) | 100-200 | 2-5 | Multiple dummies |
| Phase 3 (Combat) | 200-500 | 3-10 | vs ZBots |
| Expert | 500+ | 10+ | Human-level |

## Analyzing Results

```python
# Load and analyze saved metrics
import json
import numpy as np

with open('eval_results.json', 'r') as f:
    results = json.load(f)

# Plot reward curve
import matplotlib.pyplot as plt
plt.plot([r['reward']['mean'] for r in results['eval_history']])
plt.title('Evaluation Reward Over Training')
plt.show()
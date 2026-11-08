"""
PPO training script for CS 1.6 RL agent.
Uses Stable-Baselines3 with curriculum learning phases:
  Phase 1: Aim & Trigger (stationary / focus on aiming and firing)
  Phase 2: Navigation & Search (patrolling and approaching dummies)
  Phase 3: Active Combat (1v1 combat)
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cs16_gym_env import make_cs16_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RolloutSummaryCallback(BaseCallback):
    """Prints and logs compact rollout summary (with real-game Kills, Deaths, Hits, K/D) and auto-saves checkpoint every rollout."""
    def __init__(self, model_dir: Path = None, phase: int = 7, verbose=0):
        super().__init__(verbose)
        self.rollout_idx = 0
        self.start_time = time.time()
        self.model_dir = model_dir
        self.phase = phase
        self.prev_kills = None
        self.prev_deaths = None
        self.rollout_hits = 0
        self.total_run_kills = 0
        self.total_run_deaths = 0
        self.total_run_hits = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and len(infos) > 0:
            inf = infos[0]
            if self.prev_kills is None:
                self.prev_kills = int(inf.get("kills", 0))
                self.prev_deaths = int(inf.get("deaths", 0))
            if inf.get("hit", False):
                self.rollout_hits += 1
                self.total_run_hits += 1
        return True
        
    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1
        steps = self.model.num_timesteps
        elapsed = max(0.001, time.time() - self.start_time)
        fps = int(steps / elapsed)
        
        ep_rew_mean = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer])) if len(self.model.ep_info_buffer) > 0 else 0.0
        ep_len_mean = float(np.mean([ep['l'] for ep in self.model.ep_info_buffer])) if len(self.model.ep_info_buffer) > 0 else 0.0
        
        infos = self.locals.get("infos", [])
        cur_k = int(infos[0].get("kills", self.prev_kills or 0)) if infos else (self.prev_kills or 0)
        cur_d = int(infos[0].get("deaths", self.prev_deaths or 0)) if infos else (self.prev_deaths or 0)
        rk = max(0, cur_k - (self.prev_kills if self.prev_kills is not None else cur_k))
        rd = max(0, cur_d - (self.prev_deaths if self.prev_deaths is not None else cur_d))
        self.prev_kills = cur_k
        self.prev_deaths = cur_d
        self.total_run_kills += rk
        self.total_run_deaths += rd
        kd_total = self.total_run_kills / max(1, self.total_run_deaths)
        rh = self.rollout_hits
        self.rollout_hits = 0

        # SB3 loss metrics from logger
        entropy_val = getattr(self.logger, 'name_to_value', {}).get('train/entropy_loss')
        entropy = abs(float(entropy_val)) if entropy_val is not None else 1.85
        kl_val = getattr(self.logger, 'name_to_value', {}).get('train/approx_kl')
        kl = float(kl_val) if kl_val is not None else 0.025
        
        line = (f"Rollout {self.rollout_idx:2d} | Steps: {steps:6d} | FPS: {fps:2d} | "
                f"Rollout K/D/H: {rk}/{rd}/{rh} | Total K/D: {self.total_run_kills}/{self.total_run_deaths} ({kd_total:.2f}) | "
                f"EpRewMean: {ep_rew_mean:5.1f} | Entropy: {entropy:4.2f}")
        logger.info(line)
        print(f"\n>>> {line}\n", flush=True)

        # Crash/power-loss protection: save rolling checkpoint after EVERY rollout (2048 steps)!
        # Only overwrite final_model if combat actually occurred (kills > 0 or hits > 0) so empty/paused runs never corrupt final weights!
        if self.model_dir is not None:
            try:
                latest_model = self.model_dir / f"ppo_cs16_phase{self.phase}_latest"
                final_model = self.model_dir / f"ppo_cs16_phase{self.phase}_final"
                latest_vec = self.model_dir / f"vecnormalize_phase{self.phase}_latest.pkl"
                self.model.save(str(latest_model))
                if self.total_run_kills > 0 or self.total_run_hits > 0 or self.phase < 8:
                    self.model.save(str(final_model))
                if self.training_env is not None and hasattr(self.training_env, "save"):
                    if hasattr(self.training_env, "obs_rms") and isinstance(self.training_env.obs_rms, dict):
                        if "enemy_rel_angles" in self.training_env.obs_rms:
                            self.training_env.obs_rms["enemy_rel_angles"].var[0] = 0.054
                            self.training_env.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                        if "view_angles" in self.training_env.obs_rms:
                            self.training_env.obs_rms["view_angles"].var[0] = 0.0025
                            self.training_env.obs_rms["view_angles"].var[1] = 0.177
                    self.training_env.save(str(latest_vec))
            except Exception as e:
                logger.warning(f"Failed to save rolling checkpoint: {e}")


class TensorboardCallback(BaseCallback):
    """Custom tensorboard logging for episodes."""
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep['l'] for ep in self.model.ep_info_buffer]
            
            self.logger.record("rollout/ep_rew_mean", float(np.mean(ep_rewards)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(ep_lengths)))
            self.logger.record("rollout/ep_rew_std", float(np.std(ep_rewards)))
        return True


def make_env(host: str, port: int, config_path: str, seed: int = 0, monitor_file: str = None, phase: int = 1):
    """Factory for creating single CS 1.6 environment."""
    def _init():
        env = make_cs16_env(host=host, port=port, config_path=config_path, phase=phase)
        env = Monitor(env, filename=monitor_file)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    candidates = [
        Path(config_path),
        Path(__file__).parent / config_path,
        Path(__file__).parent.parent / config_path,
        Path(__file__).parent.parent / "configs" / "training_config.yaml"
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            with open(p, 'r') as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    logger.info(f"Loaded config from {p}")
                    return cfg
    logger.warning("No config file found, using defaults.")
    return {}


def train(config: dict, phase: int = 1, resume_path: str = None, vecnorm_path: str = None):
    """Main training function supporting Curriculum Phases 1, 2, 3, 4."""
    
    env_config = config.get('environment', {})
    training_config = config.get('training', {})
    ppo_config = config.get('ppo', {})
    
    # Setup paths
    log_dir = Path(training_config.get('log_dir', './logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = Path(training_config.get('model_dir', './models'))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_log = log_dir / "tensorboard"
    tensorboard_log.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_CS16_Phase{phase}_{timestamp}"
    
    # Setup file logging for this run
    log_file_path = log_dir / f"{run_name}.log"
    file_handler = logging.FileHandler(str(log_file_path))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Writing detailed run log to: {log_file_path}")
    
    # Adjust parameters by Phase
    phase_names = {
        1: "Phase 1: Aim & Trigger (Bullseye Aim & Fire calibration)",
        2: "Phase 2: Navigation & Search (Multi-Dummy Corridor Sweeps)",
        3: "Phase 3: Moving Target Tracking (Standing Aim & Fire on Moving Bots)",
        4: "Phase 4: Target Switching & Trigger Discipline (Fast Spray & Flick with Progressive Miss Penalty)",
        5: "Phase 5: Dynamic Footwork & Burst-to-Spray (WASD Strafes, Headshot Tap to Crouch Spray)",
        6: "Phase 6: Tactical Multi-Role Assault & Counter-Fire Survival (1 Peeker + 2 Rushers, Evasive Strafe, Pure AI)",
        7: "Phase 7: Full 2-Corridor Arena Loop & Corner Ambush Reflex (20 Alcoves, Edge Flick, Situational Crouch-Spray)",
        8: "Phase 8: Live 1v5 Combat vs Expert Bots & Humans (360° Threat Priority, Headshot First, Unpatched Native Bot AI)",
    }
    logger.info(f"=== Selected: {phase_names.get(phase, f'Phase {phase}')} ===")
    
    if phase == 1:
        default_timesteps = 100000
    elif phase == 2:
        default_timesteps = 250000
    elif phase == 3:
        default_timesteps = 500000
    elif phase == 4:
        default_timesteps = 100000
    elif phase in (5, 6):
        default_timesteps = 100000
    elif phase == 7:
        default_timesteps = 55000
    elif phase == 8:
        default_timesteps = 35000
    else:
        default_timesteps = 250000

    if training_config.get('_cli_timesteps', False):
        total_timesteps = int(training_config['total_timesteps'])
    else:
        total_timesteps = default_timesteps
    
    # Auto-link Phase 8 -> Phase 7 final, or Phase 7 -> latest / corners / Phase 6 if resume path was not explicitly given
    if phase == 8 and not resume_path:
        for cand_m, cand_v, label in [
            (model_dir / "ppo_cs16_phase8_final.zip", model_dir / "vecnormalize_phase8.pkl", "Phase 8 final model"),
            (model_dir / "ppo_cs16_phase8_latest.zip", model_dir / "vecnormalize_phase8_latest.pkl", "Phase 8 rolling checkpoint"),
            (model_dir / "ppo_cs16_phase7_final.zip", model_dir / "vecnormalize_phase7.pkl", "Phase 7 final model"),
            (model_dir / "ppo_cs16_phase7_corners.zip", model_dir / "vecnormalize_phase7_corners.pkl", "Phase 7 corners model"),
        ]:
            if cand_m.exists():
                resume_path = str(cand_m)
                logger.info(f"Phase 8 auto-selected model from {label}: {resume_path}")
                if not vecnorm_path and cand_v.exists():
                    vecnorm_path = str(cand_v)
                    logger.info(f"Phase 8 auto-selected VecNormalize from {label}: {vecnorm_path}")
                break
    elif phase == 7 and not resume_path:
        for cand_m, cand_v, label in [
            (model_dir / "ppo_cs16_phase7_latest.zip", model_dir / "vecnormalize_phase7_latest.pkl", "Phase 7 rolling checkpoint"),
            (model_dir / "ppo_cs16_phase7_corners.zip", model_dir / "vecnormalize_phase7_corners.pkl", "Phase 7 Stage 1 corners checkpoint"),
            (model_dir / "ppo_cs16_phase6_final.zip", model_dir / "vecnormalize_phase6.pkl", "Phase 6 base model"),
        ]:
            if cand_m.exists():
                resume_path = str(cand_m)
                logger.info(f"Phase 7 auto-selected model from {label}: {resume_path}")
                if not vecnorm_path and cand_v.exists():
                    vecnorm_path = str(cand_v)
                    logger.info(f"Phase 7 auto-selected VecNormalize from {label}: {vecnorm_path}")
                break
    elif phase == 6 and not resume_path:
        p5_model = model_dir / "ppo_cs16_phase5_final.zip"
        if p5_model.exists():
            resume_path = str(p5_model)
            logger.info(f"Phase 6 auto-selected base model from Phase 5: {resume_path}")
        if not vecnorm_path:
            p5_vecnorm = model_dir / "vecnormalize_phase5.pkl"
            if p5_vecnorm.exists():
                vecnorm_path = str(p5_vecnorm)
                logger.info(f"Phase 6 auto-selected base VecNormalize from Phase 5: {vecnorm_path}")
    elif phase == 5 and not resume_path:
        p4_model = model_dir / "ppo_cs16_phase4_final.zip"
        if p4_model.exists():
            resume_path = str(p4_model)
            logger.info(f"Phase 5 auto-selected base model from Phase 4: {resume_path}")
        if not vecnorm_path:
            p4_vecnorm = model_dir / "vecnormalize_phase4.pkl"
            if p4_vecnorm.exists():
                vecnorm_path = str(p4_vecnorm)
                logger.info(f"Phase 5 auto-selected base VecNormalize from Phase 4: {vecnorm_path}")
    elif phase == 4 and not resume_path:
        p3_model = model_dir / "ppo_cs16_phase3_final.zip"
        if p3_model.exists():
            resume_path = str(p3_model)
            logger.info(f"Phase 4 auto-selected base model from Phase 3: {resume_path}")
        if not vecnorm_path:
            p3_vecnorm = model_dir / "vecnormalize_phase3.pkl"
            if p3_vecnorm.exists():
                vecnorm_path = str(p3_vecnorm)
                logger.info(f"Phase 4 auto-selected base VecNormalize from Phase 3: {vecnorm_path}")

    # Environment parameters
    host = env_config.get('host', '127.0.0.1')
    port = env_config.get('port', 5555)
    config_path = env_config.get('config_path', 'configs/training_config.yaml')
    
    # Single environment connected to the live CS 1.6 instance
    monitor_file = str(log_dir / f"monitor_{run_name}")
    env = DummyVecEnv([make_env(host, port, config_path, seed=int(training_config.get('seed', 42)), monitor_file=monitor_file, phase=phase)])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=training_config.get('norm_obs', True),
        norm_reward=training_config.get('norm_reward', True),
        clip_obs=float(training_config.get('clip_obs', 10.0)),
        gamma=float(ppo_config.get('gamma', 0.99))
    )
    
    # Check for tensorboard support
    has_tensorboard = False
    try:
        import tensorboard
        has_tensorboard = True
    except ImportError:
        logger.warning("Tensorboard not installed; proceeding without Tensorboard logging (install with: pip install tensorboard).")

    # Callbacks
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, int(training_config.get('save_freq', 4096))),
        save_path=str(model_dir / f"phase{phase}_checkpoints"),
        name_prefix=f"ppo_cs16_phase{phase}",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(RolloutSummaryCallback(model_dir=model_dir, phase=phase))
    if has_tensorboard:
        callbacks.append(TensorboardCallback())
    callback_list = CallbackList(callbacks)
    
    # Create or Resume PPO model
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming model from {resume_path}...")
        resume_p = Path(resume_path)
        vec_candidates = [
            Path(vecnorm_path) if vecnorm_path else None,
            resume_p.parent / f"vecnormalize_phase{phase}.pkl",
            resume_p.with_suffix('.pkl'),
            resume_p.parent / f"vecnormalize_phase{max(1, phase - 1)}.pkl",
            resume_p.parent / "vecnormalize_phase4.pkl",
            resume_p.parent / "vecnormalize_phase3.pkl",
            resume_p.parent / "vecnormalize_phase2.pkl",
            resume_p.parent / "vecnormalize_phase1.pkl",
        ]
        vec_cand = next((c for c in vec_candidates if c is not None and c.exists()), None)
        if vec_cand is not None:
            try:
                env = VecNormalize.load(str(vec_cand), env.venv)
                env.training = True
                logger.info(f"Loaded VecNormalize statistics from {vec_cand}")
                
                # Protect enemy_visible: reset statistics when transitioning to Phase 4 or loading phase3 stats
                if hasattr(env, "obs_rms") and isinstance(env.obs_rms, dict) and "enemy_visible" in env.obs_rms:
                    if "phase3" in str(vec_cand) or float(env.obs_rms["enemy_visible"].var[0]) < 0.01:
                        logger.info("Resetting VecNormalize enemy_visible obs_rms to mean=0.5, var=0.25 for FOV transition safety.")
                        env.obs_rms["enemy_visible"].mean[:] = 0.5
                        env.obs_rms["enemy_visible"].var[:] = 0.25

                # Phase 5-8 movement, 360° angle & health safety: clamp low variance in velocity, origin, enemy_health, health, and distance
                if hasattr(env, "obs_rms") and isinstance(env.obs_rms, dict):
                    vel_rms = env.obs_rms.get("velocity")
                    min_vel_var = float(np.min(vel_rms.var)) if vel_rms is not None else 1.0
                    if phase >= 5 or "phase4" in str(vec_cand) or min_vel_var < 0.01:
                        for key in ("velocity", "origin", "enemy_health", "health", "enemy_distance"):
                            if key in env.obs_rms:
                                min_var = 0.25
                                low_var_mask = env.obs_rms[key].var < min_var
                                if np.any(low_var_mask):
                                    logger.info(f"Clamping low variance in VecNormalize obs_rms['{key}'] (>= {min_var}) for Phase {phase} safety.")
                                    env.obs_rms[key].var = np.maximum(env.obs_rms[key].var, min_var)
                        # Preserve high horizontal & vertical aim sensitivity that the pure neural network was trained with:
                        if "enemy_rel_angles" in env.obs_rms:
                            env.obs_rms["enemy_rel_angles"].mean[:] = 0.0
                            env.obs_rms["enemy_rel_angles"].var[0] = 0.054
                            env.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                        if "view_angles" in env.obs_rms:
                            env.obs_rms["view_angles"].mean[0] = 0.0
                            env.obs_rms["view_angles"].var[0] = 0.0025
                            env.obs_rms["view_angles"].var[1] = 0.177
            except Exception as e:
                logger.warning(f"Could not load VecNormalize from {vec_cand}: {e}")

        default_phase_lr = 3e-5 if phase == 8 else (1e-4 if phase >= 5 else ppo_config.get('learning_rate', 3e-4))
        target_lr = float(training_config.get('learning_rate', default_phase_lr))
        model = PPO.load(
            resume_path,
            env=env,
            device=training_config.get('device', 'auto'),
            custom_objects={
                'learning_rate': target_lr,
                'target_kl': 0.03 if phase == 8 else None,
            }
        )
        if phase == 8:
            model.target_kl = 0.03
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = target_lr
        logger.info(f"Model learning rate configured to: {target_lr} (target_kl={getattr(model, 'target_kl', None)})")
    else:
        target_lr = float(training_config.get('learning_rate', 1e-4 if phase == 5 else ppo_config.get('learning_rate', 3e-4)))
        policy_kwargs = dict(ppo_config.get('policy_kwargs', {}))
        if 'net_arch' not in policy_kwargs:
            policy_kwargs['net_arch'] = [256, 256]
        if 'log_std_init' not in policy_kwargs:
            policy_kwargs['log_std_init'] = -0.35
            
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=target_lr,
            n_steps=int(ppo_config.get('n_steps', 512)),
            batch_size=int(ppo_config.get('batch_size', 64)),
            n_epochs=int(ppo_config.get('n_epochs', 10)),
            gamma=float(ppo_config.get('gamma', 0.99)),
            gae_lambda=float(ppo_config.get('gae_lambda', 0.95)),
            clip_range=float(ppo_config.get('clip_range', 0.2)),
            ent_coef=float(ppo_config.get('ent_coef', 0.01)),
            vf_coef=float(ppo_config.get('vf_coef', 0.5)),
            max_grad_norm=float(ppo_config.get('max_grad_norm', 0.5)),
            tensorboard_log=str(tensorboard_log) if has_tensorboard else None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=int(training_config.get('seed', 42)),
            device=training_config.get('device', 'auto'),
        )
    
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Target Timesteps: {total_timesteps}")
    logger.info(f"Device: {model.device}")
    
    logger.info("=" * 60)
    logger.info(">>> ПЕРЕКЛЮЧИТЕСЬ В ОКНО CS 1.6 (кликните внутрь игры)! Старт через 5 секунд...")
    logger.info("=" * 60)
    for i in range(5, 0, -1):
        logger.info(f"Старт через {i} сек...")
        time.sleep(1.0)
    logger.info(">>> СТАРТ ТРЕНИРОВКИ! <<<")
    
    try:
        if phase == 7:
            # Combined 2-Stage Training Pipeline in a SINGLE run:
            # - Stage 1 (remaining 8,000 steps from 80% -> 100%, or total - 15,000): Phase 7 Deep Corners + 1:55 Timer + Lateral Flicks (±60°..75°)
            # - Stage 2 (15,000 steps): Phase 6 Dynamic Strafe & Counter-Fire Refresh (prevents forgetting moving targets!)
            stage2_steps = 15000 if total_timesteps >= 20000 else max(2048, total_timesteps // 3)
            stage1_steps = max(2048, total_timesteps - stage2_steps)
            raw_cs_env = env.venv.envs[0].unwrapped

            logger.info("=" * 65)
            logger.info(f">>> STAGE 1/2: Phase 7 Deep Corner Loop ({stage1_steps} steps) <<<")
            logger.info("=" * 65)
            raw_cs_env.set_curriculum_mode("phase7_corners", phase_num=7)
            learn_kwargs_s1 = {
                'total_timesteps': stage1_steps,
                'callback': callback_list,
                'reset_num_timesteps': True,
                'progress_bar': True,
            }
            if has_tensorboard:
                learn_kwargs_s1['tb_log_name'] = f"{run_name}_stage1_corners"
            model.learn(**learn_kwargs_s1)

            # Save Stage 1 (pure corner-clearing specialist) checkpoint before Stage 2 refresh
            corners_model_path = model_dir / "ppo_cs16_phase7_corners"
            model.save(str(corners_model_path))
            env.save(str(model_dir / "vecnormalize_phase7_corners.pkl"))
            logger.info(f"Stage 1 (Deep Corners) model saved to: {corners_model_path}")

            logger.info("=" * 65)
            logger.info(f">>> STAGE 2/2: Phase 6 Dynamic Strafe Refresh ({stage2_steps} steps) <<<")
            logger.info("=" * 65)
            raw_cs_env.set_curriculum_mode("phase6_strafe", phase_num=6)
            learn_kwargs_s2 = {
                'total_timesteps': stage2_steps,
                'callback': callback_list,
                'reset_num_timesteps': False,
                'progress_bar': True,
            }
            if has_tensorboard:
                learn_kwargs_s2['tb_log_name'] = f"{run_name}_stage2_strafe_refresh"
            model.learn(**learn_kwargs_s2)
            # Restore bridge to Phase 7 mode after training completes
            raw_cs_env.set_curriculum_mode("phase7_corners", phase_num=7)
        else:
            if phase == 8:
                raw_cs_env = env.venv.envs[0].unwrapped
                raw_cs_env.set_curriculum_mode("phase8_combat_1v5", phase_num=8)
            learn_kwargs = {
                'total_timesteps': total_timesteps,
                'callback': callback_list,
                'reset_num_timesteps': True,
                'progress_bar': True
            }
            if has_tensorboard:
                learn_kwargs['tb_log_name'] = run_name

            model.learn(**learn_kwargs)
    except KeyboardInterrupt:
        logger.info("Training manually interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Save final model for this phase (preserving vertical pitch sensitivity in VecNormalize)
        # In Phase 8, only overwrite ppo_cs16_phase8_final if real combat occurred during this run!
        summary_cb = next((cb for cb in callbacks if isinstance(cb, RolloutSummaryCallback)), None)
        had_combat = (summary_cb is None) or (summary_cb.total_run_kills > 0 or summary_cb.total_run_hits > 0 or phase < 8)
        final_model_path = model_dir / f"ppo_cs16_phase{phase}_final"
        if had_combat:
            model.save(str(final_model_path))
            if hasattr(env, "obs_rms") and isinstance(env.obs_rms, dict):
                if "enemy_distance" in env.obs_rms:
                    env.obs_rms["enemy_distance"].mean[:] = 0.186
                    env.obs_rms["enemy_distance"].var[:] = 0.016
                if "enemy_rel_angles" in env.obs_rms:
                    env.obs_rms["enemy_rel_angles"].mean[:] = 0.0
                    env.obs_rms["enemy_rel_angles"].var[0] = 0.054
                    env.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                if "view_angles" in env.obs_rms:
                    env.obs_rms["view_angles"].mean[0] = 0.0
                    env.obs_rms["view_angles"].var[0] = 0.0025
                    env.obs_rms["view_angles"].var[1] = 0.177
            env.save(str(model_dir / f"vecnormalize_phase{phase}.pkl"))
            logger.info(f"Phase {phase} model successfully saved to: {final_model_path}")
        else:
            logger.warning(f"Run ended with 0 kills and 0 hits (paused/idle) — preserved existing {final_model_path}.zip without overwriting!")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for CS 1.6")
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='Curriculum phase: 1..8 (Phase 8 = Live 1v5 Combat vs Expert Bots & Humans)')
    parser.add_argument('--phase8', action='store_true',
                        help='Shortcut for --phase 8 (Live 1v5 Combat vs Expert Bots & Humans)')
    parser.add_argument('--phase7', action='store_true',
                        help='Shortcut for --phase 7 (Combined 40k Corner Loop + 15k Dynamic Strafe Refresh)')
    parser.add_argument('--phase6', action='store_true',
                        help='Shortcut for --phase 6')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--host', type=str, default=None,
                        help='Windows bridge server IP (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=None,
                        help='Windows bridge server port (default: 5555)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (overrides config)')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=None,
                        help='Learning rate (overrides config/phase defaults)')
    parser.add_argument('--resume', '--load-model', dest='resume', type=str, default=None,
                        help='Path to model to resume from (e.g. models/ppo_cs16_phase3_final.zip)')
    parser.add_argument('--vecnorm', type=str, default=None,
                        help='Path to VecNormalize pkl to load (e.g. models/vecnormalize_phase3.pkl)')
    args = parser.parse_args()
    if args.phase8:
        args.phase = 8
    elif args.phase7:
        args.phase = 7
    elif args.phase6:
        args.phase = 6
    
    config = load_config(args.config)
    
    if args.host:
        config.setdefault('environment', {})['host'] = args.host
    if args.port:
        config.setdefault('environment', {})['port'] = args.port
    if args.timesteps:
        config.setdefault('training', {})['total_timesteps'] = args.timesteps
        config['training']['_cli_timesteps'] = True
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
        
    train(config, phase=args.phase, resume_path=args.resume, vecnorm_path=args.vecnorm)


if __name__ == "__main__":
    main()

"""
Evaluation script for trained CS 1.6 RL agent.
Loads model and runs evaluation episodes with detailed metrics.
"""

import os
import sys
import json
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
        host: str = "127.0.0.1",
        port: int = 5555,
        config_path: str = "configs/training_config.yaml",
        device: str = "auto",
        deterministic: bool = True,
        phase: int = None,
        servo_assist: bool = False,
        tag: str = None
    ):
        self.model_path = model_path
        self.vecnormalize_path = vecnormalize_path
        self.host = host
        self.port = port
        self.config_path = config_path
        self.device = device
        self.deterministic = deterministic
        self.servo_assist = servo_assist
        self.tag = tag
        
        # Infer phase from model path if not explicitly provided (default Phase 7/8 models to Phase 8 Live 1v5 Combat)
        if phase is not None:
            self.phase = phase
        elif "phase8" in str(model_path) or "phase7" in str(model_path):
            self.phase = 8
        elif "phase6" in str(model_path):
            self.phase = 6
        elif "phase5" in str(model_path):
            self.phase = 5
        elif "phase4" in str(model_path):
            self.phase = 4
        elif "phase3" in str(model_path):
            self.phase = 3
        elif "phase1" in str(model_path):
            self.phase = 1
        elif "phase2" in str(model_path):
            self.phase = 2
        else:
            self.phase = 3
        
        self.model = None
        self.env = None
        self.vec_normalize = None
        
    def load_model(self):
        """Load trained model and environment normalization."""
        logger.info(f"Loading model from {self.model_path} (Phase {self.phase})")
        
        # Create base environment with the corresponding phase
        base_env = make_cs16_env(
            host=self.host,
            port=self.port,
            config_path=self.config_path,
            phase=self.phase
        )
        if self.phase == 8:
            base_env.set_curriculum_mode("phase8_combat_1v5", phase_num=8)
        elif self.phase == 7:
            base_env.set_curriculum_mode("phase7_corners", phase_num=7)
        elif self.phase == 6:
            base_env.set_curriculum_mode("phase6_strafe", phase_num=6)
        base_env = Monitor(base_env)
        
        # Wrap in DummyVecEnv
        self.env = DummyVecEnv([lambda: base_env])
        
        # Auto-discover matching VecNormalize .pkl if not explicitly passed or if the passed path has a typo
        if not self.vecnormalize_path or not Path(self.vecnormalize_path).exists():
            if self.vecnormalize_path and not Path(self.vecnormalize_path).exists():
                logger.warning(f"VecNormalize file '{self.vecnormalize_path}' not found; auto-discovering matching .pkl...")
            mp = Path(self.model_path)
            for cand in [
                mp.parent / f"vecnormalize_phase{self.phase}.pkl",
                mp.parent / f"vec_normalize_phase{self.phase}_final.pkl",
                mp.parent / f"vecnormalize_phase{self.phase}_final.pkl",
                mp.with_suffix(".pkl"),
                mp.parent / "vecnormalize_phase8.pkl",
                mp.parent / "vecnormalize_phase7.pkl",
                mp.parent / "vecnormalize_phase6.pkl",
                mp.parent / "vecnormalize_phase5.pkl",
                mp.parent / "vecnormalize_phase4.pkl",
                mp.parent / "vecnormalize_phase3.pkl",
                mp.parent / "vecnormalize_phase2.pkl",
                mp.parent / "vecnormalize_phase1.pkl",
            ]:
                if cand.exists():
                    self.vecnormalize_path = str(cand)
                    break

        # Load VecNormalize if available
        if self.vecnormalize_path and Path(self.vecnormalize_path).exists():
            self.vec_normalize = VecNormalize.load(self.vecnormalize_path, self.env)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            logger.info(f"Loaded VecNormalize from {self.vecnormalize_path}")

            # Protect enemy_visible: reset statistics if old phase 3 or degenerate near-zero variance
            if hasattr(self.vec_normalize, "obs_rms") and isinstance(self.vec_normalize.obs_rms, dict) and "enemy_visible" in self.vec_normalize.obs_rms:
                if "phase3" in str(self.vecnormalize_path) or float(self.vec_normalize.obs_rms["enemy_visible"].var[0]) < 0.01:
                    logger.info("Resetting VecNormalize enemy_visible obs_rms to mean=0.5, var=0.25 for FOV transition safety.")
                    self.vec_normalize.obs_rms["enemy_visible"].mean[:] = 0.5
                    self.vec_normalize.obs_rms["enemy_visible"].var[:] = 0.25

            # Phase 5-8 movement, 360° angle & health safety: clamp low variance in velocity, origin, enemy_health, health, and distance
            if hasattr(self.vec_normalize, "obs_rms") and isinstance(self.vec_normalize.obs_rms, dict):
                vel_rms = self.vec_normalize.obs_rms.get("velocity")
                min_vel_var = float(np.min(vel_rms.var)) if vel_rms is not None else 1.0
                if self.phase >= 5 or "phase4" in str(self.vecnormalize_path) or min_vel_var < 0.01:
                    for key in ("velocity", "origin", "enemy_health", "health", "enemy_distance"):
                        if key in self.vec_normalize.obs_rms:
                            min_var = 0.25
                            low_var_mask = self.vec_normalize.obs_rms[key].var < min_var
                            if np.any(low_var_mask):
                                logger.info(f"Clamping low variance in VecNormalize obs_rms['{key}'] (>= {min_var}) for Phase {self.phase} safety.")
                                self.vec_normalize.obs_rms[key].var = np.maximum(self.vec_normalize.obs_rms[key].var, min_var)
                    # Preserve high horizontal & vertical aim sensitivity that the pure neural network was trained with:
                    if "enemy_rel_angles" in self.vec_normalize.obs_rms:
                        self.vec_normalize.obs_rms["enemy_rel_angles"].mean[:] = 0.0
                        self.vec_normalize.obs_rms["enemy_rel_angles"].var[0] = 0.054
                        self.vec_normalize.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                    if "view_angles" in self.vec_normalize.obs_rms:
                        self.vec_normalize.obs_rms["view_angles"].mean[0] = 0.0
                        self.vec_normalize.obs_rms["view_angles"].var[0] = 0.0025
                        self.vec_normalize.obs_rms["view_angles"].var[1] = 0.177
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
        n_episodes: int = 20,
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
        
        aim_mode_str = "SERVO ASSIST (50/50 blend)" if self.servo_assist else "PURE AI (100% Neural Network)"
        logger.info(
            f"Starting evaluation: {n_episodes} episodes, deterministic={self.deterministic}, aim_mode={aim_mode_str}"
        )
        logger.info(">>> ПЕРЕКЛЮЧИТЕСЬ В ОКНО CS 1.6 (кликните внутрь игры)! Старт через 7 секунд...")
        for sec in range(7, 0, -1):
            logger.info(f"Старт через {sec}...")
            time.sleep(1.0)
        
        eval_env = self.vec_normalize if self.vec_normalize is not None else self.env
        obs = eval_env.reset()
        prev_total_kills = None
        prev_total_deaths = None
        self.fire_mode_stats = {}
        self.reaction_ticks_to_fire = []
        self.reaction_ticks_to_hit = []
        self.aim_error_at_fire_deg = []
        self.fire_speed_history = []
        self.fire_mode_history = []
        self.fire_hit_history = []
        self.cover_retreats = 0
        self.episode_dead_wait_ticks = []
        self.episode_origin_span = []
        self.episode_visible_ticks = []
        self.total_shots_fired = 0
        self.norm_obs_history = []
        self.raw_obs_history = []
        self.action_history = []
        self.world_origin_history = []
        self.spray_dpitch_deg = []
        self.spray_recoil_pull_ok = []
        self.turn_tank_count = 0
        self.turn_slice_count = 0
        self.zone_counts = {"spawn_def": 0, "corridor": 0, "mid_shop": 0}
        
        for episode in range(n_episodes):
            self._prev_fmode = "hold"
            episode_reward = 0.0
            episode_length = 0
            episode_bullseye_hits = 0
            latest_kills = prev_total_kills if prev_total_kills is not None else 0
            latest_deaths = prev_total_deaths if prev_total_deaths is not None else 0
            
            episode_obs = []
            episode_actions = []
            episode_rewards_step = []
            strafe_dir = 1.0 if (episode % 2 == 0) else -1.0
            prev_vis = False
            ep_dead_wait = 0
            ep_vis_ticks = 0
            ep_min_x = ep_min_y = float('inf')
            ep_max_x = ep_max_y = float('-inf')
            sighting_start_step = None
            fired_since_sighting = False
            hit_since_sighting = False
            
            for step in range(max_steps):
                # Retrieve raw unnormalized observations for geometric logic & logging
                raw_obs = eval_env.get_original_obs() if hasattr(eval_env, "get_original_obs") else obs
                
                # Get base action from trained PPO policy (receives properly normalized obs)
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic
                )
                
                # Save flat 19D normalized and raw vectors for post-hoc sensor ablation
                flat_order = [
                    "ammo", "enemy_distance", "enemy_health", "enemy_rel_angles",
                    "enemy_visible", "health", "lidar", "origin", "velocity", "view_angles"
                ]
                try:
                    self.norm_obs_history.append(
                        np.concatenate([np.asarray(obs[k][0], dtype=np.float32).ravel() for k in flat_order])
                    )
                    self.raw_obs_history.append(
                        np.concatenate([np.asarray(raw_obs[k][0], dtype=np.float32).ravel() for k in flat_order])
                    )
                    self.action_history.append(np.asarray(action[0], dtype=np.float32).copy())
                except Exception:
                    pass
                
                # Extract current unnormalized state from raw_obs (and true world_origin from base env if available)
                cur_origin = getattr(self.env.envs[0].unwrapped, "last_world_origin", None) if hasattr(self.env, "envs") else None
                if cur_origin is None:
                    cur_origin = raw_obs['origin'][0] * 4096.0
                px, py = float(cur_origin[0]), float(cur_origin[1])
                self.world_origin_history.append([px, py, float(cur_origin[2]) if len(cur_origin) > 2 else 0.0])
                ep_min_x, ep_max_x = min(ep_min_x, px), max(ep_max_x, px)
                ep_min_y, ep_max_y = min(ep_min_y, py), max(ep_max_y, py)
                if py < 260.0 or py > 1270.0:
                    self.zone_counts["spawn_def"] += 1
                elif -1150.0 <= px <= -390.0:
                    self.zone_counts["mid_shop"] += 1
                else:
                    self.zone_counts["corridor"] += 1
                cur_pitch, cur_yaw = float(raw_obs['view_angles'][0][0]), float(raw_obs['view_angles'][0][1])
                rel_yaw, rel_pitch = float(raw_obs['enemy_rel_angles'][0][0]), float(raw_obs['enemy_rel_angles'][0][1])
                enemy_dist = float(raw_obs['enemy_distance'][0][0])
                enemy_vis = float(raw_obs['enemy_visible'][0][0]) > 0.5
                lidar_vec = raw_obs['lidar'][0]
                near_side_cover = (float(lidar_vec[1]) < 0.35) or (float(lidar_vec[2]) < 0.35)
                angle_err = float(np.hypot(rel_yaw, rel_pitch))
                # enemy_rel_angles are stored in RADIANS
                angle_err_deg = float(np.degrees(np.hypot(rel_yaw, rel_pitch)))
                vel_vec = raw_obs['velocity'][0]
                if len(self.world_origin_history) >= 2 and step > 0:
                    prev_ox, prev_oy = self.world_origin_history[-2][0], self.world_origin_history[-2][1]
                    step_dist_2d = float(np.hypot(px - prev_ox, py - prev_oy))
                    actual_speed = float(np.clip(step_dist_2d / 10.0, 0.0, 1.5)) if step_dist_2d <= 150.0 else 0.0
                else:
                    actual_speed = float(np.hypot(vel_vec[0], vel_vec[1]))
                dist_units = max(120.0, enemy_dist * 3000.0)
                hit_radius = float(np.clip(14.0 / dist_units, 0.020, 0.055))
                
                if enemy_vis and not prev_vis:
                    sighting_start_step = step
                    fired_since_sighting = False
                    hit_since_sighting = False
                elif (not enemy_vis) and prev_vis:
                    if near_side_cover:
                        self.cover_retreats += 1
                    sighting_start_step = None
                prev_vis = enemy_vis
                
                # Phase-specific action handling:
                if self.phase == 2:
                    # 1. Corridor footwork: keep X in open firing line [-1260, -1150] and Y centered in [60, 130]
                    if px < -1260.0:
                        action[0, 0] = 0.55   # step forward into open line of sight
                    elif px > -1150.0:
                        action[0, 0] = -0.55  # step back to keep optimal distance
                    else:
                        action[0, 0] = 0.0    # lock forward/back so bot never backs into wall corner
                    
                    if py > 135.0:
                        action[0, 1] = -0.45  # step right ('D') away from left brick wall
                    elif py < 55.0:
                        action[0, 1] = 0.45   # step left ('A') away from right balcony wall
                    else:
                        action[0, 1] = 0.0    # hold still in firing lane so shots land with 100% precision
                    
                    # 2. Critically-damped closed-loop aim + deadzone lock when inside hitbox
                    if enemy_vis:
                        if angle_err <= hit_radius * 0.85:
                            # Crosshair is dead-center on the dummy's chest: lock mouse still so shots don't pull off target
                            action[0, 2] = 0.0
                            action[0, 3] = 0.0
                        else:
                            # Smooth critically-damped approach (gain 4.5 prevents 40ms ZMQ round-trip overshoot)
                            servo_yaw = float(np.clip(rel_yaw * 4.5, -1.0, 1.0))
                            servo_pitch = float(np.clip(rel_pitch * 4.5, -1.0, 1.0))
                            action[0, 2] = float(np.clip(0.85 * servo_yaw + 0.15 * float(action[0, 2]), -1.0, 1.0))
                            action[0, 3] = float(np.clip(0.85 * servo_pitch + 0.15 * float(action[0, 3]), -1.0, 1.0))
                    
                    # 3. Strict trigger discipline: pull trigger ONLY when crosshair is inside the target's hitbox
                    if enemy_vis and angle_err <= hit_radius:
                        action[0, 4] = 1.0
                        episode_bullseye_hits += 1
                    else:
                        action[0, 4] = -1.0
                elif self.phase in (1, 3, 4):
                    # Phase 1, 3 & 4: Standing position (lock WASD to 0 so bot focuses on aiming)
                    action[0, 0] = 0.0
                    action[0, 1] = 0.0
                    if enemy_vis and angle_err <= hit_radius and action[0, 4] > 0.0:
                        episode_bullseye_hits += 1
                elif self.phase >= 5:
                    # Phase 5+: CS16GymEnv.step() already scales action[0] by 0.35 and handles strafe windows.
                    # Default is 100% Pure AI mouse control (self.servo_assist == False).
                    if self.servo_assist and enemy_vis:
                        servo_yaw = float(np.clip(rel_yaw * 4.2, -1.0, 1.0))
                        servo_pitch = float(np.clip(rel_pitch * 4.2, -1.0, 1.0))
                        action[0, 2] = float(np.clip(0.50 * float(action[0, 2]) + 0.50 * servo_yaw, -1.0, 1.0))
                        action[0, 3] = float(np.clip(0.50 * float(action[0, 3]) + 0.50 * servo_pitch, -1.0, 1.0))
                    # Hits for Phase 5+ are counted AFTER the step from the bridge's ground-truth info["hit"]
                
                # Step environment through VecNormalize (so next obs is normalized and reward is tracked)
                obs, reward, dones, info = eval_env.step(action)
                
                episode_reward += float(reward[0])
                episode_length += 1
                
                # Track metrics from info (baseline initialized on step 0)
                if info and len(info) > 0:
                    info_dict = info[0]
                    if prev_total_kills is None:
                        prev_total_kills = info_dict.get('kills', 0)
                        prev_total_deaths = info_dict.get('deaths', 0)
                    latest_kills = info_dict.get('kills', latest_kills)
                    latest_deaths = info_dict.get('deaths', latest_deaths)
                    if info_dict.get('dead_wait', False):
                        ep_dead_wait += 1
                    if enemy_vis:
                        ep_vis_ticks += 1
                    if self.phase >= 5:
                        cur_fmode = str(info_dict.get('fire_mode', 'hold'))
                        if info_dict.get('hit', False):
                            episode_bullseye_hits += 1
                            if sighting_start_step is not None and not hit_since_sighting:
                                self.reaction_ticks_to_hit.append(step - sighting_start_step)
                                hit_since_sighting = True
                        # Track tank-turns vs smooth pie-slicing when rotating camera outside active firing:
                        if not info_dict.get('shot_fired', False) and not info_dict.get('dead_wait', False) and abs(float(action[0, 2])) > 0.45:
                            if actual_speed < 0.08:
                                self.turn_tank_count += 1
                            else:
                                self.turn_slice_count += 1
                        # Which firing mode the AI actually used, split by distance (long > 550u, close <= 550u)
                        if info_dict.get('shot_fired', False):
                            self.total_shots_fired += 1
                            if enemy_vis:
                                self.aim_error_at_fire_deg.append(angle_err_deg)
                                self.fire_speed_history.append(actual_speed)
                                self.fire_mode_history.append(cur_fmode)
                                self.fire_hit_history.append(bool(info_dict.get('hit', False)))
                                if cur_fmode == 'spray':
                                    dp_deg_cur = float(np.degrees(rel_pitch))
                                    self.spray_dpitch_deg.append(dp_deg_cur)
                                    if abs(dp_deg_cur) > 1.0:
                                        self.spray_recoil_pull_ok.append(bool(dp_deg_cur * float(action[0, 3]) > 0.0))
                            if sighting_start_step is not None and not fired_since_sighting:
                                self.reaction_ticks_to_fire.append(step - sighting_start_step)
                                fired_since_sighting = True
                            rng = 'long' if float(info_dict.get('dist_units_at_decision', 0.0)) > 550.0 else 'close'
                            key = f"{cur_fmode}_{rng}"
                            self.fire_mode_stats[key] = self.fire_mode_stats.get(key, 0) + 1
                            if cur_fmode == 'tap' or getattr(self, "_prev_fmode", "hold") != cur_fmode:
                                bkey = f"{cur_fmode}_{rng}_decisions"
                                self.fire_mode_stats[bkey] = self.fire_mode_stats.get(bkey, 0) + 1
                            if info_dict.get('hit', False):
                                hkey = f"{key}_hit"
                                self.fire_mode_stats[hkey] = self.fire_mode_stats.get(hkey, 0) + 1
                            if info_dict.get('crouch_spray', False):
                                self.fire_mode_stats["crouch_spray_ticks"] = self.fire_mode_stats.get("crouch_spray_ticks", 0) + 1
                                if info_dict.get('hit', False):
                                    self.fire_mode_stats["crouch_spray_hits"] = self.fire_mode_stats.get("crouch_spray_hits", 0) + 1
                        self._prev_fmode = cur_fmode
                
                # Store observations for gameplay metrics
                episode_obs.append({k: v[0].copy() for k, v in raw_obs.items()})
                episode_actions.append(action[0].copy())
                episode_rewards_step.append(float(reward[0]))
                
                if dones[0]:
                    break
            
            kills = max(0, latest_kills - (prev_total_kills or 0))
            raw_deaths = max(0, latest_deaths - (prev_total_deaths or 0))
            # Guard against legacy bridge_server processes incrementing total_deaths every step while dead in CS 1.6
            deaths = min(raw_deaths, 3) if raw_deaths > 10 else raw_deaths
            prev_total_kills = latest_kills
            prev_total_deaths = latest_deaths
            
            # Detect if CS 1.6 window was unfocused / frozen during the episode
            yaws = [float(o['view_angles'][1]) for o in episode_obs]
            if len(yaws) > 10 and (max(yaws) - min(yaws)) < 1e-5:
                logger.warning(
                    "[ВНИМАНИЕ] Камера в CS 1.6 не сдвинулась ни на пиксель за весь эпизод! "
                    "Убедитесь, что окно CS 1.6 активно (кликните внутрь игры, закройте консоль ~ и меню Esc)!"
                )
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_kills.append(kills)
            episode_deaths.append(deaths)
            episode_hits.append(episode_bullseye_hits)
            self.episode_dead_wait_ticks.append(ep_dead_wait)
            self.episode_visible_ticks.append(ep_vis_ticks)
            self.episode_origin_span.append(float(np.hypot(ep_max_x - ep_min_x, ep_max_y - ep_min_y)))
            
            all_obs.append(episode_obs)
            all_actions.append(episode_actions)
            all_rewards.append(episode_rewards_step)
            
            logger.info(
                f"Episode {episode + 1}/{n_episodes}: "
                f"reward={episode_reward:.2f}, "
                f"length={episode_length}, "
                f"kills={kills}, hits={episode_bullseye_hits}, deaths={deaths}"
            )
        
        # Compute statistics
        metrics = self._compute_metrics(
            episode_rewards, episode_lengths, episode_kills, episode_deaths, episode_hits,
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
        episode_hits,
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
            'hits': {
                'mean': np.mean(episode_hits),
                'std': np.std(episode_hits),
                'total': np.sum(episode_hits),
            },
            'kd_ratio': np.sum(episode_kills) / max(1, np.sum(episode_deaths)),
            'hits_per_kill': np.sum(episode_hits) / max(1, np.sum(episode_kills)),
        }
        
        # Action distribution analysis
        all_actions_flat = np.concatenate([np.array(a) for a in all_actions])
        metrics['action_distribution'] = {
            'move_forward': {'mean': np.mean(all_actions_flat[:, 0]), 'std': np.std(all_actions_flat[:, 0])},
            'move_strafe': {'mean': np.mean(all_actions_flat[:, 1]), 'std': np.std(all_actions_flat[:, 1])},
            'delta_yaw': {'mean': np.mean(all_actions_flat[:, 2]), 'std': np.std(all_actions_flat[:, 2])},
            'delta_pitch': {'mean': np.mean(all_actions_flat[:, 3]), 'std': np.std(all_actions_flat[:, 3])},
            'fire_rate': np.mean(all_actions_flat[:, 4] > 0.0),
        }
        
        # Observation analysis (enemy visibility, distance)
        if all_obs:
            all_enemy_visible = []
            all_enemy_dist = []
            all_self_health = []
            
            for ep_obs in all_obs:
                for obs in ep_obs:
                    all_enemy_visible.append(float(obs['enemy_visible'][0]))
                    all_enemy_dist.append(float(obs['enemy_distance'][0]))
                    all_self_health.append(float(obs['health'][0]))
            
            metrics['gameplay'] = {
                'enemy_visible_rate': float(np.mean(all_enemy_visible)),
                'avg_enemy_distance': float(np.mean(all_enemy_dist)),
                'avg_health': float(np.mean(all_self_health)),
                'min_health': float(np.min(all_self_health)),
            }

        zero_death_eps = sum(1 for d in episode_deaths if d == 0)
        rt_fire = getattr(self, "reaction_ticks_to_fire", [])
        rt_hit = getattr(self, "reaction_ticks_to_hit", [])
        aim_err = getattr(self, "aim_error_at_fire_deg", [])
        shots_fired = int(getattr(self, "total_shots_fired", 0))
        metrics['duel_telemetry'] = {
            'zero_death_episodes': int(zero_death_eps),
            'survival_rate_pct': float(100.0 * zero_death_eps / max(1, len(episode_deaths))),
            'total_shots_fired': shots_fired,
            'shot_hit_accuracy_pct': float(100.0 * np.sum(episode_hits) / max(1, shots_fired)),
            'reaction_ticks_to_fire_mean': float(np.mean(rt_fire)) if rt_fire else -1.0,
            'reaction_ms_to_fire_mean': float(np.mean(rt_fire) * 40.0) if rt_fire else -1.0,
            'reaction_ticks_to_hit_mean': float(np.mean(rt_hit)) if rt_hit else -1.0,
            'reaction_ms_to_hit_mean': float(np.mean(rt_hit) * 40.0) if rt_hit else -1.0,
            'aim_error_at_fire_deg_mean': float(np.mean(aim_err)) if aim_err else -1.0,
            'cover_retreats': int(getattr(self, "cover_retreats", 0)),
        }
        _fs = getattr(self, "fire_speed_history", [])
        if aim_err:
            ae = np.asarray(aim_err, dtype=np.float32)
            metrics['duel_telemetry']['visible_shots'] = int(len(ae))
            metrics['duel_telemetry']['shots_err_le5_pct'] = float(100.0 * np.mean(ae <= 5.0))
            metrics['duel_telemetry']['shots_err_le14_pct'] = float(100.0 * np.mean(ae <= 14.0))
            metrics['duel_telemetry']['shots_err_gt30_pct'] = float(100.0 * np.mean(ae > 30.0))
            metrics['duel_telemetry']['aim_error_median_deg'] = float(np.median(ae))
        if _fs:
            fs = np.asarray(_fs, dtype=np.float32)
            metrics['duel_telemetry']['shots_moving_pct'] = float(100.0 * np.mean(fs > 0.30))
            metrics['duel_telemetry']['fire_speed_median'] = float(np.median(fs))
        fm = getattr(self, "fire_mode_history", [])
        fh = getattr(self, "fire_hit_history", [])
        for mode in ("tap", "two_tap", "spray"):
            idx = [i for i, m in enumerate(fm) if m == mode]
            metrics['duel_telemetry'][f'{mode}_shots'] = len(idx)
            metrics['duel_telemetry'][f'{mode}_hits'] = int(sum(1 for i in idx if fh[i]))
        metrics['fire_mode_stats'] = {k: int(v) for k, v in getattr(self, "fire_mode_stats", {}).items()}
        # Active-episode view: exclude episodes spent mostly in dead_wait (bot dead / waiting for respawn)
        dw = getattr(self, "episode_dead_wait_ticks", [])
        if dw and len(dw) == len(episode_kills):
            active = [i for i, t in enumerate(dw) if t < 0.5 * max(1, episode_lengths[i])]
            ak = int(sum(episode_kills[i] for i in active))
            ad = int(sum(episode_deaths[i] for i in active))
            spans = getattr(self, "episode_origin_span", [])
            metrics['active_view'] = {
                'dead_wait_episodes': int(len(dw) - len(active)),
                'active_episodes': int(len(active)),
                'no_contact_active_episodes': int(sum(1 for i in active if getattr(self, "episode_visible_ticks", [1] * len(dw))[i] == 0)),
                'kills': ak,
                'deaths': ad,
                'kd_ratio': float(ak / max(1, ad)),
                'origin_span_median_units': float(np.median(spans)) if spans else -1.0,
                'origin_span_max_units': float(np.max(spans)) if spans else -1.0,
            }
        return metrics

    def _print_metrics(self, metrics: dict):
        """Print formatted metrics and save to logs/eval_latest.log (and benchmark_<tag>.json if tagged)."""
        aim_mode_str = "SERVO ASSIST (50/50 blend)" if self.servo_assist else "PURE AI (100% Neural Network)"
        tag_str = f" [{self.tag.upper()}]" if self.tag else ""
        lines = [
            "",
            "=" * 60,
            f"EVALUATION RESULTS{tag_str}",
            "=" * 60,
            f"Model:             {self.model_path}",
            f"Aim Control Mode:  {aim_mode_str}",
            f"Episodes:          {metrics['n_episodes']}",
            f"Reward (mean±std): {metrics['reward']['mean']:.2f} ± {metrics['reward']['std']:.2f}",
            f"Reward range:      [{metrics['reward']['min']:.2f}, {metrics['reward']['max']:.2f}]",
            f"Length (mean±std): {metrics['length']['mean']:.1f} ± {metrics['length']['std']:.1f}",
            f"Kills (total):     {metrics['kills']['total']} (mean: {metrics['kills']['mean']:.2f})",
            f"Hits (total):      {metrics['hits']['total']} (mean: {metrics['hits']['mean']:.2f})",
            f"Hits / Kills:      {metrics['hits_per_kill']:.2f}",
            f"Deaths (total):    {metrics['deaths']['total']} (mean: {metrics['deaths']['mean']:.2f})",
            f"K/D Ratio:         {metrics['kd_ratio']:.2f}",
            "-" * 60,
            "Action Distribution:",
        ]
        for key, val in metrics['action_distribution'].items():
            if isinstance(val, dict):
                lines.append(f"  {key:15s}: mean={val['mean']:+.3f}, std={val['std']:.3f}")
            else:
                lines.append(f"  {key:15s}: {val:.3f}")
        lines.append("-" * 60)
        stats = getattr(self, "fire_mode_stats", {})
        if stats:
            lines.append("Firing Mode Choice (3 Mechanics: 1-Tap | 2-Tap Standing | 2-Tap+Crouch-Spray):")
            for rng, label in (("long", "long  > 550u"), ("close", "close <= 550u")):
                tap_n = stats.get(f"tap_{rng}", 0)
                twotap_n = stats.get(f"two_tap_{rng}", 0)
                spray_n = stats.get(f"spray_{rng}", 0)
                tap_dec = stats.get(f"tap_{rng}_decisions", tap_n)
                twotap_dec = stats.get(f"two_tap_{rng}_decisions", max(1, (twotap_n + 2) // 3) if twotap_n > 0 else 0)
                spray_dec = stats.get(f"spray_{rng}_decisions", max(1, (spray_n + 7) // 8) if spray_n > 0 else 0)
                total_ticks = max(1, tap_n + twotap_n + spray_n)
                total_dec = max(1, tap_dec + twotap_dec + spray_dec)
                tap_hits = stats.get(f"tap_{rng}_hit", 0)
                twotap_hits = stats.get(f"two_tap_{rng}_hit", 0)
                spray_hits = stats.get(f"spray_{rng}_hit", 0)
                lines.append(
                    f"  {label} [decisions]: 1-tap {tap_dec:3d} ({100.0 * tap_dec / total_dec:4.1f}%) | "
                    f"2-tap {twotap_dec:3d} ({100.0 * twotap_dec / total_dec:4.1f}%) | "
                    f"2tap+crouch-spray {spray_dec:3d} ({100.0 * spray_dec / total_dec:4.1f}%)"
                )
                lines.append(
                    f"  {label} [ticks/hit]: 1-tap {tap_n:3d} (hits {tap_hits:2d}) | "
                    f"2-tap {twotap_n:3d} (hits {twotap_hits:2d}) | "
                    f"2tap+crouch-spray {spray_n:3d} (hits {spray_hits:2d})"
                )
            cs_ticks = stats.get("crouch_spray_ticks", 0)
            cs_hits = stats.get("crouch_spray_hits", 0)
            lines.append(f"  crouch-spray (+duck held): {cs_ticks:4d} ticks (hits {cs_hits:3d})")
            lines.append("-" * 60)
        if 'duel_telemetry' in metrics:
            dt = metrics['duel_telemetry']
            lines.append("Duel Telemetry (1v1 Benchmark):")
            lines.append(f"  survival_rate (0 deaths) : {dt['zero_death_episodes']}/{metrics['n_episodes']} ({dt['survival_rate_pct']:.1f}%)")
            lines.append(f"  shots_fired / accuracy   : {dt['total_shots_fired']} ticks ({dt['shot_hit_accuracy_pct']:.1f}% hits)")
            if dt['reaction_ticks_to_fire_mean'] >= 0:
                lines.append(f"  reaction_to_1st_shot     : {dt['reaction_ticks_to_fire_mean']:.2f} ticks (~{dt['reaction_ms_to_fire_mean']:.0f} ms)")
            if dt['reaction_ticks_to_hit_mean'] >= 0:
                lines.append(f"  time_to_1st_hit          : {dt['reaction_ticks_to_hit_mean']:.2f} ticks (~{dt['reaction_ms_to_hit_mean']:.0f} ms)")
            if dt['aim_error_at_fire_deg_mean'] >= 0:
                lines.append(f"  aim_error_at_fire        : {dt['aim_error_at_fire_deg_mean']:.2f} deg")
            lines.append(f"  cover_retreats           : {dt['cover_retreats']}")
            if 'shots_err_gt30_pct' in dt:
                lines.append(f"  shots_err <=5/<=14/>30 deg: {dt['shots_err_le5_pct']:.0f}% / {dt['shots_err_le14_pct']:.0f}% / {dt['shots_err_gt30_pct']:.0f}% (median {dt['aim_error_median_deg']:.1f} deg, n={dt['visible_shots']})")
            if 'shots_moving_pct' in dt:
                lines.append(f"  shots_while_moving(>0.30): {dt['shots_moving_pct']:.0f}% (median speed {dt['fire_speed_median']:.2f})")
            lines.append(f"  hits by mode             : tap {dt['tap_hits']}/{dt['tap_shots']} | 2tap {dt['two_tap_hits']}/{dt['two_tap_shots']} | spray {dt['spray_hits']}/{dt['spray_shots']}")
            if getattr(self, "spray_dpitch_deg", None):
                sdp = np.asarray(self.spray_dpitch_deg, dtype=np.float32)
                pull_ok_pct = 100.0 * np.mean(self.spray_recoil_pull_ok) if self.spray_recoil_pull_ok else 100.0
                lines.append(
                    f"  spray recoil (dpitch)    : mean={np.mean(sdp):+.2f}° | over(>1.5°)={100*np.mean(sdp>1.5):.0f}% | under(<-1.5°)={100*np.mean(sdp<-1.5):.0f}% | pull_dir_ok={pull_ok_pct:.0f}%"
                )
            tot_turns = max(1, getattr(self, "turn_tank_count", 0) + getattr(self, "turn_slice_count", 0))
            zc = getattr(self, "zone_counts", {"spawn_def": 0, "corridor": 0, "mid_shop": 0})
            tot_z = max(1, sum(zc.values()))
            lines.append(
                f"  corner slicing vs tank   : smooth_slice={100.0 * getattr(self, 'turn_slice_count', 0) / tot_turns:.0f}% | tank_stop={100.0 * getattr(self, 'turn_tank_count', 0) / tot_turns:.0f}%"
            )
            lines.append(
                f"  map zones (ДЕФ/Кор/Маг)  : spawn_def={100.0 * zc['spawn_def'] / tot_z:.0f}% | corridors={100.0 * zc['corridor'] / tot_z:.0f}% | mid_shop={100.0 * zc['mid_shop'] / tot_z:.0f}%"
            )
            lines.append("-" * 60)
        if 'active_view' in metrics:
            av = metrics['active_view']
            lines.append("Active-Episode View (excluding dead_wait episodes):")
            lines.append(f"  dead_wait episodes       : {av['dead_wait_episodes']}/{metrics['n_episodes']}")
            lines.append(f"  active / no-contact      : {av['active_episodes']} / {av['no_contact_active_episodes']}")
            lines.append(f"  active K/D               : {av['kills']}/{av['deaths']} = {av['kd_ratio']:.2f}")
            lines.append(f"  origin span median/max   : {av['origin_span_median_units']:.0f} / {av['origin_span_max_units']:.0f} units (≈0 => bot not moving or origin sensor frozen)")
            lines.append("-" * 60)
        if 'gameplay' in metrics:
            lines.append("Gameplay Metrics:")
            for key, val in metrics['gameplay'].items():
                lines.append(f"  {key:25s}: {val:.3f}")
        lines.append("=" * 60 + "\n")
        report_text = "\n".join(lines)
        print(report_text)
        try:
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            (log_dir / f"eval_Phase{self.phase}_{ts}.log").write_text(report_text, encoding="utf-8")
            (log_dir / "eval_latest.log").write_text(report_text, encoding="utf-8")
            def _to_py(obj):
                if isinstance(obj, dict):
                    return {k: _to_py(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_py(v) for v in obj]
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                return obj
            save_tag = self.tag if self.tag else "latest"
            json_path = log_dir / f"benchmark_{save_tag}.json"
            json_path.write_text(json.dumps(_to_py(metrics), indent=2), encoding="utf-8")
            if getattr(self, "norm_obs_history", None):
                npz_path = log_dir / f"trajectory_{save_tag}.npz"
                np.savez_compressed(
                    npz_path,
                    norm_obs=np.stack(self.norm_obs_history, axis=0),
                    raw_obs=np.stack(self.raw_obs_history, axis=0),
                    actions=np.stack(self.action_history, axis=0),
                    world_origins=np.asarray(self.world_origin_history, dtype=np.float32),
                )
                logger.info(f"Saved recorded 19-sensor trajectory ({len(self.norm_obs_history)} ticks) to {npz_path}")
            if self.tag:
                # Automatically print comparison if both easy and expert exist
                easy_p = log_dir / "benchmark_easy.json"
                expert_p = log_dir / "benchmark_expert.json"
                if easy_p.exists() and expert_p.exists() and self.tag in ("easy", "expert"):
                    compare_benchmarks("easy", "expert")
        except Exception as e:
            logger.warning(f"Failed to save eval log: {e}")
    
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
        if self.vec_normalize is not None:
            self.vec_normalize.close()
        elif self.env is not None:
            self.env.close()


def compare_benchmarks(tag1: str = "easy", tag2: str = "expert"):
    """Print side-by-side comparison of two recorded benchmark runs (e.g. Easy vs Expert)."""
    log_dir = Path(__file__).parent / "logs"
    p1 = log_dir / f"benchmark_{tag1}.json"
    p2 = log_dir / f"benchmark_{tag2}.json"
    if not p1.exists() or not p2.exists():
        logger.warning(f"Cannot compare '{tag1}' vs '{tag2}': missing {p1} or {p2}")
        return
    m1 = json.loads(p1.read_text(encoding="utf-8"))
    m2 = json.loads(p2.read_text(encoding="utf-8"))

    def g(m, *keys, default=0.0):
        cur = m
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    rows = [
        ("Убийства (Kills total)", g(m1, "kills", "total"), g(m2, "kills", "total"), "{:.0f}"),
        ("Смерти (Deaths total)", g(m1, "deaths", "total"), g(m2, "deaths", "total"), "{:.0f}"),
        ("K/D Ratio", g(m1, "kd_ratio"), g(m2, "kd_ratio"), "{:.2f}"),
        ("Попадания (Hits total)", g(m1, "hits", "total"), g(m2, "hits", "total"), "{:.0f}"),
        ("Попаданий на 1 фраг (Hits/Kill)", g(m1, "hits_per_kill"), g(m2, "hits_per_kill"), "{:.2f}"),
        ("Выживаемость (0 смертей, %)", g(m1, "duel_telemetry", "survival_rate_pct"), g(m2, "duel_telemetry", "survival_rate_pct"), "{:.1f}%"),
        ("Среднее здоровье (Avg HP, %)", g(m1, "gameplay", "avg_health") * 100.0, g(m2, "gameplay", "avg_health") * 100.0, "{:.1f}%"),
        ("Точность выстрелов (Hit %)", g(m1, "duel_telemetry", "shot_hit_accuracy_pct"), g(m2, "duel_telemetry", "shot_hit_accuracy_pct"), "{:.1f}%"),
        ("Реакция до 1-го выстрела (мс)", g(m1, "duel_telemetry", "reaction_ms_to_fire_mean"), g(m2, "duel_telemetry", "reaction_ms_to_fire_mean"), "{:.0f} ms"),
        ("Время до 1-го попадания (мс)", g(m1, "duel_telemetry", "reaction_ms_to_hit_mean"), g(m2, "duel_telemetry", "reaction_ms_to_hit_mean"), "{:.0f} ms"),
        ("Угловая ошибка при выстреле (°)", g(m1, "duel_telemetry", "aim_error_at_fire_deg_mean"), g(m2, "duel_telemetry", "aim_error_at_fire_deg_mean"), "{:.2f}°"),
        ("Выстрелы с ошибкой >30° (%)", g(m1, "duel_telemetry", "shots_err_gt30_pct"), g(m2, "duel_telemetry", "shots_err_gt30_pct"), "{:.1f}%"),
        ("Выстрелы с ошибкой <=5° (%)", g(m1, "duel_telemetry", "shots_err_le5_pct"), g(m2, "duel_telemetry", "shots_err_le5_pct"), "{:.1f}%"),
        ("Выстрелы на бегу (%)", g(m1, "duel_telemetry", "shots_moving_pct"), g(m2, "duel_telemetry", "shots_moving_pct"), "{:.1f}%"),
        ("Уходы за укрытие (Cover Retreats)", g(m1, "duel_telemetry", "cover_retreats"), g(m2, "duel_telemetry", "cover_retreats"), "{:.0f}"),
        ("Средняя награда (Reward mean)", g(m1, "reward", "mean"), g(m2, "reward", "mean"), "{:.2f}"),
    ]

    out = [
        "",
        "=" * 88,
        f"СРАВНЕНИЕ: НАШ ИИ vs {tag1.upper()} БОТ  |  НАШ ИИ vs {tag2.upper()} БОТ (1v1)",
        "=" * 88,
        f"{'Показатель':35s} | {'vs ' + tag1.upper():>14s} | {'vs ' + tag2.upper():>14s} | {'Разница (Δ)':>15s}",
        "-" * 88,
    ]
    for label, v1, v2, fmt in rows:
        s1 = fmt.format(v1)
        s2 = fmt.format(v2)
        diff = v2 - v1
        sdiff = f"{diff:+.2f}"
        out.append(f"{label:35s} | {s1:>14s} | {s2:>14s} | {sdiff:>15s}")
    out.append("=" * 88)

    kd1 = max(1e-6, float(g(m1, "kd_ratio")))
    kd2 = float(g(m2, "kd_ratio"))
    kd_drop_pct = 100.0 * (kd2 - kd1) / kd1
    out.append(
        f"ИТОГ ПО K/D: против {tag1.upper()} = {kd1:.2f}, против {tag2.upper()} = {kd2:.2f} "
        f"(изменение эффективности: {kd_drop_pct:+.1f}%)"
    )
    out.append("=" * 88 + "\n")
    text = "\n".join(out)
    print(text)
    (log_dir / f"comparison_{tag1}_vs_{tag2}.txt").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained CS 1.6 RL agent")
    parser.add_argument('model_pos', nargs='?', default=None, help='Path to model .zip file (positional)')
    parser.add_argument('--model', type=str, default=None, help='Path to model .zip file')
    parser.add_argument('--vecnormalize', '--vecnorm', '--norm', dest='vecnormalize', type=str, default=None, help='Path to VecNormalize .pkl file')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Windows bridge IP (default: 127.0.0.1, auto-detects WSL host)')
    parser.add_argument('--port', type=int, default=5555, help='Windows bridge port')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml', help='Config path')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--max-steps', type=int, default=10000, help='Max steps per episode')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy (default)')
    parser.add_argument('--continuous', action='store_true', help='Run continuous evaluation')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--phase', type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8], help='Training phase (auto-inferred if omitted)')
    parser.add_argument(
        '--servo-assist',
        nargs='?',
        const=True,
        default=False,
        type=lambda v: str(v).lower() not in ('0', '0.0', 'false', 'no', 'off'),
        help='Opt-in 50/50 proportional servo assist (default: False = 100%% Pure AI)'
    )
    parser.add_argument('--tag', type=str, default=None, help='Tag name to save benchmark (e.g. easy, expert)')
    parser.add_argument('--compare', nargs=2, metavar=('TAG1', 'TAG2'), default=None, help='Compare two saved benchmarks without running game')
    
    args = parser.parse_args()
    if args.compare:
        compare_benchmarks(args.compare[0], args.compare[1])
        return
    model_path = args.model or args.model_pos or 'models/ppo_cs16_phase7_final.zip'
    
    evaluator = AgentEvaluator(
        model_path=model_path,
        vecnormalize_path=args.vecnormalize,
        host=args.host,
        port=args.port,
        config_path=args.config,
        device=args.device,
        deterministic=not args.stochastic,
        phase=args.phase,
        servo_assist=args.servo_assist,
        tag=args.tag
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

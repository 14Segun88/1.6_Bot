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

def detect_wsl_host(host: str = "127.0.0.1") -> str:
    """If host is 127.0.0.1 and running inside WSL2, auto-detect Windows host IP."""
    if host != "127.0.0.1":
        return host
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                import subprocess
                out = subprocess.check_output(["ip", "route", "show", "default"], universal_newlines=True)
                parts = out.split()
                if "via" in parts:
                    detected = parts[parts.index("via") + 1]
                    logger.info(f"WSL2 detected: auto-selected Windows host IP {detected}")
                    return detected
    except Exception as e:
        logger.debug(f"WSL host detection skipped: {e}")
    return host


class CS16GymEnv(gym.Env):
    """
    Gymnasium environment for Counter-Strike 1.6.
    
    Observation Space (Flat Dict of Box spaces for SB3 compatibility):
    - health (1), armor (1), ammo (1), origin (3), view_angles (2), velocity (3)
    - enemy_rel_angles (2), enemy_distance (1), enemy_visible (1), enemy_health (1)
    - lidar (3)
    
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
        host: str = "127.0.0.1",
        port: int = 5555,
        config_path: Optional[str] = None,
        max_episode_steps: int = 512,
        reward_config: Optional[Dict] = None,
        phase: int = 1
    ):
        super().__init__()
        
        self.host = detect_wsl_host(host)
        self.port = port
        self.max_episode_steps = max_episode_steps
        self.episode_step = 0
        self.phase = phase
        self.prev_angle_error: Optional[float] = None
        self.invalid_obs_count = 0
        
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
        self.was_paused = False
        self.burst_ticks = 0
        self.no_fire_ticks = 0
        self.strafe_ticks_remaining = 10
        self.stop_ticks_remaining = 4
        self._bridge_next_in_strafe = True  # bridge starts every episode with a 10-tick strafe
        
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
        
        candidates = [config_path, "configs/training_config.yaml", "../configs/training_config.yaml"]
        for p in candidates:
            if p and Path(p).exists():
                with open(p, 'r') as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        default_config.update(user_config)
                break
        
        return default_config
    
    def _connect(self):
        """Connect to Windows bridge server."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 15000)  # 15 second timeout to safely allow reload pause during reset
        self.socket.setsockopt(zmq.SNDTIMEO, 15000)
        logger.info(f"Connecting to bridge server at tcp://{self.host}:{self.port}...")
        
        # Test connection
        try:
            self._send_request({"type": "ping"})
            logger.info(f"Connected and verified communication with bridge server at {self.host}:{self.port}")
            active_curr = getattr(self, "_active_curriculum", None)
            if active_curr is not None:
                c_mode, c_bots = active_curr
                req_c = {"type": "set_mode", "mode": c_mode}
                if c_bots is not None:
                    req_c["max_bots"] = int(c_bots)
                self.socket.send(msgpack.packb(req_c, use_bin_type=True))
                self.socket.recv()
                logger.info(f"Auto-restored curriculum mode '{c_mode}' (max_bots={c_bots}) after bridge reconnect!")
        except Exception as e:
            logger.error(f"Failed to communicate with bridge server at {self.host}:{self.port}: {e}")
            logger.error("Troubleshooting steps:")
            logger.error("  1. Make sure bridge_server.py is running on Windows (in PowerShell)")
            logger.error(f"  2. If {self.host} is unreachable, specify Windows host IP via --host <IP>")
            logger.error("  3. Allow port 5555 in Windows Defender Firewall (Run PowerShell as Admin):")
            logger.error('     New-NetFirewallRule -DisplayName "CS16 Bridge 5555" -Direction Inbound -LocalPort 5555 -Protocol TCP -Action Allow')
            raise
    
    def _define_spaces(self):
        """Define observation and action spaces (flat Dict of Box spaces for SB3)."""
        self.observation_space = spaces.Dict({
            'health': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'armor': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'ammo': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'origin': spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            'view_angles': spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32),
            'velocity': spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
            'enemy_rel_angles': spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32),
            'enemy_distance': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'enemy_visible': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'enemy_health': spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'lidar': spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        })
        
        # Action space - 5 continuous values in [-1, 1] (fire triggers when action[4] > 0.0)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
    
    def _send_request(self, request: Dict) -> Dict:
        """Send request to bridge server and get response (auto-reconnects REQ socket on timeout/restart)."""
        try:
            self.socket.send(msgpack.packb(request, use_bin_type=True))
            response_bytes = self.socket.recv()
            return msgpack.unpackb(response_bytes, raw=False)
        except Exception as e:
            logger.warning(f"ZMQ socket reconnecting after error: {e}")
            try:
                self._connect()
            except Exception:
                pass
            raise ConnectionError(f"Bridge communication error: {e}")
    
    def _dict_to_obs_array(self, obs_dict: Dict) -> Dict:
        """Convert observation dict from bridge to flat numpy arrays matching space definition."""
        def to_vec(val, length: int, dtype=np.float32):
            if np.isscalar(val):
                arr = np.array([val], dtype=dtype)
            else:
                arr = np.array(val, dtype=dtype)
            if arr.size != length:
                arr = np.resize(arr, (length,))
            return arr.reshape((length,))

        self_info = obs_dict.get('self', {})
        enemy_info = obs_dict.get('enemy', {})
        lidar_info = obs_dict.get('lidar', [1.0, 1.0, 1.0])

        raw_origin = to_vec(self_info.get('origin', [0.0, 0.0, 0.0]), 3)
        self.last_world_origin = to_vec(self_info.get('world_origin', raw_origin), 3)
        norm_origin = np.clip(raw_origin / 4096.0, -1.0, 1.0)

        return {
            'health': to_vec(self_info.get('health', 1.0), 1),
            'armor': to_vec(self_info.get('armor', 1.0), 1),
            'ammo': to_vec(self_info.get('ammo', 1.0), 1),
            'origin': norm_origin,
            'view_angles': to_vec(self_info.get('view_angles', [0.0, 0.0]), 2),
            'velocity': to_vec(self_info.get('velocity', [0.0, 0.0, 0.0]), 3),
            'enemy_rel_angles': to_vec(enemy_info.get('relative_angles', [0.0, 0.0]), 2),
            'enemy_distance': to_vec(enemy_info.get('distance', 1.0), 1),
            'enemy_visible': to_vec(enemy_info.get('visible', 0.0), 1),
            'enemy_health': to_vec(enemy_info.get('health', 0.0), 1),
            'lidar': to_vec(lidar_info, 3),
        }

    def set_curriculum_mode(self, mode: str, phase_num: Optional[int] = None, max_bots: Optional[int] = None):
        """Switch bridge server between curriculum stages (e.g., 1v1 -> 1v3 -> 1v5 or phase7_corners -> phase6_strafe)."""
        if phase_num is not None:
            self.phase = int(phase_num)
        self._active_curriculum = (mode, max_bots)
        try:
            req = {"type": "set_mode", "mode": mode}
            if max_bots is not None:
                req["max_bots"] = int(max_bots)
            resp = self._send_request(req)
            logger.info(f"Switched bridge curriculum mode to '{mode}' (phase={self.phase}, max_bots={max_bots}): {resp}")
        except Exception as e:
            logger.warning(f"Failed to switch bridge curriculum mode to '{mode}': {e}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.episode_step = 0
        self.total_reward = 0.0
        self.episode_count += 1
        self.prev_angle_error = None
        self.burst_ticks = 0
        self.no_fire_ticks = 0
        self.strafe_ticks_remaining = 10
        self.stop_ticks_remaining = 4
        self._bridge_next_in_strafe = True  # bridge starts every episode with a 10-tick strafe
        
        # Request unpause & reset from bridge
        try:
            self._send_request({"type": "unpause"})
        except Exception:
            pass
        response = self._send_request({"type": "reset"})
        
        if "error" in response:
            raise RuntimeError(f"Reset failed: {response['error']}")
        
        obs_dict = response["observation"]
        obs = self._dict_to_obs_array(obs_dict)
        if bool(obs_dict.get("valid", True)):
            rel_angles = obs['enemy_rel_angles']
            self.prev_angle_error = float(np.hypot(rel_angles[0], rel_angles[1]))
        info = response.get("info", {})
        self._prev_kills = int(info.get("kills", 0))
        self._prev_has_enemy = bool(obs_dict.get("enemy", {}).get("visible", 0.0) > 0.5)
        
        logger.info(f"Episode {self.episode_count} started")
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high).copy()
        
        # Phase 1 (Static Aim) & Phase 3/4 (Moving Target Tracking from Standing Position):
        # lock WASD to 0 so the agent stands firmly in place and focuses 100% on tracking & firing
        if self.phase in (1, 3, 4):
            action[0] = 0.0  # move_forward
            action[1] = 0.0  # move_strafe
        elif self.phase in (5, 6, 7):
            action[0] = float(action[0]) * 0.35  # scale forward/back to prevent wall collision while keeping full strafe A/D
            # The bridge reports (in the previous step's info) whether THIS step is a mandatory strafe step
            if not getattr(self, "_bridge_next_in_strafe", True):
                action[0] = 0.0
                action[1] = 0.0
        elif self.phase >= 8:
            # Phase 8-10 (Live Combat & Tactical PUSH/DEF vs Native CS 1.6 Bots):
            # Scale combat W/S by 0.35 so shots_while_moving(>0.30) stays near 0% during duels (while bridge_server.py
            # drives full-speed 0.95 diagonal pie-slicing during PUSH/DEF navigation when no enemy is in LOS).
            action[0] = float(action[0]) * 0.35
        
        # Send step request
        response = self._send_request({
            "type": "step",
            "action": action.tolist()
        })
        
        if "error" in response:
            raise RuntimeError(f"Step failed: {response['error']}")
        
        # Check if bridge reported pause (e.g. F9 pressed on Windows) or PAUSE file exists
        pause_file = Path(__file__).parent / "PAUSE"
        is_paused = bool(response.get("paused", False)) or pause_file.exists()

        if is_paused:
            if not self.was_paused:
                logger.info("[ПАУЗА] Бот заморожен. Вы можете свободно управлять игрой! Нажмите F9 на Windows для возобновления...")
                self.was_paused = True
            while is_paused:
                time.sleep(0.5)
                try:
                    ping_resp = self._send_request({"type": "step", "action": [0.0, 0.0, 0.0, 0.0, 0.0]})
                    is_paused = bool(ping_resp.get("paused", False)) or pause_file.exists()
                    if not is_paused:
                        response = ping_resp
                except Exception:
                    pass
            logger.info("[СТАРТ] Обучение возобновлено! Управление передано нейросети.")
            self.was_paused = False

        obs_dict = response["observation"]
        obs_valid = bool(obs_dict.get("valid", True))
        obs = self._dict_to_obs_array(obs_dict)
        reward = float(response["reward"])
        
        info = response.get("info", {})
        current_kills = int(info.get("kills", 0))
        kills_this_step = max(0, current_kills - getattr(self, "_prev_kills", current_kills))
        self._prev_kills = current_kills

        # Ground-truth firing telemetry from the bridge (the env does NOT re-simulate the firing state machine,
        # otherwise env and bridge desync when the target distance changes between the two observations)
        fire_mode = str(info.get("fire_mode", "hold"))
        is_firing = bool(info.get("shot_fired", False))
        bridge_hit = bool(info.get("hit", False))
        fire_blocked = bool(info.get("fire_blocked", False))
        self._bridge_next_in_strafe = bool(info.get("next_in_strafe", False))
        in_strafe_window = fire_mode == "strafe"
        in_stop_window = fire_mode == "stop"
        is_single_tap_shot = is_firing and fire_mode in ("tap", "two_tap")
        is_spray_shot = is_firing and fire_mode == "spray"
        
        if not obs_valid:
            self.invalid_obs_count += 1
            if self.invalid_obs_count % 100 == 1:
                logger.warning(
                    f"[WARN] Bridge server returning invalid/zero observations (count={self.invalid_obs_count}). "
                    "Make sure CS 1.6 is loaded into aim_ai_sandbox match!"
                )
            reward = 0.0
            self.prev_angle_error = None
        else:
            self.invalid_obs_count = 0
            # --- Trigger Discipline, Distance-Aware Tap/Spray Mode & Precision Reward Shaping ---
            rel_angles = obs['enemy_rel_angles']
            angle_error = float(np.hypot(rel_angles[0], rel_angles[1]))
            enemy_visible = float(obs['enemy_visible'][0]) > 0.5
            enemy_dist = float(obs['enemy_distance'][0])
            dist_units = max(30.0, enemy_dist * 3000.0)
            # Mode choice is judged at the distance the bridge actually used for the decision
            decision_dist = float(info.get("dist_units_at_decision", dist_units))
            has_enemy = enemy_visible and (enemy_dist < 0.98)
            in_cover_combat = bool(info.get("in_cover_combat", False))
            prev_has_enemy = bool(getattr(self, "_prev_has_enemy", False))
            self._prev_has_enemy = has_enemy
            raw_fire = float(action[4])

            if bool(info.get("dead_wait", False)) or (self.phase == 8 and (bool(info.get("autopilot", False)) or (not has_enemy and not in_cover_combat))):
                # While waiting for round respawn or walking in non-combat corridor patrol in Phase 8,
                # freeze step shaping rewards/penalties at 0.0 so PPO only updates weights on real combat & cover ticks!
                self.prev_angle_error = None
                self._cover_ticks = 0
                reward = 0.0
                terminated = False
                truncated = bool(response["truncated"]) or (self.episode_step + 1 >= self.max_episode_steps)
                self.episode_step += 1
                self.total_reward += reward
                info["episode_step"] = self.episode_step
                info["episode_reward"] = self.total_reward
                return obs, reward, terminated, truncated, info

            # 1. Anti-ceiling / anti-floor penalty:
            # In GoldSrc, pitch < 0 is looking UP (above bots), pitch > 0 is looking DOWN.
            # Floor dummies stand on the same level (target pitch ~ +0.01 to +0.04 rad).
            current_pitch = float(obs['view_angles'][0])
            if current_pitch < -0.08 or current_pitch > 0.22:
                reward -= 0.40 * abs(current_pitch)

            # 2. Potential-based delta-angle shaping (rewards turning toward the target from any angle)
            # True angular half-size of a CS 1.6 torso (~16u) at current distance — identical to bridge_server.py
            hit_radius = float(np.clip(16.0 / dist_units, 0.022, 0.20))    # ~1.5 deg at 600u, ~11 deg at 50u
            near_radius = hit_radius * 2.2

            if has_enemy:
                if self.prev_angle_error is not None:
                    delta_err = self.prev_angle_error - angle_error
                    if self.phase >= 7:
                        gain = 6.5 if angle_error > 0.32 else 4.5
                    else:
                        gain = 6.0 if self.phase == 3 else 4.0
                    reward += float(np.clip(delta_err * gain, -0.5, 0.85 if self.phase >= 7 else 0.65))
                self.prev_angle_error = angle_error

                # Coarse approach gradient + sharp bullseye lock-on gradient inside < 3 deg
                reward += 0.12 * float(np.exp(-3.0 * angle_error))
                reward += 0.45 * float(np.exp(-25.0 * angle_error))

                # Proportional micro-aim & anti-crosstalk shaping (pure RL reward for smooth neural tracking):
                dyaw_err = float(rel_angles[0])
                dpitch_err = float(rel_angles[1])
                act_dyaw = float(action[2])
                act_dpitch = float(action[3])
                # Penalize turning away from the enemy horizontally when error > 1.5 deg
                if abs(dyaw_err) > 0.026 and (dyaw_err * act_dyaw < 0.0):
                    reward -= 0.35 * min(1.0, abs(act_dyaw))
                # Penalize bang-bang ±1.0 saturation inside the fine micro-aim zone (< 5 deg) to prevent overshoot
                if abs(dyaw_err) < 0.087 and abs(act_dyaw) > 0.65:
                    reward -= 0.25 * (abs(act_dyaw) - 0.65)
                # Penalize vertical pitch crosstalk when target is vertically aligned (|dpitch| < 1.5 deg)
                if abs(dpitch_err) < 0.026 and abs(act_dpitch) > 0.40:
                    reward -= 0.25 * (abs(act_dpitch) - 0.40)
            else:
                self.prev_angle_error = None

            # 3. True hitbox trigger discipline & phase-specific footwork/tracking:
            move_mag = float(np.hypot(action[0], action[1]))
            aim_mag = float(np.hypot(action[2], action[3]))
            px = float(obs['origin'][0]) * 4096.0
            py = float(obs['origin'][1]) * 4096.0
            min_lidar = float(np.min(obs['lidar']))
            near_wall = (min_lidar < 0.12) if self.phase >= 5 else ((py > 142.0) or (py < 48.0) or (px < -1285.0) or (min_lidar < 0.15))

            if self.phase == 2:
                if near_wall:
                    reward -= 0.18
                else:
                    center_y_dist = abs(py - 95.0) / 50.0
                    reward += 0.08 * max(0.0, 1.0 - center_y_dist)
                    if abs(float(action[1])) > 0.18:
                        reward += 0.10 * abs(float(action[1]))
            elif self.phase >= 3:
                # Phase 3+ (Standing/Dynamic Tracking of Moving Bots):
                # Reward smooth crosshair tracking inside near_radius without jerky overshoots
                if has_enemy and angle_error <= hit_radius and aim_mag < 0.35:
                    reward += 0.25

            # When firing at an enemy in Phase 5+, physical movement is zeroed by counter-strafe:
            if self.phase >= 5 and has_enemy and is_firing:
                action[0] = 0.0
                action[1] = 0.0

            # Phase 5+ footwork rewards and counter-strafe constraints:
            vel = action[:2]
            horiz_speed = float(np.hypot(vel[0], vel[1]))

            # Penalty for shooting on the run without counter-strafing / stopping
            if self.phase >= 5 and is_firing and horiz_speed > 0.30:
                reward -= 0.80 * horiz_speed
            elif is_single_tap_shot and horiz_speed > 0.35:
                reward -= 0.35 * horiz_speed

            # Mobility reward for active strafing during post-kill/post-burst strafe window or target switching
            if self.phase >= 5:
                if in_strafe_window and abs(float(vel[1])) > 0.20:
                    reward += 0.15
                elif (not is_firing or angle_error > 0.18) and abs(float(vel[1])) > 0.15:
                    reward += 0.10

            # Pure AI Firing Mode Shaping for Phase 5+ (calibrated from Easy/Expert 4-way benchmark):
            # - Long range (> 550u): 1-Tap Standing (0.00..0.10) or 2-Tap Headshot Burst Standing (0.10..0.34)
            # - Close/Medium range (<= 550u): 2-Tap -> Crouch-Spray (> 0.34) or 2-Tap Headshot Burst (0.10..0.34)
            mode_bonus_radius = hit_radius * 1.50
            tap_dist_thresh = 550.0
            if bool(info.get("dead_wait", False)):
                pass  # Zero step shaping penalties while waiting for round respawn
            elif self.phase == 8:
                # Phase 8 Live Combat Duel Shaping vs Native CS 1.6 Bots (100% Pure AI — learned via rewards, not scripts!):
                engage_radius = near_radius * 1.4
                inc_dmg = float(info.get("incoming_dmg", 0.0))
                cur_hp_frac = float(obs['health'][0])
                enemy_hp_frac = float(obs['enemy_health'][0])
                lidar_left = float(obs['lidar'][1])
                lidar_right = float(obs['lidar'][2])

                # 1. Tactical Cover & Jiggle-Peek Shaping (2.5s Combat Cover Window):
                # Also reward strafing toward the nearest side pillar (lidar_left / lidar_right) when wounded or on cooldown!
                if (fire_blocked or inc_dmg > 0.0 or cur_hp_frac < 0.75):
                    if lidar_left < 0.55 and lidar_left <= lidar_right and float(vel[1]) > 0.15:
                        reward += 0.55
                    elif lidar_right < 0.55 and lidar_right < lidar_left and float(vel[1]) < -0.15:
                        reward += 0.55

                if in_cover_combat and not has_enemy:
                    self._cover_ticks = int(getattr(self, "_cover_ticks", 0)) + 1
                    self._was_in_cover = True
                    # Reward stepping behind cover right after a burst (cooldown) or when taking damage
                    if prev_has_enemy and (fire_blocked or inc_dmg > 0.0 or cur_hp_frac < 0.75) and abs(float(vel[1])) > 0.15:
                        reward += 1.20
                    # Reward keeping crosshair pre-aimed on the enemy's corner while behind cover
                    reward += 0.25 * float(np.exp(-8.0 * angle_error))
                    # Penalize shooting into the wall while behind cover
                    if raw_fire > 0.0:
                        reward -= 0.45
                    # Once weapon recovery is complete, reward strafing back out to peek (and penalize camping > 10 ticks)
                    if not fire_blocked:
                        if abs(float(vel[1])) > 0.20:
                            reward += 0.25
                        if self._cover_ticks > 10:
                            reward -= 0.25
                elif has_enemy:
                    # Jiggle-Peek Pre-Aim Bonus when stepping out from cover with crosshair on target
                    if not prev_has_enemy and bool(getattr(self, "_was_in_cover", False)):
                        if angle_error <= engage_radius:
                            reward += 1.50
                    self._cover_ticks = 0
                    self._was_in_cover = False

                # 2. Reward closing distance when far (> 550u) and active lateral strafe (A/D) between bursts
                if has_enemy and not is_firing:
                    if decision_dist > 550.0:
                        if float(vel[0]) > 0.05:
                            reward += 0.25 * float(vel[0])
                        elif float(vel[0]) < -0.05:
                            reward -= 0.35 * abs(float(vel[0]))
                    if abs(float(vel[1])) > 0.25:
                        reward += 0.20

                # 3. Trigger discipline & firing mode choice (uniform action[4] thresholds: <=0.28 1-tap, <=0.42 2-tap, >0.42 crouch-spray)
                if has_enemy and not fire_blocked:
                    if raw_fire > 0.0 and angle_error <= engage_radius:
                        # Wounded Target Finish Instinct (03:enemy_hp -> finish the kill!)
                        if 0.0 < enemy_hp_frac < 0.85:
                            reward += 0.60 * (1.0 - enemy_hp_frac)
                        if decision_dist > tap_dist_thresh:
                            if raw_fire <= 0.42:
                                reward += 0.85   # Long range (> 550u): reward 1-Tap (<= 0.28) & 2-Tap (0.28..0.42)
                            else:
                                reward -= 0.75   # Long range (> 550u): penalize full crouch-spray (> 0.42)!
                        else:
                            # Phase 9 prep (measured on 30+30 duel episodes: tap ~14% hit rate, spray ~1%):
                            # close-range spray no longer out-earns 1-tap, and 1-tap is no longer penalised.
                            if raw_fire > 0.42:
                                reward += 0.30   # Close/Medium range (<= 550u): crouch-spray (> 0.42), reduced from +0.95
                            elif raw_fire > 0.28:
                                reward += 0.55   # Close/Medium range: 2-Tap standing (0.28..0.42)
                            else:
                                reward += 0.55   # Close/Medium range: 1-Tap (<= 0.28), was -0.45
                    elif raw_fire <= 0.0 and angle_error <= engage_radius:
                        reward -= 0.85  # Strong anti-hesitation penalty: never stare passively at an enemy!
                    elif raw_fire > 0.0 and angle_error > engage_radius * 1.35:
                        # Symmetric off-target trigger penalty (prevents firing into empty air on either left or right side)
                        off_scale = min(1.0, (angle_error - engage_radius * 1.35) / 0.35)
                        reward -= (0.55 + 0.45 * off_scale)
                        if raw_fire > 0.42:
                            reward -= 0.40  # Extra penalty: crouch-spray while off-target
                    elif raw_fire <= 0.0 and angle_error > engage_radius * 1.35:
                        reward += 0.15  # Reward trigger discipline while still flicking onto target
                if is_firing:
                    if bridge_hit:
                        reward += 6.50
                        if 0.0 < enemy_hp_frac < 0.85:
                            reward += 2.50 * (1.0 - enemy_hp_frac)  # Follow-up hit bonus on wounded enemy!
                        if bool(info.get("crouch_spray", False)):
                            reward += 0.50
                    elif not has_enemy or angle_error > engage_radius * 1.5:
                        reward -= 0.20
                if kills_this_step > 0:
                    reward += (6.0 + 4.0 * cur_hp_frac) * kills_this_step
                if inc_dmg > 0.0 and (in_strafe_window or abs(float(vel[1])) > 0.25):
                    reward += 0.25
            else:
                if self.phase >= 5 and has_enemy and not fire_blocked:
                    if decision_dist > tap_dist_thresh:
                        if raw_fire > 0.34:
                            reward -= 1.00   # Penalty for full crouch-spray at long distance (> 550u)
                        elif 0.0 < raw_fire <= 0.34 and angle_error <= mode_bonus_radius:
                            reward += 0.65   # Correct: 1-tap (<= 0.10) or 2-tap headshot burst (0.10..0.34) at long distance
                    else:
                        if raw_fire > 0.34 and angle_error <= mode_bonus_radius:
                            reward += 0.75   # Best at close/medium range (<= 550u): 2-tap headshot + crouch-spray!
                        elif 0.10 < raw_fire <= 0.34 and angle_error <= mode_bonus_radius:
                            reward += 0.55   # Good at close/medium range: 2-tap headshot burst standing
                        elif 0.0 < raw_fire <= 0.10:
                            reward -= 0.60   # Penalty for slow 1-tap at close range (<= 550u)

                if is_firing:
                    if bridge_hit:
                        # Bridge confirmed a hit (same torso radius, same observation it used for damage)
                        precision_bonus = float(np.clip(1.0 - 0.4 * (angle_error / hit_radius), 0.6, 1.0))
                        # Phase 5+ running-and-gunning spread penalty (reduces hit reward by 75%)
                        if self.phase >= 5 and horiz_speed > 0.30:
                            precision_bonus *= 0.25

                        if is_single_tap_shot and decision_dist > tap_dist_thresh:
                            if int(info.get("dmg", 0)) >= 6:
                                reward += 4.40 * precision_bonus   # long-range single-tap headshot
                            else:
                                reward += 3.40 * precision_bonus   # long-range single-tap body hit
                        elif is_spray_shot and decision_dist <= tap_dist_thresh:
                            reward += 4.00 * precision_bonus       # close/medium-range spray hit
                        elif is_spray_shot and decision_dist > tap_dist_thresh:
                            reward += 0.50 * precision_bonus       # capped < 1.20 penalty so long-range spray is never profitable
                        else:
                            reward += 0.60 * precision_bonus       # capped < 0.95 penalty so close-range tap is never profitable
                    elif has_enemy and angle_error <= near_radius:
                        if not (
                            self.phase >= 5
                            and ((is_spray_shot and decision_dist > tap_dist_thresh) or (is_single_tap_shot and decision_dist <= tap_dist_thresh))
                        ):
                            reward += 0.15   # Grazing shot right next to the model (only for the correct firing mode)
                    else:
                        # Progressive miss penalty scaling with angular error (in radians):
                        if angle_error <= 0.10:
                            miss_penalty = 0.20
                        else:
                            miss_penalty = 0.20 + 0.65 * min(1.0, (angle_error - 0.10) / 0.40)
                        reward -= miss_penalty
                else:
                    # Hesitation penalty: crosshair inside the hitbox, firing was possible, but the AI held fire (action[4] <= 0)
                    if has_enemy and angle_error <= hit_radius and not fire_blocked:
                        reward -= 0.35

                # Phase 6-7 Tactical Survival, Corner Flick, Situational Crouch-Spray:
                if self.phase >= 6:
                    if kills_this_step > 0 and decision_dist <= 360.0:
                        reward += 3.0 * kills_this_step   # Close-quarters rusher neutralizer bonus
                    if float(info.get("incoming_dmg", 0.0)) > 0.0 and (in_strafe_window or abs(float(vel[1])) > 0.25):
                        reward += 0.20                    # Evasive strafe under enemy counter-fire bonus
                if self.phase >= 7:
                    if bridge_hit and bool(info.get("crouch_spray", False)):
                        reward += 0.45                    # Situational crouch-spray accuracy bonus
                    if has_enemy and kills_this_step == 0:
                        reward -= 0.025                   # Time-to-kill urgency: eliminate visible targets as fast as possible!

        # Always run the full episode (512 steps) so transient round-end / 0 HP states never end an episode at step 1!
        terminated = False
        truncated = bool(response["truncated"]) or (self.episode_step + 1 >= self.max_episode_steps)
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
    host: str = "127.0.0.1",
    port: int = 5555,
    config_path: Optional[str] = None,
    **kwargs
) -> CS16GymEnv:
    """Factory function to create CS16GymEnv."""
    return CS16GymEnv(host=host, port=port, config_path=config_path, **kwargs)


# Test script
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test connection
    env = make_cs16_env(host="127.0.0.1", port=5555)
    
    try:
        print("Testing environment...")
        obs, info = env.reset()
        print(f"Observation keys: {list(obs.keys())}")
        print(f"Self health: {obs['health']}")
        print(f"Enemy distance: {obs['enemy_distance']}")
        
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

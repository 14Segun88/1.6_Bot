"""
TCP/ZeroMQ bridge server running on Windows.
Connects memory_reader + game_controller to Linux training client.
"""

import zmq
import msgpack
import threading
import time
import logging
import signal
import sys
from typing import Dict, Optional
import numpy as np
from pynput.keyboard import Key
from pynput.mouse import Button

from memory_reader import MemoryReader, PlayerState
from game_controller import GameController

logger = logging.getLogger(__name__)

class BridgeServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5555):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        
        self.memory_reader = MemoryReader()
        self.game_controller = GameController()
        
        self.running = False
        self.current_observation: Optional[Dict] = None
        self.last_reward = 0.0
        self.episode_step = 0
        self.total_kills = 0
        self.total_deaths = 0
        
        # Reward tracking
        self.virtual_health = 100.0
        self.prev_health = 100.0
        self.prev_enemy_dist = 3000.0
        self.prev_ammo = 30
        self.shots_fired = 0
        self.hits_dealt = 0
        self.burst_ticks = 0
        self.no_fire_ticks = 0
        self.burst_id = 0
        self._counter_strafed_burst = 0
        self.strafe_ticks_remaining = 10
        self.stop_ticks_remaining = 4
        self.strafe_dir = 1.0
        self._reset_fire_counters()
        
    def start(self):
        """Start the bridge server."""
        # Disable Windows Console QuickEdit Mode so accidental mouse clicks in PowerShell never freeze the server
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_stdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
                # Clear ENABLE_QUICK_EDIT_MODE (0x0040), set ENABLE_EXTENDED_FLAGS (0x0080)
                new_mode = (mode.value & ~0x0040) | 0x0080
                kernel32.SetConsoleMode(h_stdin, new_mode)
        except Exception:
            pass

        # Connect to CS 1.6 first
        logger.info("Connecting to CS 1.6...")
        if not self.memory_reader.connect():
            logger.error("Failed to connect to CS 1.6. Make sure the game is running and you're in a match.")
            logger.error("Run as Administrator!")
            return
        
        # Initialize ZeroMQ REP socket with LINGER=0 so Ctrl+C never hangs on context.term()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 40)  # 40ms (25 Hz) so enemy corner-rush carousel runs smoothly even while idle/paused
        
        # Start game controller
        self.game_controller.start()
        
        self.running = True
        logger.info(f"Bridge server started on tcp://{self.host}:{self.port}")
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._main_loop()
    
    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)
    
    def stop(self):
        """Stop the bridge server."""
        self.running = False
        self.game_controller.stop()
        if self.socket:
            try:
                self.socket.setsockopt(zmq.LINGER, 0)
                self.socket.close()
            except Exception:
                pass
        try:
            self.context.destroy(linger=0)
        except Exception:
            pass
        self.memory_reader.close()
        logger.info("Bridge server stopped")
    
    def _main_loop(self):
        """Main server loop - handles requests from Linux client."""
        while self.running:
            try:
                # Wait for request from Linux
                message = self.socket.recv()
                request = msgpack.unpackb(message, raw=False)
                
                # Handle different request types
                response = self._handle_request(request)
                
                # Send response
                self.socket.send(msgpack.packb(response, use_bin_type=True))
                
            except zmq.Again:
                # Idle/paused 25 Hz tick: keep driving the enemy corner-rush carousel & real-bullet respawn!
                try:
                    local_p = self.memory_reader.get_local_player()
                    if local_p is not None:
                        self.memory_reader.get_enemies(local_p.team, local_p.origin)
                except Exception:
                    pass
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Send error response
                try:
                    self.socket.send(msgpack.packb({"error": str(e)}, use_bin_type=True))
                except:
                    pass
    
    def _handle_request(self, request: Dict) -> Dict:
        """Handle incoming request from Linux client."""
        req_type = request.get("type", "step")
        
        if req_type == "reset":
            return self._handle_reset(request)
        elif req_type == "step":
            return self._handle_step(request)
        elif req_type == "observation":
            return self._handle_observation()
        elif req_type == "bots_snapshot":
            return self.memory_reader.get_5v5_bots_snapshot()
        elif req_type == "ping":
            return {"status": "ok", "tick": int(time.time() * 1000)}
        elif req_type == "unpause":
            self.game_controller.is_paused = False
            return {"status": "unpaused"}
        elif req_type == "set_mode":
            mode = str(request.get("mode", "phase8_combat_1v5"))
            is_dyn = mode == "phase6_strafe"
            is_real_combat = mode == "phase8_combat_1v5"
            self.memory_reader.real_combat_mode = is_real_combat
            self.memory_reader.dynamic_strafe_mode = is_dyn
            self.memory_reader.reset_lap_to_spawn()
            now_m = time.time()
            self.lap_start_time = now_m
            self.last_kill_time = now_m
            self.lap_kills = 0
            self._no_hit_fire_steps = 0
            logger.info(
                f"=== [CURRICULUM MODE SWITCH] Mode set to '{mode}' (real_combat_1v5={is_real_combat}, dynamic_strafe={is_dyn})! ==="
            )
            return {"status": "ok", "mode": mode}
        else:
            return {"error": f"Unknown request type: {req_type}"}
    
    def _handle_reset(self, request: Dict) -> Dict:
        """Handle environment reset."""
        logger.info("Reset requested")
        
        # Reset game state (may need to send console commands)
        self._reset_game_state()
        
        # Reset tracking variables
        is_seamless_combat = bool(getattr(self.memory_reader, "real_combat_mode", False)) and getattr(self, "episode_step", 0) > 0
        self.episode_step = 0
        self.last_reward = 0.0
        if not is_seamless_combat:
            self.virtual_health = 100.0
            self.prev_health = 100.0
            self.prev_enemy_dist = 3000.0
            self.prev_ammo = 30
            self.shots_fired = 0
            self.hits_dealt = 0
            self.burst_ticks = 0
            self.no_fire_ticks = 0
            self.strafe_ticks_remaining = 10
            self.stop_ticks_remaining = 4
            self._reset_fire_counters()
        
        # Get initial observation
        obs = self._get_observation()
        self.current_observation = obs
        logger.info(
            f"Reset complete (local_pev=0x{(getattr(self.memory_reader, 'local_pev_addr', None) or 0):X}, "
            f"hp={float(getattr(self.memory_reader, 'local_real_hp', 100.0)):.1f}, "
            f"kills={self.total_kills}, deaths={self.total_deaths})"
        )
        
        return {
            "observation": obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {"kills": self.total_kills, "deaths": self.total_deaths}
        }
    
    def _handle_step(self, request: Dict) -> Dict:
        """Handle environment step with action from agent."""
        if self.game_controller.is_paused:
            obs = self._get_observation()
            return {
                "observation": obs,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "paused": True,
                "info": {"paused": True, "kills": self.total_kills, "deaths": self.total_deaths}
            }

        action = request.get("action", [0, 0, 0, 0, 0])
        
        enemy_visible = False
        in_cover_combat = False
        dist_u = 600.0
        rel_yaw_pre = 0.0
        rel_pitch_pre = 0.0
        loop_leg = 0
        patrol_wp = [-1255.0, 600.0, 1.5707963]
        if self.current_observation and self.current_observation.get("valid", False):
            enemy_obs_pre = self.current_observation.get("enemy", {})
            enemy_visible = enemy_obs_pre.get("visible", 0.0) > 0.5
            in_cover_combat = bool(self.current_observation.get("in_cover_combat", False))
            dist_u = max(30.0, float(enemy_obs_pre.get("distance", 1.0)) * 3000.0)
            rel_angles_pre = enemy_obs_pre.get("relative_angles", [0.0, 0.0])
            rel_yaw_pre = float(rel_angles_pre[0])
            rel_pitch_pre = float(rel_angles_pre[1])
            loop_leg = int(self.current_observation.get("loop_leg", 0))
            patrol_wp = self.current_observation.get("patrol_waypoint", [-1255.0, 600.0, 1.5707963])

        # Phase 5-8 firing & patrol controller. The NEURAL NETWORK chooses the mode via action[4]:
        #   action[4] <= 0.0          : HOLD fire (trigger released)
        #   0.00 < action[4] <= 0.14  : 1-TAP STANDING ("одиночный в голову стоя")
        #   0.14 < action[4] <= 0.24  : 2-TAP STANDING ("2 пули в голову стоя")
        #   action[4] > 0.24          : 2-TAP + CROUCH-SPRAY ("2 пули стоя + зажим в приседе")
        # When no enemy is visible in the current corridor, the agent patrols forward along the 4-leg loop
        # (Spawn -> Corridor 1 -> Enemy Spawn -> Corridor 2 -> Own Spawn), and counter-strafes to an instant
        # stop the moment a bot pops out from any corner!
        is_real_combat = bool(getattr(self.memory_reader, "real_combat_mode", False))
        TAP_COOLDOWN_TICKS = 3 if is_real_combat else 2
        TWO_TAP_HOLD_TICKS = 3
        TWO_TAP_PAUSE_TICKS = 3 if is_real_combat else 2
        SPRAY_BURST_TICKS = 10 if is_real_combat else 8
        SPRAY_PAUSE_TICKS = 3 if is_real_combat else 2
        CYCLE_MAX_SHOTS = 12 if is_real_combat else 4

        strafe_cmd = float(action[1])
        fwd_cmd = float(action[0])
        raw_fire = float(action[4])
        single_shot_pulse = False
        continuous_spray = False
        current_spray_tick = 0
        fire_mode = "hold"
        fire_blocked = False  # True while a mandatory strafe/stop/cooldown makes firing impossible this tick
        in_patrol_phase = False

        # Instant counter-strafe stop when a bot pops out from a corner while the player was patrolling:
        # In real_combat_mode, lines 586-631 execute the 35ms counter-strafe key impulse on the VERY SAME tick,
        # so stop_ticks_remaining = 0 allows immediate first-tick return fire against Expert bots!
        if enemy_visible and getattr(self, "_was_patrolling", False):
            self._was_patrolling = False
            self.strafe_ticks_remaining = 0
            self.stop_ticks_remaining = 0 if is_real_combat else 2
            self.burst_id = getattr(self, "burst_id", 0) + 1

        in_strafe_phase = getattr(self, "strafe_ticks_remaining", 0) > 0

        if not enemy_visible and not in_cover_combat and not in_strafe_phase and getattr(self, "stop_ticks_remaining", 0) == 0:
            fire_mode = "patrol"
            fire_blocked = True
            in_patrol_phase = True
            self._was_patrolling = True
            self._reset_fire_counters()
            self.no_fire_ticks = 0
            fwd_cmd = 0.85
            strafe_cmd = 0.0
        elif in_strafe_phase:
            self._was_patrolling = False
            fire_mode = "strafe"
            fire_blocked = True
            self.strafe_ticks_remaining -= 1
            self._reset_fire_counters()
            if abs(strafe_cmd) < 0.65:
                strafe_cmd = 0.85 * float(getattr(self, "strafe_dir", 1.0))
            if self.strafe_ticks_remaining == 0:
                self.stop_ticks_remaining = 1 if is_real_combat else 4
                self.burst_id = getattr(self, "burst_id", 0) + 1
        elif getattr(self, "stop_ticks_remaining", 0) > 0:
            self._was_patrolling = False
            fire_mode = "stop"
            fire_blocked = True
            self.stop_ticks_remaining -= 1
            self._reset_fire_counters()
            fwd_cmd = 0.0
            strafe_cmd = 0.0
        else:
            self._was_patrolling = False
            # 100% PURE AI COMBAT: forward, strafe, aim, and trigger come directly from the neural network's action[0..4]!
            fwd_cmd = float(action[0]) * 0.50 if is_real_combat else 0.0
            strafe_cmd = float(action[1])
            if getattr(self, "trigger_cooldown", 0) > 0:
                # Mandatory weapon recovery time between taps/bursts (movement remains 100% controlled by action[0..1])
                fire_mode = "cooldown"
                fire_blocked = True
                self.trigger_cooldown -= 1
                self.burst_ticks = 0
                self.recoil_ticks = 0
                if self.trigger_cooldown == 0:
                    self.strafe_dir = -float(getattr(self, "strafe_dir", 1.0))
                    if getattr(self, "cycle_shots", 0) >= CYCLE_MAX_SHOTS:
                        self.cycle_shots = 0
                        is_dyn_only = bool(getattr(self.memory_reader, "dynamic_strafe_mode", False)) and not is_real_combat
                        self.strafe_ticks_remaining = 4 if is_dyn_only else 0
                        self.stop_ticks_remaining = 2 if is_dyn_only else 0
            elif enemy_visible and (raw_fire > 0.0 or getattr(self, "spray_ticks", 0) > 0):
                if getattr(self, "spray_ticks", 0) == 0 and self.no_fire_ticks > 0:
                    self.burst_id = getattr(self, "burst_id", 0) + 1
                self.no_fire_ticks = 0
                # Stop legs during the exact ticks of active firing so shots land with counter-strafed accuracy:
                fwd_cmd = 0.0
                strafe_cmd = 0.0
                # 100% PURE AI Firing Mode Choice via action[4] (uniform thresholds across ALL distances — no distance if-else):
                #   - 0.00 < action[4] <= 0.28 : 1-TAP STANDING
                #   - 0.28 < action[4] <= 0.42 : 2-TAP STANDING
                #   - action[4] > 0.42         : 2-TAP + CROUCH-SPRAY
                active_mode = getattr(self, "active_burst_mode", None)
                if getattr(self, "spray_ticks", 0) == 0 or active_mode not in ("two_tap", "crouch_spray"):
                    if raw_fire <= 0.28:
                        active_mode = "one_tap"
                    elif raw_fire <= 0.42:
                        active_mode = "two_tap"
                    else:
                        active_mode = "crouch_spray"
                    self.active_burst_mode = active_mode

                if active_mode == "one_tap":
                    # 1) SINGLE-TAP ("одиночный в голову стоя"): 1 bullet pulse (15ms), then 4-tick (~170ms) reset
                    fire_mode = "tap"
                    single_shot_pulse = True
                    self.burst_ticks = 1
                    self.spray_ticks = 0
                    self.recoil_ticks = 0
                    self.active_burst_mode = None
                    self.trigger_cooldown = TAP_COOLDOWN_TICKS
                    self.cycle_shots = getattr(self, "cycle_shots", 0) + 1
                    self.clip_bullets_used = getattr(self, "clip_bullets_used", 0) + 1
                elif active_mode == "two_tap":
                    # 2) 2-TAP HEADSHOT BURST STANDING ("2 пули в голову стоя"):
                    #    Hold LMB for 3 ticks (~130ms -> exactly 2 AK/M4 bullets at 0ms & 95.5ms) aimed at the head,
                    #    then 4-tick (~170ms) pause so each 2-bullet burst is crisp and distinct!
                    fire_mode = "two_tap"
                    continuous_spray = True
                    self.spray_ticks = getattr(self, "spray_ticks", 0) + 1
                    current_spray_tick = self.spray_ticks
                    self.burst_ticks = 2
                    self.recoil_ticks = 0
                    if self.spray_ticks >= TWO_TAP_HOLD_TICKS:
                        self.spray_ticks = 0
                        self.active_burst_mode = None
                        self.trigger_cooldown = TWO_TAP_PAUSE_TICKS
                        self.cycle_shots = getattr(self, "cycle_shots", 0) + 2
                        self.clip_bullets_used = getattr(self, "clip_bullets_used", 0) + 2
                else:
                    # 3) 2-TAP INTO CROUCH-SPRAY ("2 пули стоя в голову + зажим в приседе"):
                    #    Ticks 1..2 (~90ms, first 2 bullets): STANDING headshot aim point (duck = 0.0, recoil_ticks = 0)
                    #    Ticks 3..10 (8 ticks = ~350ms, bullets 3..6): FULL CROUCH (+duck held continuously) + pull down to chest!
                    fire_mode = "spray"
                    continuous_spray = True
                    self.spray_ticks = getattr(self, "spray_ticks", 0) + 1
                    current_spray_tick = self.spray_ticks
                    self.burst_ticks = 1 + self.spray_ticks
                    self.recoil_ticks = max(0, self.spray_ticks - 2)
                    if self.spray_ticks >= SPRAY_BURST_TICKS or (self.spray_ticks >= 2 and raw_fire <= 0.42):
                        self.spray_ticks = 0
                        self.active_burst_mode = None
                        self.trigger_cooldown = SPRAY_PAUSE_TICKS
                        self.cycle_shots = getattr(self, "cycle_shots", 0) + 4
                        self.clip_bullets_used = getattr(self, "clip_bullets_used", 0) + 6
            else:
                # HOLD fire (AI chose action[4] <= 0 while no burst is active): 100% AI movement!
                fire_mode = "hold"
                strafe_cmd = float(action[1])
                self.no_fire_ticks += 1
                self.spray_ticks = 0
                self.active_burst_mode = None
                self.recoil_ticks = 0
                self.burst_ticks = 0

        is_firing = single_shot_pulse or continuous_spray
        # "2 пули стоя в голову -> зажим в приседе":
        # - Ticks 1..2 (current_spray_tick <= 2): ALWAYS standing so the first 2 bullets (0ms & 95.5ms) go dead-center into the HEAD!
        # - Ticks 3..10 (current_spray_tick >= 3 and fire_mode == "spray"): Hold +duck continuously for ~350ms so the model visually crouches and sprays!
        if continuous_spray and fire_mode == "spray" and current_spray_tick >= 3:
            duck_active = 1.0
        else:
            duck_active = 0.0
            if getattr(self, "_duck_released_once", False) is False or self.game_controller.current_action.duck:
                try:
                    self.game_controller.release_scancode('duck')
                    self.game_controller.pressed_keys.discard('duck')
                    self.game_controller.current_action.duck = False
                    self.game_controller.target_action.duck = False
                    self.game_controller.keyboard.release(Key.ctrl_l)
                    self._duck_released_once = True
                except Exception:
                    pass

        # Convert action array to dict:
        action_dict = {
            "move_forward": fwd_cmd,
            "move_strafe": strafe_cmd,
            "delta_yaw": float(action[2]),
            "delta_pitch": float(action[3]),
            "fire": 1.0 if continuous_spray else 0.0,
            "duck": duck_active,
        }
        
        # Soft-clamp vertical pitch within [-0.10, +0.20] rad (~ -6 deg up to +11 deg down)
        # and guide 4-leg loop navigation & corridor geofence
        if self.current_observation and self.current_observation.get("valid", False):
            self_obs = self.current_observation.get("self", {})
            cur_pitch = float(self_obs.get("view_angles", [0.0, 0.0])[0])
            if cur_pitch < -0.10 and action_dict["delta_pitch"] < 0.0:
                action_dict["delta_pitch"] = 0.35
            elif cur_pitch > 0.20 and action_dict["delta_pitch"] > 0.0:
                action_dict["delta_pitch"] = -0.35

            orig = self_obs.get("world_origin", self_obs.get("origin", [0.0, 0.0, 0.0]))
            px, py = float(orig[0]), float(orig[1])
            cur_yaw = float(self_obs.get("world_yaw", self_obs.get("view_angles", [0.0, 0.0])[1]))
            cos_y, sin_y = float(np.cos(cur_yaw)), float(np.sin(cur_yaw))
            wp_x, wp_y, leg_yaw = float(patrol_wp[0]), float(patrol_wp[1]), float(patrol_wp[2])
            if getattr(self.memory_reader, "dynamic_strafe_mode", False):
                loop_leg = 0
                wp_x, wp_y, leg_yaw = -1255.0, 210.0, float(np.pi / 2.0)

            # Detect physical wall stall (holding WASD while position is blocked by a wall)
            prev_pos = getattr(self, "_prev_xy", (px, py))
            moved_dist = float(np.hypot(px - prev_pos[0], py - prev_pos[1]))
            self._prev_xy = (px, py)
            cmd_mag = float(np.hypot(action_dict["move_forward"], action_dict["move_strafe"]))
            if cmd_mag > 0.20 and moved_dist < 0.6:
                self._stuck_ticks = getattr(self, "_stuck_ticks", 0) + 1
            else:
                self._stuck_ticks = 0

            if in_patrol_phase:
                # Smoothly align camera yaw down the active corridor (POSITIVE yaw_err turns toward target_yaw!)
                target_yaw = leg_yaw + 0.06 * float(np.sin(time.time() * 2.0))
                yaw_err = (target_yaw - cur_yaw + np.pi) % (2.0 * np.pi) - np.pi
                action_dict["delta_yaw"] = float(np.clip(yaw_err * 2.2, -0.90, 0.90))
                pitch_err = 0.0 - cur_pitch
                action_dict["delta_pitch"] = float(np.clip(pitch_err * 2.0, -0.35, 0.35))

                # Turn to face down the corridor FIRST (|yaw_err| <= 0.38 rad ~ 22 deg) before walking forward,
                # so the AI NEVER walks backward ('S') with its back to the bots!
                if abs(yaw_err) > 0.38:
                    action_dict["move_forward"] = 0.0
                    action_dict["move_strafe"] = 0.0
                else:
                    to_wx, to_wy = wp_x - px, wp_y - py
                    norm_w = max(1.0, float(np.hypot(to_wx, to_wy)))
                    vx = 0.85 * (to_wx / norm_w)
                    vy = 0.85 * (to_wy / norm_w)
                    action_dict["move_forward"] = float(np.clip(vx * cos_y + vy * sin_y, 0.0, 1.0))
                    action_dict["move_strafe"] = float(np.clip(-vx * sin_y + vy * cos_y, -1.0, 1.0))
            elif cmd_mag < 1e-4:
                action_dict["move_forward"] = 0.0
                action_dict["move_strafe"] = 0.0
            else:
                # Combat strafe geofence:
                # In Phase 8 (is_real_combat=True), NEVER push the AI out of side cover alcoves/pillars back into the open center!
                # Only apply unstuck nudge if physically blocked against a wall for >= 5 ticks.
                is_real_combat = bool(getattr(self.memory_reader, "real_combat_mode", False))
                vx = action_dict["move_forward"] * cos_y - action_dict["move_strafe"] * sin_y
                vy = action_dict["move_forward"] * sin_y + action_dict["move_strafe"] * cos_y
                override_xy = False

                if not is_real_combat:
                    if loop_leg == 0:
                        # Left Corridor (+Y, center X = -1255, walkable X in [-1315..-1190])
                        if px < -1305.0 and vx < 0.15:
                            vx = 0.70
                            override_xy = True
                        elif px > -1200.0 and vx > -0.15:
                            vx = -0.70
                            override_xy = True
                        if py < 120.0 and vy < 0.15:
                            vy = 0.65
                            override_xy = True
                        elif getattr(self.memory_reader, "dynamic_strafe_mode", False) and py > 265.0 and vy > -0.15:
                            vy = -0.65
                            override_xy = True
                        elif py > 1425.0 and vy > -0.15:
                            vy = -0.65
                            override_xy = True
                    elif loop_leg == 1:
                        # Enemy Spawn (+X, center Y = 1375, walkable Y in [1255..1450])
                        if py < 1270.0 and vy < 0.15:
                            vy = 0.70
                            override_xy = True
                        elif py > 1440.0 and vy > -0.15:
                            vy = -0.70
                            override_xy = True
                        if px < -1305.0 and vx < 0.15:
                            vx = 0.65
                            override_xy = True
                        elif px > -235.0 and vx > -0.15:
                            vx = -0.65
                            override_xy = True
                    elif loop_leg == 2:
                        # Right Corridor (-Y, center X = -285, walkable X in [-350..-225])
                        if px < -340.0 and vx < 0.15:
                            vx = 0.70
                            override_xy = True
                        elif px > -235.0 and vx > -0.15:
                            vx = -0.70
                            override_xy = True
                        if py < 120.0 and vy < 0.15:
                            vy = 0.65
                            override_xy = True
                        elif py > 1425.0 and vy > -0.15:
                            vy = -0.65
                            override_xy = True
                    else:
                        # Own Spawn (-X, center Y = 155, walkable Y in [85..275])
                        if py < 95.0 and vy < 0.15:
                            vy = 0.70
                            override_xy = True
                        elif py > 265.0 and vy > -0.15:
                            vy = -0.70
                            override_xy = True
                        if px < -1305.0 and vx < 0.15:
                            vx = 0.65
                            override_xy = True
                        elif px > -235.0 and vx > -0.15:
                            vx = -0.65
                            override_xy = True

                if self._stuck_ticks >= 5:
                    to_wx, to_wy = wp_x - px, wp_y - py
                    norm_w = max(1.0, float(np.hypot(to_wx, to_wy)))
                    vx = 0.75 * (to_wx / norm_w)
                    vy = 0.75 * (to_wy / norm_w)
                    override_xy = True

                if override_xy or (not is_real_combat and abs(action_dict["move_forward"]) > 0.20):
                    action_dict["move_forward"] = float(np.clip(vx * cos_y + vy * sin_y, -1.0, 1.0))
                    action_dict["move_strafe"] = float(np.clip(-vx * sin_y + vy * cos_y, -1.0, 1.0))

            # In Phase 8 (real_combat_mode), allow full 360-degree camera rotation when enemy_visible or in_cover_combat is True
            # so the AI can hold pre-aim on corners behind cover and snap 180° onto flankers!
            is_real_combat = bool(getattr(self.memory_reader, "real_combat_mode", False))
            yaw_diff = (cur_yaw - leg_yaw + np.pi) % (2.0 * np.pi) - np.pi
            max_corridor_yaw = (3.15 if is_real_combat else 1.88) if (enemy_visible or in_cover_combat) else 1.30
            if abs(yaw_diff) > max_corridor_yaw and yaw_diff * action_dict["delta_yaw"] > 0.0:
                action_dict["delta_yaw"] = -0.55 * float(np.sign(yaw_diff))

            # Periodic Loop & Aim Telemetry Log (every 40 steps or on leg change)
            prev_log_leg = getattr(self, "_last_logged_leg", -1)
            if loop_leg != prev_log_leg or (self.episode_step % 40 == 0):
                self._last_logged_leg = loop_leg
                leg_names = {0: "Corridor1(+Y)", 1: "EnemySpawn(+X)", 2: "Corridor2(-Y)", 3: "OwnSpawn(-X)"}
                rel_pre = self.current_observation.get("enemy", {}).get("relative_angles", [0.0, 0.0])
                lap_el = time.time() - getattr(self, "lap_start_time", time.time())
                lap_rem = max(0.0, 115.0 - lap_el)
                t_mm, t_ss = int(lap_rem) // 60, int(lap_rem) % 60
                lap_k = int(getattr(self, "lap_kills", 0))
                lap_total_corners = len(getattr(self.memory_reader, "AMBUSH_CORNERS", ())) or 16
                logger.info(
                    f"[LOOP TELEMETRY] Timer={t_mm}:{t_ss:02d}/1:55 | Targets={lap_k}/{lap_total_corners} | "
                    f"leg={loop_leg}({leg_names.get(loop_leg, '?')}) mode={fire_mode} "
                    f"pos=({px:.0f},{py:.0f}) yaw={np.degrees(cur_yaw):.1f}° leg_yaw={np.degrees(leg_yaw):.1f}° | "
                    f"enemy_vis={enemy_visible} rel_yaw={np.degrees(float(rel_pre[0])):+.1f}° dist={dist_u:.0f}u"
                )

            # Counter-strafe & Strict Stop:
            # - In Phase 8 (is_real_combat=True), ONLY counter-strafe and lock WASD during active firing (`is_firing`) or `stop` ticks,
            #   so the AI is never glued to the floor while aiming (`abs(rel_yaw_pre) <= 0.16`) or between bursts (`cooldown`)!
            # - In static corner phases, keep the wider stop lock.
            should_lock_legs = (
                (is_firing or fire_mode == "stop")
                if is_real_combat
                else (is_firing or fire_mode in ("stop", "cooldown") or abs(rel_yaw_pre) <= 0.16)
            )
            if not in_strafe_phase and not in_patrol_phase and should_lock_legs:
                action_dict["move_forward"] = 0.0
                action_dict["move_strafe"] = 0.0

                if single_shot_pulse:
                    # Pure enemy-blind trigger-pull steadying during the 15ms single-tap click
                    # In real_combat_mode (moving native CS 1.6 bots), keep 100% tracking speed (1.0) so crosshair tracks running targets!
                    damp = 1.0 if is_real_combat else 0.35
                    action_dict["delta_yaw"] *= damp
                    action_dict["delta_pitch"] *= damp

                # Check what movement keys were active right at the end of the strafe
                was_pressed = set(self.game_controller.pressed_keys)
                cur_act_strafe = float(getattr(self.game_controller.current_action, 'move_strafe', 0.0))
                cur_act_fwd = float(getattr(self.game_controller.current_action, 'move_forward', 0.0))

                opp_key = None
                if 'left' in was_pressed or cur_act_strafe > 0.15:
                    opp_key = 'right'  # was moving left (A) -> tap D to cancel inertia
                elif 'right' in was_pressed or cur_act_strafe < -0.15:
                    opp_key = 'left'   # was moving right (D) -> tap A to cancel inertia
                elif 'forward' in was_pressed or cur_act_fwd > 0.15:
                    opp_key = 'back'   # was moving forward (W) -> tap S
                elif 'back' in was_pressed or cur_act_fwd < -0.15:
                    opp_key = 'forward' # was moving back (S) -> tap W

                # Release all active movement keys immediately
                try:
                    for k in ('forward', 'back', 'left', 'right'):
                        self.game_controller.release_scancode(k)
                        self.game_controller.pressed_keys.discard(k)
                    self.game_controller.current_action.move_forward = 0.0
                    self.game_controller.current_action.move_strafe = 0.0
                    self.game_controller.target_action.move_forward = 0.0
                    self.game_controller.target_action.move_strafe = 0.0
                except Exception:
                    pass

                # Apply a 35ms counter-strafe impulse at the start of the stop phase (before the 1st bullet!)
                burst_id = getattr(self, "burst_id", 0)
                if opp_key and getattr(self, "_counter_strafed_burst", -1) != burst_id:
                    self._counter_strafed_burst = burst_id
                    try:
                        self.game_controller.tap_scancode(opp_key, hold_sec=0.035)
                    except Exception:
                        pass
        
        # Send action to game
        self.game_controller.set_action(action_dict)

        # Execute explicit 1-bullet hardware tap in single-tap mode, or guarantee LMB release when not spraying
        if single_shot_pulse:
            try:
                self.game_controller.target_action.fire = False
                self.game_controller.current_action.fire = False
                self.game_controller.mouse.press(Button.left)
                time.sleep(0.015)
                self.game_controller.mouse.release(Button.left)
            except Exception:
                pass
        elif not continuous_spray:
            try:
                self.game_controller.target_action.fire = False
                self.game_controller.current_action.fire = False
                self.game_controller.mouse.release(Button.left)
            except Exception:
                pass
        
        # Wait for game to process (low-latency 8ms tick for fast reaction & counter-strafing)
        time.sleep(0.008)

        # Auto-select primary rifle (slot 1) if the player ever switched to secondary pistol (USP/Glock)
        if getattr(self.memory_reader, "local_holding_secondary", False):
            now_s = time.time()
            if now_s - getattr(self, "_last_slot1_tap", 0.0) > 1.0:
                self._last_slot1_tap = now_s
                try:
                    self.game_controller.tap_scancode('slot1', hold_sec=0.015)
                except Exception:
                    pass
        
        # Get new observation
        obs = self._get_observation()
        self.current_observation = obs
        
        # Calculate reward
        reward, info = self._calculate_reward(obs)
        
        # Check if agent landed a direct hit on the currently tracked target:
        # When live GoldSrc server bots (_server_origins) are active, ONLY real engine bullet hits (pev->health drops)
        # count toward killing and teleporting a bot, so bots NEVER vanish prematurely on geometric near-misses!
        hit_registered = False
        hit_dmg = 0
        eng_kills, eng_hits = self.memory_reader.pop_engine_telemetry()
        has_live_server_bots = bool(getattr(self.memory_reader, "_server_origins", None))
        if eng_hits > 0:
            hit_registered = True
            hit_dmg = 2
        killed_now = eng_kills > 0

        if not has_live_server_bots and is_firing and obs.get("valid", False):
            enemy_info = obs.get("enemy", {})
            rel = enemy_info.get("relative_angles", [0.0, 0.0])
            angle_err = float(np.hypot(rel[0], rel[1]))
            dist_units = max(30.0, float(enemy_info.get("distance", 1.0)) * 3000.0)
            # Angular half-size of a CS 1.6 torso (~16u) and head (~5u) at the current distance
            hit_radius = float(np.clip(16.0 / dist_units, 0.022, 0.20))
            head_radius = float(np.clip(5.0 / dist_units, 0.018, 0.05))
            if enemy_info.get("visible", 0.0) > 0.5 and angle_err <= hit_radius:
                # - Long range (> 360u) single tap inside head radius: 6 damage (1-tap kill), body: 3 damage
                # - Close/Medium range (<= 360u) spray tick: 3 damage (fast 2-tick burst kill!), slow tap: 2 damage
                if dist_units > 360.0:
                    if single_shot_pulse and angle_err <= head_radius:
                        dmg = 6
                    elif single_shot_pulse:
                        dmg = 3
                    else:
                        dmg = 2
                else:
                    if continuous_spray:
                        dmg = 3
                    else:
                        dmg = 2
                if not killed_now:
                    if self.memory_reader.register_hit_on_current_target(hits_to_kill=6, hit_damage=dmg):
                        killed_now = True
                    if bool(getattr(self.memory_reader, "last_hit_counted", True)):
                        hit_registered = True
                        hit_dmg = max(hit_dmg, dmg)
                else:
                    hit_registered = True
                    hit_dmg = max(hit_dmg, dmg)

        # Initialize 1:55 (115.0s) Lap Time-Trial Timer if not yet started, and sync to in-game CS 1.6 HUD timer!
        now_t = time.time()
        if not hasattr(self, "lap_start_time"):
            self.lap_start_time = now_t
            self.last_kill_time = now_t
            self.lap_kills = 0
            self._last_hud_timer_sync = now_t
            self.memory_reader.set_hud_round_timer(115.0)
        lap_elapsed = now_t - self.lap_start_time
        lap_rem = max(0.0, 115.0 - lap_elapsed)
        if now_t - getattr(self, "_last_hud_timer_sync", 0.0) >= 5.0:
            self._last_hud_timer_sync = now_t
            self.memory_reader.set_hud_round_timer(lap_rem)

        if killed_now:
            n_kills = max(1, int(eng_kills))
            self.total_kills += n_kills
            self.lap_kills = int(getattr(self, "lap_kills", 0)) + n_kills
            self.virtual_health = min(100.0, float(getattr(self, "virtual_health", 100.0)) + 18.0 * n_kills)
            self._reset_fire_counters()
            self.no_fire_ticks = 0
            self.strafe_dir = -float(getattr(self, "strafe_dir", 1.0))
            try:
                self.game_controller.mouse.release(Button.left)
                self.game_controller.current_action.fire = False
                self.game_controller.target_action.fire = False
                self.game_controller.release_scancode('duck')
                self.game_controller.pressed_keys.discard('duck')
                self.game_controller.current_action.duck = False
                self.game_controller.target_action.duck = False
                self.game_controller.keyboard.release(Key.ctrl_l)
                self.game_controller.tap_scancode('buyammo1', hold_sec=0.01)
            except Exception:
                pass

            # Time-trial speed bonus ("чем быстрее он ликвидирует все цели тем лучше", 1:55 = 115.0s timer):
            kill_split_sec = max(0.1, now_t - getattr(self, "last_kill_time", now_t))
            self.last_kill_time = now_t
            speed_kill_bonus = (6.0 * max(0.0, 1.0 - kill_split_sec / 8.0) + 3.0 * (lap_rem / 115.0)) * n_kills
            reward += (10.0 * n_kills) + speed_kill_bonus
            t_mm, t_ss = int(lap_rem) // 60, int(lap_rem) % 60
            lap_total_corners = len(getattr(self.memory_reader, "AMBUSH_CORNERS", ())) or 16
            logger.info(
                f"[CORNER CLEARED {self.lap_kills}/{lap_total_corners}] split={kill_split_sec:.2f}s | "
                f"Timer={t_mm}:{t_ss:02d}/1:55 | speed_bonus=+{speed_kill_bonus:.2f}"
            )

            # Full Lap Clear Bonus when all static corner targets around the loop are eliminated before 1:55!
            if self.lap_kills >= lap_total_corners:
                lap_clear_bonus = 25.0 + 50.0 * (lap_rem / 115.0)
                reward += lap_clear_bonus
                logger.info(
                    f"[LAP CLEARED {lap_total_corners}/{lap_total_corners}!] Completed in {lap_elapsed:.1f}s "
                    f"(Time Left: {t_mm}:{t_ss:02d} / 1:55) | Lap Speed Bonus: +{lap_clear_bonus:.1f}! Teleporting to spawn for next lap..."
                )
                self.lap_kills = 0
                self.lap_start_time = now_t
                self.last_kill_time = now_t
                self._last_hud_timer_sync = now_t
                if not getattr(self.memory_reader, "dynamic_strafe_mode", False) and not getattr(self.memory_reader, "real_combat_mode", False):
                    self.memory_reader.reset_lap_to_spawn()
                else:
                    self.memory_reader.set_hud_round_timer(115.0)

            info["kills"] = self.total_kills
            # Refresh observation after the corner kill: if the corner is now clear (no visible enemy),
            # immediately allow patrol forward along the loop to the next corner!
            obs = self._get_observation()
            self.current_observation = obs
            is_dyn_only = bool(getattr(self.memory_reader, "dynamic_strafe_mode", False)) and not bool(
                getattr(self.memory_reader, "real_combat_mode", False)
            )
            if float(obs.get("enemy", {}).get("visible", 0.0)) > 0.5:
                # In real_combat_mode (e.g. 2v1 or 5v5), when a 2nd enemy is visible right after killing the 1st,
                # NEVER block fire for 5 ticks (225ms) — transfer aim and shoot the 2nd enemy immediately!
                self.strafe_ticks_remaining = 3 if is_dyn_only else 0
                self.stop_ticks_remaining = 2 if is_dyn_only else 0
            else:
                self.strafe_ticks_remaining = 0
                self.stop_ticks_remaining = 0
        elif lap_elapsed >= 115.0:
            # 1:55 (115.0s) Lap Timer Expired!
            is_real_combat = bool(getattr(self.memory_reader, "real_combat_mode", False))
            lap_total_corners = 5 if is_real_combat else (len(getattr(self.memory_reader, "AMBUSH_CORNERS", ())) or 16)
            uncleared = max(0, lap_total_corners - int(getattr(self, "lap_kills", 0)))
            timeout_pen = 20.0 * (uncleared / float(lap_total_corners))
            reward -= timeout_pen
            logger.info(
                f"[TIMER 1:55 EXPIRED!] Cleared {getattr(self, 'lap_kills', 0)}/{lap_total_corners} targets in 1:55 "
                f"(penalty=-{timeout_pen:.1f})."
            )
            self.lap_kills = 0
            self.lap_start_time = now_t
            self.last_kill_time = now_t
            self._last_hud_timer_sync = now_t
            self._no_hit_fire_steps = 0
            if not is_real_combat:
                self.memory_reader.reset_lap_to_spawn()
            else:
                self.memory_reader.set_hud_round_timer(115.0)

        # Phase 6-8 Enemy Counter-Fire & Standard 100->0 HUD Health Tracking:
        raw_incoming = float(obs.get("incoming_dmg", 0.0)) if obs.get("valid", False) else 0.0
        actual_incoming_dmg = raw_incoming
        player_died = bool(getattr(self.memory_reader, "_pending_player_died", False))
        self.memory_reader._pending_player_died = False
        player_dead_wait = bool(getattr(self.memory_reader, "local_is_dead", False))
        self.virtual_health = float(getattr(self.memory_reader, "local_real_hp", 100.0))
        if player_died:
            self.total_deaths += 1
            info["deaths"] = self.total_deaths
            self.virtual_health = 0.0
            self.prev_health = 100.0
            logger.info(f"[PLAYER DIED 0 HP] AI killed by enemy fire (total deaths={self.total_deaths})!")
            self.memory_reader.reset_lap_to_spawn()

        if obs.get("valid", False) and "self" in obs:
            obs["self"]["health"] = float(np.clip(self.virtual_health / 100.0, 0.0, 1.0))
        if actual_incoming_dmg > 0.0:
            reward -= actual_incoming_dmg * 0.08
        if player_died:
            reward -= 8.0
        self.prev_health = self.virtual_health

        # Track whether the AI was overridden by the corridor autopilot this tick (patrol / wall-unstuck)
        autopilot_active = bool(in_patrol_phase or getattr(self, "_stuck_ticks", 0) >= 5 or player_dead_wait)
        if autopilot_active:
            reward = 0.0  # Never assign positive/negative RL reward to random neural outputs while autopilot is steering!

        self.last_reward = reward
        self.episode_step += 1

        # Ground-truth firing telemetry for cs16_gym_env.py (it must NOT re-simulate the firing state machine)
        info["fire_mode"] = fire_mode                  # hold / tap / spray / cooldown / strafe / stop / patrol
        info["shot_fired"] = bool(is_firing)
        info["hit"] = bool(hit_registered)
        info["dmg"] = int(hit_dmg)
        info["fire_blocked"] = bool(fire_blocked)
        info["next_in_strafe"] = bool(getattr(self, "strafe_ticks_remaining", 0) > 0)
        info["dist_units_at_decision"] = float(dist_u)
        info["incoming_dmg"] = float(actual_incoming_dmg)
        info["player_died"] = bool(player_died)
        info["dead_wait"] = bool(player_dead_wait)
        info["autopilot"] = bool(autopilot_active)
        info["in_cover_combat"] = bool(in_cover_combat)
        info["crouch_spray"] = bool(duck_active > 0.5)
        info["loop_leg"] = int(loop_leg)
        info["lap_time_remaining"] = float(lap_rem)
        info["lap_kills"] = int(getattr(self, "lap_kills", 0))
        
        # Check termination: episodes run for the full 512 steps (~18s), while individual round deaths are tracked via info["deaths"]
        terminated = False
        truncated = self.episode_step >= 512  # 512 steps (~18s) per episode for frequent PPO feedback
        if (terminated or truncated) and not bool(getattr(self.memory_reader, "real_combat_mode", False)):
            try:
                self.game_controller.target_action.fire = False
                self.game_controller.current_action.fire = False
                self.game_controller.mouse.release(Button.left)
                self.game_controller.release_all()
            except Exception:
                pass
        
        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    
    def _handle_observation(self) -> Dict:
        """Return current observation without stepping."""
        obs = self._get_observation()
        return {"observation": obs}
    
    def _reset_game_state(self):
        """Reset the game to initial state for a new episode."""
        # In Phase 8 Live Combat ("вечный круг" vs Native CS 1.6 bots), NEVER interrupt gameplay,
        # NEVER press reload, and NEVER freeze or jerk the camera between 512-step rollout windows!
        if bool(getattr(self.memory_reader, "real_combat_mode", False)) and getattr(self, "episode_step", 0) > 0:
            self.game_controller.is_paused = False
            return

        # 1. Release all inputs to avoid stuck keys/buttons
        self.burst_ticks = 0
        self.no_fire_ticks = 0
        self.burst_id = 0
        self._counter_strafed_burst = 0
        self._was_patrolling = False
        self.strafe_ticks_remaining = 6
        self.stop_ticks_remaining = 4
        self.strafe_dir = 1.0
        self._reset_fire_counters()
        self.game_controller.is_paused = False
        self.game_controller.release_all()
        try:
            self.game_controller.release_scancode('duck')
            self.game_controller.pressed_keys.discard('duck')
            self.game_controller.current_action.duck = False
            self.game_controller.target_action.duck = False
            self.game_controller.keyboard.release(Key.ctrl_l)
        except Exception:
            pass
        
        # 2. Replenish reserve ammo without pressing 'reload' (clip is kept at 30 in memory_reader.py)
        try:
            self.game_controller.tap_scancode('buyammo1')
        except Exception as e:
            logger.debug(f"Ammo reset exception: {e}")
        time.sleep(0.02)
        
        # 3. Recenter camera toward visible dummy or active loop corridor direction (up to 25 iterations)
        if not self.game_controller.is_paused:
            for _ in range(25):
                obs = self._get_observation()
                if not obs.get("valid", False):
                    break
                enemy_obs = obs.get("enemy", {})
                if float(enemy_obs.get("visible", 0.0)) > 0.5:
                    rel = enemy_obs.get("relative_angles", [0.0, 0.0])
                    rel_yaw, rel_pitch = float(rel[0]), float(rel[1])
                else:
                    self_obs = obs.get("self", {})
                    cur_pitch = float(self_obs.get("view_angles", [0.0, 0.0])[0])
                    cur_yaw = float(self_obs.get("world_yaw", self_obs.get("view_angles", [0.0, 0.0])[1]))
                    patrol_wp = obs.get("patrol_waypoint", [-1255.0, 600.0, 1.5707963])
                    leg_yaw = float(patrol_wp[2])
                    rel_yaw = (leg_yaw - float(cur_yaw) + np.pi) % (2.0 * np.pi) - np.pi
                    rel_pitch = 0.02 - float(cur_pitch)
                
                # Check target threshold
                yaw_aligned = abs(rel_yaw) < 0.05
                pitch_aligned = abs(rel_pitch) < 0.03
                if yaw_aligned and pitch_aligned:
                    break
                
                # Proportional mouse nudge with minimum step to prevent stall
                mx = int(np.clip(-rel_yaw * 800.0, -220, 220))
                my = int(np.clip(rel_pitch * 800.0, -140, 140))
                if not yaw_aligned and mx == 0:
                    mx = -1 if rel_yaw > 0 else 1
                if not pitch_aligned and my == 0:
                    my = 1 if rel_pitch > 0 else -1
                    
                self.game_controller.mouse.move(mx, my)
                time.sleep(0.015)
        
        # Could inject "kill" command to respawn
        # self._send_console_command("kill")
    
    def _reset_fire_counters(self):
        """Clear all per-engagement firing counters (tap cooldown, spray burst, recoil, cycle)."""
        self.burst_ticks = 0
        self.spray_ticks = 0
        self.active_burst_mode = None
        self.recoil_ticks = 0
        self.trigger_cooldown = 0
        self.cycle_shots = 0

    def _get_observation(self) -> Dict:
        """Get observation from memory reader."""
        obs = self.memory_reader.get_observation(
            burst_ticks=getattr(self, "burst_ticks", 0),
            recoil_ticks=getattr(self, "recoil_ticks", 0),
        )
        if obs is None:
            # Return zero observation if reading failed
            return self._zero_observation()
        return obs
    
    def _zero_observation(self) -> Dict:
        """Return zero-filled observation."""
        return {
            'self': {
                'health': 0.0, 'armor': 0.0, 'ammo': 0.0,
                'origin': [0, 0, 0], 'view_angles': [0, 0], 'velocity': [0, 0, 0]
            },
            'enemy': {
                'relative_angles': [0, 0], 'distance': 1.0, 'visible': 0.0, 'health': 0.0
            },
            'lidar': [1.0, 1.0, 1.0],
            'valid': False,
            'tick': int(time.time() * 1000)
        }
    
    def _calculate_reward(self, obs: Dict) -> tuple:
        """Calculate dense reward for RL agent with anti-exploit fixes."""
        info = {"kills": self.total_kills, "deaths": self.total_deaths}
        if not obs.get('valid', True):
            return 0.0, info
            
        reward = 0.0
        
        self_info = obs.get('self', {})
        enemy_info = obs.get('enemy', {})
        
        health = self_info.get('health', 0) * 100
        ammo = self_info.get('ammo', 0) * 30
        enemy_dist = enemy_info.get('distance', 1.0) * 3000
        enemy_visible = enemy_info.get('visible', 0) > 0.5
        rel_yaw = enemy_info.get('relative_angles', [0, 0])[0]
        rel_pitch = enemy_info.get('relative_angles', [0, 0])[1]
        
        # Get absolute pitch for ceiling penalty (view_angles is [pitch, yaw])
        view_angles = self_info.get('view_angles', [0, 0])
        abs_pitch = abs(float(view_angles[0])) if len(view_angles) > 0 else 0.0
        
        # Primary reward shaping (delta angle, lock-on, and trigger discipline)
        # is computed in CS16GymEnv.step() on Linux where action & state history reside.
        if health < self.prev_health:
            reward -= (self.prev_health - health) * 0.05
        
        # Update tracking
        self.prev_health = health
        self.prev_enemy_dist = enemy_dist
        self.prev_ammo = ammo
        
        return reward, info
    
    def _check_terminated(self, obs: Dict) -> bool:
        """Check if episode should terminate."""
        if not obs.get('valid', True):
            return False  # Do not terminate on memory read failures
            
        self_info = obs.get('self', {})
        health = self_info.get('health', 0) * 100
        
        # Terminate if dead
        if health <= 0:
            self.total_deaths += 1
            return True
        
        return False


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Allow custom host/port via command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5555)
    args = parser.parse_args()
    
    server = BridgeServer(host=args.host, port=args.port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()

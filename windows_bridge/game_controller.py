"""
Game input controller for CS 1.6 using pynput.
Emulates keyboard and mouse input for the RL agent.
Run as Administrator on Windows.
"""

import time
import threading
import logging
import ctypes
from ctypes import wintypes
from typing import Dict, List, Optional
from dataclasses import dataclass
from pynput import keyboard, mouse
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import numpy as np

logger = logging.getLogger(__name__)


# DirectInput / SDL hardware scan codes (layout-independent: works on both EN and RU keyboard layouts)
SCAN_CODES = {
    'forward': 0x11,   # W (Ц)
    'back': 0x1F,      # S (Ы)
    'left': 0x1E,      # A (Ф)
    'right': 0x20,     # D (В)
    'reload': 0x13,    # R (К)
    'use': 0x12,       # E (У)
    'jump': 0x39,      # Space
    'duck': 0x1D,      # Left Ctrl
    'buyammo1': 0x33,  # , (Б)
    'slot1': 0x02,     # 1 (Primary rifle)
}


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

@dataclass
class ActionState:
    """Current state of all inputs."""
    move_forward: float = 0.0    # -1 to 1 (S to W)
    move_strafe: float = 0.0     # -1 to 1 (A to D)
    delta_yaw: float = 0.0       # Mouse X delta
    delta_pitch: float = 0.0     # Mouse Y delta
    fire: bool = False           # Left click
    jump: bool = False           # Space
    duck: bool = False           # Ctrl
    reload: bool = False         # R
    use: bool = False            # E

class GameController:
    def __init__(self, sensitivity: float = 1.0, mouse_smoothing: float = 0.5):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        
        self.sensitivity = sensitivity
        self.mouse_smoothing = mouse_smoothing
        
        # Current key states
        self.pressed_keys: set = set()
        self.current_action = ActionState()
        self.target_action = ActionState()
        
        # Key mappings
        self.key_map = {
            'forward': 'w',
            'back': 's',
            'left': 'a',
            'right': 'd',
            'jump': Key.space,
            'duck': Key.ctrl_l,
            'reload': 'r',
            'use': 'e',
        }
        
        # Smoothing thread and subpixel mouse accumulator
        self.running = False
        self.smooth_thread: Optional[threading.Thread] = None
        self.mouse_acc_x = 0.0
        self.mouse_acc_y = 0.0
        
        # Fire rate limiting (tap fire cooldown)
        self.last_fire_time = 0.0
        self.fire_cooldown = 0.28  # 280ms = zero upward recoil climb on AK-47 single taps
        
        # Pause state and hardware hotkey polling (F9 / F8 / Pause)
        self.is_paused = False
        self.pause_hotkey_was_pressed = False
        self.last_pause_toggle = 0.0
        self.user32 = getattr(ctypes, 'windll', None)
        if self.user32:
            self.user32 = self.user32.user32
        
    def start(self):
        """Start the smoothing thread."""
        self.running = True
        self.smooth_thread = threading.Thread(target=self._smoothing_loop, daemon=True)
        self.smooth_thread.start()
        logger.info("Game controller started. Press F9, F8 or Pause to Pause/Resume bot.")
            
    def toggle_pause(self):
        """Toggle pause state with debounce cooldown and release all inputs."""
        now = time.time()
        if now - self.last_pause_toggle < 0.4:
            return  # Ignore rapid bounce triggers
        self.last_pause_toggle = now

        self.is_paused = not self.is_paused
        self.release_all()
        if self.is_paused:
            print("\n" + "=" * 55)
            print(" [ПАУЗА] Бот заморожен! У вас полное ручное управление.")
            print(" Нажмите F9 или F8 еще раз, чтобы возобновить обучение.")
            print("=" * 55 + "\n")
            logger.info("Bot PAUSED by user via hotkey")
        else:
            print("\n" + "=" * 55)
            print(" [СТАРТ] Обучение возобновлено! Управление передано боту.")
            print("=" * 55 + "\n")
            logger.info("Bot RESUMED by user via hotkey")
    
    def stop(self):
        """Stop and release all keys."""
        self.running = False
        self.release_all()
        if self.smooth_thread:
            self.smooth_thread.join(timeout=1.0)
        logger.info("Game controller stopped")
    
    def set_action(self, action: Dict[str, float]):
        """
        Set target action from RL agent.
        action: dict with keys [move_forward, move_strafe, delta_yaw, delta_pitch, fire, jump, duck, reload, use]
        Values normalized to [-1, 1] or [0, 1] for fire.
        """
        if self.is_paused:
            return
            
        self.target_action.move_forward = np.clip(action.get('move_forward', 0), -1, 1)
        self.target_action.move_strafe = np.clip(action.get('move_strafe', 0), -1, 1)
        self.target_action.delta_yaw = np.clip(action.get('delta_yaw', 0), -1, 1)
        self.target_action.delta_pitch = np.clip(action.get('delta_pitch', 0), -1, 1)
        self.target_action.fire = action.get('fire', 0) > 0.5
        self.target_action.jump = action.get('jump', 0) > 0.5
        self.target_action.duck = action.get('duck', 0) > 0.5
        self.target_action.reload = action.get('reload', 0) > 0.5
        self.target_action.use = action.get('use', 0) > 0.5
    
    def _smoothing_loop(self):
        """Smooth transitions between actions at 100Hz."""
        dt = 0.01  # 100Hz
        while self.running:
            start = time.perf_counter()
            
            # Direct Windows hardware polling for F9 (0x78) or F8 (0x77)
            if self.user32:
                try:
                    key_down = bool(
                        (self.user32.GetAsyncKeyState(0x78) & 0x8000) or
                        (self.user32.GetAsyncKeyState(0x77) & 0x8000)
                    )
                    if key_down and not self.pause_hotkey_was_pressed:
                        self.toggle_pause()
                    self.pause_hotkey_was_pressed = key_down
                except Exception:
                    pass

            if self.is_paused:
                elapsed = time.perf_counter() - start
                time.sleep(max(0.01, dt - elapsed))
                continue
            
            # Smooth movement keys
            self._smooth_movement(dt)
            
            # Smooth mouse movement
            self._smooth_mouse(dt)
            
            # Handle discrete actions
            self._handle_discrete_actions()
            
            elapsed = time.perf_counter() - start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
    
    def _smooth_movement(self, dt: float):
        """Smoothly transition movement keys."""
        # Forward/Back (0.15 threshold for responsive WASD movement in Phase 2+)
        self._update_key('forward', 'back', self.target_action.move_forward, 
                         self.current_action.move_forward, 0.15, dt)
        
        # Strafe Left/Right
        self._update_key('left', 'right', self.target_action.move_strafe,
                         self.current_action.move_strafe, 0.15, dt)
    
    def _update_key(self, pos_key: str, neg_key: str, target: float, current: float, 
                    threshold: float, dt: float):
        """Update a pair of opposite keys based on target value."""
        # Exponential smoothing
        alpha = 1.0 - np.exp(-dt * 30)  # Fast response
        new_current = current + (target - current) * alpha
        
        # Determine which keys to press
        press_pos = new_current > threshold
        press_neg = new_current < -threshold
        
        # Update key states using hardware scan codes (works regardless of EN/RU keyboard layout)
        if press_pos and pos_key not in self.pressed_keys:
            self.press_scancode(pos_key)
            self.pressed_keys.add(pos_key)
        elif not press_pos and pos_key in self.pressed_keys:
            self.release_scancode(pos_key)
            self.pressed_keys.discard(pos_key)
        
        if press_neg and neg_key not in self.pressed_keys:
            self.press_scancode(neg_key)
            self.pressed_keys.add(neg_key)
        elif not press_neg and neg_key in self.pressed_keys:
            self.release_scancode(neg_key)
            self.pressed_keys.discard(neg_key)
        
        # Update current (for next frame)
        if pos_key == 'forward':
            self.current_action.move_forward = new_current
        elif pos_key == 'left':
            self.current_action.move_strafe = new_current
    
    def _smooth_mouse(self, dt: float):
        """Move mouse using subpixel accumulation (100% Pure AI aim calibrated to trained network weights)."""
        base_px_per_sec = 460.0 * self.sensitivity
        dyaw = float(self.target_action.delta_yaw)
        dpitch = float(self.target_action.delta_pitch)
        yaw_scale = 0.28 + 0.72 * (abs(dyaw) ** 1.30)
        pitch_scale = 0.30 + 0.70 * (abs(dpitch) ** 1.25)
        yaw_accel = 1.0 + 4.5 * (max(0.0, (abs(dyaw) - 0.55) / 0.45) ** 2)
        pitch_accel = 1.0 + 2.0 * (max(0.0, (abs(dpitch) - 0.55) / 0.45) ** 2)

        # Positive delta_yaw turns left (towards rel_yaw > 0); positive delta_pitch turns down (towards rel_pitch > 0)
        step_x = -dyaw * base_px_per_sec * yaw_scale * yaw_accel * dt
        step_y = dpitch * (320.0 * self.sensitivity) * pitch_scale * pitch_accel * dt
        
        self.mouse_acc_x += step_x
        self.mouse_acc_y += step_y
        
        move_x = int(self.mouse_acc_x)
        move_y = int(self.mouse_acc_y)
        
        if move_x != 0 or move_y != 0:
            self.mouse.move(move_x, move_y)
            self.mouse_acc_x -= move_x
            self.mouse_acc_y -= move_y
    
    def _handle_discrete_actions(self):
        """Handle discrete actions (fire, jump, duck, etc.)."""
        # Fire: hold left mouse button continuously while target_action.fire is True
        # (enables real 2-tap into full-auto crouch spray in CS 1.6)
        if self.target_action.fire:
            if not self.current_action.fire:
                self.mouse.press(Button.left)
                self.current_action.fire = True
        else:
            if self.current_action.fire:
                self.mouse.release(Button.left)
                self.current_action.fire = False
        
        # Jump
        self._handle_key('jump', self.target_action.jump, self.current_action.jump)
        
        # Duck
        self._handle_key('duck', self.target_action.duck, self.current_action.duck)
        
        # Reload (tap)
        if self.target_action.reload and not self.current_action.reload:
            self.tap_scancode('reload')
            self.current_action.reload = True
        elif not self.target_action.reload:
            self.current_action.reload = False
        
        # Use (tap)
        if self.target_action.use and not self.current_action.use:
            self.tap_scancode('use')
            self.current_action.use = True
        elif not self.target_action.use:
            self.current_action.use = False
    
    def press_scancode(self, key_name: str):
        """Press a key using hardware scan code (KEYEVENTF_SCANCODE = 0x0008)."""
        scan = SCAN_CODES.get(key_name)
        if scan and self.user32:
            inp = _INPUT(type=1)
            inp.u.ki = _KEYBDINPUT(0, scan, 0x0008, 0, 0)
            self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def release_scancode(self, key_name: str):
        """Release a key using hardware scan code (KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP = 0x000A)."""
        scan = SCAN_CODES.get(key_name)
        if scan and self.user32:
            inp = _INPUT(type=1)
            inp.u.ki = _KEYBDINPUT(0, scan, 0x0008 | 0x0002, 0, 0)
            self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def tap_scancode(self, key_name: str, hold_sec: float = 0.02):
        """Tap (press + release) a key by hardware scan code."""
        self.press_scancode(key_name)
        time.sleep(hold_sec)
        self.release_scancode(key_name)

    def _handle_key(self, key_name: str, target: bool, current: bool):
        """Handle a single key press/release."""
        if target and not current:
            self.press_scancode(key_name)
            self.pressed_keys.add(key_name)
            setattr(self.current_action, key_name, True)
        elif not target and current:
            self.release_scancode(key_name)
            self.pressed_keys.discard(key_name)
            setattr(self.current_action, key_name, False)
    
    def release_all(self):
        """Release all keys and mouse buttons."""
        for key_name in list(SCAN_CODES.keys()):
            try:
                self.release_scancode(key_name)
            except Exception:
                pass
        self.pressed_keys.clear()
        
        # Release mouse
        try:
            self.mouse.release(Button.left)
            self.mouse.release(Button.right)
        except:
            pass
        
        # Reset action states
        self.current_action = ActionState()
        self.target_action = ActionState()
        self.mouse_acc_x = 0.0
        self.mouse_acc_y = 0.0
        self.last_fire_time = 0.0
    
    def center_mouse(self):
        """Move mouse to center of screen (useful for reset)."""
        # Get screen size
        import ctypes
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        self.mouse.position = (screen_w // 2, screen_h // 2)
        self.last_mouse_move = (0, 0)


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    controller = GameController(sensitivity=2.0)
    
    print("Testing game controller...")
    print("Press Ctrl+C to stop")
    
    try:
        controller.start()
        
        # Test movement
        print("Moving forward...")
        controller.set_action({'move_forward': 1.0})
        time.sleep(2)
        
        print("Strafing right...")
        controller.set_action({'move_forward': 0.0, 'move_strafe': 1.0})
        time.sleep(1)
        
        print("Looking right...")
        controller.set_action({'move_strafe': 0.0, 'delta_yaw': 0.5})
        time.sleep(1)
        
        print("Firing...")
        controller.set_action({'delta_yaw': 0.0, 'fire': 1.0})
        time.sleep(0.5)
        
        print("Stopping...")
        controller.set_action({})
        time.sleep(0.5)
        
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        print("Test complete")

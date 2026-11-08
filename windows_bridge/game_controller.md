# Windows Bridge: Game Controller (game_controller.py)

```python
"""
Game input controller for CS 1.6 using pynput.
Emulates keyboard and mouse input for the RL agent.
Run as Administrator on Windows.
"""

import time
import threading
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pynput import keyboard, mouse
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import numpy as np

logger = logging.getLogger(__name__)

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
        
        # Smoothing thread
        self.running = False
        self.smooth_thread: Optional[threading.Thread] = None
        self.last_mouse_move = (0, 0)
        
    def start(self):
        """Start the smoothing thread."""
        self.running = True
        self.smooth_thread = threading.Thread(target=self._smoothing_loop, daemon=True)
        self.smooth_thread.start()
        logger.info("Game controller started")
    
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
        # Forward/Back
        self._update_key('forward', 'back', self.target_action.move_forward, 
                         self.current_action.move_forward, 0.1, dt)
        
        # Strafe Left/Right
        self._update_key('left', 'right', self.target_action.move_strafe,
                         self.current_action.move_strafe, 0.1, dt)
    
    def _update_key(self, pos_key: str, neg_key: str, target: float, current: float, 
                    threshold: float, dt: float):
        """Update a pair of opposite keys based on target value."""
        # Exponential smoothing
        alpha = 1.0 - np.exp(-dt * 30)  # Fast response
        new_current = current + (target - current) * alpha
        
        # Determine which keys to press
        press_pos = new_current > threshold
        press_neg = new_current < -threshold
        
        # Update key states
        if press_pos and pos_key not in self.pressed_keys:
            self.keyboard.press(self.key_map[pos_key])
            self.pressed_keys.add(pos_key)
        elif not press_pos and pos_key in self.pressed_keys:
            self.keyboard.release(self.key_map[pos_key])
            self.pressed_keys.discard(pos_key)
        
        if press_neg and neg_key not in self.pressed_keys:
            self.keyboard.press(self.key_map[neg_key])
            self.pressed_keys.add(neg_key)
        elif not press_neg and neg_key in self.pressed_keys:
            self.keyboard.release(self.key_map[neg_key])
            self.pressed_keys.discard(neg_key)
        
        # Update current (for next frame)
        if pos_key == 'forward':
            self.current_action.move_forward = new_current
        elif pos_key == 'left':
            self.current_action.move_strafe = new_current
    
    def _smooth_mouse(self, dt: float):
        """Smooth mouse movement for aiming."""
        # Convert normalized [-1, 1] to pixels per frame
        max_pixels_per_frame = 50 * self.sensitivity
        
        target_x = self.target_action.delta_yaw * max_pixels_per_frame
        target_y = self.target_action.delta_pitch * max_pixels_per_frame
        
        # Smooth mouse movement
        alpha = 1.0 - np.exp(-dt * 20)
        smooth_x = self.last_mouse_move[0] + (target_x - self.last_mouse_move[0]) * alpha
        smooth_y = self.last_mouse_move[1] + (target_y - self.last_mouse_move[1]) * alpha
        
        # Move mouse relative
        if abs(smooth_x) > 0.5 or abs(smooth_y) > 0.5:
            self.mouse.move(int(smooth_x), int(smooth_y))
            self.last_mouse_move = (smooth_x, smooth_y)
        else:
            self.last_mouse_move = (0, 0)
    
    def _handle_discrete_actions(self):
        """Handle discrete actions (fire, jump, duck, etc.)."""
        # Fire (left click)
        if self.target_action.fire and not self.current_action.fire:
            self.mouse.press(Button.left)
            self.current_action.fire = True
        elif not self.target_action.fire and self.current_action.fire:
            self.mouse.release(Button.left)
            self.current_action.fire = False
        
        # Jump
        self._handle_key('jump', self.target_action.jump, self.current_action.jump)
        
        # Duck
        self._handle_key('duck', self.target_action.duck, self.current_action.duck)
        
        # Reload (tap)
        if self.target_action.reload and not self.current_action.reload:
            self.keyboard.tap(self.key_map['reload'])
            self.current_action.reload = True
        elif not self.target_action.reload:
            self.current_action.reload = False
        
        # Use (tap)
        if self.target_action.use and not self.current_action.use:
            self.keyboard.tap(self.key_map['use'])
            self.current_action.use = True
        elif not self.target_action.use:
            self.current_action.use = False
    
    def _handle_key(self, key_name: str, target: bool, current: bool):
        """Handle a single key press/release."""
        key = self.key_map[key_name]
        if target and not current:
            self.keyboard.press(key)
            self.pressed_keys.add(key_name)
            setattr(self.current_action, key_name, True)
        elif not target and current:
            self.keyboard.release(key)
            self.pressed_keys.discard(key_name)
            setattr(self.current_action, key_name, False)
    
    def release_all(self):
        """Release all keys and mouse buttons."""
        for key_name in list(self.pressed_keys):
            key = self.key_map.get(key_name)
            if key:
                try:
                    self.keyboard.release(key)
                except:
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
        self.last_mouse_move = (0, 0)
    
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
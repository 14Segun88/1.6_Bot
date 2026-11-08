# Windows Bridge: Bridge Server (bridge_server.py)

```python
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
        self.prev_health = 100
        self.prev_enemy_dist = 3000.0
        self.prev_ammo = 30
        self.shots_fired = 0
        self.hits_dealt = 0
        
    def start(self):
        """Start the bridge server."""
        # Initialize ZeroMQ REP socket
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        
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
            self.socket.close()
        self.context.term()
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
                # Timeout - continue loop
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
        elif req_type == "ping":
            return {"status": "ok", "tick": int(time.time() * 1000)}
        else:
            return {"error": f"Unknown request type: {req_type}"}
    
    def _handle_reset(self, request: Dict) -> Dict:
        """Handle environment reset."""
        logger.info("Reset requested")
        
        # Reset game state (may need to send console commands)
        self._reset_game_state()
        
        # Reset tracking variables
        self.episode_step = 0
        self.last_reward = 0.0
        self.prev_health = 100
        self.prev_enemy_dist = 3000.0
        self.prev_ammo = 30
        self.shots_fired = 0
        self.hits_dealt = 0
        
        # Get initial observation
        obs = self._get_observation()
        self.current_observation = obs
        
        return {
            "observation": obs,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {"kills": 0, "deaths": 0}
        }
    
    def _handle_step(self, request: Dict) -> Dict:
        """Handle environment step with action from agent."""
        action = request.get("action", [0, 0, 0, 0, 0])
        
        # Convert action array to dict
        action_dict = {
            "move_forward": float(action[0]),
            "move_strafe": float(action[1]),
            "delta_yaw": float(action[2]),
            "delta_pitch": float(action[3]),
            "fire": float(action[4]),
        }
        
        # Send action to game
        self.game_controller.set_action(action_dict)
        
        # Wait for game to process (1-2 ticks)
        time.sleep(0.02)  # ~50Hz
        
        # Get new observation
        obs = self._get_observation()
        self.current_observation = obs
        
        # Calculate reward
        reward, info = self._calculate_reward(obs)
        self.last_reward = reward
        self.episode_step += 1
        
        # Check termination
        terminated = self._check_terminated(obs)
        truncated = self.episode_step >= 10000  # Max episode length
        
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
        """Reset the game to initial state."""
        # This is tricky with external client - options:
        # 1. Send console commands via memory writing (complex)
        # 2. Rely on map reset (mp_restartgame)
        # 3. Just respawn (if deathmatch mode)
        
        # For now, just release all keys and wait
        self.game_controller.release_all()
        time.sleep(0.5)
        
        # Could inject "kill" command to respawn
        # self._send_console_command("kill")
    
    def _get_observation(self) -> Dict:
        """Get observation from memory reader."""
        obs = self.memory_reader.get_observation()
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
            'tick': int(time.time() * 1000)
        }
    
    def _calculate_reward(self, obs: Dict) -> tuple:
        """Calculate dense reward for RL agent."""
        reward = 0.0
        info = {"kills": self.total_kills, "deaths": self.total_deaths}
        
        self_info = obs.get('self', {})
        enemy_info = obs.get('enemy', {})
        
        health = self_info.get('health', 0) * 100
        ammo = self_info.get('ammo', 0) * 30
        enemy_dist = enemy_info.get('distance', 1.0) * 3000
        enemy_visible = enemy_info.get('visible', 0) > 0.5
        rel_yaw = enemy_info.get('relative_angles', [0, 0])[0]
        rel_pitch = enemy_info.get('relative_angles', [0, 0])[1]
        
        # 1. Aim reward (cosine of angle difference)
        if enemy_visible:
            aim_reward = np.cos(rel_yaw) * np.cos(rel_pitch)
            reward += aim_reward * 0.1
        
        # 2. Approach reward (get closer to enemy)
        if enemy_dist < self.prev_enemy_dist and enemy_visible:
            reward += (self.prev_enemy_dist - enemy_dist) * 0.001
        
        # 3. Health penalty (took damage)
        if health < self.prev_health:
            reward -= (self.prev_health - health) * 0.05
        
        # 4. Ammo tracking (detect shots fired)
        if ammo < self.prev_ammo:
            shots = self.prev_ammo - ammo
            self.shots_fired += shots
            reward -= shots * 0.02  # Penalty for shooting
        
        # 5. Hit detection (would need game event hook - placeholder)
        # For now, estimate from health changes of enemy
        # This requires reading enemy health over time
        
        # Update tracking
        self.prev_health = health
        self.prev_enemy_dist = enemy_dist
        self.prev_ammo = ammo
        
        return reward, info
    
    def _check_terminated(self, obs: Dict) -> bool:
        """Check if episode should terminate."""
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
```

## Running the Bridge Server

```bash
# On Windows (run as Administrator)
cd Cs 1.6_Bot/windows_bridge
pip install -r requirements_windows.txt
python bridge_server.py --host 0.0.0.0 --port 5555
```

## Network Protocol

**Request (Linux → Windows):**
```json
// Reset
{"type": "reset"}

// Step
{"type": "step", "action": [0.8, 0.0, -0.1, 0.0, 1.0]}

// Get observation only
{"type": "observation"}

// Ping
{"type": "ping"}
```

**Response (Windows → Linux):**
```json
// Step/Reset response
{
  "observation": {...},
  "reward": 0.15,
  "terminated": false,
  "truncated": false,
  "info": {"kills": 2, "deaths": 0}
}

// Ping response
{"status": "ok", "tick": 1234567890}
```

## Firewall Note

Allow port 5555 on Windows Firewall for the Linux machine's IP:
```powershell
New-NetFirewallRule -DisplayName "CS16 RL Bridge" -Direction Inbound -LocalPort 5555 -Protocol TCP -Action Allow -RemoteAddress <LINUX_IP>
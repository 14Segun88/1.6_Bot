# Windows Bridge: Memory Reader (memory_reader.py)

```python
"""
Memory reader for CS 1.6 hl.exe using pymem.
Reads player state, enemy positions, and game info.
Run as Administrator on Windows.
"""

import pymem
import pymem.process
import struct
import time
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    def distance_to(self, other: 'Vector3') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

@dataclass
class PlayerState:
    address: int
    origin: Vector3
    view_angles: Vector3
    health: int
    armor: int
    team: int
    flags: int
    velocity: Vector3
    view_offset: Vector3
    punch_angle: Vector3
    ammo_clip: int
    is_alive: bool
    is_local: bool = False

class MemoryReader:
    # Offsets (UPDATE THESE FOR YOUR hl.exe BUILD!)
    OFFSETS = {
        'origin': 0x44,
        'view_angles': 0x4C,
        'health': 0xFC,
        'armor': 0x100,
        'team': 0xF0,
        'flags': 0x104,
        'velocity': 0x7C,
        'view_offset': 0x64,
        'punch_angle': 0x68,
        'active_weapon': 0x1268,
        'weapon_clip': 0x1274,
    }
    
    # Signatures for finding base addresses
    SIGNATURES = {
        'local_player': b'8B 0D ?? ?? ?? ?? 83 F9 FF 74 07',
        'entity_list': b'8B 0D ?? ?? ?? ?? 85 C9 74 0A 8B 01 FF 50 0C',
    }
    
    def __init__(self, process_name: str = "hl.exe"):
        self.process_name = process_name
        self.pm: Optional[pymem.Pymem] = None
        self.client_base: int = 0
        self.local_player_ptr: int = 0
        self.entity_list_ptr: int = 0
        self._connect()
    
    def _connect(self) -> bool:
        """Attach to hl.exe process and find base addresses."""
        try:
            self.pm = pymem.Pymem(self.process_name)
            logger.info(f"Attached to {self.process_name} (PID: {self.pm.process_id})")
            
            # Get hl.exe module base
            client_module = pymem.process.module_from_name(self.pm.process_handle, "hl.exe")
            self.client_base = client_module.lpBaseOfDll
            logger.info(f"hl.exe base address: 0x{self.client_base:X}")
            
            # Find pointers via signature scanning
            self.local_player_ptr = self._find_pointer(self.SIGNATURES['local_player'], 2)
            self.entity_list_ptr = self._find_pointer(self.SIGNATURES['entity_list'], 2)
            
            if not self.local_player_ptr or not self.entity_list_ptr:
                logger.error("Failed to find critical pointers!")
                return False
            
            logger.info(f"Local player ptr: 0x{self.local_player_ptr:X}")
            logger.info(f"Entity list ptr: 0x{self.entity_list_ptr:X}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def _find_pointer(self, pattern: bytes, offset: int) -> int:
        """Scan for signature and return dereferenced pointer."""
        try:
            # Simple pattern scan (implement properly for production)
            pattern_bytes = self._parse_pattern(pattern)
            address = self._pattern_scan(self.client_base, pattern_bytes)
            if address:
                # Read the pointer at offset
                ptr_addr = address + offset
                ptr = self.pm.read_int(ptr_addr)
                return ptr
        except Exception as e:
            logger.error(f"Pattern scan failed: {e}")
        return 0
    
    def _parse_pattern(self, pattern: bytes) -> bytes:
        """Convert IDA-style pattern to bytes with wildcards."""
        parts = pattern.split()
        result = bytearray()
        for part in parts:
            if part == b'??':
                result.append(0xCC)  # Wildcard marker
            else:
                result.append(int(part, 16))
        return bytes(result)
    
    def _pattern_scan(self, base: int, pattern: bytes) -> int:
        """Scan memory for pattern. Simplified - use proper scanner in production."""
        # This is a placeholder - implement proper pattern scanning
        # For now, return hardcoded offsets (UPDATE THESE!)
        if pattern.startswith(b'\x8B\x0D'):
            # These are example offsets - REPLACE WITH ACTUAL VALUES
            return base + 0x123456  # Example offset
        return 0
    
    def read_vector3(self, address: int) -> Vector3:
        """Read 3 floats (12 bytes) as Vector3."""
        try:
            data = self.pm.read_bytes(address, 12)
            x, y, z = struct.unpack('<fff', data)
            return Vector3(x, y, z)
        except:
            return Vector3()
    
    def read_player(self, address: int, is_local: bool = False) -> Optional[PlayerState]:
        """Read full player state from entity address."""
        try:
            if not address or address < 0x10000:
                return None
            
            # Check if player is valid/alive
            health = self.pm.read_int(address + self.OFFSETS['health'])
            if health <= 0 or health > 1000:
                return None
            
            team = self.pm.read_int(address + self.OFFSETS['team'])
            if team not in (1, 2):  # Only T (1) and CT (2)
                return None
            
            origin = self.read_vector3(address + self.OFFSETS['origin'])
            view_angles = self.read_vector3(address + self.OFFSETS['view_angles'])
            velocity = self.read_vector3(address + self.OFFSETS['velocity'])
            view_offset = self.read_vector3(address + self.OFFSETS['view_offset'])
            punch_angle = self.read_vector3(address + self.OFFSETS['punch_angle'])
            
            armor = self.pm.read_int(address + self.OFFSETS['armor'])
            flags = self.pm.read_int(address + self.OFFSETS['flags'])
            
            # Read ammo
            weapon_handle = self.pm.read_int(address + self.OFFSETS['active_weapon'])
            ammo_clip = 0
            if weapon_handle:
                # Weapon handle is index, need to resolve from entity list
                weapon_addr = self.get_entity_by_index(weapon_handle & 0xFFF)
                if weapon_addr:
                    ammo_clip = self.pm.read_int(weapon_addr + self.OFFSETS['weapon_clip'])
            
            return PlayerState(
                address=address,
                origin=origin,
                view_angles=view_angles,
                health=health,
                armor=armor,
                team=team,
                flags=flags,
                velocity=velocity,
                view_offset=view_offset,
                punch_angle=punch_angle,
                ammo_clip=ammo_clip,
                is_alive=health > 0,
                is_local=is_local
            )
        except Exception as e:
            logger.debug(f"Failed to read player at 0x{address:X}: {e}")
            return None
    
    def get_entity_by_index(self, index: int) -> int:
        """Get entity address from entity list by index."""
        try:
            if index <= 0 or index > 2048:
                return 0
            entry_addr = self.entity_list_ptr + index * 0x10
            return self.pm.read_int(entry_addr)
        except:
            return 0
    
    def get_local_player(self) -> Optional[PlayerState]:
        """Get local player state."""
        try:
            local_addr = self.pm.read_int(self.local_player_ptr)
            return self.read_player(local_addr, is_local=True)
        except:
            return None
    
    def get_enemies(self, local_team: int) -> List[PlayerState]:
        """Get all alive enemy players."""
        enemies = []
        for i in range(1, 33):  # Max 32 players
            if i == 1:  # Skip local player (usually index 1)
                continue
            entity_addr = self.get_entity_by_index(i)
            if entity_addr:
                player = self.read_player(entity_addr)
                if player and player.is_alive and player.team != local_team:
                    enemies.append(player)
        return enemies
    
    def get_observation(self) -> Optional[Dict]:
        """Get full observation for RL agent."""
        local = self.get_local_player()
        if not local or not local.is_alive:
            return None
        
        enemies = self.get_enemies(local.team)
        
        # Find closest visible enemy
        closest_enemy = None
        min_dist = float('inf')
        
        for enemy in enemies:
            dist = local.origin.distance_to(enemy.origin)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        
        # Calculate relative angles to closest enemy
        rel_yaw = 0.0
        rel_pitch = 0.0
        visible = False
        
        if closest_enemy:
            # Calculate angle from local to enemy
            delta = closest_enemy.origin - local.origin
            # Simplified - need proper angle calculation
            rel_yaw = np.arctan2(delta.y, delta.x) - local.view_angles.y
            rel_pitch = -np.arctan2(delta.z, np.sqrt(delta.x**2 + delta.y**2)) - local.view_angles.x
            
            # Normalize angles
            rel_yaw = (rel_yaw + np.pi) % (2 * np.pi) - np.pi
            rel_pitch = np.clip(rel_pitch, -np.pi/2, np.pi/2)
            
            # TODO: Implement line of sight trace
            visible = True  # Placeholder
        
        # Lidar (simplified - raycast forward, left, right)
        lidar = [1000.0, 1000.0, 1000.0]  # Placeholder
        
        return {
            'self': {
                'health': local.health / 100.0,
                'armor': local.armor / 100.0,
                'ammo': local.ammo_clip / 30.0,
                'origin': local.origin.to_list(),
                'view_angles': [local.view_angles.x, local.view_angles.y],
                'velocity': local.velocity.to_list(),
            },
            'enemy': {
                'relative_angles': [rel_yaw, rel_pitch],
                'distance': min(min_dist / 3000.0, 1.0) if closest_enemy else 1.0,
                'visible': float(visible),
                'health': closest_enemy.health / 100.0 if closest_enemy else 0.0,
            },
            'lidar': lidar,
            'tick': int(time.time() * 1000),
        }
    
    def close(self):
        if self.pm:
            self.pm.close_process()


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reader = MemoryReader()
    
    if reader.local_player_ptr:
        print("Memory reader initialized successfully!")
        print("Offsets need to be updated for your hl.exe build.")
        print("Use Cheat Engine to find actual offsets.")
    else:
        print("Failed to initialize. Run as Administrator.")
    
    reader.close()
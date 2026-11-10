"""
Precision Memory Reader for CS 1.6 GoldClient.
Uses verified static offsets:
- Local Player: client.dll + 0x199E3C (Pos X,Y,Z + Forward vector)
- Target Dummies: hw.dll + 0x288970 + i * 0xE0 (aim_ai_sandbox dummies)
"""

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import struct
import math
import time
import logging
from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    def distance_to(self, other: 'Vector3') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

@dataclass
class PlayerState:
    address: int
    origin: Vector3
    view_angles: Vector3  # pitch, yaw in radians
    health: int = 100
    armor: int = 100
    team: int = 2        # 2 = CT, 1 = T
    flags: int = 0
    velocity: Vector3 = None
    view_offset: Vector3 = None
    punch_angle: Vector3 = None
    ammo_clip: int = 30
    is_alive: bool = True
    is_local: bool = False

    def __post_init__(self):
        if self.velocity is None: self.velocity = Vector3()
        if self.view_offset is None: self.view_offset = Vector3(0, 0, 17)
        if self.punch_angle is None: self.punch_angle = Vector3()

class MemoryReader:
    # Verified static offsets
    CLIENT_POS_OFFSET = 0x199E3C    # client.dll + 0x199E3C (x, y, z, fx, fy, fz)
    DUMMY_TABLE_OFFSET = 0x288970   # hw.dll + 0x288970 + (idx * 0xE0)

    def __init__(self, process_name: str = "hl.exe"):
        self.process_name = process_name
        self.hProc = None
        self.pid = None
        self.hw_base = None
        self.client_base = None
        self.mp_base = None
        self.last_connect = 0.0
        self.last_yaw = 0.0
        self.zero_read_count = 0
        self.ptr_addr = 0x199E3C  # Compatibility flag
        
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32
        
        # Optional auto-discovered offsets (written by find_offsets.py)
        self.custom_offsets = self._load_custom_offsets()
        
        # Track hits per target slot/grid so both moving bots (Phase 3) and static dummies (Phase 1-2) work seamlessly
        self.target_hit_counts: Dict[int, int] = {}
        self.eliminated_until: Dict[tuple, float] = {}
        self.current_target_key: Optional[tuple] = None
        self.last_counted_shot_time: float = 0.0
        self._local_pos_history: List[tuple] = []
        # Lateral strafe & reactive dodge state per dummy slot
        self._strafe_state: Dict[int, dict] = {}
        self._entvars_Scanned: bool = False
        self._mp_patched: bool = False
        self._server_origins: List[tuple] = []  # [(pev_addr, csbot_addr, init_x, init_y, init_z)]
        self.local_pev_addr: Optional[int] = None
        self.static_enemies: bool = False  # Phase 5/6: Corner-rush + armed bots + instant corner respawn!
        self.real_combat_mode: bool = True  # Default to Phase 8 Live 1v5 Combat (100-0 HP, bots see & shoot AI)
        self.local_real_hp: float = 100.0
        self._prev_local_hp: float = 100.0
        self._pending_player_died: bool = False
        self._pending_engine_kills: int = 0
        self._pending_engine_hits: int = 0
        self._pending_player_real_dmg: float = 0.0

    # Spray recoil compensation (degrees of aim-point lowering per consecutive spray tick).
    # 1 tick = 1 env step (~70-150 ms, i.e. ~1-1.5 M4A1 bullets). Tune by spraying at a wall.
    RECOIL_COMP_DEG_PER_TICK: float = 0.6
    RECOIL_COMP_MAX_DEG: float = 2.4

    # Verified Hull-1 walkable deep corner-pocket positions around all 4 legs of the $2000$ (aim_ai_sandbox.bsp) arena loop:
    # Tuple format: (leg_id, corner_x, corner_y, peek_x, peek_y, corner_name)
    # - Leg 0: Left Corridor (+Y, X ~ -1255, Y: 155 -> 1375) — 4 deep left brick pockets (X=-1395, +65°..+74° left flick) + 3 deep right stone alcoves (X=-1122/-1095, -60°..-76° right flick)
    # - Leg 1: Enemy Spawn (+X, Y ~ 1375, X: -1255 -> -285) — deep right stone corner pocket (-345, 1135, -70° flick)
    # - Leg 2: Right Corridor (-Y, X ~ -285, Y: 1375 -> 155) — 4 deep outer brick pockets (X=-145, +65°..+74° left flick) + 3 deep inner stone alcoves (X=-418/-445, -60°..-76° right flick)
    # - Leg 3: Own Spawn (-X, Y ~ 155, X: -285 -> -1255) — deep left stone corner pocket (-1195, 385, -70° flick)
    AMBUSH_CORNERS = (
        # Leg 0: Left Corridor (moving +Y, deep lateral pockets 1..7)
        (0, -1395.0, 310.0,  -1395.0, 310.0,  "1_L0_BRICK_1"),
        (0, -1122.0, 595.0,  -1122.0, 595.0,  "2_L0_STONE_1"),
        (0, -1395.0, 670.0,  -1395.0, 670.0,  "3_L0_BRICK_2"),
        (0, -1122.0, 915.0,  -1122.0, 915.0,  "4_L0_STONE_2"),
        (0, -1395.0, 990.0,  -1395.0, 990.0,  "5_L0_BRICK_3"),
        (0, -1095.0, 1225.0, -1095.0, 1225.0, "6_L0_STONE_3"),
        (0, -1395.0, 1366.0, -1395.0, 1366.0, "7_L0_BRICK_4"),
        # Leg 1: Enemy Spawn (moving +X, deep behind right corner of Stone Pillar #3)
        (1, -345.0,  1135.0, -345.0,  1135.0, "8_L1_STONE_CORNER"),
        # Leg 2: Right Corridor (moving -Y, symmetric 7 deep lateral pockets)
        (2, -145.0,  1214.0, -145.0,  1214.0, "9_L2_BRICK_1"),
        (2, -418.0,  925.0,  -418.0,  925.0,  "10_L2_STONE_1"),
        (2, -145.0,  854.0,  -145.0,  854.0,  "11_L2_BRICK_2"),
        (2, -418.0,  605.0,  -418.0,  605.0,  "12_L2_STONE_2"),
        (2, -145.0,  534.0,  -145.0,  534.0,  "13_L2_BRICK_3"),
        (2, -445.0,  295.0,  -445.0,  295.0,  "14_L2_STONE_3"),
        (2, -145.0,  177.0,  -145.0,  177.0,  "15_L2_BRICK_4"),
        # Leg 3: Own Spawn (moving -X, deep behind left corner of Stone Pillar #1)
        (3, -1195.0, 385.0,  -1195.0, 385.0,  "16_L3_STONE_CORNER"),
    )

    # Exact BSP wall AABBs (xmin, ymin, xmax, ymax) for 360-degree Line-of-Sight occlusion & Lidar:
    SOLID_WALL_AABBS = (
        # 3 Central grey stone blocks + recessed central divider
        (-1168.0, 335.0, -372.0, 548.0),
        (-1168.0, 652.0, -372.0, 868.0),
        (-1168.0, 972.0, -372.0, 1185.0),
        (-1102.0, 548.0, -438.0, 972.0),
        # 4 Left-corridor outer brick half-walls
        (-1540.0, 215.0,  -1332.0, 272.0),
        (-1540.0, 572.0,  -1332.0, 632.0),
        (-1540.0, 892.0,  -1332.0, 952.0),
        (-1540.0, 1252.0, -1332.0, 1328.0),
        # 4 Right-corridor outer brick half-walls
        (-212.0,  215.0,  0.0,     272.0),
        (-212.0,  572.0,  0.0,     632.0),
        (-212.0,  892.0,  0.0,     952.0),
        (-212.0,  1252.0, 0.0,     1328.0),
    )

    MAX_ACTIVE_CAROUSEL_BOTS: int = 12

    @staticmethod
    def _segment_intersects_aabb(x0: float, y0: float, x1: float, y1: float,
                                 xmin: float, ymin: float, xmax: float, ymax: float,
                                 pad: float = 0.0) -> bool:
        """2D Liang-Barsky segment-AABB intersection test with optional corner clearance padding."""
        if pad != 0.0:
            xmin -= pad
            ymin -= pad
            xmax += pad
            ymax += pad
        dx = x1 - x0
        dy = y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            if abs(p) < 1e-8:
                if q < 0.0:
                    return False
            else:
                t = q / p
                if p < 0.0:
                    if t > t1:
                        return False
                    if t > t0:
                        t0 = t
                else:
                    if t < t0:
                        return False
                    if t < t1:
                        t1 = t
        return t0 <= t1

    def get_loop_leg(self, px: float, py: float) -> int:
        """
        Update and return current leg (0..3) of the 2-corridor arena loop:
        - Leg 0: Left Corridor (+Y, X ~ -1255, Y: 155 -> 1375)
        - Leg 1: Enemy Spawn (+X, Y ~ 1375, X: -1255 -> -285)
        - Leg 2: Right Corridor (-Y, X ~ -285, Y: 1375 -> 155)
        - Leg 3: Own Spawn (-X, Y ~ 155, X: -285 -> -1255)
        """
        if not hasattr(self, "_used_corners_this_lap"):
            self._used_corners_this_lap = set()
        leg = getattr(self, "_loop_leg", None)
        prev_leg = leg
        prev_xy = getattr(self, "_prev_leg_xy", None)
        self._prev_leg_xy = (px, py)
        jumped = prev_xy is not None and math.hypot(px - prev_xy[0], py - prev_xy[1]) > 220.0
        if leg is None or jumped:
            if py < 310.0:
                leg = 0 if px <= -1020.0 else (2 if px >= -520.0 else 3)
            elif py > 1210.0:
                leg = 2 if px >= -520.0 else (0 if px <= -1020.0 else 1)
            else:
                leg = 0 if px < -770.0 else 2
        else:
            if leg == 0 and py >= 1330.0:
                leg = 1
            elif leg == 0 and py <= 195.0 and getattr(self, "_patrol_dir_sign", 1) < 0:
                leg = 3
            elif leg == 1 and px >= -500.0:
                leg = 2
            elif leg == 1 and px <= -1040.0 and getattr(self, "_patrol_dir_sign", 1) < 0:
                leg = 0
            elif leg == 2 and py <= 195.0:
                leg = 3
            elif leg == 2 and py >= 1330.0 and getattr(self, "_patrol_dir_sign", 1) < 0:
                leg = 1
            elif leg == 3 and px <= -1040.0:
                leg = 0
            elif leg == 3 and px >= -500.0 and getattr(self, "_patrol_dir_sign", 1) < 0:
                leg = 2
            # Re-sync if player spawned or moved outside the current leg's corridor
            elif leg == 0 and px > -980.0:
                leg = 2 if py > 300.0 else 3
            elif leg == 2 and px < -560.0:
                leg = 0 if py < 1200.0 else 1
            elif leg == 1 and py < 1100.0:
                leg = 0 if px < -770.0 else (2 if py > 300.0 else 3)
            elif leg == 3 and py > 420.0:
                leg = 0 if px < -770.0 else (2 if py < 1200.0 else 1)

        # When advancing to a new leg of the loop, reset the "used this lap" flag for the far legs
        # ((leg + 2) % 4 and (leg + 3) % 4) so each corner lives strictly ONCE per full lap!
        if prev_leg is not None and leg != prev_leg:
            refresh_legs = {(leg + 2) % 4, (leg + 3) % 4}
            self._used_corners_this_lap = {
                c_idx for c_idx in self._used_corners_this_lap
                if self.AMBUSH_CORNERS[c_idx][0] not in refresh_legs
            }
            logger.info(f"[LOOP LAP] Player advanced Leg {prev_leg} -> Leg {leg} (refreshed legs {sorted(refresh_legs)} for next lap)")

        self._loop_leg = leg
        return leg

    @staticmethod
    def get_leg_waypoint(leg: int, px: float, py: float, dir_sign: int = 1) -> tuple:
        """
        Return (wp_x, wp_y, leg_yaw_rad) for advancing along the given loop leg.
        dir_sign = +1 (clockwise: 0->1->2->3->0) or -1 (counter-clockwise: 0->3->2->1->0).
        """
        if dir_sign >= 0:
            if leg == 0:
                return -1255.0, min(1375.0, max(py + 220.0, 260.0)), math.pi / 2.0
            elif leg == 1:
                return min(-285.0, max(px + 240.0, -1100.0)), 1375.0, 0.0
            elif leg == 2:
                return -285.0, max(155.0, min(py - 220.0, 1260.0)), -math.pi / 2.0
            else:
                return max(-1255.0, min(px - 240.0, -420.0)), 155.0, math.pi
        else:
            if leg == 0:
                return -1255.0, max(155.0, min(py - 220.0, 1260.0)), -math.pi / 2.0
            elif leg == 3:
                return min(-285.0, max(px + 240.0, -1100.0)), 155.0, 0.0
            elif leg == 2:
                return -285.0, min(1375.0, max(py + 220.0, 260.0)), math.pi / 2.0
            else:
                return max(-1255.0, min(px - 240.0, -420.0)), 1375.0, math.pi

    @staticmethod
    def _perimeter_param(px: float, py: float) -> float:
        """
        Map any 2D position (px, py) on $2000$ onto a continuous 1D perimeter coordinate s in [0, 4380)
        going clockwise around the loop:
          - Leg 0 (Left Corridor, X ~ -1255, Y: 155 -> 1375): s in [0, 1220)
          - Leg 1 (Top Spawn,     Y ~ 1375, X: -1255 -> -285): s in [1220, 2190)
          - Leg 2 (Right Corridor,X ~ -285, Y: 1375 -> 155):   s in [2190, 3410)
          - Leg 3 (Bottom Spawn,  Y ~ 155,  X: -285 -> -1255): s in [3410, 4380)
        """
        if py < 280.0 and px > -1140.0:
            # Leg 3: Bottom Spawn
            return 3410.0 + float(np.clip((-285.0 - px) / 970.0, 0.0, 1.0)) * 970.0
        elif py > 1240.0 and px > -1140.0 and px < -380.0:
            # Leg 1: Top Spawn
            return 1220.0 + float(np.clip((px - (-1255.0)) / 970.0, 0.0, 1.0)) * 970.0
        elif px >= -770.0:
            # Leg 2: Right Corridor
            return 2190.0 + float(np.clip((1375.0 - py) / 1220.0, 0.0, 1.0)) * 1220.0
        else:
            # Leg 0: Left Corridor
            return float(np.clip((py - 155.0) / 1220.0, 0.0, 1.0)) * 1220.0

    def _select_upcoming_corner(self, slot_idx: int, pev_addr: int, px: float, py: float) -> tuple:
        """
        Select the next UNUSED ambush corner ahead of the player along the 4-leg loop.
        Each corner lives strictly ONCE per lap ("живут только 1 раз за круг").
        Returns (corner_idx, c_leg, hx, hy, pkx, pky, cname).
        """
        if not hasattr(self, "_used_corners_this_lap"):
            self._used_corners_this_lap = set()
        cur_leg = self.get_loop_leg(px, py)

        # Compute monotonically increasing distance along the 4-leg loop starting from (px, py) on cur_leg
        leg_lengths = {0: 1220.0, 1: 970.0, 2: 1220.0, 3: 970.0}
        if cur_leg == 0:
            rem_cur = max(0.0, 1375.0 - py)
        elif cur_leg == 1:
            rem_cur = max(0.0, -285.0 - px)
        elif cur_leg == 2:
            rem_cur = max(0.0, py - 155.0)
        else:
            rem_cur = max(0.0, px - (-1255.0))

        def _loop_forward_dist(c_leg: int, cx: float, cy: float) -> float:
            leg_offset = (c_leg - cur_leg) % 4
            if leg_offset == 0:
                if cur_leg == 0:
                    return cy - py
                elif cur_leg == 1:
                    return cx - px
                elif cur_leg == 2:
                    return py - cy
                else:
                    return px - cx
            dist = rem_cur
            for step_l in range(1, leg_offset):
                dist += leg_lengths[(cur_leg + step_l) % 4]
            if c_leg == 0:
                dist += max(0.0, cy - 155.0)
            elif c_leg == 1:
                dist += max(0.0, cx - (-1255.0))
            elif c_leg == 2:
                dist += max(0.0, 1375.0 - cy)
            else:
                dist += max(0.0, -285.0 - cx)
            return dist

        occupied_indices = set()
        for other_pev, ost in self._strafe_state.items():
            if other_pev != pev_addr and ost.get("active", False):
                if "corner_idx" in ost:
                    occupied_indices.add(int(ost["corner_idx"]))

        candidates = []
        for idx, c in enumerate(self.AMBUSH_CORNERS):
            if idx in self._used_corners_this_lap or idx in occupied_indices:
                continue
            c_leg, hx, hy, pkx, pky, cname = c
            fdist = _loop_forward_dist(c_leg, pkx, pky)
            if fdist >= -35.0:
                candidates.append((fdist, idx, c))

        # If all uncleared corners ahead are already occupied by other live bots, wrap around to any unoccupied uncleared corner
        if not candidates:
            for idx, c in enumerate(self.AMBUSH_CORNERS):
                if idx in self._used_corners_this_lap or idx in occupied_indices:
                    continue
                c_leg, hx, hy, pkx, pky, cname = c
                fdist = _loop_forward_dist(c_leg, pkx, pky)
                candidates.append((fdist + 4400.0, idx, c))

        # Only clear _used_corners_this_lap when all 16 corners have been KILLED!
        if not candidates:
            if len(self._used_corners_this_lap) >= len(self.AMBUSH_CORNERS):
                self._used_corners_this_lap.clear()
            for idx, c in enumerate(self.AMBUSH_CORNERS):
                if idx in occupied_indices:
                    continue
                c_leg, hx, hy, pkx, pky, cname = c
                fdist = _loop_forward_dist(c_leg, pkx, pky)
                if fdist >= -35.0:
                    candidates.append((fdist, idx, c))
            if not candidates:
                idx = int(slot_idx) % len(self.AMBUSH_CORNERS)
                c = self.AMBUSH_CORNERS[idx]
                candidates.append((100.0, idx, c))

        candidates.sort(key=lambda item: item[0])
        _, chosen_idx, chosen_c = candidates[0]
        c_leg, hx, hy, pkx, pky, cname = chosen_c
        return chosen_idx, c_leg, hx, hy, pkx, pky, cname

    def pop_engine_telemetry(self) -> tuple:
        """Return and clear (real_engine_kills, real_engine_hits) detected directly via pev->health drops and scoreboard frags (+0x164)."""
        sb_frags = int(getattr(self, "_pending_scoreboard_frags", 0))
        self._pending_scoreboard_frags = 0
        now_t = time.time()
        if self._pending_engine_kills > 0:
            self._last_eng_kill_pop_t = now_t
            k = max(self._pending_engine_kills, sb_frags)
        elif sb_frags > 0 and (now_t - float(getattr(self, "_last_eng_kill_pop_t", 0.0))) > 1.0:
            self._last_eng_kill_pop_t = now_t
            k = sb_frags
            self._pending_engine_hits = max(1, self._pending_engine_hits)
            logger.info(f"[SCOREBOARD FRAG DETECTED] +1 Kill verified via local_pev->frags (+0x164)!")
        else:
            k = 0
        h = self._pending_engine_hits
        self._pending_engine_kills = 0
        self._pending_engine_hits = 0
        return k, h

    def set_hud_round_timer(self, seconds: float):
        """Set the in-game CS 1.6 HUD round timer (CHudTimer at client.dll + 0x1F3090) to count down from `seconds`."""
        if not self.hProc or not self.client_base:
            return
        try:
            hud_timer_base = self.client_base + 0x1F3090
            # Write pending round duration (float) at +0x20, zero flash timer at +0x24, clear red-expired flag (byte) at +0x28
            payload = struct.pack("<ffB", max(1.0, float(seconds)), 0.0, 0)
            self.kernel32.WriteProcessMemory(
                self.hProc, ctypes.c_void_p(hud_timer_base + 0x20), payload, 9, None
            )
        except Exception:
            pass

    def reset_lap_to_spawn(self):
        """
        Teleport the local player back to the starting point in Own Spawn (-1255, 160, -143) facing +90° (Corridor 1),
        reset all 16 lap corners, re-stage all bots from Corner #1, and reset the 1:55 HUD timer!
        In real_combat_mode (native CS 1.6 game bots), do NOT teleport the player or bots — let CS 1.6 rounds handle spawns!
        """
        if hasattr(self, "_used_corners_this_lap"):
            self._used_corners_this_lap.clear()
        self._loop_leg = 0
        self._last_player_x = -1255.0
        self._last_player_y = 160.0
        for ost in getattr(self, "_strafe_state", {}).values():
            if isinstance(ost, dict):
                ost["active"] = False
                ost["in_los"] = False
        self.local_real_hp = 100.0
        self._prev_local_hp = 100.0
        if getattr(self, "real_combat_mode", False):
            # Native CS 1.6 game bots mode: only clean up any leftover >100 HP sentinel, never teleport!
            if self.hProc and self.local_pev_addr:
                try:
                    lp_buf = (ctypes.c_char * 4)()
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), lp_buf, 4, None):
                        lp_hp = struct.unpack("<f", lp_buf)[0]
                        if not math.isfinite(lp_hp) or lp_hp > 100.5 or (lp_hp > 0.5 and lp_hp < 100.0):
                            self.kernel32.WriteProcessMemory(
                                self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                            )
                            self.kernel32.WriteProcessMemory(
                                self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x1BC), struct.pack("<f", 100.0), 4, None
                            )
                except Exception:
                    pass
            return
        if self.hProc and self.local_pev_addr:
            try:
                spawn_pos = struct.pack("<3f", -1255.0, 160.0, -143.0)
                zero_vec = struct.pack("<3f", 0.0, 0.0, 0.0)
                spawn_ang = struct.pack("<3f", 0.0, 90.0, 0.0)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x08), spawn_pos, 12, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x14), spawn_pos, 12, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x20), zero_vec, 12, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x2C), zero_vec, 12, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x50), spawn_ang, 12, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x74), spawn_ang, 12, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), struct.pack("<f", 100.0), 4, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x1BC), struct.pack("<f", 100.0), 4, None)
            except Exception as e:
                logger.warning(f"Failed to teleport local player to spawn: {e}")
        self.set_hud_round_timer(115.0)

    def _teleport_bot_to_corner(
        self,
        pev_addr: int,
        csbot_addr: int,
        st: dict,
        now: float,
        wait_sec: float = 0.15,
        upper_side: bool = False,
    ) -> tuple:
        """
        Instantly place a live CCSBot STATICALLY at the next UNUSED corner ahead of the player,
        reset its HP to 1000.0 sentinel, zero its velocity, and hold position!
        """
        px = float(getattr(self, "_last_player_x", -1255.0))
        py = float(getattr(self, "_last_player_y", 160.0))
        slot_idx = int(st.get("slot_idx", 0))
        corner_idx, c_leg, hx, hy, pkx, pky, cname = self._select_upcoming_corner(slot_idx, pev_addr, px, py)
        sx, sy, sz = float(pkx), float(pky), -143.0

        pos_bytes = struct.pack("<3f", float(sx), float(sy), float(sz))
        zero_vec = struct.pack("<3f", 0.0, 0.0, 0.0)
        # Write pev->origin (+0x08), pev->oldorigin (+0x14), pev->velocity (+0x20), pev->basevelocity (+0x2C)
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x08), pos_bytes, 12, None)
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x14), pos_bytes, 12, None)
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), zero_vec, 12, None)
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x2C), zero_vec, 12, None)
        # Keep sentinel health = 10000.0 (+0x160) & takedamage = 2.0 (+0x16C) so real bullets deal measurable damage without permanent DEAD_DEAD
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x160), struct.pack("<f", 10000.0), 4, None)
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x16C), struct.pack("<f", 2.0), 4, None)
        # Zero CCSBot move commands
        cmd_zero = struct.pack("<3fH", 0.0, 0.0, 0.0, 0)
        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x9D4), cmd_zero, 14, None)

        self._respawn_seq = getattr(self, "_respawn_seq", 0) + 1
        st["role"] = "static_corner_shooter"
        st["corner_idx"] = corner_idx
        st["corner_leg"] = c_leg
        st["corner_name"] = cname
        st["corner_slot"] = (sx, sy, sz)
        st["peek_target"] = (float(pkx), float(pky))
        st["jump_peek"] = False
        st["armoury_visit"] = False
        st["armoury_tried"] = False
        st["phase"] = "corner_hold"
        st["phase_t"] = now
        st["wait_until"] = now
        st["real_dmg_taken"] = 0.0
        st["hit_boost_until"] = 0.0
        st["flinch_until"] = 0.0
        st["visible_since"] = now
        st["in_los"] = False
        st["last_pos"] = (sx, sy)
        st["last_lat"] = sx if c_leg in (0, 2) else sy
        st["stuck_t"] = now
        st["dir"] = 1.0
        self.target_hit_counts.pop(pev_addr, None)
        return sx, sy, sz

    def register_hit_on_current_target(self, hits_to_kill: int = 6, hit_damage: int = 1) -> bool:
        """
        Record a direct hit on the currently tracked enemy.
        On lethal damage (>= hits_to_kill), INSTANTLY kills & stages the bot at the next unused corner along the loop!
        """
        now = time.time()
        self.last_hit_counted = False
        if self.current_target_key is None or (now - self.last_counted_shot_time) < 0.10:
            return False
        self.last_counted_shot_time = now
        self.last_hit_counted = True
        slot_addr, grid_x, grid_y = self.current_target_key
        st = self._strafe_state.get(slot_addr)
        if st is not None:
            st["flinch_until"] = now + 0.35

        self.target_hit_counts[slot_addr] = self.target_hit_counts.get(slot_addr, 0) + max(1, int(hit_damage))
        if self.target_hit_counts[slot_addr] >= hits_to_kill:
            if st is not None and st.get("csbot_addr"):
                self._teleport_bot_to_corner(
                    slot_addr,
                    st["csbot_addr"],
                    st,
                    now,
                    wait_sec=0.25,
                    upper_side=bool(st.get("upper_side", False)),
                )
            else:
                self.eliminated_until[("slot", slot_addr)] = now + 1.5
            self.target_hit_counts[slot_addr] = 0
            logger.info(f"[TARGET ELIMINATED] Enemy 0x{slot_addr:X} killed & staged at next corner ({st.get('corner_name') if st else '?'})!")
            return True
        return False

    @staticmethod
    def _load_custom_offsets() -> Optional[Dict]:
        """Load offsets.json (origin / pitch / yaw) produced by find_offsets.py, if present."""
        path = Path(__file__).parent / "offsets.json"
        if not path.exists():
            logger.warning("offsets.json not found - using legacy client.dll + 0x199E3C layout. "
                           "Run find_offsets.py if observations are invalid.")
            return None
        try:
            cfg = json.loads(path.read_text())
            parsed = {}
            for key in ("origin", "pitch", "yaw"):
                parsed[key] = (cfg[key]["module"].lower(), int(str(cfg[key]["offset"]), 0))
            parsed["degrees"] = str(cfg.get("angles_unit", "degrees")).lower().startswith("deg")
            parsed["eye_z"] = float(cfg.get("eye_z_offset", 0.0))
            parsed["exclude_slots"] = set(int(s) for s in cfg.get("exclude_dummy_slots", []))
            logger.info(f"Loaded custom offsets from {path}: " +
                        ", ".join(f"{k}={parsed[k][0]}+0x{parsed[k][1]:X}" for k in ("origin", "pitch", "yaw")) +
                        f", eye_z={parsed['eye_z']}, exclude_slots={sorted(parsed['exclude_slots'])}")
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse {path}: {e} - falling back to legacy layout")
            return None

    def _module_base(self, name: str) -> Optional[int]:
        return {"hw.dll": self.hw_base, "client.dll": self.client_base}.get(name)

    def _read_floats(self, addr: int, count: int) -> Optional[tuple]:
        size = 4 * count
        buf = (ctypes.c_char * size)()
        if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(addr), buf, size, None):
            return None
        return struct.unpack_from(f"<{count}f", buf, 0)

    def _get_local_player_custom(self) -> Optional[PlayerState]:
        """Read local player using offsets.json (origin vec3 + pitch/yaw floats)."""
        co = self.custom_offsets
        o_base = self._module_base(co["origin"][0])
        p_base = self._module_base(co["pitch"][0])
        y_base = self._module_base(co["yaw"][0])
        if not (o_base and p_base and y_base):
            return None
        target = o_base + co["origin"][1]
        pos = self._read_floats(target, 3)
        pitch = self._read_floats(p_base + co["pitch"][1], 1)
        yaw = self._read_floats(y_base + co["yaw"][1], 1)
        if pos is None or pitch is None or yaw is None:
            if time.time() - self.last_connect > 2.0:
                logger.warning("ReadProcessMemory failed. Auto-reconnecting to CS 1.6...")
                self.close()
                self.connect()
            return None
        x, y, z = pos
        p, yw = pitch[0], yaw[0]
        bad = any(math.isnan(v) or math.isinf(v) for v in (x, y, z, p, yw))
        if bad or abs(x) > 8192 or abs(y) > 8192 or (x == 0.0 and y == 0.0 and z == 0.0):
            self.zero_read_count += 1
            if self.zero_read_count >= 25 and (time.time() - self.last_connect > 3.0):
                logger.warning("Player memory returned zeros/invalid pose (ensure you spawned on map in CS 1.6). Re-attaching to hl.exe...")
                self.zero_read_count = 0
                self.close()
                self.connect()
            return None
        self.zero_read_count = 0
        if co["degrees"]:
            p, yw = math.radians(p), math.radians(yw)
        # GoldSrc: pitch > 0 looks down; wrap yaw into [-pi, pi)
        pitch_rad = max(-math.pi / 2, min(math.pi / 2, p))
        yaw_rad = (yw + math.pi) % (2 * math.pi) - math.pi
        self.last_yaw = yaw_rad
        eye_z = float(co["eye_z"])
        punch_vec = Vector3(0.0, 0.0, 0.0)
        if self.local_pev_addr:
            vo_z = self._read_floats(self.local_pev_addr + 0x17C, 1)
            if vo_z is not None and math.isfinite(vo_z[0]) and 8.0 <= vo_z[0] <= 32.0:
                eye_z = float(vo_z[0])
            pa = self._read_floats(self.local_pev_addr + 0x68, 3)
            if pa is not None and all(math.isfinite(v) and abs(v) <= 45.0 for v in pa):
                punch_vec = Vector3(math.radians(pa[0]), math.radians(pa[1]), math.radians(pa[2]))
        return PlayerState(
            address=target,
            origin=Vector3(x, y, z + eye_z),  # exact standing/crouching eye position
            view_angles=Vector3(pitch_rad, yaw_rad, 0.0),
            punch_angle=punch_vec,
            health=100,
            armor=100,
            team=2,
            ammo_clip=30,
            is_alive=True,
            is_local=True
        )

    def connect(self) -> bool:
        """Attach to CS 1.6 and locate hw.dll and client.dll."""
        self.last_connect = time.time()
        pid = wintypes.DWORD()
        for cls in [b"SDL_app", b"Valve001"]:
            h = self.user32.FindWindowA(cls, None)
            if h:
                self.user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                if pid.value:
                    break
        
        if not pid.value:
            # Fallback: find process by executable name (hl.exe)
            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)
                ]
            snap_p = self.kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
            if snap_p and snap_p != -1:
                try:
                    pe32 = PROCESSENTRY32W()
                    pe32.dwSize = ctypes.sizeof(PROCESSENTRY32W)
                    if self.kernel32.Process32FirstW(snap_p, ctypes.byref(pe32)):
                        while True:
                            if pe32.szExeFile.lower() == self.process_name.lower():
                                pid.value = pe32.th32ProcessID
                                break
                            if not self.kernel32.Process32NextW(snap_p, ctypes.byref(pe32)):
                                break
                finally:
                    self.kernel32.CloseHandle(snap_p)

        if not pid.value:
            logger.error(f"CS 1.6 process '{self.process_name}' not found")
            return False
        
        self.pid = pid.value
        # Request VM_WRITE (0x0020) | VM_OPERATION (0x0008) so we can strafe frozen dummies left/right in place
        self.hProc = self.kernel32.OpenProcess(
            PROCESS_VM_READ | 0x0020 | 0x0008 | PROCESS_QUERY_INFORMATION,
            False,
            self.pid
        )
        if not self.hProc:
            self.hProc = self.kernel32.OpenProcess(
                PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, 
                False, 
                self.pid
            )
        if not self.hProc:
            logger.error("Failed to open process (run as Administrator)")
            return False
        
        # Locate both hw.dll and client.dll
        class MODULEENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
                ("ProccntUsage", wintypes.DWORD), ("modBaseAddr", ctypes.c_void_p),
                ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
                ("szModule", ctypes.c_wchar * 256), ("szExePath", ctypes.c_wchar * 260)
            ]
        
        snap = self.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, self.pid)
        if not snap or snap == -1:
            return False
        
        try:
            me32 = MODULEENTRY32W()
            me32.dwSize = ctypes.sizeof(MODULEENTRY32W)
            if self.kernel32.Module32FirstW(snap, ctypes.byref(me32)):
                while True:
                    mname = me32.szModule.lower()
                    if mname == "hw.dll":
                        self.hw_base = int(me32.modBaseAddr)
                    elif mname == "client.dll":
                        self.client_base = int(me32.modBaseAddr)
                    elif mname == "mp.dll":
                        self.mp_base = int(me32.modBaseAddr)
                    if not self.kernel32.Module32NextW(snap, ctypes.byref(me32)):
                        break
        finally:
            self.kernel32.CloseHandle(snap)
            
        if not self.hw_base or not self.client_base:
            logger.error(f"Modules missing: hw.dll={bool(self.hw_base)}, client.dll={bool(self.client_base)}")
            return False

        self._entvars_Scanned = False
        self._mp_patched = False
        if self.mp_base:
            self._patch_mp_dll_for_strafing()
        logger.info(
            f"Connected: PID={self.pid}, hw.dll=0x{self.hw_base:X}, "
            f"client.dll=0x{self.client_base:X}, mp.dll=0x{(self.mp_base or 0):X}"
        )
        return True

    def get_local_player(self) -> Optional[PlayerState]:
        """Read local player position and view angles from client.dll."""
        if not self.hProc or not self.client_base:
            if time.time() - self.last_connect > 3.0:
                self.connect()
            return None
        
        if self.custom_offsets:
            return self._get_local_player_custom()
        
        target = self.client_base + self.CLIENT_POS_OFFSET
        buf = (ctypes.c_char * 24)()
        if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(target), buf, 24, None):
            if time.time() - self.last_connect > 2.0:
                logger.warning("ReadProcessMemory failed. Auto-reconnecting to CS 1.6...")
                self.close()
                self.connect()
            return None
        
        x, y, z, fx, fy, fz = struct.unpack_from("<6f", buf, 0)
        if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isnan(fx) or math.isnan(fy) or math.isnan(fz):
            return None
        
        fwd_norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        if abs(x) > 8000 or abs(y) > 8000 or fwd_norm < 0.2 or fwd_norm > 2.0:
            self.zero_read_count += 1
            if self.zero_read_count >= 25 and (time.time() - self.last_connect > 3.0):
                logger.warning("Player memory returned zeros/invalid pose (ensure you spawned on map in CS 1.6). Re-attaching to hl.exe...")
                self.zero_read_count = 0
                self.close()
                self.connect()
            return None
        
        self.zero_read_count = 0
        if abs(fx) > 1e-5 or abs(fy) > 1e-5:
            yaw_rad = math.atan2(fy, fx)
            self.last_yaw = yaw_rad
        else:
            yaw_rad = self.last_yaw
            
        fz_clamped = max(-1.0, min(1.0, float(fz)))
        pitch_rad = -math.asin(fz_clamped)
        
        return PlayerState(
            address=target,
            origin=Vector3(x, y, z),
            view_angles=Vector3(pitch_rad, yaw_rad, 0.0),
            health=100,
            armor=100,
            team=2,
            ammo_clip=30,
            is_alive=True,
            is_local=True
        )

    def _set_mp_bot_quota(self, target_bots: Optional[int]) -> None:
        """
        Keep max_combat_bots stored on MemoryReader without mutating mp.dll's internal bot_quota cvar_t,
        because lowering bot_quota causes CCSBotManager::MaintainBotQuota() to free CCSBot heap objects mid-frame.
        Surplus bots are safely parked in static sv.edicts (get_enemies) without freeing heap memory.
        """
        return

    def _patch_mp_dll_for_strafing(self):
        """
        Patch ReGameDLL_CS (mp.dll) in memory so enemy bots physically strafe left/right via
        GoldSrc's native g_engfuncs.pfnRunPlayerMove without ever chasing or knifing the player:
        1. Set bot_stop = 1.0f (mp.dll + 0x171D14) so CCSBot::Update() returns at 0x1001f628
           before any combat/chase/knife logic runs.
        2. Set bot_freeze = 0.0f (mp.dll + 0x171AE4) and bot_zombie = 1.0f (mp.dll + 0x1719B0).
        3. NOP the instructions in CBot::BotThink() (0x100a0543, 0x100a0556, 0x100a055c) that zero
           m_buttonFlags (+0x9E0), m_forwardSpeed (+0x9D4), and m_strafeSpeed (+0x9D8) before
           CBot::UpdatePlayer() calls g_engfuncs.pfnRunPlayerMove.
        4. NOP the call to CCSBot::UpdateLookAngles() in CCSBot::Upkeep() (0x1001efed) so bots keep
           pev->v_angle = (0, 180, 0) facing down the corridor (-X) while strafing along ±Y.
        """
        if not self.hProc or not self.mp_base:
            return
        try:
            f_one = struct.pack("<f", 1.0)
            f_zero = struct.pack("<f", 0.0)
            orig_mimic = bytes.fromhex(
                "558bec51d9eed905d01a1710d8d1dfe0538bd9ddd9f6c4417a0bddd832c05b8be55dc20400a170"
            )
            prev_mimic = bytes.fromhex("6a01e87974020032c0c20400") + orig_mimic[12:]
            new_mimic = bytes.fromhex(
                "dbe38b81c005000085c075068b81c40500003b81d4050000740650e8d0730200dbe332c0c20400"
            )
            patches = [
                (0xA0543, bytes.fromhex("668986e0090000"), b"\x90" * 7),  # m_buttonFlags zeroing
                (0xA0556, bytes.fromhex("d996d4090000"), b"\x90" * 6),    # m_forwardSpeed zeroing
                (0xA055C, bytes.fromhex("d996d8090000"), b"\x90" * 6),    # m_strafeSpeed zeroing
                (0x1EFED, bytes.fromhex("e85e600a00"), b"\x90" * 5),      # UpdateLookAngles call
                (0xA0A60, orig_mimic, new_mimic),
            ]

            # Enable bot_allow_pistols, shotguns, sub_machine_guns, rifles, machine_guns, snipers
            for wep_cvar_rva in (0x171C24, 0x171C10, 0x171BFC, 0x171BE8, 0x171BD4, 0x171BAC):
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + wep_cvar_rva), f_one, 4, None)
            # Ensure bot_knives_only = 0.0
            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x171B98), f_zero, 4, None)

            if getattr(self, "real_combat_mode", False):
                # Phase 8: 100% NATIVE CS 1.6 GAME BOTS (CCSBot::Update, native vision, native pathfinding & shooting):
                # Keep bot_stop = 0.0, bot_freeze = 0.0, bot_zombie = 0.0, and unconditionally restore original mp.dll bytes!
                self._set_mp_bot_quota(getattr(self, "max_combat_bots", None))
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x171D14), f_zero, 4, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x171AE4), f_zero, 4, None)
                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x1719B0), f_zero, 4, None)
                old_prot = wintypes.DWORD()
                restored_any = False
                for rva, orig_bytes, nop_bytes in patches:
                    target_va = self.mp_base + rva
                    n = len(orig_bytes)
                    cur_buf = (ctypes.c_char * n)()
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(target_va), cur_buf, n, None):
                        cur = bytes(cur_buf)
                        if cur != orig_bytes and cur in (nop_bytes, prev_mimic):
                            if self.kernel32.VirtualProtectEx(
                                self.hProc, ctypes.c_void_p(target_va), n, 0x40, ctypes.byref(old_prot)
                            ):
                                self.kernel32.WriteProcessMemory(
                                    self.hProc, ctypes.c_void_p(target_va), orig_bytes, n, None
                                )
                                self.kernel32.VirtualProtectEx(
                                    self.hProc, ctypes.c_void_p(target_va), n, old_prot.value, ctypes.byref(old_prot)
                                )
                                restored_any = True
                if restored_any or self._mp_patched:
                    self._mp_patched = False
                    logger.info(
                        "[NATIVE CS 1.6 BOTS] Unpatched mp.dll and enabled native CCSBot AI (bot_stop=0, bot_zombie=0, UpdateLookAngles restored)!"
                    )
                return

            has_live_bots = getattr(self, "_last_active_count", 0) > 0
            f_stop = f_one if has_live_bots else f_zero
            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x171D14), f_stop, 4, None)
            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x171AE4), f_zero, 4, None)
            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x1719B0), f_stop, 4, None)

            if self._mp_patched:
                return
            old_prot = wintypes.DWORD()
            patched_count = 0
            problems = []
            for rva, orig_bytes, nop_bytes in patches:
                target_va = self.mp_base + rva
                n = len(orig_bytes)
                cur_buf = (ctypes.c_char * n)()
                if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(target_va), cur_buf, n, None):
                    problems.append(f"RVA 0x{rva:X}: read failed (err {ctypes.GetLastError()})")
                    continue
                cur = bytes(cur_buf)
                if cur not in (orig_bytes, nop_bytes, prev_mimic):
                    problems.append(f"RVA 0x{rva:X}: unexpected bytes {cur.hex()} (different mp.dll build?)")
                    continue
                if not self.kernel32.VirtualProtectEx(
                    self.hProc, ctypes.c_void_p(target_va), n, 0x40, ctypes.byref(old_prot)
                ):
                    problems.append(f"RVA 0x{rva:X}: VirtualProtectEx failed (err {ctypes.GetLastError()})")
                    continue
                ok = self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(target_va), nop_bytes, n, None)
                self.kernel32.VirtualProtectEx(
                    self.hProc, ctypes.c_void_p(target_va), n, old_prot.value, ctypes.byref(old_prot)
                )
                if ok:
                    patched_count += 1
                else:
                    problems.append(f"RVA 0x{rva:X}: WriteProcessMemory failed (err {ctypes.GetLastError()})")
            if patched_count == len(patches):
                self._mp_patched = True
                logger.info(
                    f"[STRAFE ENGINE] Patched mp.dll (0x{self.mp_base:X}) CBot::BotThink & Upkeep "
                    f"for native GoldSrc pfnRunPlayerMove lateral strafing!"
                )
            elif not getattr(self, "_warned_patch", False):
                self._warned_patch = True
                logger.warning(f"[STRAFE ENGINE] mp.dll patch incomplete ({patched_count}/{len(patches)}): {problems}")
        except Exception as e:
            logger.warning(f"[STRAFE ENGINE] Failed to patch mp.dll: {e}")

    def _scan_server_entvars(self, dummy_coords: List[tuple]):
        """
        Scan writable process memory once for GoldSrc server entvars_t (pev) and CCSBot structs.
        Verified by:
        - Standing player hull signature at pev + 0x0DC: mins=(-16,-16,-36), maxs=(16,16,36), size=(32,32,72)
        - Self-referential pointer invariant: pev->pContainingEntity (pev + 0x208) == edict (pev - 0x80)
        - CCSBot* pvPrivateData at edict + 0x7C (pev - 0x04) with CCSBot vtable (mp.dll + 0x13958C)
        """
        if self._entvars_Scanned or not self.hProc:
            return
        self._entvars_Scanned = True
        self._last_scan_t = time.time()
        hull_sig = struct.pack("<9f", -16.0, -16.0, -36.0, 16.0, 16.0, 36.0, 32.0, 32.0, 72.0)
        csbot_vtable = (self.mp_base + 0x13958C) if self.mp_base else None

        class MEMORY_BASIC_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BaseAddress", ctypes.c_void_p),
                ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD),
                ("RegionSize", ctypes.c_size_t),
                ("State", wintypes.DWORD),
                ("Protect", wintypes.DWORD),
                ("Type", wintypes.DWORD),
            ]

        mbi = MEMORY_BASIC_INFORMATION()
        addr = 0x00010000
        max_addr = 0x7FFF0000
        found_bots = []
        seen_pev = set()
        hull_hits = 0
        vt_sig = struct.pack("<I", csbot_vtable) if csbot_vtable else None
        CH = 16 * 1024 * 1024   # read big regions (e.g. engine Hunk > 64 MB) in 16 MB chunks
        OV = 0x400              # overlap so structs crossing chunk borders are not missed

        def _validate(pev_addr: int, csbot_ptr: int) -> None:
            # pev->pContainingEntity (pev + 0x208) must be the edict (pev - 0x80);
            # edict->pvPrivateData (pev - 0x04) must be the CCSBot object, whose +0x04 is pev.
            if pev_addr in seen_pev:
                return
            b = (ctypes.c_char * 0x218)()
            if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(pev_addr - 0x04), b, 0x218, None):
                return
            priv = struct.unpack_from("<I", b, 0x00)[0]
            edict_ptr = struct.unpack_from("<I", b, 0x04 + 0x208)[0]
            ox, oy, oz = struct.unpack_from("<3f", b, 0x04 + 0x08)
            if priv != csbot_ptr or edict_ptr != (pev_addr - 0x80):
                return
            if csbot_vtable is not None:
                vt_buf = (ctypes.c_char * 8)()
                if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(csbot_ptr), vt_buf, 8, None):
                    return
                vt_ptr, bot_pev = struct.unpack_from("<2I", vt_buf, 0)
                if vt_ptr != csbot_vtable or bot_pev != pev_addr:
                    ent_flags = struct.unpack_from("<I", b, 0x04 + 0x1A4)[0]
                    # In GoldSrc (common/const.h): FL_CLIENT = (1<<3) = 0x0008, FL_FAKECLIENT = (1<<13) = 0x2000
                    if bot_pev == pev_addr and (ent_flags & 0x2008) == 0x0008 and math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oz):
                        self.local_pev_addr = pev_addr  # Human host player entity (CBasePlayer)
                    return  # not a CCSBot (e.g. the human host) -> never drive it
            if not (math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oz)):
                return
            seen_pev.add(pev_addr)
            found_bots.append((pev_addr, csbot_ptr, ox, oy, oz))

        while addr < max_addr:
            res = self.kernel32.VirtualQueryEx(self.hProc, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
            if not res:
                break
            base = mbi.BaseAddress or 0
            size = mbi.RegionSize or 0
            # MEM_COMMIT = 0x1000, PAGE_READWRITE = 0x04 / PAGE_EXECUTE_READWRITE = 0x40
            if mbi.State == 0x1000 and (mbi.Protect & 0xFF) in (0x04, 0x40) and not (mbi.Protect & 0x100) and size > 0:
                off = 0
                while off < size:
                    rd_start = max(0, off - OV)
                    rd_end = min(size, off + CH + OV)
                    n = rd_end - rd_start
                    buf = (ctypes.c_char * n)()
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(base + rd_start), buf, n, None):
                        raw = bytes(buf)
                        lo = off - rd_start
                        hi = min(off + CH, size) - rd_start
                        # 1) Robust path: CCSBot objects start with the CCSBot vtable pointer, +0x04 = pev
                        if vt_sig is not None:
                            pos = lo
                            while True:
                                i = raw.find(vt_sig, pos, hi + 3)
                                if i == -1 or i >= hi:
                                    break
                                pos = i + 4
                                if (base + rd_start + i) % 4 or i + 8 > n:
                                    continue
                                pev_addr = struct.unpack_from("<I", raw, i + 4)[0]
                                if pev_addr > 0x10000:
                                    _validate(pev_addr, base + rd_start + i)
                        # 2) Fallback: standing hull mins/maxs/size at pev + 0x0DC
                        pos = lo
                        while True:
                            i = raw.find(hull_sig, pos, hi + len(hull_sig))
                            if i == -1 or i >= hi:
                                break
                            pos = i + 4
                            pev_off = i - 0x0DC
                            if pev_off < 4:
                                continue
                            hull_hits += 1
                            csbot_ptr = struct.unpack_from("<I", raw, pev_off - 0x04)[0]
                            if csbot_ptr > 0x10000:
                                _validate(base + rd_start + pev_off, csbot_ptr)
                    off += CH
            addr = base + max(4096, size)

        self._server_origins = found_bots
        if found_bots:
            self._locate_local_pev_from_bots(found_bots, seen_pev)
        logger.info(
            f"[STRAFE ENGINE] Scan done: mp.dll=0x{(self.mp_base or 0):X}, CCSBot vtable=0x{(csbot_vtable or 0):X}, "
            f"hull hits={hull_hits}, validated CCSBot entities={len(found_bots)}, local_pev=0x{(self.local_pev_addr or 0):X}"
        )
        for pev_addr, csbot_ptr, ox, oy, oz in found_bots:
            logger.info(f"[STRAFE ENGINE]   bot pev=0x{pev_addr:X} CCSBot=0x{csbot_ptr:X} origin=({ox:.0f}, {oy:.0f}, {oz:.0f})")
        if not found_bots:
            logger.warning("[STRAFE ENGINE] No CCSBot entities found yet (will rescan every 15 s).")

    def _locate_local_pev_from_bots(
        self,
        bots_list: list,
        seen_pev: Optional[set] = None,
        local_origin: Optional[Vector3] = None,
    ) -> None:
        """
        Locate Host Player (edict #1) in sv.edicts around the bot edicts using 1 KB chunked reads
        (so unmapped guard pages before sv.edicts[0] never abort ReadProcessMemory) and verify against
        client local_origin when available.
        """
        if not self.hProc or not bots_list:
            return
        bot_pevs = {b[0] for b in bots_list}
        if seen_pev:
            bot_pevs |= set(seen_pev)
        min_bot_pev = min(b[0] for b in bots_list)
        max_bot_pev = max(b[0] for b in bots_list)
        scan_base = max(0x10000, (min_bot_pev - 0x4000) & ~0x3FF)
        scan_end = (max_bot_pev + 0x18000 + 0x3FF) & ~0x3FF
        csbot_vtable = (self.mp_base + 0x13958C) if self.mp_base else None
        ca_vtable = (self.mp_base + 0x1359FC) if self.mp_base else None
        ref_x = local_origin.x if local_origin is not None else getattr(self, "_last_player_x", None)
        ref_y = local_origin.y if local_origin is not None else getattr(self, "_last_player_y", None)

        best_cand = None
        best_score = 1e18
        armoury_list = []
        chunk_step = 0x400
        chunk_read = chunk_step + 0x220
        for chunk_addr in range(scan_base, scan_end, chunk_step):
            raw_buf = (ctypes.c_char * chunk_read)()
            n_read = chunk_read
            if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(chunk_addr), raw_buf, chunk_read, None):
                # Fallback to exact 1 KB chunk if the +0x220 overlap crossed into an unmapped page
                raw_buf = (ctypes.c_char * chunk_step)()
                n_read = chunk_step
                if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(chunk_addr), raw_buf, chunk_step, None):
                    continue
            raw = bytes(raw_buf)
            max_off = min(chunk_step, n_read - 0x210)
            for off in range(4, max_off, 4):
                cand_pev = chunk_addr + off
                if cand_pev in bot_pevs:
                    continue
                c_edict = struct.unpack_from("<I", raw, off + 0x208)[0]
                if c_edict != (cand_pev - 0x80):
                    continue
                c_priv = struct.unpack_from("<I", raw, off - 0x04)[0]
                if c_priv <= 0x10000:
                    continue
                cx, cy, cz = struct.unpack_from("<3f", raw, off + 0x08)
                if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cz)):
                    continue
                if abs(cx) < 1.0 and abs(cy) < 1.0:
                    continue
                if abs(cx) > 4096.0 or abs(cy) > 4096.0:
                    continue
                vt_buf = (ctypes.c_char * 8)()
                if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(c_priv), vt_buf, 8, None):
                    continue
                vt_ptr, ent_pev = struct.unpack_from("<2I", vt_buf, 0)
                if ent_pev != cand_pev or (csbot_vtable is not None and vt_ptr == csbot_vtable):
                    continue
                flags = struct.unpack_from("<I", raw, off + 0x1A4)[0]
                hp_val = struct.unpack_from("<f", raw, off + 0x160)[0]
                max_hp = struct.unpack_from("<f", raw, off + 0x1B0)[0]
                # Record verified spawn floor weapons (CArmoury / armoury_entity: exact vtable mp.dll + 0x1359FC)
                if ca_vtable is not None and vt_ptr == ca_vtable and (flags & 0x2008) == 0 and max_hp == 0.0 and cand_pev > max_bot_pev:
                    arm_buf = (ctypes.c_char * 12)()
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(c_priv + 0x88), arm_buf, 12, None):
                        m_item, m_cnt, m_init = struct.unpack("<3i", bytes(arm_buf))
                        if m_item == 4 and 0 <= m_cnt <= 30 and 1 <= m_init <= 30:
                            armoury_list.insert(0, (cand_pev, c_priv, cx, cy, cz))
                        elif 0 <= m_item <= 13 and 0 <= m_cnt <= 30 and 1 <= m_init <= 30:
                            armoury_list.append((cand_pev, c_priv, cx, cy, cz))
                    continue
                # In GoldSrc (common/const.h): FL_CLIENT = 0x0008, FL_FAKECLIENT = 0x2000.
                # Only the human host player (CBasePlayer, edict #1) has FL_CLIENT=1 and FL_FAKECLIENT=0!
                if (flags & 0x2008) != 0x0008:
                    continue
                if not math.isfinite(max_hp) or max_hp < 50.0:
                    continue
                pos_dist = math.hypot(cx - ref_x, cy - ref_y) if (ref_x is not None and ref_y is not None) else 50.0
                # Host player is edict #1 (before min_bot_pev), has FL_CLIENT (0x0008), and matches client local_origin
                score = pos_dist
                if cand_pev > min_bot_pev:
                    score += 200.0
                if score < best_score:
                    best_score = score
                    best_cand = (cand_pev, cx, cy, cz, hp_val, pos_dist)

        if armoury_list:
            self._spawn_armoury_pevs = armoury_list
        if best_cand is not None:
            cand_pev, cx, cy, cz, hp_val, pos_dist = best_cand
            self.local_pev_addr = cand_pev
            logger.info(
                f"[STRAFE ENGINE] Located human host player local_pev=0x{cand_pev:X} "
                f"at ({cx:.0f}, {cy:.0f}, {cz:.0f}) | hp={hp_val:.1f} | client_dist={pos_dist:.1f}u"
            )

    def get_enemies(self, local_team: int = 2, local_origin: Optional[Vector3] = None) -> List[PlayerState]:
        """Read enemy bots and drive real GoldSrc lateral strafing via CCSBot + pfnRunPlayerMove."""
        if not self.hProc or not self.hw_base:
            if time.time() - self.last_connect > 2.0:
                self.close()
                self.connect()
            return []

        if self.mp_base:
            self._patch_mp_dll_for_strafing()
        elif not getattr(self, "_warned_no_mp", False):
            self._warned_no_mp = True
            logger.warning("[STRAFE ENGINE] mp.dll is not loaded in hl.exe (are you hosting the game via 'New Game'?)")

        if local_origin is not None:
            if not self._local_pos_history or math.hypot(
                local_origin.x - self._local_pos_history[-1][0],
                local_origin.y - self._local_pos_history[-1][1]
            ) > 8.0:
                self._local_pos_history.append((local_origin.x, local_origin.y))
                if len(self._local_pos_history) > 250:
                    self._local_pos_history.pop(0)

        enemies = []
        if not hasattr(self, "_enemy_vel_cache"):
            self._enemy_vel_cache = {}
        now = time.time()
        seen_positions = set()

        # (Re)scan whenever fewer than MAX_ACTIVE_CAROUSEL_BOTS live bots are known, or every 8s in real_combat_mode
        # so adding/kicking bots (e.g. 5v5 -> 2v1) or joining a team from Spectator is detected automatically!
        is_rc = bool(getattr(self, "real_combat_mode", False))
        scan_interval = 3.0 if getattr(self, "_last_active_count", 0) == 0 else (8.0 if is_rc else 10.0)
        if (
            is_rc
            or self.local_pev_addr is None
            or getattr(self, "_last_active_count", 0) < self.MAX_ACTIVE_CAROUSEL_BOTS
        ) and now - getattr(self, "_last_scan_t", 0.0) > scan_interval:
            self._last_scan_t = now
            self._entvars_Scanned = False
        if not self._entvars_Scanned:
            self._scan_server_entvars([])

        # Ensure self.local_pev_addr (Host Player edict #1) and _spawn_armoury_pevs are always located:
        if (self.local_pev_addr is None or not getattr(self, "_spawn_armoury_pevs", None)) and self._server_origins:
            self._locate_local_pev_from_bots(self._server_origins, local_origin=local_origin)

        # Primary path: Drive & read live server CCSBot entities directly via pev + csbot_ptr
        if self._server_origins:
            player_x = local_origin.x if local_origin is not None else -1255.0
            player_y = local_origin.y if local_origin is not None else 160.0
            self._last_player_x = player_x
            self._last_player_y = player_y
            cur_leg = self.get_loop_leg(player_x, player_y)
            upper_side = bool(player_y >= 763.0)
            active_count = 0

            # Latch local player's spawn side (1 = Top/T spawn Y > 763, 2 = Bottom/CT spawn Y <= 763) when in spawn zone at start/respawn:
            if not hasattr(self, "local_spawn_side"):
                self.local_spawn_side = 1 if player_y > 763.0 else 2
            prev_p_dist = math.hypot(
                player_x - float(getattr(self, "_prev_check_px", player_x)),
                player_y - float(getattr(self, "_prev_check_py", player_y)),
            )
            self._prev_check_px = player_x
            self._prev_check_py = player_y
            if (prev_p_dist > 350.0 or getattr(self, "_local_was_dead", False)) and (player_y < 480.0 or player_y > 1040.0):
                self.local_spawn_side = 1 if player_y > 763.0 else 2

            # Standard 100 -> 0 HP HUD counter for the local player (NEVER lock at 1000 HP!):
            # When enemy bullets hit the AI, the CS 1.6 HUD health counter visibly drops from 100 down to 0.
            # If health drops <= 1.0 (death) or had leftover > 100.0 from an old run, reset to 100.0 HP.
            local_team_id = 0
            local_cbase = 0
            if self.local_pev_addr:
                lp_buf = (ctypes.c_char * 4)()
                df_buf = (ctypes.c_char * 4)()
                vo_buf = (ctypes.c_char * 4)()
                fr_buf = (ctypes.c_char * 4)()
                lo_buf = (ctypes.c_char * 12)()
                if not getattr(self, "local_is_dead", False) and self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x08), lo_buf, 12, None):
                    lx_p, ly_p, _ = struct.unpack("<3f", lo_buf)
                    if math.isfinite(lx_p) and math.isfinite(ly_p) and math.hypot(lx_p - player_x, ly_p - player_y) > 300.0:
                        self._locate_local_pev_from_bots(self._server_origins, local_origin=local_origin)
                # Track official GoldSrc scoreboard frags (pev->frags at +0x164) so NO player kill is ever missed!
                if getattr(self, "real_combat_mode", False) and self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x164), fr_buf, 4, None):
                    lp_frags = struct.unpack("<f", fr_buf)[0]
                    if math.isfinite(lp_frags):
                        prev_frags = getattr(self, "_prev_local_frags", None)
                        if prev_frags is not None and lp_frags > prev_frags + 0.5:
                            frag_diff = int(round(lp_frags - prev_frags))
                            if 1 <= frag_diff <= 5:
                                self._pending_scoreboard_frags = getattr(self, "_pending_scoreboard_frags", 0) + frag_diff
                        self._prev_local_frags = lp_frags
                if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), lp_buf, 4, None):
                    lp_hp = struct.unpack("<f", lp_buf)[0]
                    lp_dead = 0
                    lp_view_z = 17.0
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x170), df_buf, 4, None):
                        lp_dead = struct.unpack("<i", df_buf)[0]
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x17C), vo_buf, 4, None):
                        lp_view_z = struct.unpack("<f", vo_buf)[0]
                    if not math.isfinite(lp_hp) or lp_hp > 100.5:
                        # Clean up any leftover 1000.0 sentinel HP -> restore standard 100 HP on HUD!
                        lp_hp = 100.0
                        self._prev_local_hp = 100.0
                        self.local_real_hp = 100.0
                        self.kernel32.WriteProcessMemory(
                            self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                        )
                        self.kernel32.WriteProcessMemory(
                            self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x1BC), struct.pack("<f", 100.0), 4, None
                        )
                    elif lp_dead != 0 or lp_hp <= 0.5:
                        self.local_is_dead = True
                        # Player reached 0 HP -> record death ONCE per CS 1.6 round (with 4.5s cooldown matching CS 1.6 round restart delay)
                        if not getattr(self, "_local_was_dead", False) and (now - getattr(self, "_last_player_death_t", 0.0)) > 4.5:
                            self._local_was_dead = True
                            self._last_player_death_t = now
                            self._pending_player_died = True
                            self._pending_player_real_dmg = max(10.0, float(getattr(self, "_prev_local_hp", 100.0)))
                            logger.info(f"[PLAYER DEATH DETECTED] Local player reached 0 HP (deadflag={lp_dead}, hp={lp_hp:.1f})")
                        self.local_real_hp = 0.0 if getattr(self, "real_combat_mode", False) else 100.0
                        self._prev_local_hp = 100.0
                        if not getattr(self, "real_combat_mode", False):
                            self.local_is_dead = False
                            self.kernel32.WriteProcessMemory(
                                self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                            )
                            self.kernel32.WriteProcessMemory(
                                self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x1BC), struct.pack("<f", 100.0), 4, None
                            )
                            if lp_dead != 0:
                                self.kernel32.WriteProcessMemory(
                                    self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x170), struct.pack("<i", 0), 4, None
                                )
                                self.kernel32.WriteProcessMemory(
                                    self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x16C), struct.pack("<f", 2.0), 4, None
                                )
                                self.kernel32.WriteProcessMemory(
                                    self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x17C), struct.pack("<f", 17.0), 4, None
                                )
                    else:
                        self.local_is_dead = False
                        if getattr(self, "_local_was_dead", False) and (player_y < 480.0 or player_y > 1040.0):
                            self.local_spawn_side = 1 if player_y > 763.0 else 2
                        if lp_hp >= 50.0 and lp_dead == 0:
                            self._local_was_dead = False
                        prev_lp = float(getattr(self, "_prev_local_hp", lp_hp))
                        if lp_hp < prev_lp - 0.5:
                            dmg_taken = prev_lp - lp_hp
                            self._pending_player_real_dmg = float(getattr(self, "_pending_player_real_dmg", 0.0)) + dmg_taken
                        self._prev_local_hp = lp_hp
                        self.local_real_hp = lp_hp
                # Read local player's CBasePlayer* at edict->pvPrivateData (local_pev_addr - 0x04)
                cb_buf = (ctypes.c_char * 4)()
                if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.local_pev_addr - 0x04), cb_buf, 4, None):
                    local_cbase = struct.unpack("<I", cb_buf)[0]
                    if local_cbase > 0x10000:
                        lw_buf = (ctypes.c_char * 24)()
                        if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(local_cbase + 0x5C0), lw_buf, 24, None):
                            l_prim, l_pist, _lk, _lg, _lb, l_act = struct.unpack("<6I", lw_buf)
                            clip30 = struct.pack("<i", 30)
                            for lw_ptr in (l_prim, l_pist, l_act):
                                if lw_ptr > 0x10000:
                                    self.kernel32.WriteProcessMemory(
                                        self.hProc, ctypes.c_void_p(lw_ptr + 0xCC), clip30, 4, None
                                    )
                            wep_id = 28
                            if l_prim > 0x10000:
                                wid_buf = (ctypes.c_char * 4)()
                                if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(l_prim + 0xAC), wid_buf, 4, None):
                                    wep_id = struct.unpack("<i", wid_buf)[0]
                            self.local_holding_secondary = bool(l_prim > 0x10000 and l_act != l_prim)
                            self.local_wrong_primary = bool(l_prim > 0x10000 and wep_id != 28)
                            self.local_missing_primary = bool(l_prim <= 0x10000 or wep_id != 28)
                            if self.local_missing_primary and not self.local_is_dead and self.mp_base:
                                arm_list = getattr(self, "_spawn_armoury_pevs", None)
                                if arm_list:
                                    a_pev, a_priv, _, _, _ = arm_list[0]
                                    ca_vt = self.mp_base + 0x1359FC
                                    vt_chk = (ctypes.c_char * 8)()
                                    pv_chk = (ctypes.c_char * 4)()
                                    ed_chk = (ctypes.c_char * 4)()
                                    valid_arm = False
                                    if (
                                        self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(a_priv), vt_chk, 8, None)
                                        and self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(a_pev - 0x04), pv_chk, 4, None)
                                        and self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x208), ed_chk, 4, None)
                                    ):
                                        a_vt, a_ent_pev = struct.unpack("<2I", vt_chk)
                                        a_pv = struct.unpack("<I", pv_chk)[0]
                                        a_ed = struct.unpack("<I", ed_chk)[0]
                                        valid_arm = (a_vt == ca_vt and a_ent_pev == a_pev and a_pv == a_priv and a_ed == (a_pev - 0x80))
                                    if valid_arm:
                                        pz_a = -143.0
                                        pos_a = struct.pack("<3f", float(player_x), float(player_y), pz_a)
                                        abs_min = struct.pack("<3f", float(player_x) - 36.0, float(player_y) - 36.0, pz_a - 36.0)
                                        abs_max = struct.pack("<3f", float(player_x) + 36.0, float(player_y) + 36.0, pz_a + 36.0)
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x08), pos_a, 12, None)
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x14), pos_a, 12, None)
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x0C4), abs_min, 12, None)
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x0D0), abs_max, 12, None)
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x118), struct.pack("<i", 0), 4, None)
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_pev + 0x10C), struct.pack("<i", 1), 4, None)
                                        # CArmoury::m_iItem (+0x88) = 4 (AK47), m_iCount (+0x8C) = 30
                                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(a_priv + 0x88), struct.pack("<2i", 4, 30), 8, None)
                                    else:
                                        self._spawn_armoury_pevs = []
                                        self._entvars_Scanned = False

            # Pure Spawn-Zone Side Latching on $2000$ (aim_ai_sandbox.bsp):
            # - Top Spawn (Team 1 / T): Y > 1020.0 (center Y = 1375.0)
            # - Bottom Spawn (Team 2 / CT): Y < 500.0 (center Y = 155.0)
            # IMPORTANT: Only latch `local_spawn_side` when the player is first detected or teleports/respawns (> 300u jump),
            # NEVER when walking normally into the enemy spawn zone!
            local_team_id = int(getattr(self, "local_spawn_side", 2))

            has_opposing_team_in_match = False
            bot_teams = {}
            bot_pev_cache = {}
            for pev_addr, csbot_addr, init_x, init_y, init_z in self._server_origins:
                pev_buf = (ctypes.c_char * 0x214)()
                if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(pev_addr), pev_buf, 0x214, None):
                    continue
                bot_pev_cache[pev_addr] = pev_buf
                ox, oy, oz = struct.unpack_from("<3f", pev_buf, 0x08)
                health = struct.unpack_from("<f", pev_buf, 0x160)[0]
                deadflag = struct.unpack_from("<i", pev_buf, 0x170)[0]
                st_b = self._strafe_state.get(pev_addr)
                prev_side = int(st_b.get("spawn_side", 0)) if st_b is not None else 0
                prev_b_pos = st_b.get("last_world_xy", (ox, oy)) if st_b is not None else (init_x, init_y)
                jumped_to_spawn = math.hypot(ox - prev_b_pos[0], oy - prev_b_pos[1]) > 300.0 and (oy < 500.0 or oy > 1020.0)
                was_dead = bool(st_b.get("counted_death", False)) if st_b is not None else False
                in_spawn_full_hp = (deadflag == 0 and health >= 98.0 and (oy < 460.0 or oy > 1060.0))
                if st_b is not None:
                    st_b["last_world_xy"] = (ox, oy)

                # In 1v1 / 1v5 (up to 5 bots total), every bot in the match is an opposing enemy!
                if len(self._server_origins) <= 5:
                    b_team = 1 if local_team_id == 2 else 2
                elif jumped_to_spawn or (was_dead and in_spawn_full_hp) or (prev_side == 0 and in_spawn_full_hp):
                    b_team = 1 if oy > 763.0 else 2
                elif prev_side in (1, 2):
                    b_team = prev_side
                elif init_y > 1020.0 or init_y < 500.0:
                    b_team = 1 if init_y > 763.0 else 2
                else:
                    b_team = 1 if local_team_id == 2 else 2

                if st_b is not None:
                    st_b["spawn_side"] = b_team
                bot_teams[pev_addr] = b_team
                if b_team != local_team_id:
                    has_opposing_team_in_match = True

            max_combat_bots = getattr(self, "max_combat_bots", 1)
            if max_combat_bots is None:
                max_combat_bots = 1
            enemy_slot_idx = 0
            for bot_idx, (pev_addr, csbot_addr, init_x, init_y, init_z) in enumerate(self._server_origins):
                if getattr(self, "real_combat_mode", False) and has_opposing_team_in_match and bot_teams.get(pev_addr) == local_team_id:
                    continue  # Strictly skip allied teammate bot in 5v5 / 2v1 mode!
                pev_buf = bot_pev_cache.get(pev_addr)
                if pev_buf is None:
                    continue
                ox, oy, oz = struct.unpack_from("<3f", pev_buf, 0x08)
                vx, vy, vz = struct.unpack_from("<3f", pev_buf, 0x20)
                health = struct.unpack_from("<f", pev_buf, 0x160)[0]
                deadflag = struct.unpack_from("<i", pev_buf, 0x170)[0]
                flags = struct.unpack_from("<I", pev_buf, 0x1A4)[0]

                if deadflag != 0 or not math.isfinite(health) or health <= 0.0:
                    st_dead = self._strafe_state.get(pev_addr)
                    if st_dead is not None:
                        if getattr(self, "real_combat_mode", False) and not st_dead.get("counted_death", False) and not st_dead.get("parked_curriculum", False):
                            st_dead["counted_death"] = True
                            # In 1v1 / 1v3 / 1v5 (no allied bots), every active enemy bot death is 100% a kill by our AI!
                            was_ai_target = (
                                len(self._server_origins) <= 5
                                or (pev_addr == getattr(self, "_locked_target_addr", None))
                                or ((now - float(st_dead.get("last_ai_hit_t", 0.0))) < 5.0)
                            )
                            if was_ai_target:
                                self._pending_engine_kills += 1
                                self._pending_engine_hits += 1
                                logger.info(f"[NATIVE BOT KILLED] Native CS 1.6 bot 0x{pev_addr:X} eliminated by AI!")
                                # In 1v1 mode, restore AI health to 100 HP upon winning the duel so the next unparked 1v1 duel starts at 100 HP!
                                if max_combat_bots == 1 and self.local_pev_addr and not getattr(self, "local_is_dead", False):
                                    self.kernel32.WriteProcessMemory(
                                        self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                                    )
                                    self.local_real_hp = 100.0
                                    self._prev_local_hp = 100.0
                            st_dead["last_ai_hit_t"] = 0.0
                        st_dead["active"] = False
                    continue

                # Count slot index ONLY among ALIVE enemy bots so when bot_0 dies in 1v1, bot_1 immediately steps in!
                cur_enemy_slot = enemy_slot_idx
                enemy_slot_idx += 1

                if getattr(self, "real_combat_mode", False):
                    is_client_idle = (now - float(getattr(self, "_last_obs_t", now))) > 1.2
                    if is_client_idle and self.local_pev_addr and not getattr(self, "local_is_dead", False):
                        self.kernel32.WriteProcessMemory(
                            self.hProc, ctypes.c_void_p(self.local_pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                        )
                        self.local_real_hp = 100.0
                        self._prev_local_hp = 100.0
                    eff_max_bots = 0 if is_client_idle else max_combat_bots
                    if eff_max_bots is not None and cur_enemy_slot >= int(eff_max_bots):
                        # Park surplus (or pause-idled) bot directly under its own spawn CNavArea and empty its clip so it NEVER shoots during AUTO-PAUSE!
                        b_side_p = bot_teams.get(pev_addr, 1 if local_team_id == 2 else 2)
                        park_x = -768.0 + max(0, cur_enemy_slot - 1) * 45.0
                        park_y = 1375.0 if b_side_p == 1 else 155.0
                        park_z = -420.0
                        p_bytes = struct.pack("<3f", float(park_x), float(park_y), float(park_z))
                        z_vec = struct.pack("<3f", 0.0, 0.0, 0.0)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x08), p_bytes, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x14), p_bytes, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), z_vec, 12, None)
                        # Set clip ammo = 0 while parked so parked bots cannot fire a single bullet
                        bw_buf = (ctypes.c_char * 24)()
                        if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x5C0), bw_buf, 24, None):
                            b_prim, b_pist, _, _, _, b_act = struct.unpack("<6I", bw_buf)
                            zero_clip = struct.pack("<i", 0)
                            for bw_ptr in (b_prim, b_pist, b_act):
                                if bw_ptr > 0x10000:
                                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(bw_ptr + 0xCC), zero_clip, 4, None)
                        st_p = self._strafe_state.get(pev_addr)
                        if st_p is None:
                            st_p = {"active": False, "csbot_addr": csbot_addr, "parked_curriculum": True}
                            self._strafe_state[pev_addr] = st_p
                        else:
                            st_p["active"] = False
                            st_p["parked_curriculum"] = True
                        continue

                    # Phase 8: 100% NATIVE CS 1.6 GAME BOTS (CCSBot::Update, native vision, native pathfinding & shooting):
                    # NEVER teleport bots, NEVER overwrite their v_angle (+0x50), velocity (+0x20), or m_forwardSpeed (+0x9D4)!
                    # Only clean up any leftover 10000.0 sentinel HP from Phase 7 once so bots have standard 100->0 HP.
                    st_nat = self._strafe_state.get(pev_addr)
                    if (st_nat is not None and st_nat.get("parked_curriculum", False)) or oz < -240.0 or ox > 2000.0:
                        # Unpark bot back to the OPPOSITE spawn zone from our player so it never spawns behind/on top of our player!
                        ox = -768.0 + max(0, cur_enemy_slot) * 85.0
                        oy = 155.0 if player_y > 763.0 else 1375.0
                        oz = -143.0
                        vx, vy, vz = 0.0, 0.0, 0.0
                        unp_bytes = struct.pack("<3f", float(ox), float(oy), float(oz))
                        z_vec = struct.pack("<3f", 0.0, 0.0, 0.0)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x08), unp_bytes, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x14), unp_bytes, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), z_vec, 12, None)
                        # Restore 30 bullets in clip when unparked into active combat
                        bw_buf = (ctypes.c_char * 24)()
                        if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x5C0), bw_buf, 24, None):
                            b_prim, b_pist, _, _, _, b_act = struct.unpack("<6I", bw_buf)
                            full_clip = struct.pack("<i", 30)
                            for bw_ptr in (b_prim, b_pist, b_act):
                                if bw_ptr > 0x10000:
                                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(bw_ptr + 0xCC), full_clip, 4, None)
                        if st_nat is not None:
                            st_nat["parked_curriculum"] = False
                            st_nat["prev_hp"] = float(health)
                    if health > 100.5:
                        health = 100.0
                        self.kernel32.WriteProcessMemory(
                            self.hProc, ctypes.c_void_p(pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                        )
                    st_nat = self._strafe_state.get(pev_addr)
                    if st_nat is None:
                        st_nat = {
                            "active": True,
                            "csbot_addr": csbot_addr,
                            "phase": "native_csbot",
                            "corner_name": "NATIVE_AI",
                            "prev_hp": float(health),
                            "counted_death": False,
                            "spawn_side": bot_teams.get(pev_addr, 1 if oy > 763.0 else 2),
                            "last_world_xy": (ox, oy),
                            "last_ai_hit_t": 0.0,
                        }
                        self._strafe_state[pev_addr] = st_nat
                    else:
                        st_nat["active"] = True
                        st_nat["phase"] = "native_csbot"
                        st_nat["corner_name"] = "NATIVE_AI"
                        st_nat["counted_death"] = False
                        prev_b_hp = float(st_nat.get("prev_hp", health))
                        if health < prev_b_hp - 0.5:
                            if len(self._server_origins) <= 5 or pev_addr == getattr(self, "_locked_target_addr", None):
                                self._pending_engine_hits += 1
                                st_nat["last_ai_hit_t"] = now
                        st_nat["prev_hp"] = float(health)
                        self.target_hit_counts[pev_addr] = min(6, int(round((100.0 - float(health)) / 100.0 * 6.0)))
                    active_count += 1
                    key = (round(ox / 8.0), round(oy / 8.0), round(oz / 8.0))
                    if key in seen_positions:
                        continue
                    seen_positions.add(key)
                    enemies.append(PlayerState(
                        address=pev_addr,
                        origin=Vector3(ox, oy, oz + 15.0),
                        view_angles=Vector3(0.0, math.pi, 0.0),
                        velocity=Vector3(vx if math.isfinite(vx) else 0.0, vy if math.isfinite(vy) else 0.0, vz if math.isfinite(vz) else 0.0),
                        health=max(1, int(round(health))),
                        team=bot_teams.get(pev_addr, 1),
                        is_alive=True,
                        is_local=False,
                    ))
                    continue

                if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
                    vx, vy, vz = 0.0, 0.0, 0.0
                    z_vel = struct.pack("<3f", 0.0, 0.0, 0.0)
                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), z_vel, 12, None)
                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x2C), z_vel, 12, None)

                if not (
                    math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oz)
                    and abs(ox) < 8000.0 and abs(oy) < 8000.0 and abs(oz) < 8000.0
                ):
                    ox, oy, oz = -1410.0, 315.0, -143.0
                    st_bad = self._strafe_state.get(pev_addr)
                    if st_bad is not None:
                        st_bad["active"] = False

                if getattr(self, "static_enemies", False):
                    # Phase 5 Static Curriculum mode
                    v_angle_bytes = struct.pack("<3f", 0.0, 180.0, 0.0)
                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x50), v_angle_bytes, 12, None)
                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x74), v_angle_bytes, 12, None)
                    cmd_bytes = struct.pack("<3fH", 0.0, 0.0, 0.0, 0)
                    self.kernel32.WriteProcessMemory(
                        self.hProc, ctypes.c_void_p(csbot_addr + 0x9D4), cmd_bytes, 14, None
                    )
                    zero_vel = struct.pack("<3f", 0.0, 0.0, 0.0)
                    self.kernel32.WriteProcessMemory(
                        self.hProc, ctypes.c_void_p(pev_addr + 0x38), zero_vel, 12, None
                    )
                    vx, vy, vz = 0.0, 0.0, 0.0
                else:
                    # Dynamic Multi-Corner Ambush & 1v5 Live Combat Carousel:
                    is_active_rusher = active_count < self.MAX_ACTIVE_CAROUSEL_BOTS
                    if not is_active_rusher:
                        st_park = self._strafe_state.get(pev_addr)
                        opp_side = player_x < -770.0
                        if st_park is None or st_park.get("active", True) or st_park.get("opp_side") != opp_side:
                            park_idx = max(0, bot_idx - self.MAX_ACTIVE_CAROUSEL_BOTS)
                            park_x = (-135.0 if opp_side else -1410.0)
                            park_y = 330.0 + (park_idx % 6) * 145.0
                            park_z = -143.0
                            p_bytes = struct.pack("<3f", float(park_x), float(park_y), float(park_z))
                            z_vec = struct.pack("<3f", 0.0, 0.0, 0.0)
                            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x08), p_bytes, 12, None)
                            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x14), p_bytes, 12, None)
                            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), z_vec, 12, None)
                            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x160), struct.pack("<f", 10000.0), 4, None)
                            self._strafe_state[pev_addr] = {
                                "active": False,
                                "opp_side": opp_side,
                                "csbot_addr": csbot_addr,
                                "last_pos": (park_x, park_y),
                            }
                        cmd_zero = struct.pack("<3fH", 0.0, 0.0, 0.0, 0)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x9D4), cmd_zero, 14, None)
                        continue

                    slot_idx = active_count
                    active_count += 1

                    st = self._strafe_state.get(pev_addr)
                    on_balcony = oz > -115.0
                    # Recycle any bot that was left behind on a completed leg so upcoming corridors are NEVER empty!
                    stale_corner = False
                    if st is not None and st.get("active", False) and not getattr(self, "dynamic_strafe_mode", False):
                        c_leg = int(st.get("corner_leg", 0))
                        if c_leg not in (cur_leg, (cur_leg + 1) % 4):
                            stale_corner = True

                    if (
                        st is None
                        or not st.get("active", False)
                        or stale_corner
                        or on_balcony
                    ):
                        st = {
                            "active": True,
                            "slot_idx": slot_idx,
                            "csbot_addr": csbot_addr,
                            "upper_side": upper_side,
                            "dir": 1.0 if (slot_idx % 2 == 0) else -1.0,
                            "speed": 185.0 + (slot_idx % 4) * 16.0,
                            "last_flip_t": now,
                            "last_dodge_t": 0.0,
                            "last_lat": ox,
                            "last_pos": (ox, oy),
                            "stuck_t": now,
                            "next_juke_dt": 0.95 + (slot_idx % 4) * 0.25,
                            "phase": "corner_wait",
                            "phase_t": now,
                            "wait_until": now + 0.15 + slot_idx * 0.35,
                            "hit_boost_until": 0.0,
                            "real_dmg_taken": 0.0,
                        }
                        self._strafe_state[pev_addr] = st
                        ox, oy, oz = self._teleport_bot_to_corner(
                            pev_addr,
                            csbot_addr,
                            st,
                            now,
                            wait_sec=0.15 + slot_idx * 0.35,
                            upper_side=upper_side,
                        )
                        vx, vy, vz = 0.0, 0.0, 0.0
                    else:
                        st["slot_idx"] = slot_idx
                        st["csbot_addr"] = csbot_addr
                        st["last_pos"] = (ox, oy)

                        # Detect REAL GoldSrc bullet hits via sentinel pev->health (10000.0):
                        # Every real bullet hit drops health below 10000.0 -> record damage,
                        # and when accumulated real damage >= 100 HP, INSTANTLY kill & stage at the next unused corner ahead!
                        if health < 9999.5:
                            real_dmg = max(0.0, 10000.0 - float(health))
                            self.kernel32.WriteProcessMemory(
                                self.hProc, ctypes.c_void_p(pev_addr + 0x160), struct.pack("<f", 10000.0), 4, None
                            )
                            if 1.0 <= real_dmg <= 500.0 and st.get("phase") != "corner_wait":
                                st["real_dmg_taken"] = float(st.get("real_dmg_taken", 0.0)) + real_dmg
                                st["flinch_until"] = now + 0.35
                                self._pending_engine_hits += 1
                                eq_hits = min(6, int(round(st["real_dmg_taken"] / 100.0 * 6.0)))
                                if eq_hits > self.target_hit_counts.get(pev_addr, 0):
                                    self.target_hit_counts[pev_addr] = eq_hits

                                if st["real_dmg_taken"] >= 100.0:
                                    self._pending_engine_kills += 1
                                    killed_c_idx = st.get("corner_idx")
                                    if killed_c_idx is not None:
                                        if not hasattr(self, "_used_corners_this_lap"):
                                            self._used_corners_this_lap = set()
                                        self._used_corners_this_lap.add(int(killed_c_idx))
                                    ox, oy, oz = self._teleport_bot_to_corner(
                                        pev_addr,
                                        csbot_addr,
                                        st,
                                        now,
                                        wait_sec=0.25,
                                        upper_side=upper_side,
                                    )
                                    logger.info(
                                        f"[ENGINE KILL] Bot 0x{pev_addr:X} killed -> staged at next unused corner {st.get('corner_name')}!"
                                    )
                        elif health < 9950.0 or health > 10050.0:
                            self.kernel32.WriteProcessMemory(
                                self.hProc, ctypes.c_void_p(pev_addr + 0x160), struct.pack("<f", 10000.0), 4, None
                            )

                    cur_phase = st.get("phase", "corner_wait")
                    c_leg = int(st.get("corner_leg", cur_leg))
                    pkx, pky = st.get("peek_target", (ox, oy))

                    # Ensure bot has a rifle/pistol equipped with infinite clip (m_iClip = 30) and zero equip timer:
                    wep_buf = (ctypes.c_char * 24)()
                    primary_ptr = 0
                    if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x5C0), wep_buf, 24, None):
                        primary_ptr, pistol_ptr, _knife_ptr, _g_ptr, _b_ptr, active_ptr = struct.unpack("<6I", wep_buf)
                        self.kernel32.WriteProcessMemory(
                            self.hProc, ctypes.c_void_p(csbot_addr + 0x2AC8), struct.pack("<f", 0.0), 4, None
                        )
                        clip_bytes = struct.pack("<i", 30)
                        for w_ptr in (primary_ptr, pistol_ptr, active_ptr):
                            if w_ptr > 0x10000:
                                self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(w_ptr + 0xCC), clip_bytes, 4, None)

                    # Check true 2D Line-of-Sight between this bot (ox, oy) and the AI player (player_x, player_y)
                    dist_to_ai = math.hypot(player_x - ox, player_y - oy)
                    bot_sees_ai = dist_to_ai <= 1450.0
                    if bot_sees_ai:
                        for xmin, ymin, xmax, ymax in self.SOLID_WALL_AABBS:
                            if self._segment_intersects_aabb(ox, oy, player_x, player_y, xmin, ymin, xmax, ymax, pad=0.0):
                                bot_sees_ai = False
                                break

                    is_live_combat = bool(getattr(self, "real_combat_mode", False)) or bool(getattr(self, "dynamic_strafe_mode", False))
                    fwd_spd_cmd = 0.0
                    str_spd_cmd = 0.0
                    buttons = 0

                    if getattr(self, "dynamic_strafe_mode", False) and slot_idx < 3:
                        # Stage 2 Dynamic Strafe in Corridor 1 (X in [-1315..-1195], Y = 380, 560, 740)
                        lane_y = 380.0 + slot_idx * 180.0
                        if abs(oy - lane_y) > 45.0 or ox < -1330.0 or ox > -1175.0:
                            ox = -1255.0 + (slot_idx - 1) * 35.0
                            oy = lane_y
                            oz = -143.0
                            p_bytes = struct.pack("<3f", float(ox), float(oy), float(oz))
                            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x08), p_bytes, 12, None)
                            self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x14), p_bytes, 12, None)
                        s_dir = float(st.get("dir", 1.0))
                        if ox <= -1310.0:
                            s_dir = 1.0
                        elif ox >= -1200.0:
                            s_dir = -1.0
                        elif now - float(st.get("last_flip_t", 0.0)) > float(st.get("next_juke_dt", 0.95)):
                            s_dir = -s_dir
                            st["last_flip_t"] = now
                        st["dir"] = s_dir
                        st["corner_leg"] = 0
                        st["peek_target"] = (ox, oy)
                        st["phase"] = "dynamic_strafe"
                        s_spd = float(st.get("speed", 205.0))
                        vx_bot = s_dir * s_spd
                        vel_bytes = struct.pack("<3f", float(vx_bot), 0.0, 0.0)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), vel_bytes, 12, None)
                        vx, vy, vz = vx_bot, 0.0, 0.0
                    else:
                        # "corner_hold": Bot waits at its corner ambush position until the AI approaches
                        st["saw_ai_at"] = 0.0
                        ox, oy, oz = float(pkx), float(pky), -143.0
                        hold_bytes = struct.pack("<3f", ox, oy, oz)
                        z_vec = struct.pack("<3f", 0.0, 0.0, 0.0)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x08), hold_bytes, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x14), hold_bytes, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x20), z_vec, 12, None)
                        self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x2C), z_vec, 12, None)
                        st["phase"] = "corner_hold"
                        st["last_pos"] = (ox, oy)
                        vx, vy, vz = 0.0, 0.0, 0.0

                    self.kernel32.WriteProcessMemory(
                        self.hProc, ctypes.c_void_p(pev_addr + 0x210), struct.pack("<f", 0.0), 4, None
                    )
                    if flags & 0x1000:
                        self.kernel32.WriteProcessMemory(
                            self.hProc, ctypes.c_void_p(pev_addr + 0x1A4), struct.pack("<I", flags & ~0x1000), 4, None
                        )

                    # Point bot's v_angle (+0x50) and model angles (+0x74) directly at the AI player!
                    aim_yaw_rad = math.atan2(player_y - oy, player_x - ox)
                    aim_yaw_deg = math.degrees(aim_yaw_rad)
                    v_angle_bytes = struct.pack("<3f", 0.0, float(aim_yaw_deg), 0.0)
                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x50), v_angle_bytes, 12, None)
                    self.kernel32.WriteProcessMemory(self.hProc, ctypes.c_void_p(pev_addr + 0x74), v_angle_bytes, 12, None)

                    # Write m_forwardSpeed (+0x9D4), m_strafeSpeed (+0x9D8), m_verticalSpeed (+0x9DC), m_buttonFlags (+0x9E0)
                    cmd_bytes = struct.pack("<3fH", float(fwd_spd_cmd), float(str_spd_cmd), 0.0, int(buttons))
                    self.kernel32.WriteProcessMemory(
                        self.hProc, ctypes.c_void_p(csbot_addr + 0x9D4), cmd_bytes, 14, None
                    )

                key = (round(ox / 8.0), round(oy / 8.0), round(oz / 8.0))
                if key in seen_positions:
                    continue
                seen_positions.add(key)

                enemies.append(PlayerState(
                    address=pev_addr,
                    origin=Vector3(ox, oy, oz + 15.0),  # Upper chest/neck height (matches player eye_z +17.03)
                    view_angles=Vector3(0.0, math.pi, 0.0),
                    velocity=Vector3(vx, vy, vz),
                    health=100,
                    team=1,
                    is_alive=True,
                    is_local=False,
                ))

            # Heartbeat diagnostics every 5 s: proves whether commands reach the bots and whether they move
            if now - getattr(self, "_last_diag_t", 0.0) > 5.0:
                self._last_diag_t = now
                try:
                    pev0, bot0 = self._server_origins[0][0], self._server_origins[0][1]
                    o_buf = (ctypes.c_char * 12)()
                    c_buf = (ctypes.c_char * 8)()
                    s_buf = (ctypes.c_char * 4)()
                    self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(pev0 + 0x08), o_buf, 12, None)
                    self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(bot0 + 0x9D4), c_buf, 8, None)
                    self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.mp_base + 0x171D14), s_buf, 4, None)
                    o0 = struct.unpack("<3f", bytes(o_buf))
                    fs, ss = struct.unpack("<2f", bytes(c_buf))
                    bstop = struct.unpack("<f", bytes(s_buf))[0]
                    st0 = self._strafe_state.get(pev0, {})
                    ph0 = st0.get("phase", "?")
                    cn0 = st0.get("corner_name", "?")
                    used_n = len(getattr(self, "_used_corners_this_lap", ()))
                    logger.info(
                        f"[AMBUSH ENGINE] leg={cur_leg} player=({player_x:.0f},{player_y:.0f}) "
                        f"used_corners={used_n}/{len(self.AMBUSH_CORNERS)} visible={len(enemies)}/{active_count} patched={self._mp_patched} "
                        f"bot_stop={bstop:.0f} | bot0 corner={cn0} phase={ph0} pos=({o0[0]:.0f},{o0[1]:.0f},{o0[2]:.0f})"
                    )
                except Exception as e:
                    logger.warning(f"[AMBUSH ENGINE] diag failed: {e}")

            # When server CCSBot entities are active, return `enemies` directly
            self._last_active_count = active_count
            if active_count > 0:
                return enemies

        # Do NOT fall back to DUMMY_TABLE_OFFSET when static_enemies is False, because sv.edicts[5..15] on $2000$
        # contains floor weapons (armoury_entity) and spawn points when 0 bots are spawned!
        if not getattr(self, "static_enemies", False):
            self._last_active_count = 0
            if now - getattr(self, "_last_nobot_warn_t", 0.0) > 5.0:
                self._last_nobot_warn_t = now
                logger.warning(
                    "[AMBUSH ENGINE] На карте 0 живых ботов CCSBot! Добавьте ботов в CS 1.6 (в консоли: bot_add или меню H -> Add Bot)!"
                )
            return []

        # Fallback path if server entvars scan returned empty
        base_table = self.hw_base + self.DUMMY_TABLE_OFFSET
        exclude = self.custom_offsets["exclude_slots"] if self.custom_offsets else set()
        for idx in range(5, 16):
            if idx in exclude:
                continue
            ent_addr = base_table + idx * 0xE0
            buf = (ctypes.c_char * 12)()
            if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(ent_addr), buf, 12, None):
                ex, ey, ez = struct.unpack_from("<3f", buf, 0)
                if abs(ex) > 10.0 and abs(ey) > 10.0 and abs(ex) < 4000.0 and abs(ey) < 4000.0 and -220.0 <= ez <= -100.0:
                    key = (round(ex / 8.0), round(ey / 8.0), round(ez / 8.0))
                    if key in seen_positions:
                        continue
                    seen_positions.add(key)
                    enemies.append(PlayerState(
                        address=ent_addr,
                        origin=Vector3(ex, ey, ez + 15.0),
                        view_angles=Vector3(0, 0, 0),
                        velocity=Vector3(0.0, 0.0, 0.0),
                        health=100,
                        team=1,
                        is_alive=True,
                        is_local=False
                    ))
        return enemies

    def get_observation(self, burst_ticks: int = 0, recoil_ticks: int = 0) -> Optional[Dict]:
        """
        Compute full observation vector for RL agent.

        burst_ticks : 0..2 -> hold / single-tap (aim point = head), >= 3 -> spray (aim point = upper chest)
        recoil_ticks: number of consecutive spray ticks already fired in the current burst. The aim point is
                      lowered by RECOIL_COMP_DEG_PER_TICK per tick so that crosshair + view punch lands on target.
        """
        self._last_obs_t = time.time()
        local = self.get_local_player()
        if not local:
            return None
        
        enemies = self.get_enemies(local.team, local.origin)
        closest_enemy = None
        min_dist = float('inf')
        best_score = float('inf')
        rel_yaw = 0.0
        rel_pitch = 0.0
        visible = 0.0
        
        # Vertical aim point on the target silhouette (with true standing/crouching eye_z in get_local_player):
        # - Opening 2 bullets of tap or burst (burst_ticks <= 3): dead-center HEADSHOT (+5.5 above oz+15.0 = oz+20.5)
        # - Sustained spray (burst_ticks >= 4): upper chest (+0.0 = oz+15.0) + real GoldSrc engine punchangle (2.0 * pev->punchangle)
        dz_offset = 5.5 if burst_ticks <= 3 else 0.0
        punch_pitch_rad = float(local.punch_angle.x) if local.punch_angle else 0.0
        punch_yaw_rad = float(local.punch_angle.y) if local.punch_angle else 0.0
        if abs(punch_pitch_rad) > 1e-4 or abs(punch_yaw_rad) > 1e-4:
            # In GoldSrc (CS 1.6), bullets fire along view_angles + 2.0 * punchangle.
            # Including 2.0 * punchangle directly in the aim error lets the Pure AI see true weapon kick and pull down naturally!
            recoil_comp_rad = -2.0 * punch_pitch_rad
            recoil_yaw_comp_rad = 2.0 * punch_yaw_rad
        else:
            recoil_comp_rad = math.radians(
                min(self.RECOIL_COMP_DEG_PER_TICK * max(0, int(recoil_ticks)), self.RECOIL_COMP_MAX_DEG)
            )
            recoil_yaw_comp_rad = 0.0
        
        px, py = local.origin.x, local.origin.y
        cur_leg = self.get_loop_leg(px, py)
        wp_x, wp_y, leg_yaw = self.get_leg_waypoint(cur_leg, px, py)
        real_combat = bool(getattr(self, "real_combat_mode", False))

        # Exact 360-degree Line-of-Sight check:
        # In Phase 8 (real_combat_mode=True), check true 360° LOS across the entire arena (up to 1600u, any corridor/leg).
        # In Phase 7 corner curriculum, engage static corner bots within <= 385u on the active leg.
        def _has_corridor_los(e: PlayerState) -> bool:
            ex, ey = e.origin.x, e.origin.y
            d = math.hypot(ex - px, ey - py)
            if real_combat:
                if d < 20.0 or d > 1650.0:
                    return False
                for xmin, ymin, xmax, ymax in self.SOLID_WALL_AABBS:
                    if self._segment_intersects_aabb(px, py, ex, ey, xmin, ymin, xmax, ymax, pad=0.0):
                        return False
                return True
            max_eng_d = 650.0 if getattr(self, "dynamic_strafe_mode", False) else 385.0
            if d < 25.0 or d > max_eng_d:
                return False
            st_c = self._strafe_state.get(e.address)
            if st_c is not None and int(st_c.get("corner_leg", cur_leg)) != cur_leg:
                return False
            # Once a corner bot on the current leg has been revealed with 8-unit doorway clearance,
            # latch `in_los = True` until that corner bot is killed so the AI NEVER skips it!
            if st_c is not None and st_c.get("in_los", False):
                return True
            req_pad = 0.0 if getattr(self, "dynamic_strafe_mode", False) else 8.0
            for xmin, ymin, xmax, ymax in self.SOLID_WALL_AABBS:
                if self._segment_intersects_aabb(px, py, ex, ey, xmin, ymin, xmax, ymax, pad=req_pad):
                    return False
            if st_c is not None:
                st_c["in_los"] = True
            return True

        corridor_enemies = [e for e in enemies if _has_corridor_los(e)]
        now = time.time()
        # Expire old elimination cooldowns (so respawned/moving bots become active again)
        expired_keys = [k for k, t_exp in self.eliminated_until.items() if now >= t_exp]
        for k in expired_keys:
            self.eliminated_until.pop(k, None)

        active_enemies = [
            e for e in corridor_enemies
            if ("slot", e.address) not in self.eliminated_until
            and ("pos", round(e.origin.x / 24.0), round(e.origin.y / 24.0)) not in self.eliminated_until
        ]
        if not active_enemies and corridor_enemies and self.eliminated_until:
            logger.info("All corridor targets cycled! Resetting eliminated target list...")
            self.eliminated_until.clear()
            self.target_hit_counts.clear()
            active_enemies = corridor_enemies

        prev_locked = getattr(self, "_locked_target_addr", None)
        self.current_target_key = None
        target_vel = [0.0, 0.0, 0.0]
        incoming_dmg = float(getattr(self, "_pending_player_real_dmg", 0.0))
        self._pending_player_real_dmg = 0.0
        for enemy in active_enemies:
            dist = local.origin.distance_to(enemy.origin)
            st_e = self._strafe_state.get(enemy.address)
            if st_e is not None:
                if not st_e.get("in_los", False):
                    st_e["in_los"] = True
                    st_e["visible_since"] = now

            # Velocity lead prediction for moving targets (compensates 45ms frame latency at 200+ u/s strafe speed)
            lead_dt = min(0.055, dist / 4500.0) if (real_combat or getattr(self, "dynamic_strafe_mode", False)) else 0.0
            dx = (enemy.origin.x + enemy.velocity.x * lead_dt) - local.origin.x
            dy = (enemy.origin.y + enemy.velocity.y * lead_dt) - local.origin.y
            dz = (enemy.origin.z + dz_offset) - local.origin.z
            
            target_yaw = math.atan2(dy, dx)
            dist_2d = max(1.0, math.hypot(dx, dy))
            # GoldSrc pitch > 0 = looking down: adding recoil_comp_rad aims slightly LOWER during spray
            target_pitch = -math.atan2(dz, dist_2d) + recoil_comp_rad
            
            cand_yaw = (target_yaw - (local.view_angles.y + recoil_yaw_comp_rad) + math.pi) % (2 * math.pi) - math.pi
            cand_pitch = target_pitch - local.view_angles.x
            angle_err = math.hypot(cand_yaw, cand_pitch)
            
            # Target Priority Scoring:
            # - In Phase 8 (real_combat_mode) or Dynamic Strafe: Threat Priority Selector!
            #   Prioritizes closest threat + smallest angular flick (fast spray transfer) + wounded targets + lock stickiness.
            # - In Phase 7 static corners: sequential corner_idx order.
            if not real_combat and not getattr(self, "dynamic_strafe_mode", False) and st_e is not None and "corner_idx" in st_e:
                score = float(st_e["corner_idx"]) * 10.0 + (dist / 400.0)
            else:
                stickiness = -0.25 if (prev_locked is not None and enemy.address == prev_locked) else 0.0
                wounded_bonus = -0.30 * (1.0 - float(enemy.health) / 100.0)
                score = (dist / 450.0) + 0.35 * angle_err + stickiness + wounded_bonus
            if score < best_score:
                best_score = score
                min_dist = dist
                closest_enemy = enemy
                self.current_target_key = (
                    enemy.address,
                    round(enemy.origin.x / 24.0),
                    round(enemy.origin.y / 24.0)
                )
                rel_yaw = cand_yaw
                rel_pitch = cand_pitch
                target_vel = [
                    float(np.clip(enemy.velocity.x / 250.0, -1.0, 1.0)),
                    float(np.clip(enemy.velocity.y / 250.0, -1.0, 1.0)),
                    float(np.clip(enemy.velocity.z / 250.0, -1.0, 1.0)),
                ]
        self._locked_target_addr = closest_enemy.address if closest_enemy is not None else None

        # In real_combat_mode, allow full 360-degree (±180° yaw) visibility when an enemy has clear LOS so the AI can
        # whip around 180° if flanked from behind!
        max_vis_yaw_deg = 180.0 if real_combat else 135.0
        in_cover_combat = False
        tactical_mode = getattr(self, "_last_tactical_mode", "PUSH_PILLARS")
        threat_yaw = leg_yaw
        if closest_enemy is not None and abs(rel_yaw) <= math.radians(max_vis_yaw_deg) and abs(rel_pitch) <= math.radians(40.0):
            visible = 1.0
            self._last_seen_enemy_t = now
            self._last_seen_enemy_addr = closest_enemy.address
        else:
            visible = 0.0
            # 2.5-Second Tactical Cover / Jiggle-Peek Memory:
            # If the AI just stepped behind a pillar/corner wall during an active firefight (< 2.5s since last LOS)
            # and that enemy is still alive, keep tracking the enemy's angular direction behind the wall so the Pure AI
            # can hold pre-aim on the corner edge and jiggle-peek back out without autopilot hijacking movement!
            last_addr = getattr(self, "_last_seen_enemy_addr", None)
            last_seen_dt = now - float(getattr(self, "_last_seen_enemy_t", 0.0))
            engaged_enemy = next((e for e in enemies if e.address == last_addr), None) if last_addr is not None else None
            if engaged_enemy is None:
                self._last_seen_enemy_t = 0.0
                self._last_seen_enemy_addr = None
            elif real_combat and last_seen_dt <= 2.5:
                dist_cov = local.origin.distance_to(engaged_enemy.origin)
                if dist_cov <= 1350.0:
                    in_cover_combat = True
                    min_dist = dist_cov
                    closest_enemy = engaged_enemy
                    dx_c = engaged_enemy.origin.x - local.origin.x
                    dy_c = engaged_enemy.origin.y - local.origin.y
                    dz_c = (engaged_enemy.origin.z + dz_offset) - local.origin.z
                    tyaw_c = math.atan2(dy_c, dx_c)
                    d2d_c = max(1.0, math.hypot(dx_c, dy_c))
                    tpitch_c = -math.atan2(dz_c, d2d_c)
                    rel_yaw = (tyaw_c - local.view_angles.y + math.pi) % (2.0 * math.pi) - math.pi
                    rel_pitch = tpitch_c - local.view_angles.x

            # Phase 10: Autonomous PUSH / DEF Tactical Intelligence & Diagonal Spawn Awareness on $2000$
            tactical_mode = "PUSH_PILLARS"
            threat_yaw = leg_yaw
            if real_combat:
                prev_t_xy = getattr(self, "_prev_tactical_xy", None)
                self._prev_tactical_xy = (px, py)
                respawned_now = (
                    prev_t_xy is None
                    or math.hypot(px - prev_t_xy[0], py - prev_t_xy[1]) > 240.0
                    or (now - getattr(self, "_round_spawn_t", 0.0) > 28.0 and (py < 310.0 or py > 1210.0))
                )
                if respawned_now:
                    self._round_spawn_t = now
                    self._spawn_is_bottom = py < 763.0
                    self._spawn_is_left = px < -770.0
                    self._round_id = getattr(self, "_round_id", 0) + 1
                    self._initial_tactic = ("DEF_POCKET", "PUSH_PILLARS", "DEF_POCKET", "PUSH_FLANK_OPP")[self._round_id % 4]

                prev_edist = float(getattr(self, "_prev_any_enemy_dist", min_dist))
                prev_edist_t = float(getattr(self, "_prev_any_enemy_t", now - 0.04))
                dt_e = max(0.02, min(0.5, now - prev_edist_t))
                if enemies:
                    nearest_any = min(enemies, key=lambda e: local.origin.distance_to(e.origin))
                    cur_any_dist = local.origin.distance_to(nearest_any.origin)
                    if not in_cover_combat:
                        min_dist = cur_any_dist
                    ex_n, ey_n = nearest_any.origin.x, nearest_any.origin.y
                else:
                    cur_any_dist = min_dist
                    ex_n = -320.0 if getattr(self, "_spawn_is_left", True) else -1220.0
                    ey_n = 1365.0 if getattr(self, "_spawn_is_bottom", True) else 165.0

                approach_vel = (prev_edist - cur_any_dist) / dt_e if math.isfinite(cur_any_dist) and math.isfinite(prev_edist) else 0.0
                self._prev_any_enemy_dist = cur_any_dist
                self._prev_any_enemy_t = now

                time_since_spawn = now - float(getattr(self, "_round_spawn_t", now))
                self_hp_frac = float(np.clip(getattr(self, "local_real_hp", 100.0) / 100.0, 0.0, 1.0))
                init_tac = getattr(self, "_initial_tactic", "DEF_POCKET")

                enemy_rushing_us = (approach_vel > 110.0 and cur_any_dist < 1150.0)
                multi_threat_in_los = len(corridor_enemies) >= 2
                if multi_threat_in_los or self_hp_frac < 0.65 or enemy_rushing_us or (init_tac == "DEF_POCKET" and time_since_spawn < 2.2):
                    tactical_mode = "DEF_POCKET"
                elif init_tac == "PUSH_FLANK_OPP" and time_since_spawn < 4.5 and (py < 340.0 or py > 1180.0):
                    tactical_mode = "PUSH_FLANK_OPP"
                else:
                    tactical_mode = "PUSH_PILLARS"
                self._last_tactical_mode = tactical_mode

                if tactical_mode == "DEF_POCKET":
                    on_left = px < -770.0
                    on_bot = py < 763.0
                    jiggle_phase = math.sin(now * 4.8)
                    if py < 360.0 or py > 1160.0:
                        pocket_y = 188.0 if on_bot else 1352.0
                        if on_left:
                            wp_x = -1262.0 if jiggle_phase > 0.15 else -1375.0
                            wp_y = pocket_y
                        else:
                            wp_x = -278.0 if jiggle_phase > 0.15 else -165.0
                            wp_y = pocket_y
                        threat_yaw = (math.pi / 2.0) if on_bot else (-math.pi / 2.0)
                        leg_yaw = threat_yaw
                    else:
                        alcove_ys = (188.0, 520.0, 840.0, 1200.0) if on_bot else (1352.0, 995.0, 675.0, 315.0)
                        nearest_ay = min(alcove_ys, key=lambda ay: abs(py - ay))
                        if on_left:
                            wp_x = -1255.0 if jiggle_phase > 0.20 else -1365.0
                        else:
                            wp_x = -285.0 if jiggle_phase > 0.20 else -175.0
                        wp_y = nearest_ay
                        threat_yaw = math.atan2(ey_n - py, ex_n - px)
                        leg_yaw = (math.pi / 2.0) if ey_n > py else (-math.pi / 2.0)
                elif tactical_mode == "PUSH_FLANK_OPP":
                    on_bot = py < 763.0
                    target_corridor_x = -285.0 if getattr(self, "_spawn_is_left", True) else -1255.0
                    wp_x = target_corridor_x
                    wp_y = 240.0 if on_bot else 1300.0
                    if abs(px - target_corridor_x) < 290.0:
                        wp_y = 520.0 if on_bot else 1010.0
                        threat_yaw = (math.pi / 2.0) if on_bot else (-math.pi / 2.0)
                    else:
                        threat_yaw = math.atan2((520.0 if on_bot else 1010.0) - py, target_corridor_x - px)
                    leg_yaw = threat_yaw
                else:
                    s_self = self._perimeter_param(px, py)
                    s_enemy = self._perimeter_param(ex_n, ey_n)
                    cw_dist = (s_enemy - s_self) % 4380.0
                    ccw_dist = (s_self - s_enemy) % 4380.0
                    prev_dir = int(getattr(self, "_patrol_dir_sign", 1))
                    if cw_dist + 120.0 < ccw_dist:
                        dir_sign = 1
                    elif ccw_dist + 120.0 < cw_dist:
                        dir_sign = -1
                    else:
                        dir_sign = prev_dir
                    self._patrol_dir_sign = dir_sign
                    wp_x, wp_y, leg_yaw = self.get_leg_waypoint(cur_leg, px, py, dir_sign=dir_sign)

                    if cur_leg in (0, 2) and 540.0 <= py <= 960.0 and abs(ex_n - px) > 650.0 and 450.0 <= ey_n <= 1080.0:
                        tactical_mode = "PUSH_MID_SHOP"
                        wp_x = -770.0
                        wp_y = 600.0 if abs(py - 600.0) < abs(py - 920.0) else 920.0
                        threat_yaw = math.atan2(ey_n - py, ex_n - px)
                        leg_yaw = threat_yaw
                    elif cur_leg == 0:
                        weave = 42.0 * math.sin(py / 140.0)
                        wp_x = -1265.0 + weave
                        look_ahead_y = min(1375.0, py + 320.0) if dir_sign >= 0 else max(155.0, py - 320.0)
                        look_ahead_x = -1180.0 if (look_ahead_y > 1250.0 or look_ahead_y < 260.0) else -1255.0
                        threat_yaw = math.atan2(look_ahead_y - py, look_ahead_x - px)
                        leg_yaw = threat_yaw
                    elif cur_leg == 2:
                        weave = 42.0 * math.sin(py / 140.0)
                        wp_x = -275.0 + weave
                        look_ahead_y = max(155.0, py - 320.0) if dir_sign >= 0 else min(1375.0, py + 320.0)
                        look_ahead_x = -360.0 if (look_ahead_y > 1250.0 or look_ahead_y < 260.0) else -285.0
                        threat_yaw = math.atan2(look_ahead_y - py, look_ahead_x - px)
                        leg_yaw = threat_yaw
                    else:
                        on_bot_sp = py < 763.0
                        corridor_yaw = (math.pi / 2.0) if on_bot_sp else (-math.pi / 2.0)
                        if px <= -960.0 or px >= -580.0:
                            threat_yaw = corridor_yaw
                        else:
                            threat_yaw = math.atan2(ey_n - py, ex_n - px)
                        leg_yaw = threat_yaw

                if getattr(self, "local_missing_primary", False) and (py < 360.0 or py > 1160.0):
                    wp_x = -780.0
                    wp_y = 165.0 if py < 763.0 else 1365.0
                    leg_yaw = math.atan2(wp_y - py, wp_x - px)
                    threat_yaw = leg_yaw

        # Compute real 3-ray wall clearance lidar [front, left, right] in [0, 1] across the full 2-corridor arena
        yaw = local.view_angles.y
        lidar_rays = []
        for ray_angle in (yaw, yaw + math.pi / 2.0, yaw - math.pi / 2.0):
            rx, ry = math.cos(ray_angle), math.sin(ray_angle)
            t_min = 200.0
            # Outer arena walls: X in [-1505, -35], Y in [25, 1505]
            if rx > 1e-4 and px < -35.0:
                t_min = min(t_min, max(0.0, (-35.0 - px) / rx))
            elif rx < -1e-4 and px > -1505.0:
                t_min = min(t_min, max(0.0, (-1505.0 - px) / rx))
            if ry > 1e-4 and py < 1505.0:
                t_min = min(t_min, max(0.0, (1505.0 - py) / ry))
            elif ry < -1e-4 and py > 25.0:
                t_min = min(t_min, max(0.0, (25.0 - py) / ry))
            # Check ray against SOLID_WALL_AABBS
            end_x, end_y = px + rx * t_min, py + ry * t_min
            for xmin, ymin, xmax, ymax in self.SOLID_WALL_AABBS:
                if self._segment_intersects_aabb(px, py, end_x, end_y, xmin, ymin, xmax, ymax):
                    for s in (0.2, 0.4, 0.6, 0.8):
                        if self._segment_intersects_aabb(px, py, px + rx * t_min * s, py + ry * t_min * s, xmin, ymin, xmax, ymax):
                            t_min *= s
                            end_x, end_y = px + rx * t_min, py + ry * t_min
                            break
            lidar_rays.append(float(np.clip(t_min / 200.0, 0.0, 1.0)))

        # Express view_angles[1] and origin in the active corridor's canonical frame so VecNormalize
        # from Phase 6/7 never saturates (+10/-10) when the player rotates 90/180/270 degrees around the loop!
        # In Phase 8 (real_combat_mode), keep canonical view_angles[1] clamped within [-0.35, +0.35] so 180° turns
        # onto flanking native CS 1.6 bots NEVER saturate the PPO policy's VecNormalize!
        rel_corridor_yaw = (local.view_angles.y - leg_yaw + math.pi) % (2.0 * math.pi) - math.pi
        if real_combat:
            rel_corridor_yaw = float(np.clip(rel_corridor_yaw, -0.35, 0.35))

        return {
            'self': {
                'health': float(np.clip(getattr(self, "local_real_hp", 100.0) / 100.0, 0.0, 1.0)),
                'armor': 1.0,
                'ammo': 1.0,
                'origin': [-1200.0, 95.0, float(local.origin.z)],
                'world_origin': local.origin.to_list(),
                'view_angles': [float(local.view_angles.x), float(rel_corridor_yaw)],
                'world_yaw': float(local.view_angles.y),
                'velocity': target_vel,
            },
            'enemy': {
                'relative_angles': [rel_yaw, rel_pitch],
                'distance': min(min_dist / 3000.0, 1.0) if closest_enemy else 1.0,
                'visible': visible,
                'health': max(0.0, 1.0 - self.target_hit_counts.get(closest_enemy.address, 0) / 6.0) if closest_enemy else 0.0,
            },
            'lidar': lidar_rays,
            'loop_leg': int(cur_leg),
            'patrol_waypoint': [float(wp_x), float(wp_y), float(threat_yaw)],
            'tactical_mode': tactical_mode,
            'visible_enemies_count': int(len(corridor_enemies)),
            'in_cover_combat': bool(in_cover_combat),
            'incoming_dmg': float(incoming_dmg),
            'valid': True,
            'tick': int(time.time() * 1000)
        }

    def _has_line_of_sight(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True 360-degree 2D Line-of-Sight check between two points against SOLID_WALL_AABBS."""
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 15.0 or d > 1650.0:
            return False
        for xmin, ymin, xmax, ymax in self.SOLID_WALL_AABBS:
            if self._segment_intersects_aabb(x0, y0, x1, y1, xmin, ymin, xmax, ymax, pad=0.0):
                return False
        return True

    def get_5v5_bots_snapshot(self) -> dict:
        """
        Read all live/dead CCSBot entities (up to 10-12 bots in a 5v5 Expert match) directly from mp.dll,
        track each bot's kills, deaths, damage dealt, hits, firing modes, and return per-bot 19-D observations
        and 5-D actions for Top-3 MVP analysis and Behavioral Cloning / Policy Distillation.
        """
        if not self.hProc or not self.hw_base:
            if time.time() - self.last_connect > 2.0:
                self.close()
                self.connect()
            return {"valid": False, "bots": []}

        self.real_combat_mode = True
        if self.mp_base:
            self._patch_mp_dll_for_strafing()

        now = time.time()
        if not hasattr(self, "_mvp_bot_stats"):
            self._mvp_bot_stats = {}

        if not self._entvars_Scanned or (len(self._server_origins) < 10 and now - getattr(self, "_last_scan_t", 0.0) > 8.0):
            self._last_scan_t = now
            self._entvars_Scanned = False
            self._scan_server_entvars([])

        if not self._server_origins:
            return {"valid": False, "bots": []}

        raw_bots = []
        for idx, (pev_addr, csbot_addr, init_x, init_y, init_z) in enumerate(self._server_origins):
            pev_buf = (ctypes.c_char * 0x214)()
            if not self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(pev_addr), pev_buf, 0x214, None):
                continue
            ox, oy, oz = struct.unpack_from("<3f", pev_buf, 0x08)
            vx, vy, vz = struct.unpack_from("<3f", pev_buf, 0x20)
            pitch_deg, yaw_deg, _ = struct.unpack_from("<3f", pev_buf, 0x50)
            punch_p, punch_y, _ = struct.unpack_from("<3f", pev_buf, 0x68)
            frags_engine = struct.unpack_from("<f", pev_buf, 0x154)[0]
            health = struct.unpack_from("<f", pev_buf, 0x160)[0]
            deadflag = struct.unpack_from("<i", pev_buf, 0x170)[0]
            view_ofs_z = struct.unpack_from("<f", pev_buf, 0x17C)[0]
            buttons = struct.unpack_from("<I", pev_buf, 0x19C)[0]
            flags = struct.unpack_from("<I", pev_buf, 0x1A4)[0]

            if not (math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oz)):
                continue
            if math.isfinite(health) and health > 100.5:
                health = 100.0
                self.kernel32.WriteProcessMemory(
                    self.hProc, ctypes.c_void_p(pev_addr + 0x160), struct.pack("<f", 100.0), 4, None
                )

            team_id = 0
            fwd_spd, str_spd, btn_flags = 0.0, 0.0, 0
            if csbot_addr > 0x10000:
                tm_buf = (ctypes.c_char * 4)()
                if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x1C8), tm_buf, 4, None):
                    t_val = struct.unpack("<i", tm_buf)[0]
                    if t_val in (1, 2):
                        team_id = t_val
                cmd_buf = (ctypes.c_char * 16)()
                if self.kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(csbot_addr + 0x9D4), cmd_buf, 16, None):
                    fwd_spd, str_spd = struct.unpack_from("<2f", cmd_buf, 0)
                    btn_flags = struct.unpack_from("<H", cmd_buf, 12)[0]
                    if not math.isfinite(fwd_spd):
                        fwd_spd = 0.0
                    if not math.isfinite(str_spd):
                        str_spd = 0.0

            st = self._mvp_bot_stats.get(pev_addr)
            init_frags = max(0, int(round(frags_engine))) if (math.isfinite(frags_engine) and 0.0 <= frags_engine <= 5000.0) else 0
            if st is None:
                # Fallback team assignment by initial spawn side on $2000$ (Y > 763 = Top/T=1, Y <= 763 = Bottom/CT=2)
                fallback_team = 1 if oy > 763.0 else 2
                st = {
                    "bot_slot": idx + 1,
                    "pev_addr": pev_addr,
                    "team": team_id if team_id in (1, 2) else fallback_team,
                    "kills": init_frags,
                    "deaths": 0,
                    "dmg_dealt": float(init_frags * 100.0),
                    "hits_landed": int(init_frags * 4),
                    "prev_hp": float(health) if (deadflag == 0 and health > 0) else 100.0,
                    "prev_frags": float(init_frags),
                    "was_dead": bool(deadflag != 0 or health <= 0.0),
                    "burst_ticks": 0,
                    "prev_yaw": math.radians(yaw_deg),
                    "prev_pitch": -math.radians(pitch_deg),
                    "last_death_t": 0.0,
                }
                self._mvp_bot_stats[pev_addr] = st
            else:
                if team_id in (1, 2):
                    st["team"] = team_id
                if init_frags > st["kills"]:
                    st["kills"] = init_frags
                    st["dmg_dealt"] = max(st["dmg_dealt"], float(init_frags * 100.0))
                    st["hits_landed"] = max(st["hits_landed"], int(init_frags * 4))

            is_alive = bool(deadflag == 0 and math.isfinite(health) and health > 0.5)
            raw_bots.append({
                "pev_addr": pev_addr,
                "st": st,
                "is_alive": is_alive,
                "ox": ox, "oy": oy, "oz": oz,
                "vx": vx, "vy": vy, "vz": vz,
                "yaw_rad": math.radians(yaw_deg),
                "pitch_rad": -math.radians(pitch_deg),
                "punch_mag": math.hypot(punch_p, punch_y),
                "frags_engine": float(frags_engine) if math.isfinite(frags_engine) else 0.0,
                "health": float(health) if is_alive else 0.0,
                "buttons": int(buttons) | int(btn_flags),
                "is_ducking": bool((buttons & 4) or (btn_flags & 4) or (flags & 2) or (view_ofs_z < 15.0)),
                "fwd_spd": fwd_spd,
                "str_spd": str_spd,
            })

        # Ensure balanced 2-team assignment if all bots reported identical team_id at +0x1C8
        teams_present = {b["st"]["team"] for b in raw_bots}
        if len(teams_present) < 2 and len(raw_bots) >= 2:
            for i, b in enumerate(raw_bots):
                b["st"]["team"] = 1 if (i % 2 == 0) else 2

        # Detect damage drops & deaths, and credit the attacker bot looking most directly at the victim with LOS!
        for victim in raw_bots:
            vst = victim["st"]
            if not victim["is_alive"]:
                if not vst["was_dead"] and (now - vst["last_death_t"]) > 2.0:
                    vst["was_dead"] = True
                    vst["last_death_t"] = now
                    vst["deaths"] += 1
                    dmg_drop = max(15.0, vst["prev_hp"])
                    vst["prev_hp"] = 100.0
                    # Find best alive opposing bot aiming at victim
                    best_attacker = None
                    best_ang = 0.45
                    for cand in raw_bots:
                        if cand["pev_addr"] == victim["pev_addr"] or not cand["is_alive"]:
                            continue
                        if cand["st"]["team"] == vst["team"]:
                            continue
                        dx = victim["ox"] - cand["ox"]
                        dy = victim["oy"] - cand["oy"]
                        tyaw = math.atan2(dy, dx)
                        dyaw = abs((tyaw - cand["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi)
                        if dyaw < best_ang and self._has_line_of_sight(cand["ox"], cand["oy"], victim["ox"], victim["oy"]):
                            best_ang = dyaw
                            best_attacker = cand["st"]
                    if best_attacker is not None:
                        best_attacker["kills"] += 1
                        best_attacker["hits_landed"] += 1
                        best_attacker["dmg_dealt"] += dmg_drop
            else:
                if vst["was_dead"]:
                    vst["was_dead"] = False
                    vst["prev_hp"] = victim["health"]
                elif victim["health"] < vst["prev_hp"] - 0.5:
                    dmg_drop = vst["prev_hp"] - victim["health"]
                    vst["prev_hp"] = victim["health"]
                    best_attacker = None
                    best_ang = 0.45
                    for cand in raw_bots:
                        if cand["pev_addr"] == victim["pev_addr"] or not cand["is_alive"]:
                            continue
                        if cand["st"]["team"] == vst["team"]:
                            continue
                        dx = victim["ox"] - cand["ox"]
                        dy = victim["oy"] - cand["oy"]
                        tyaw = math.atan2(dy, dx)
                        dyaw = abs((tyaw - cand["yaw_rad"] + math.pi) % (2 * math.pi) - math.pi)
                        if dyaw < best_ang and self._has_line_of_sight(cand["ox"], cand["oy"], victim["ox"], victim["oy"]):
                            best_ang = dyaw
                            best_attacker = cand["st"]
                    if best_attacker is not None:
                        best_attacker["hits_landed"] += 1
                        best_attacker["dmg_dealt"] += dmg_drop
                else:
                    vst["prev_hp"] = victim["health"]

            # Also sync engine frag counter increments if higher
            if victim["frags_engine"] > vst["prev_frags"] + 0.5:
                diff_f = int(round(victim["frags_engine"] - vst["prev_frags"]))
                if diff_f > 0 and diff_f <= 3:
                    vst["kills"] = max(vst["kills"], vst["kills"])
                vst["prev_frags"] = victim["frags_engine"]

        # Build per-bot 19-D observation & 5-D action snapshot
        out_bots = []
        for b in raw_bots:
            st = b["st"]
            px, py, pz = b["ox"], b["oy"], b["oz"]
            yaw = b["yaw_rad"]
            pitch = b["pitch_rad"]

            # Find closest visible / priority enemy for this bot
            best_enemy = None
            best_score = 1e9
            rel_yaw, rel_pitch, min_dist, visible = 0.0, 0.0, 3000.0, 0.0
            target_vel = [0.0, 0.0, 0.0]
            enemy_hp_norm = 0.0

            if b["is_alive"]:
                for other in raw_bots:
                    if other["pev_addr"] == b["pev_addr"] or not other["is_alive"]:
                        continue
                    if other["st"]["team"] == st["team"]:
                        continue
                    if not self._has_line_of_sight(px, py, other["ox"], other["oy"]):
                        continue
                    dx = other["ox"] - px
                    dy = other["oy"] - py
                    dz = other["oz"] - pz
                    dist = math.hypot(dx, dy)
                    tyaw = math.atan2(dy, dx)
                    tpitch = -math.atan2(dz, max(1.0, dist))
                    cyaw = (tyaw - yaw + math.pi) % (2 * math.pi) - math.pi
                    cpitch = tpitch - pitch
                    a_err = math.hypot(cyaw, cpitch)
                    score = (dist / 450.0) + 0.35 * a_err
                    if score < best_score:
                        best_score = score
                        best_enemy = other
                        min_dist = dist
                        rel_yaw = cyaw
                        rel_pitch = cpitch
                        visible = 1.0 if abs(cyaw) <= math.radians(135.0) else 0.0
                        enemy_hp_norm = float(np.clip(other["health"] / 100.0, 0.0, 1.0))
                        target_vel = [
                            float(np.clip(other["vx"] / 250.0, -1.0, 1.0)),
                            float(np.clip(other["vy"] / 250.0, -1.0, 1.0)),
                            float(np.clip(other["vz"] / 250.0, -1.0, 1.0)),
                        ]

            # Compute 3-ray lidar for this bot
            lidar_rays = []
            for ray_angle in (yaw, yaw + math.pi / 2.0, yaw - math.pi / 2.0):
                rx, ry = math.cos(ray_angle), math.sin(ray_angle)
                t_min = 200.0
                if rx > 1e-4 and px < -35.0:
                    t_min = min(t_min, max(0.0, (-35.0 - px) / rx))
                elif rx < -1e-4 and px > -1505.0:
                    t_min = min(t_min, max(0.0, (-1505.0 - px) / rx))
                if ry > 1e-4 and py < 1505.0:
                    t_min = min(t_min, max(0.0, (1505.0 - py) / ry))
                elif ry < -1e-4 and py > 25.0:
                    t_min = min(t_min, max(0.0, (25.0 - py) / ry))
                end_x, end_y = px + rx * t_min, py + ry * t_min
                for xmin, ymin, xmax, ymax in self.SOLID_WALL_AABBS:
                    if self._segment_intersects_aabb(px, py, end_x, end_y, xmin, ymin, xmax, ymax):
                        for s in (0.2, 0.4, 0.6, 0.8):
                            if self._segment_intersects_aabb(px, py, px + rx * t_min * s, py + ry * t_min * s, xmin, ymin, xmax, ymax):
                                t_min *= s
                                end_x, end_y = px + rx * t_min, py + ry * t_min
                                break
                lidar_rays.append(float(np.clip(t_min / 200.0, 0.0, 1.0)))

            cur_leg = self.get_loop_leg(px, py)
            _, _, leg_yaw = self.get_leg_waypoint(cur_leg, px, py)
            rel_corridor_yaw = float(np.clip((yaw - leg_yaw + math.pi) % (2.0 * math.pi) - math.pi, -0.35, 0.35))

            # Project velocity into bot's forward and left-strafe axes
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            fwd_vel = (b["vx"] * cos_y + b["vy"] * sin_y) / 250.0
            left_vel = (-b["vx"] * sin_y + b["vy"] * cos_y) / 250.0
            act_fwd = float(np.clip(b["fwd_spd"] / 250.0 if abs(b["fwd_spd"]) > 5.0 else fwd_vel, -1.0, 1.0))
            act_str = float(np.clip(-b["str_spd"] / 250.0 if abs(b["str_spd"]) > 5.0 else left_vel, -1.0, 1.0))

            # Firing state & action[4] encoding in our 3-mechanic scale:
            # IMPORTANT: Never label on-target combat ticks as -0.60 ("hold fire") during inter-shot pauses of CCSBot!
            # Only label -0.60 when no enemy is visible OR when the bot is still turning > 12° (0.21 rad) onto the target!
            is_firing = bool((b["buttons"] & 1) or b["punch_mag"] > 0.25)
            if is_firing and b["is_alive"]:
                st["burst_ticks"] += 1
            else:
                st["burst_ticks"] = 0

            on_target = visible > 0.5 and abs(rel_yaw) <= 0.21 and abs(rel_pitch) <= 0.18
            if visible < 0.5 or (not is_firing and not on_target):
                act_fire = -0.60
                fmode = "hold"
            elif b["is_ducking"] or st["burst_ticks"] >= 4 or min_dist <= 500.0:
                act_fire = 0.55   # 2-Tap -> Crouch-Spray (> 0.34)
                fmode = "spray"
            elif st["burst_ticks"] >= 2 or min_dist <= 750.0:
                act_fire = 0.24   # 2-Tap Standing (0.10..0.34)
                fmode = "two_tap"
            else:
                act_fire = 0.06   # 1-Tap Standing (0.00..0.10)
                fmode = "tap"

            # Smooth mouse delta actions (combining actual frame turn with target closed-loop tracking)
            frame_dyaw = (yaw - st["prev_yaw"] + math.pi) % (2 * math.pi) - math.pi
            frame_dpitch = pitch - st["prev_pitch"]
            st["prev_yaw"] = yaw
            st["prev_pitch"] = pitch
            if visible > 0.5:
                act_dyaw = float(np.clip(0.65 * (rel_yaw * 3.2) + 0.35 * (frame_dyaw * 4.0), -1.0, 1.0))
                act_dpitch = float(np.clip(0.65 * (rel_pitch * 3.2) + 0.35 * (frame_dpitch * 4.0), -1.0, 1.0))
            else:
                act_dyaw = float(np.clip(frame_dyaw * 3.5, -1.0, 1.0))
                act_dpitch = float(np.clip(frame_dpitch * 3.5, -1.0, 1.0))

            obs_dict = {
                "health": [float(np.clip(b["health"] / 100.0, 0.0, 1.0))],
                "armor": [1.0],
                "ammo": [1.0],
                "origin": [-1200.0 / 4096.0, 95.0 / 4096.0, float(pz) / 4096.0],
                "view_angles": [float(pitch), float(rel_corridor_yaw)],
                "velocity": target_vel,
                "enemy_rel_angles": [float(rel_yaw), float(rel_pitch)],
                "enemy_distance": [float(min(min_dist / 3000.0, 1.0))],
                "enemy_visible": [float(visible)],
                "enemy_health": [float(enemy_hp_norm)],
                "lidar": lidar_rays,
            }

            out_bots.append({
                "bot_slot": st["bot_slot"],
                "pev_addr": b["pev_addr"],
                "team": st["team"],
                "is_alive": b["is_alive"],
                "kills": st["kills"],
                "deaths": st["deaths"],
                "dmg_dealt": round(st["dmg_dealt"], 1),
                "hits_landed": st["hits_landed"],
                "pos": [round(px, 1), round(py, 1), round(pz, 1)],
                "speed_2d": round(math.hypot(b["vx"], b["vy"]), 1),
                "is_firing": is_firing,
                "is_ducking": b["is_ducking"],
                "fire_mode": fmode,
                "burst_ticks": st["burst_ticks"],
                "enemy_dist": round(min_dist, 1) if visible > 0.5 else -1.0,
                "obs": obs_dict,
                "act": [act_fwd, act_str, act_dyaw, act_dpitch, act_fire],
            })

        return {"valid": True, "bots": out_bots, "tick": int(now * 1000)}

    def close(self):
        if self.hProc:
            try:
                self.kernel32.CloseHandle(self.hProc)
            except Exception:
                pass
            self.hProc = None
        self.hw_base = None
        self.client_base = None
        self.mp_base = None
        self.pid = None
        self._mp_patched = False
        self._entvars_Scanned = False
        self.local_pev_addr = None
        self._server_origins = []
        self._spawn_armoury_pevs = []
        if hasattr(self, "_strafe_state") and isinstance(self._strafe_state, dict):
            self._strafe_state.clear()

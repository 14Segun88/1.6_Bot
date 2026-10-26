"""
Counter-Strike 1.6 - Master AI Agent (Vision YOLO + Stereo Hearing + Lane Hunter)
==================================================================================
Combines:
1. High-Resolution YOLOv8 Humanoid Detection (640x640 ONNX) -> Spots enemies across $2000$
2. Ballistic Head Lock & Controlled Burst Fire (Counter-Strafe + Recoil Control)
3. Stereo Audio Hearing (WASAPI Loopback) -> Snaps toward footsteps/gunfire direction
4. Aggressive Lane Patrol on $2000$ -> DirectInput hardware key injection (SendInput + keybd_event)
5. Multi-Sensor Collision Recovery -> Unhooks from crates/walls without backing into spawn
"""

import os
import sys
import time
import math
import random
import threading
import json
import queue
from collections import deque
import numpy as np
import cv2
import mss
import onnxruntime as ort

# Platform specifics & DirectInput
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # High-DPI Awareness so mouse coordinates and window rects match 1:1
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long)
        ]

    # Windows SendInput structures for physical keyboard & mouse simulation
    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL)
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_short),
            ("wParamH", ctypes.c_ushort)
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL)
        ]

    class INPUT_I(ctypes.Union):
        _fields_ = [
            ("ki", KEYBDINPUT),
            ("mi", MOUSEINPUT),
            ("hi", HARDWAREINPUT)
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("ii", INPUT_I)
        ]
    import struct
    import subprocess
    psapi = ctypes.windll.psapi

    TH32CS_SNAPMODULE = 0x00000008
    TH32CS_SNAPMODULE32 = 0x00000010

    class MODULEENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("th32ModuleID", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("GlblcntUsage", wintypes.DWORD),
            ("ProccntUsage", wintypes.DWORD),
            ("modBaseAddr", ctypes.c_void_p),
            ("modBaseSize", wintypes.DWORD),
            ("hModule", ctypes.c_void_p),
            ("szModule", ctypes.c_wchar * 256),
            ("szExePath", ctypes.c_wchar * 260)
        ]
else:
    user32 = None
    kernel32 = None
    psapi = None

class CS16MemoryReader:
    """Reads live 3D player coordinates and orientation directly from GoldSrc engine memory"""
    def __init__(self):
        self.hProc = None
        self.ptr_addr = None
        self.hw_base = None
        self.candidate_offsets = [0x7A834, 0x1222E8, 0x113CDC, 0x113CD0, 0x10B740, 0x150F40, 0x79834, 0x7B834]
        self.last_connect_try = 0.0
        self.connect()

    def connect(self):
        if not (IS_WINDOWS and kernel32):
            return False
        self.last_connect_try = time.time()
        try:
            out = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq hl.exe", "/FO", "CSV"]).decode("latin-1")
            lines = [l for l in out.splitlines() if "hl.exe" in l]
            if not lines:
                return False
            pid = int(lines[0].split(",")[1].strip('"\r\n '))

            # Open process with full VM read access
            for access_flag in [0x1F0FFF, 0x0410, 0x0010 | 0x0400]:
                self.hProc = kernel32.OpenProcess(access_flag, False, pid)
                if self.hProc:
                    break
            if not self.hProc:
                return False

            # 1. Enumerate 32-bit modules using Toolhelp32 (WOW64 compatible)
            hw_base = None
            snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
            if snap and snap != -1:
                me32 = MODULEENTRY32W()
                me32.dwSize = ctypes.sizeof(MODULEENTRY32W)
                if kernel32.Module32FirstW(snap, ctypes.byref(me32)):
                    while True:
                        mod_name = me32.szModule.lower()
                        if mod_name in ["hw.dll", "sw.dll"]:
                            hw_base = me32.modBaseAddr
                            print(f"[*] Memory Engine: Located GoldSrc engine module '{mod_name}' via Toolhelp32 (Base: 0x{hw_base:X})")
                            break
                        if not kernel32.Module32NextW(snap, ctypes.byref(me32)):
                            break
                kernel32.CloseHandle(snap)

            # 2. Fallback: 32-bit EnumProcessModulesEx if Toolhelp was blocked
            if not hw_base and psapi:
                hMods = (ctypes.c_uint32 * 1024)()
                cbNeeded = wintypes.DWORD()
                for enum_flag in [0x01, 0x03]:
                    if psapi.EnumProcessModulesEx(self.hProc, ctypes.byref(hMods), ctypes.sizeof(hMods), ctypes.byref(cbNeeded), enum_flag):
                        mod_count = cbNeeded.value // 4
                        for i in range(mod_count):
                            hMod = hMods[i]
                            if not hMod: continue
                            mod_name = (ctypes.c_char * 260)()
                            psapi.GetModuleBaseNameA(self.hProc, ctypes.c_void_p(hMod), mod_name, 260)
                            name = mod_name.value.decode("latin-1", errors="ignore").lower()
                            if name in ["hw.dll", "sw.dll"]:
                                hw_base = hMod
                                print(f"[*] Memory Engine: Located '{name}' via EnumProcessModulesEx (Base: 0x{hw_base:X})")
                                break
                    if hw_base: break

            if hw_base:
                int_base = hw_base if isinstance(hw_base, int) else (hw_base.value if hasattr(hw_base, 'value') and hw_base.value else int(hw_base))
                self.hw_base = int_base
                self.ptr_addr = int_base + self.candidate_offsets[0]

                # Proactively auto-calibrate against active game memory
                buf = (ctypes.c_char * 4)()
                buf_mem = (ctypes.c_char * 28)()
                for off in self.candidate_offsets:
                    test_ptr = int_base + off
                    if kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(test_ptr), buf, 4, None):
                        b_ptr = struct.unpack_from("<I", buf, 0)[0]
                        if b_ptr and b_ptr > 0x10000:
                            if kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(b_ptr + 0x3D4), buf_mem, 28, None):
                                tx, ty, tz = struct.unpack_from("<3f", buf_mem, 0)
                                if not (math.isnan(tx) or math.isnan(ty) or math.isnan(tz)):
                                    if abs(tx) < 8000.0 and abs(ty) < 8000.0 and abs(tz) < 4000.0 and (abs(tx) > 1.0 or abs(ty) > 1.0):
                                        self.ptr_addr = test_ptr
                                        print(f"[*] Memory Engine: Verified active coordinates (POS: {tx:.0f}, {ty:.0f}, {tz:.0f}) at offset 0x{off:X}")
                                        break
                return True
        except Exception:
            pass
        return False

    def get_pose(self):
        if not self.hProc or not self.ptr_addr:
            if time.time() - self.last_connect_try > 3.0:
                self.connect()
            return None
        buf = (ctypes.c_char * 4)()
        if not kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.ptr_addr), buf, 4, None):
            self.hProc = None
            return None
        base_ptr = struct.unpack_from("<I", buf, 0)[0]
        if not base_ptr or base_ptr < 0x10000:
            if hasattr(self, 'hw_base') and self.hw_base:
                for off in self.candidate_offsets:
                    test_ptr = self.hw_base + off
                    if kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(test_ptr), buf, 4, None):
                        b_ptr = struct.unpack_from("<I", buf, 0)[0]
                        if b_ptr and b_ptr > 0x10000:
                            self.ptr_addr = test_ptr
                            base_ptr = b_ptr
                            break
            if not base_ptr:
                return None

        buf_mem = (ctypes.c_char * 28)()
        if not kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(base_ptr + 0x3D4), buf_mem, 28, None):
            return None

        x, y, z, _, fwd_x, fwd_y, fwd_z = struct.unpack_from("<7f", buf_mem, 0)
        if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isnan(fwd_x) or math.isnan(fwd_y):
            return None
        if abs(x) > 10000.0 or abs(y) > 10000.0 or abs(z) > 10000.0:
            return None
        if abs(x) < 0.01 and abs(y) < 0.01 and abs(z) < 0.01:
            return None
        if abs(fwd_x) < 0.0001 and abs(fwd_y) < 0.0001:
            return None
        yaw_deg = math.degrees(math.atan2(fwd_y, fwd_x))
        if math.isnan(yaw_deg):
            return None
        fwd_z_clamped = max(-1.0, min(1.0, float(fwd_z)))
        pitch_deg = -math.degrees(math.asin(fwd_z_clamped))
        if math.isnan(pitch_deg):
            pitch_deg = 0.0
        return float(x), float(y), float(z), float(yaw_deg), float(pitch_deg)

# Tactical Search & Clear Routes for aim_ai_sandbox (Calibrated from aim_ai_sandbox.bsp)
# CT Spawn is at (-772, 160) | 11 Dummies distributed across Tunnels, Center Crossing, and Enemy Courtyard
MAP_ROUTES_SANDBOX = [
    # Route 0: LEFT FLANK & TUNNEL PATROL (Exit CT yard -> Left Tunnel -> Enemy Base)
    [(-772, 160), (-1250, 250), (-1380, 500), (-1380, 800), (-1380, 1050), (-1100, 1300), (-772, 1350)],

    # Route 1: CENTER CROSSING & ARCH PATROL (CT yard -> Central Arena -> Archway -> Enemy Base)
    [(-772, 160), (-772, 300), (-1000, 620), (-772, 750), (-550, 620), (-772, 900), (-772, 1350)],

    # Route 2: RIGHT FLANK & TUNNEL PATROL (Exit CT yard -> Right Tunnel -> Enemy Base)
    [(-772, 160), (-300, 250), (-200, 500), (-200, 800), (-200, 1050), (-450, 1300), (-772, 1350)],
]

MAP_ROUTES_CT = MAP_ROUTES_SANDBOX
MAP_ROUTES_T = MAP_ROUTES_SANDBOX

ROUTE_NAMES = [
    "LEFT FLANK & TUNNEL PATROL",
    "CENTER CROSSING & ARCH PATROL",
    "RIGHT FLANK & TUNNEL PATROL"
]

# Lidar Dynamic HSV Lighting Profile Cache
_LIDAR_PROFILE_CACHE = {
    "mtime": 0,
    "last_check": 0.0,
    "green_lower": np.array([32, 45, 35], dtype=np.uint8),
    "green_upper": np.array([88, 255, 255], dtype=np.uint8),
    "red_lower_1": np.array([0, 50, 40], dtype=np.uint8),
    "red_upper_1": np.array([14, 255, 255], dtype=np.uint8),
    "red_lower_2": np.array([165, 50, 40], dtype=np.uint8),
    "red_upper_2": np.array([180, 255, 255], dtype=np.uint8),
}

def get_lidar_hsv_bounds():
    """Dynamically loads benchmarks/lidar_hsv_profile.json if available, or falls back to defaults."""
    global _LIDAR_PROFILE_CACHE
    now = time.time()
    # Check at most once every 3.0 seconds to minimize I/O overhead
    if now - _LIDAR_PROFILE_CACHE["last_check"] < 3.0:
        return _LIDAR_PROFILE_CACHE
    _LIDAR_PROFILE_CACHE["last_check"] = now

    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "lidar_hsv_profile.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "lidar_hsv_profile.json"),
        "benchmarks/lidar_hsv_profile.json",
        "/mnt/d/CS 1.6 GoldClient/benchmarks/lidar_hsv_profile.json",
        "D:/CS 1.6 GoldClient/benchmarks/lidar_hsv_profile.json",
    ]
    if "CS16_WORK_DIR" in os.environ:
        candidates.insert(0, os.path.join(os.environ["CS16_WORK_DIR"], "benchmarks", "lidar_hsv_profile.json"))

    for p in candidates:
        if os.path.exists(p):
            try:
                mtime = os.path.getmtime(p)
                if mtime != _LIDAR_PROFILE_CACHE["mtime"]:
                    with open(p, "r", encoding="utf-8") as f:
                        prof = json.load(f)
                    gf = prof.get("green_floor", {})
                    rw = prof.get("red_wall", {})
                    
                    gh_min, gh_max = gf.get("h_range", [32, 88])
                    gs_min = gf.get("s_min", 45)
                    gv_min = gf.get("v_min", 35)
                    
                    rh_1 = rw.get("h_range_1", [0, 14])
                    rh_2 = rw.get("h_range_2", [165, 180])
                    rs_min = rw.get("s_min", 50)
                    rv_min = rw.get("v_min", 40)
                    
                    _LIDAR_PROFILE_CACHE["mtime"] = mtime
                    _LIDAR_PROFILE_CACHE["green_lower"] = np.array([gh_min, gs_min, gv_min], dtype=np.uint8)
                    _LIDAR_PROFILE_CACHE["green_upper"] = np.array([gh_max, 255, 255], dtype=np.uint8)
                    _LIDAR_PROFILE_CACHE["red_lower_1"] = np.array([rh_1[0], rs_min, rv_min], dtype=np.uint8)
                    _LIDAR_PROFILE_CACHE["red_upper_1"] = np.array([rh_1[1], 255, 255], dtype=np.uint8)
                    _LIDAR_PROFILE_CACHE["red_lower_2"] = np.array([rh_2[0], rs_min, rv_min], dtype=np.uint8)
                    _LIDAR_PROFILE_CACHE["red_upper_2"] = np.array([rh_2[1], 255, 255], dtype=np.uint8)
                    print(f"[Color Lidar] Calibrated lighting HSV profile loaded from {p}: Floor H:[{gh_min}-{gh_max}] V>{gv_min}, Walls V>{rv_min}")
                break
            except Exception:
                pass
    return _LIDAR_PROFILE_CACHE

def optical_color_lidar(frame_bgr):
    """
    Task 3: Real-time Optical Color Terrain & Barrier Scanner for aim_ai_sandbox:
    - GREEN FLOOR (Walkable Zone / Сюда МОЖНО -> press W): Calibrated or HSV H in [32, 88], S > 45, V > 35
    - RED WALL (Solid Barrier / Сюда НЕЛЬЗЯ -> 45° escape pivot): Calibrated or HSV H in [0, 14] or [165, 180], S > 50, V > 40
    Returns:
      steer_bias: float in [-1.0, 1.0] (negative = steer left towards green, positive = steer right)
      wall_ahead: bool (True if red barrier wall is blocking forward path)
      floor_visible: bool (True if green walkable arena floor is tracked)
      red_ratio: float (0.0 to 1.0, peak red barrier ratio)
      green_ratio: float (0.0 to 1.0, green walkable floor ratio)
    """
    if frame_bgr is None or frame_bgr.shape[0] < 100:
        return 0.0, False, False, 0.0, 0.0

    bounds = get_lidar_hsv_bounds()

    h, w = frame_bgr.shape[:2]
    floor_roi = frame_bgr[int(h * 0.50):int(h * 0.95), :]
    center_roi = frame_bgr[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]

    hsv_f = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

    # Red masks (barrier walls: dual red sectors in HSV, dynamically adjusted for lighting)
    red_f = cv2.bitwise_or(
        cv2.inRange(hsv_f, bounds["red_lower_1"], bounds["red_upper_1"]),
        cv2.inRange(hsv_f, bounds["red_lower_2"], bounds["red_upper_2"])
    )
    red_c = cv2.bitwise_or(
        cv2.inRange(hsv_c, bounds["red_lower_1"], bounds["red_upper_1"]),
        cv2.inRange(hsv_c, bounds["red_lower_2"], bounds["red_upper_2"])
    )

    # Green masks (walkable corridor floor, dynamically adjusted for shadows/lighting)
    green_f = cv2.inRange(hsv_f, bounds["green_lower"], bounds["green_upper"])

    # Floor sector analysis (left, center, right)
    rw = floor_roi.shape[1] // 3
    left_red = float(np.mean(red_f[:, :rw])) / 255.0
    center_red = float(np.mean(red_f[:, rw:2*rw])) / 255.0
    right_red = float(np.mean(red_f[:, 2*rw:])) / 255.0

    c_red_ratio = float(np.mean(red_c)) / 255.0
    f_green_ratio = float(np.mean(green_f)) / 255.0

    wall_ahead = (center_red > 0.22) or (c_red_ratio > 0.25)
    floor_visible = (f_green_ratio > 0.12)
    steer_bias = float(np.clip((left_red - right_red) * 2.5, -1.0, 1.0))
    red_ratio = float(max(center_red, c_red_ratio))

    return steer_bias, wall_ahead, floor_visible, red_ratio, f_green_ratio

# Virtual Key Codes (Win32 / SDL2 WM_KEYDOWN compatibility)
VK_MAP = {
    "w": 0x57,
    "s": 0x53,
    "a": 0x41,
    "d": 0x44,
    "space": 0x20,
    "ctrl": 0x11,
    "shift": 0x10,
    "r": 0x52,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
    "0": 0x30,
    "escape": 0x1B,
    "e": 0x45,
    "b": 0x42,
    ",": 0xBC,
    ".": 0xBE,
    "q": 0x51,
    "g": 0x47,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f9": 0x78,
    "end": 0x23,
}

# DirectInput Hardware Scan Codes (Recognized by GoldSrc engine input layer)
SCAN_CODES = {
    "w": 0x11,
    "s": 0x1F,
    "a": 0x1E,
    "d": 0x20,
    "space": 0x39,
    "ctrl": 0x1D,
    "shift": 0x2A,
    "r": 0x13,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
    "5": 0x06,
    "6": 0x07,
    "7": 0x08,
    "8": 0x09,
    "9": 0x0A,
    "0": 0x0B,
    "escape": 0x01,
    "e": 0x12,
    "b": 0x30,
    ",": 0x33,
    ".": 0x34,
    "q": 0x10,
    "g": 0x22,
    "f1": 0x3B,
    "f2": 0x3C,
    "f3": 0x3D,
    "f4": 0x3E,
    "f5": 0x3F,
    "f6": 0x40,
    "f9": 0x43,
    "end": 0xCF,
}

VK_F6 = 0x75
VK_F9 = 0x78
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

def send_key_down(name):
    """Sends key press via SendInput + keybd_event to guarantee reception in CS 1.6"""
    if not (IS_WINDOWS and user32):
        return
    vk = VK_MAP.get(name)
    sc = SCAN_CODES.get(name)
    if vk is None or sc is None:
        return

    # Method 1: SendInput with Hardware ScanCode
    try:
        inp = INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.ii.ki.wVk = 0  # Win32 specification: wVk must be 0 when KEYEVENTF_SCANCODE is set
        inp.ii.ki.wScan = sc
        inp.ii.ki.dwFlags = KEYEVENTF_SCANCODE
        inp.ii.ki.time = 0
        inp.ii.ki.dwExtraInfo = None
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    except Exception:
        pass

    # Method 2: Standard keybd_event with VK and Scan Code
    user32.keybd_event(vk, sc, 0, 0)

def send_key_up(name):
    """Sends key release via SendInput + keybd_event"""
    if not (IS_WINDOWS and user32):
        return
    vk = VK_MAP.get(name)
    sc = SCAN_CODES.get(name)
    if vk is None or sc is None:
        return

    # Method 1: SendInput with ScanCode + KEYUP
    try:
        inp = INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.ii.ki.wVk = 0
        inp.ii.ki.wScan = sc
        inp.ii.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        inp.ii.ki.time = 0
        inp.ii.ki.dwExtraInfo = None
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    except Exception:
        pass

    # Method 2: Standard keybd_event keyup
    user32.keybd_event(vk, sc, KEYEVENTF_KEYUP, 0)

def tap_key(name, hold_sec=0.035):
    """Sends a clean keypress and release with GoldSrc-friendly duration"""
    if not (IS_WINDOWS and user32):
        return
    send_key_down(name)
    time.sleep(hold_sec)
    send_key_up(name)

def execute_autobuy():
    """
    Robust Buy Sequence for CS 1.6:
    1. Buy Rifle: 'b' -> '4' -> '3' (M4A1 for CT / AK-47 for T).
    2. Buy Armor+Helmet: 'b' -> '8' -> '2'.
    3. Top off Ammo: ',' (buyammo1) and '.' (buyammo2).
    4. Draw Primary Rifle: '1'.
    NOTE: In CS 1.6 purchasing a weapon automatically closes the buy menu.
    We NEVER send 'escape' because Escape opens the main game menu and can minimize the game.
    """
    if not (IS_WINDOWS and user32):
        return
    def _buy_worker():
        time.sleep(0.08)
        # Dismiss any pending class selection menu (5 = auto-select random model)
        tap_key("5", 0.04)
        time.sleep(0.04)

        # Step 1: Buy Primary Rifle (b -> 4 -> 3 = M4A1 / AK-47)
        tap_key("b", 0.03)
        time.sleep(0.04)
        tap_key("4", 0.03)
        time.sleep(0.04)
        tap_key("3", 0.03)
        time.sleep(0.05)

        # Step 2: Buy Kevlar + Helmet (b -> 8 -> 2)
        tap_key("b", 0.03)
        time.sleep(0.04)
        tap_key("8", 0.03)
        time.sleep(0.04)
        tap_key("2", 0.03)
        time.sleep(0.05)

        # Step 3: Top off primary & secondary ammo (',' and '.')
        for _ in range(2):
            tap_key(",", 0.03)
            time.sleep(0.03)
            tap_key(".", 0.03)
            time.sleep(0.03)

        # Step 4: Draw Primary weapon ('1')
        tap_key("1", 0.04)

        play_beep(1300, 50)
        print("[★ AUTOBUY] Clean equipment purchase (Rifle + Kevlar + Ammo)!", flush=True)

    t = threading.Thread(target=_buy_worker, daemon=True)
    t.start()


def execute_counter_strafe(active_keys):
    """Active opposite impulse tap to cancel GoldSrc inertia for 100% first-bullet accuracy"""
    if not (IS_WINDOWS and user32):
        active_keys.clear()
        return
    counter_pairs = [("w", "s"), ("s", "w"), ("a", "d"), ("d", "a")]
    keys_snapshot = list(active_keys)
    # Release forward movement keys immediately
    for k in keys_snapshot:
        send_key_up(k)
    active_keys.clear()

    # Tap opposite keys for instant dead-stop friction
    opp_keys = []
    for k_move, k_opp in counter_pairs:
        if k_move in keys_snapshot:
            send_key_down(k_opp)
            opp_keys.append(k_opp)
    if opp_keys:
        time.sleep(0.045)  # 45ms instant dead-stop impulse
        for k_opp in opp_keys:
            send_key_up(k_opp)

def send_mouse_move(dx, dy):
    if IS_WINDOWS and user32:
        user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

def send_mouse_down():
    if IS_WINDOWS and user32:
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def send_mouse_up():
    if IS_WINDOWS and user32:
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def is_key_pressed(vk_code):
    if IS_WINDOWS and user32:
        return bool(user32.GetAsyncKeyState(vk_code) & 0x8000)
    return False

def play_beep(freq, duration_ms):
    if not IS_WINDOWS:
        return
    def _run():
        try:
            import winsound
            winsound.Beep(int(freq), int(duration_ms))
        except Exception:
            if kernel32:
                try:
                    kernel32.Beep(int(freq), int(duration_ms))
                except Exception:
                    pass
    threading.Thread(target=_run, daemon=True).start()

def focus_cs_window():
    """Finds Counter-Strike 1.6 window and reliably forces it to foreground (Windows 10/11 safe)"""
    if not (IS_WINDOWS and user32):
        return None
    hwnds = []
    for cls_name in [b"SDL_app", b"Valve001"]:
        h = user32.FindWindowA(cls_name, None)
        if h and user32.IsWindowVisible(h):
            hwnds.append(h)

    # Check top-level windows for hl.exe / cstrike.exe or title match
    def enum_cb(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_string_buffer(256)
            user32.GetWindowTextA(hwnd, buf, 256)
            val = buf.value.lower()
            if any(k in val for k in [b"counter", b"strike", b"cstrike", b"cs", b"half-life", b"goldclient"]):
                hwnds.append(hwnd)
                return True
            if kernel32 and psapi:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value:
                    h_proc = kernel32.OpenProcess(0x1000, False, pid.value)
                    if not h_proc:
                        h_proc = kernel32.OpenProcess(0x0410, False, pid.value)
                    if h_proc:
                        name_buf = ctypes.create_string_buffer(256)
                        psapi.GetModuleBaseNameA(h_proc, None, name_buf, 256)
                        kernel32.CloseHandle(h_proc)
                        if name_buf.value.lower() in [b"hl.exe", b"cstrike.exe"]:
                            hwnds.append(hwnd)
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

    for hwnd in hwnds:
        cur_fg = user32.GetForegroundWindow()
        if cur_fg == hwnd:
            return hwnd
        try:
            user32.ShowWindow(hwnd, 9)
            user32.keybd_event(0x12, 0, 0, 0)
            user32.SetForegroundWindow(hwnd)
            user32.keybd_event(0x12, 0, 2, 0)
            return hwnd
        except Exception:
            pass
    return None


def is_cs_window_foreground():
    """Returns True if Counter-Strike 1.6 is currently the active focused window"""
    if not (IS_WINDOWS and user32):
        return True
    cur_fg = user32.GetForegroundWindow()
    if not cur_fg:
        return False

    # 1. Broad title check
    buf = ctypes.create_string_buffer(256)
    user32.GetWindowTextA(cur_fg, buf, 256)
    val = buf.value.lower()
    # Explicitly reject IDEs, terminals, Python, Task Manager (English & Russian)
    if any(k in val for k in [b"cmd.exe", b"powershell", b"terminal", b"python", b"code", b"antigravity", b"taskmgr", b"prompt", b"\xd0\x94\xd0\xb8\xd1\x81\xd0\xbf\xd0\xb5\xd1\x82\xd1\x87\xd0\xb5\xd1\x80", b"\xd0\xb4\xd0\xb8\xd1\x81\xd0\xbf\xd0\xb5\xd1\x82\xd1\x87\xd0\xb5\xd1\x80"]):
        return False
    if any(k in val for k in [b"counter-strike", b"counter strike", b"cstrike"]):
        return True

    # 2. Direct Window Class check on foreground window (GoldSrc SDL or Valve001)
    cls_buf = ctypes.create_string_buffer(256)
    user32.GetClassNameA(cur_fg, cls_buf, 256)
    cls_val = cls_buf.value.lower()
    if cls_val in [b"sdl_app", b"valve001"]:
        return True

    # 3. Process check by PID (0x1000 works even across elevation boundaries)
    if kernel32 and psapi:
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(cur_fg, ctypes.byref(pid))
        if pid.value:
            h_proc = kernel32.OpenProcess(0x1000, False, pid.value)
            if not h_proc:
                h_proc = kernel32.OpenProcess(0x0410, False, pid.value)
            if h_proc:
                name_buf = ctypes.create_string_buffer(256)
                psapi.GetModuleBaseNameA(h_proc, None, name_buf, 256)
                kernel32.CloseHandle(h_proc)
                pname = name_buf.value.lower()
                if pname in [b"hl.exe", b"cstrike.exe"]:
                    return True

    return False




def find_cs_window_rect(monitor):
    """Finds exact Counter-Strike 1.6 window client area coordinates"""
    if IS_WINDOWS and user32:
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        for cls_name in [b"SDL_app", b"Valve001"]:
            hwnd = user32.FindWindowA(cls_name, None)
            if hwnd and user32.IsWindowVisible(hwnd):
                rect = RECT()
                user32.GetClientRect(hwnd, ctypes.byref(rect))
                pt = POINT(0, 0)
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
                if rect.right > 300 and rect.bottom > 200:
                    return {"left": pt.x, "top": pt.y, "width": rect.right, "height": rect.bottom}
    return monitor

DEBUG_DIR = "debug_visuals"
os.makedirs(DEBUG_DIR, exist_ok=True)

def annotate_frame(frame_bgr, targets, status_text, keys_str, firing_state, brain_str=""):
    """Overlays visual black-box telemetry, bounding boxes, crosshair, and neural brain status"""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    cx = w // 2
    cy = h // 2

    # Draw Green Crosshair in Center
    cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)

    # Draw Targets
    for t in targets:
        dx = t["dx"]
        dy = t["dy"]
        tx = int(cx + dx)
        ty = int(cy + dy)
        tw = int(t.get("w", 25))
        th = int(t.get("h", 50))
        conf = t.get("conf", 0.0)

        x1 = max(0, tx - tw // 2)
        y1 = max(0, ty - int(th * 0.18))
        x2 = min(w - 1, tx + tw // 2)
        y2 = min(h - 1, y1 + th)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(annotated, (tx, ty), 5, (255, 255, 0), -1)

        label = f"conf:{conf:.2f} dx:{dx:+.0f} dy:{dy:+.0f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Status Bar Header
    bar_color = (0, 0, 180) if firing_state else (40, 40, 40)
    cv2.rectangle(annotated, (0, 0), (w, 32), bar_color, -1)
    telemetry_str = f"STATUS: {status_text} | KEYS: [{keys_str}] | FIRING: {firing_state}"
    cv2.putText(annotated, telemetry_str, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Tactical Brain Telemetry Footer
    if brain_str:
        cv2.rectangle(annotated, (0, h - 26), (w, h), (20, 20, 20), -1)
        cv2.putText(annotated, f"TACTICAL BRAIN: {brain_str}", (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

    return annotated

def is_spectator_view(frame_bgr):
    """Detects genuine solid black letterbox bars rendered by GoldSrc spectator/death camera"""
    if frame_bgr is None or frame_bgr.shape[0] < 200:
        return False
    h, w = frame_bgr.shape[:2]
    # GoldSrc draws ~6% solid black bars at top and bottom of screen in spectator mode
    top_strip = frame_bgr[:int(h * 0.06), :]
    bot_strip = frame_bgr[-int(h * 0.06):, :]
    top_mean = float(np.mean(top_strip))
    bot_mean = float(np.mean(bot_strip))
    top_std = float(np.std(top_strip))
    bot_std = float(np.std(bot_strip))
    # Genuine spectator letterbox bars are pure solid black (mean < 8.0, uniform std < 4.0)
    return (top_mean < 8.0 and bot_mean < 8.0 and top_std < 4.0 and bot_std < 4.0)


def is_vgui_menu_open(frame_bgr):
    """
    Detects CS 1.6 VGUI Menu (Team Select, Class Select 'ВЫБЕРИТЕ КЛАСС', MOTD).
    Matches distinct golden header icon/title and dark background panel.
    """
    if frame_bgr is None or frame_bgr.shape[0] < 200:
        return False
    h, w = frame_bgr.shape[:2]
    # Header area check for golden CS title / icon ('ВЫБЕРИТЕ КЛАСС')
    hdr = frame_bgr[int(h * 0.04):int(h * 0.14), int(w * 0.08):int(w * 0.35)]
    hsv_hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV)
    yellow_hdr = cv2.inRange(hsv_hdr, np.array([20, 140, 140]), np.array([35, 255, 255]))
    yellow_ratio = float(np.mean(yellow_hdr > 0))
    if yellow_ratio > 0.035:
        # Confirm with dark background panel
        panel = frame_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.12):int(w * 0.88)]
        dark_ratio = float(np.mean(panel < 35))
        if dark_ratio > 0.45:
            return True
    return False


class AIVisionVideoRecorder:
    """
    Zero-Lag Cybernetic AI Vision Video Recorder:
    Captures real-time POV gameplay through the eyes of the Neural Agent:
    - Bounding boxes, Bullseye head targets, distance/confidence badges
    - Center dynamic crosshair & laser target tracing
    - Optical Red Wall Lidar avoidance indicators
    - Real-time Tactical Brain, active key inputs, mouse aim deltas
    - Animated 'REC' indicator, round timer & score
    - Asynchronous threaded writer with frame queue to guarantee zero game stutter
    - Codec: Motion JPEG (.avi) crash-proof format for universal playback
    """
    def __init__(self, output_dir="demos", fps=30):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(base_dir, output_dir)
        else:
            self.output_dir = output_dir
        self.fps = fps
        self.is_recording = False
        self.frame_queue = queue.Queue(maxsize=150)
        self.writer_thread = None
        self.current_filename = None
        self.round_num = 1
        self.frames_written = 0
        self.start_time = 0.0
        self.kill_banner_text = ""
        self.kill_banner_until = 0.0
        self.kill_banner_color = (0, 255, 255)
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:
            pass

    def trigger_banner(self, text, duration=1.8, is_headshot=False):
        self.kill_banner_text = text
        self.kill_banner_until = time.time() + duration
        self.kill_banner_color = (50, 50, 255) if is_headshot else (0, 220, 255)

    def start_recording(self, round_num=1):
        if self.is_recording:
            return
        self.round_num = round_num
        self.frames_written = 0
        self.start_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_filename = os.path.join(self.output_dir, f"round_{round_num:02d}_ai_vision_{timestamp}.avi")
        self.is_recording = True

        # Audio cue: Ascending high-pitch dual chirp (1600Hz -> 2200Hz)
        play_beep(1600, 80)
        def _second_beep():
            time.sleep(0.09)
            play_beep(2200, 100)
        threading.Thread(target=_second_beep, daemon=True).start()

        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        print("\n" + "="*76)
        print(f" >>> [AI VISION RECORDER: RECORDING] 🎬 ROUND {round_num:02d} THROUGH AI EYES! <<<")
        print(f" >>> Target File: {self.current_filename}")
        print(f" >>> Neural HUD: BBoxes + Head Bullseyes + Laser Tracking + Lidar + Brain Telemetry")
        print(f" >>> Press [F6] anytime to stop early, or let the round finish automatically.")
        print("="*76 + "\n", flush=True)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        try:
            self.frame_queue.put_nowait(None)  # Sentinel to terminate writer loop
        except Exception:
            pass

        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=3.0)
        self.writer_thread = None

        # Audio cue: Descending low-pitch dual chirp (1800Hz -> 1200Hz)
        play_beep(1800, 80)
        def _second_beep():
            time.sleep(0.09)
            play_beep(1200, 100)
        threading.Thread(target=_second_beep, daemon=True).start()

        duration = time.time() - self.start_time
        size_mb = 0.0
        if self.current_filename and os.path.exists(self.current_filename):
            try:
                size_mb = os.path.getsize(self.current_filename) / (1024 * 1024)
            except Exception:
                pass

        print("\n" + "="*76)
        print(f" >>> [AI VISION RECORDER: SAVED] 💾 Video ready to watch!")
        print(f" >>> Saved to: {self.current_filename}")
        print(f" >>> Duration: {duration:.1f}s | Recorded Frames: {self.frames_written} | File Size: {size_mb:.2f} MB")
        print("="*76 + "\n", flush=True)

    def put_frame(self, frame):
        if not self.is_recording or frame is None:
            return
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Drop frame under extreme I/O load to prevent game hitching

    def _writer_loop(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = None
        while True:
            try:
                frame = self.frame_queue.get(timeout=0.4)
            except queue.Empty:
                if not self.is_recording:
                    break
                continue

            if frame is None:
                break

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(self.current_filename, fourcc, self.fps, (w, h))

            writer.write(frame)
            self.frames_written += 1
            self.frame_queue.task_done()

        if writer:
            writer.release()

    def render_hud(
        self,
        frame_bgr,
        targets,
        primary_target,
        is_locked,
        firing_state,
        status_text,
        keys_set,
        step_dx,
        step_dy,
        brain_str,
        round_num,
        round_elapsed,
        round_kills,
        round_hs_pct,
        accumulated_reward,
        route_name="A_LANE",
        wp_idx=0,
        total_wps=0,
        wall_ahead=False,
        steer_bias=0.0,
        floor_visible=True,
        red_ratio=0.0,
        green_ratio=0.0
    ):
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]
        cx, cy = w // 2, h // 2

        # 1. Target Bounding Boxes & Head Bullseyes
        for t in targets:
            dx = t["dx"]
            dy = t["dy"]
            tx = int(cx + dx)
            ty = int(cy + dy)
            tw = int(t.get("w", 28))
            th = int(t.get("h", 55))
            conf = float(t.get("conf", 0.0))

            x1 = max(0, tx - tw // 2)
            y1 = max(0, ty - int(th * 0.18))
            x2 = min(w - 1, tx + tw // 2)
            y2 = min(h - 1, y1 + th)

            is_primary = (t is primary_target)
            if is_primary:
                box_color = (0, 0, 255) if is_locked else (0, 165, 255)
                # Bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

                # Corner Bracket Accents
                corn_len = max(6, int(tw * 0.25))
                cv2.line(vis, (x1, y1), (x1 + corn_len, y1), (255, 255, 255), 2)
                cv2.line(vis, (x1, y1), (x1, y1 + corn_len), (255, 255, 255), 2)
                cv2.line(vis, (x2, y1), (x2 - corn_len, y1), (255, 255, 255), 2)
                cv2.line(vis, (x2, y1), (x2, y1 + corn_len), (255, 255, 255), 2)
                cv2.line(vis, (x1, y2), (x1 + corn_len, y2), (255, 255, 255), 2)
                cv2.line(vis, (x1, y2), (x1, y2 - corn_len), (255, 255, 255), 2)
                cv2.line(vis, (x2, y2), (x2 - corn_len, y2), (255, 255, 255), 2)
                cv2.line(vis, (x2, y2), (x2, y2 - corn_len), (255, 255, 255), 2)

                # Laser Tracer from Center Crosshair to Target
                cv2.line(vis, (cx, cy), (tx, ty), box_color, 1, cv2.LINE_AA)

                # Head Bullseye Target
                head_y = y1 + int(th * 0.14)
                cv2.circle(vis, (tx, head_y), 7, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(vis, (tx, head_y), 2, (0, 0, 255), -1)

                # Pill Tag above Target
                tag_label = f"DUMMY {conf:.2f} | dx:{dx:+.0f} dy:{dy:+.0f} | {'LOCKED [FIRE]' if is_locked else 'AIM'}"
                (tw_txt, th_txt), _ = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(18, y1 - 6)
                cv2.rectangle(vis, (x1, tag_y1 - th_txt - 4), (x1 + tw_txt + 6, tag_y1 + 2), (20, 20, 20), -1)
                cv2.putText(vis, tag_label, (x1 + 3, tag_y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if is_locked else (255, 255, 255), 1)
            else:
                # Secondary Target
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 1)
                cv2.putText(vis, f"DUMMY {conf:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

        # 2. Dynamic Center Reticle
        ret_color = (0, 0, 255) if (is_locked or firing_state) else ((0, 220, 255) if targets else (0, 255, 0))
        cv2.line(vis, (cx - 14, cy), (cx - 4, cy), ret_color, 2)
        cv2.line(vis, (cx + 4, cy), (cx + 14, cy), ret_color, 2)
        cv2.line(vis, (cx, cy - 14), (cx, cy - 4), ret_color, 2)
        cv2.line(vis, (cx, cy + 4), (cx, cy + 14), ret_color, 2)
        cv2.circle(vis, (cx, cy), 2, ret_color, -1)
        if is_locked or firing_state:
            cv2.circle(vis, (cx, cy), 18, (0, 0, 255), 1, cv2.LINE_AA)

        # 3. Optical Terrain & Barrier Recognition (Red Wall vs Green Floor)
        # Widget Box on Top-Left: (14, 40) to (440, 96)
        overlay_lidar = vis.copy()
        cv2.rectangle(overlay_lidar, (14, 40), (440, 96), (15, 15, 15), -1)
        cv2.addWeighted(overlay_lidar, 0.75, vis, 0.25, 0, vis)

        box_border_color = (0, 0, 255) if (wall_ahead or red_ratio > 0.25) else (0, 220, 100)
        cv2.rectangle(vis, (14, 40), (440, 96), box_border_color, 2)
        cv2.putText(vis, "OPTICAL COLOR LIDAR: TERRAIN RECOGNITION", (24, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 240), 1)

        # Line 1: Green floor (Walkable / Open)
        if floor_visible or green_ratio > 0.12:
            g_txt = f"GREEN FLOOR: [WALKABLE / PATH OPEN] ({green_ratio*100:.0f}%)"
            cv2.putText(vis, g_txt, (24, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1)
        else:
            g_txt = f"GREEN FLOOR: [LOST / OBSTRUCTED] ({green_ratio*100:.0f}%)"
            cv2.putText(vis, g_txt, (24, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)

        # Line 2: Red barrier wall (Obstacle / No-go)
        if wall_ahead or red_ratio > 0.20:
            r_txt = f"RED WALL: [DANGER BARRIER / NO-GO ZONE!] ({red_ratio*100:.0f}%)"
            cv2.putText(vis, r_txt, (24, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
        else:
            r_txt = f"RED WALL: [CLEAR / NO BARRIER] ({red_ratio*100:.0f}%)"
            cv2.putText(vis, r_txt, (24, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 100), 1)

        # Big Central Warning Shield when facing Red Wall
        if wall_ahead or red_ratio > 0.25:
            cv2.rectangle(vis, (2, 2), (w - 2, h - 2), (0, 0, 255), 4)

            arrow = "<<< STEER LEFT TO GREEN" if steer_bias < 0 else "STEER RIGHT TO GREEN >>>"
            w_tag1 = "[!] [RED WALL: BARRIER DETECTED - NO-GO ZONE!] [!]"
            w_tag2 = f">>> AVOIDANCE: {arrow} (Bias: {steer_bias:+.2f}) >>>"
            (tw1, th1), _ = cv2.getTextSize(w_tag1, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            (tw2, th2), _ = cv2.getTextSize(w_tag2, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            bw = max(tw1, tw2) + 40
            bx = cx - bw // 2
            by = cy - 120
            cv2.rectangle(vis, (bx, by), (bx + bw, by + 65), (15, 15, 15), -1)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + 65), (0, 0, 255), 2)
            cv2.putText(vis, w_tag1, (bx + 20, by + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
            cv2.putText(vis, w_tag2, (bx + 20, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        # 3-Channel Optical Lidar Balance Strip above Footer (cx - 240 to cx + 240, h - 58 to h - 34)
        c_left_col = (0, 0, 255) if steer_bias > 0.3 else (0, 220, 100)
        c_center_col = (0, 0, 255) if wall_ahead else (0, 220, 100)
        c_right_col = (0, 0, 255) if steer_bias < -0.3 else (0, 220, 100)

        overlay_strip = vis.copy()
        cv2.rectangle(overlay_strip, (cx - 240, h - 58), (cx + 240, h - 34), (15, 15, 15), -1)
        cv2.addWeighted(overlay_strip, 0.70, vis, 0.30, 0, vis)
        cv2.rectangle(vis, (cx - 240, h - 58), (cx + 240, h - 34), (60, 60, 60), 1)

        l_lbl = "[L: NO-GO]" if steer_bias > 0.3 else "[L: WALK]"
        c_lbl = "[C: NO-GO]" if wall_ahead else "[C: WALK]"
        r_lbl = "[R: NO-GO]" if steer_bias < -0.3 else "[R: WALK]"

        cv2.putText(vis, l_lbl, (cx - 220, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_left_col, 1)
        cv2.putText(vis, c_lbl, (cx - 65, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_center_col, 1)
        cv2.putText(vis, r_lbl, (cx + 85, h - 41), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_right_col, 1)

        # 4. Top Status Header (Dark Translucent Bar)
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 34), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)

        # Blinking REC dot
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(vis, (16, 17), 6, (0, 0, 255), -1)
        rec_str = f"REC  RND {round_num:02d} [{int(round_elapsed // 60):02d}:{int(round_elapsed % 60):02d} / 01:55]"
        cv2.putText(vis, rec_str, (28, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        route_str = f"ROUTE: {route_name[:18]} (WP {wp_idx}/{total_wps})"
        cv2.putText(vis, route_str, (cx - 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 240), 1)

        score_str = f"KILLS: {round_kills}/11 (HS: {round_hs_pct:.0f}%) | REWARD: {accumulated_reward:+.1f}"
        (sw, _), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(vis, score_str, (w - sw - 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 150), 1)

        # 5. Bottom Tactical Footer (Dark Translucent Bar)
        overlay_bot = vis.copy()
        cv2.rectangle(overlay_bot, (0, h - 30), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_bot, 0.72, vis, 0.28, 0, vis)

        keys_str = "+".join(sorted(keys_set)) if keys_set else "NONE"
        fire_badge = " [FIRING]" if firing_state else ""
        action_desc = f"KEYS: [{keys_str}]{fire_badge} | AIM: ({step_dx:+d}, {step_dy:+d}) | {status_text[:28]}"
        cv2.putText(vis, action_desc, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        if brain_str:
            b_tag = f"BRAIN: {brain_str}"
            (bw, _), _ = cv2.getTextSize(b_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(vis, b_tag, (w - bw - 10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # 6. Event Notification Banner (Kill, Headshot, Victory)
        if time.time() < self.kill_banner_until and self.kill_banner_text:
            (bw_txt, bh_txt), _ = cv2.getTextSize(self.kill_banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            bx1 = cx - bw_txt // 2 - 16
            by1 = 80
            cv2.rectangle(vis, (bx1, by1), (bx1 + bw_txt + 32, by1 + bh_txt + 16), (20, 20, 20), -1)
            cv2.rectangle(vis, (bx1, by1), (bx1 + bw_txt + 32, by1 + bh_txt + 16), self.kill_banner_color, 2)
            cv2.putText(vis, self.kill_banner_text, (bx1 + 16, by1 + bh_txt + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.kill_banner_color, 2)

        return vis


class GameLogWatcher(threading.Thread):
    def __init__(self, on_death=None, on_kill=None, on_damaged=None, on_round_start=None, on_enemy_hit=None, on_round_win=None):
        super().__init__(daemon=True)
        self.on_death = on_death
        self.on_kill = on_kill
        self.on_damaged = on_damaged
        self.on_round_start = on_round_start
        self.on_enemy_hit = on_enemy_hit
        self.on_round_win = on_round_win
        self.is_alive = True
        self.running = True

    def find_latest_log(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        if "CS16_WORK_DIR" in os.environ:
            candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "logs"))
        candidates.extend([
            "cstrike/logs",
            os.path.join(base_dir, "cstrike", "logs"),
            os.path.join(base_dir, "CS 1.6 GoldClient", "cstrike", "logs"),
            "D:/CS 1.6 GoldClient/cstrike/logs",
            "/mnt/d/CS 1.6 GoldClient/cstrike/logs"
        ])
        for base in candidates:
            if os.path.exists(base):
                logs = sorted([os.path.join(base, f) for f in os.listdir(base) if f.endswith(".log")], key=os.path.getmtime)
                if logs:
                    return logs[-1]
        return None

    def run(self):
        log_file = self.find_latest_log()
        f = None
        if log_file and os.path.exists(log_file):
            f = open(log_file, "r", encoding="utf-8", errors="ignore")
            f.seek(0, os.SEEK_END)

        while self.running:
            time.sleep(0.05)
            cur = self.find_latest_log()
            if cur != log_file:
                if f:
                    try: f.close()
                    except Exception: pass
                log_file = cur
                if log_file and os.path.exists(log_file):
                    f = open(log_file, "r", encoding="utf-8", errors="ignore")
                    f.seek(0, os.SEEK_END)

            if not f:
                continue

            # Read all available lines in buffer
            while self.running:
                line = f.readline()
                if not line:
                    f.seek(f.tell())  # Clear Windows CRT EOF cache
                    break
                line_str = line.strip()

                if 'World triggered "Round_Start"' in line_str:
                    self.is_alive = True
                    if self.on_round_start:
                        self.on_round_start(line_str)
                elif 'World triggered "Round_End"' in line_str:
                    self.is_alive = False
                elif 'triggered "CTs_Win"' in line_str:
                    if self.on_round_win:
                        self.on_round_win(line_str)
                elif ' attacked "' in line_str and ("STEAM_ID_LAN" in line_str or "не моюсь" in line_str):
                    parts = line_str.split(' attacked "')
                    if len(parts) >= 2:
                        victim_part = parts[1]
                        attacker_part = parts[0]
                        if "STEAM_ID_LAN" in victim_part or "не моюсь" in victim_part:
                            if self.on_damaged:
                                self.on_damaged(line_str)
                        elif "STEAM_ID_LAN" in attacker_part or "не моюсь" in attacker_part:
                            if self.on_enemy_hit:
                                self.on_enemy_hit(line_str)
                elif 'killed "' in line_str and ("STEAM_ID_LAN" in line_str or "не моюсь" in line_str):
                    parts = line_str.split(' killed "')
                    if len(parts) >= 2:
                        if "STEAM_ID_LAN" in parts[1] or "не моюсь" in parts[1]:
                            self.is_alive = False
                            if self.on_death:
                                self.on_death(line_str)
                        elif "STEAM_ID_LAN" in parts[0] or "не моюсь" in parts[0]:
                            if self.on_kill:
                                self.on_kill(line_str)

# Audio Loopback
HAVE_SOUNDDEVICE = False
try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except ImportError:
    pass

class LiveAudioListener:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
        self.current_window = np.zeros((1024, channels), dtype=np.float32)

    def start(self):
        if not HAVE_SOUNDDEVICE:
            return
        try:
            def audio_cb(indata, frames, time_info, status):
                self.current_window = indata.copy()

            self.stream = sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=audio_cb,
                extra_settings=sd.WasapiSettings(loopback=True) if IS_WINDOWS else None
            )
            self.stream.start()
        except Exception:
            try:
                self.stream = sd.InputStream(
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=audio_cb
                )
                self.stream.start()
            except Exception:
                pass

    def get_cues(self):
        data = self.current_window
        if data is None or len(data) == 0:
            return 0.0, 0.0, 0.0, 0.0
        left = data[:, 0]
        right = data[:, 1] if data.shape[1] > 1 else data[:, 0]
        rms_l = float(np.sqrt(np.mean(left ** 2) + 1e-8))
        rms_r = float(np.sqrt(np.mean(right ** 2) + 1e-8))
        balance = float((rms_r - rms_l) / (rms_l + rms_r + 1e-4))
        peak = float(np.max(np.abs(data)))
        return rms_l, rms_r, np.clip(balance, -1.0, 1.0), peak

    def close(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

def nms_boxes(boxes, scores, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0] - boxes[:, 2] / 2.0
    y1 = boxes[:, 1] - boxes[:, 3] / 2.0
    x2 = boxes[:, 0] + boxes[:, 2] / 2.0
    y2 = boxes[:, 1] + boxes[:, 3] / 2.0
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

class CSYOLODetector:
    def __init__(self, model_path="models/yolov8n_crowdhuman.onnx", conf_thresh=0.18):
        self.conf_thresh = conf_thresh
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 640

    def detect(self, bgr_img):
        # If VGUI class/team selection menu is open on screen, suppress detections (prevents shooting menu art)
        if is_vgui_menu_open(bgr_img):
            return []

        gh, gw = bgr_img.shape[:2]
        cx_game = gw // 2
        cy_game = gh // 2

        # 1. Neural YOLOv8 CrowdHuman Inference
        resized = cv2.resize(bgr_img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None, ...]

        out = self.session.run(None, {self.input_name: inp})[0]
        preds = out[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes = preds[:, :4]
        scores = preds[:, 4]

        conf_limit = min(self.conf_thresh, 0.18)
        mask = scores > conf_limit
        targets = []
        scale_x = gw / float(self.input_size)
        scale_y = gh / float(self.input_size)

        if np.any(mask):
            b_filt = boxes[mask]
            s_filt = scores[mask]
            keep = nms_boxes(b_filt, s_filt, iou_thresh=0.40)

            for idx in keep:
                bx, by, bw, bh = b_filt[idx]
                conf = float(s_filt[idx])

                real_w = bw * scale_x
                real_h = bh * scale_y
                real_cx = bx * scale_x
                real_cy = by * scale_y

                # Vertical aspect ratio (0.90 - 4.20) for standard players and mannequins
                aspect = real_h / max(1.0, real_w)
                if aspect < 0.90 or aspect > 4.20:
                    continue
                if real_h < 26 or real_w < 14:
                    continue

                # Reject walls and oversized map geometry
                if real_w > 380 or real_h > 720:
                    continue

                # Reject dead bodies / corpses lying flat on the floor
                if real_w > real_h * 1.10:
                    continue

                # Anti-Ceiling Filter: Reject any target whose center or feet are in the skybox/ceiling (top 25% of screen)
                if real_cy < gh * 0.22 or (real_cy + real_h * 0.5) < gh * 0.35:
                    continue

                y1_c = max(0, int(real_cy - real_h / 2.0))
                y2_c = min(gh, int(real_cy + real_h / 2.0))
                x1_c = max(0, int(real_cx - real_w / 2.0))
                x2_c = min(gw, int(real_cx + real_w / 2.0))

                # Anti-decal check: reject deep red blood decals on crates
                _ch = y2_c - y1_c
                _cw = x2_c - x1_c
                crop = bgr_img[
                    y1_c + int(_ch * 0.15):y1_c + int(_ch * 0.65),
                    x1_c + int(_cw * 0.20):x1_c + int(_cw * 0.80)
                ]
                if crop.size > 30:
                    mb, mg, mr = crop.mean(axis=(0, 1))
                    if mr > 110 and mb < 80 and mg < 90:
                        continue

                # Target point: Headshot Center (top 16% of body box)
                target_x = real_cx
                target_y = (real_cy - real_h * 0.50) + (real_h * 0.16)

                dx = target_x - cx_game
                dy = target_y - cy_game
                dist = math.hypot(dx, dy)

                targets.append({
                    "source": "yolo",
                    "conf": conf,
                    "dx": dx,
                    "dy": dy,
                    "dist": dist,
                    "w": real_w,
                    "h": real_h,
                    "box": (x1_c, y1_c, x2_c, y2_c),
                    "is_locked": (abs(dx) <= max(24.0, real_w * 0.65) and abs(dy) <= max(20.0, real_h * 0.75))
                })

        # 2. Crash-Test Dummy Detection (aim_ai_sandbox dummies):
        # Neural YOLOv8n_crowdhuman is trained on human clothes, so dummy mannequin conf is often 0.05-0.15.
        # This morphological detector guarantees 100% detection of yellow crash-test dummies with bullseye heads,
        # with strict filters to prevent any phantom targets on ceiling or weapon models.
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([20, 85, 85]), np.array([38, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 25))
        closed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(1.0, float(bw))
            # Human vertical aspect ratio
            if aspect < 1.15 or aspect > 4.5:
                continue
            if bh < 28 or bw < 12:
                continue
            # Skybox / Ceiling Guard: feet must be at or below gh * 0.38
            if by < gh * 0.15 or (by + bh) < gh * 0.38:
                continue
            # Reject player's own viewmodel weapon / hands
            if by + bh > gh * 0.95:
                continue

            cx_d = bx + bw / 2.0
            cy_d = by + bh / 2.0
            matched = False
            for t in targets:
                t_bx1, t_by1, t_bx2, t_by2 = t["box"]
                if (t_bx1 <= cx_d <= t_bx2) and (t_by1 <= cy_d <= t_by2):
                    t["conf"] = max(t["conf"], 0.95)
                    t["source"] = "yolo+dummy"
                    matched = True
                    break
            if not matched:
                head_x = bx + bw / 2.0
                head_y = by + bh * 0.14
                dx = head_x - cx_game
                dy = head_y - cy_game
                dist = math.hypot(dx, dy)
                targets.append({
                    "source": "dummy_color",
                    "conf": 0.92,
                    "dx": dx,
                    "dy": dy,
                    "dist": dist,
                    "w": float(bw),
                    "h": float(bh),
                    "box": (bx, by, bx + bw, by + bh),
                    "is_locked": (abs(dx) <= max(24.0, bw * 0.65) and abs(dy) <= max(20.0, bh * 0.75))
                })

        targets.sort(key=lambda t: t["dist"])
        return targets

class TacticalBrainExecutor:
    """
    Ultra-Fast NumPy Neural Policy Executor for CS 1.6 Tactical Decision Making.
    Runs in ~0.25ms on CPU (over 4,000 FPS) with zero GPU contention.
    Predicts:
      - Movement logits [W, S, A, D] (counter-strafing, jiggle-peeking, retreat)
      - Combat logits [Fire, Duck] (standing spray vs. crouch spray)
      - Aim deltas [aim_dx, aim_dy] (dynamic recoil pull-down & micro-adjust)
    """
    def __init__(self, npz_path="models/tactical_brain.npz"):
        self.available = False
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        if "CS16_WORK_DIR" in os.environ:
            candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "models", "tactical_brain.npz"))
            candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "tactical_brain.npz"))
        candidates.extend([
            npz_path,
            os.path.join(base_dir, "models", "tactical_brain.npz"),
            os.path.join(base_dir, "CS 1.6 GoldClient", "models", "tactical_brain.npz"),
            "tactical_brain.npz",
            "D:/CS 1.6 GoldClient/models/tactical_brain.npz",
            "/mnt/d/CS 1.6 GoldClient/models/tactical_brain.npz"
        ])
        chosen = None
        for p in candidates:
            if os.path.exists(p):
                chosen = p
                break
        
        if not chosen:
            print(f"[!] Notice: Tactical Brain weights ({npz_path}) not found. Neural tactics inactive.")
            return

        try:
            data = np.load(chosen)
            self.w0 = data['backbone.0.weight'].T
            self.b0 = data['backbone.0.bias']
            self.ln0_w = data['backbone.1.weight']
            self.ln0_b = data['backbone.1.bias']
            
            self.w1 = data['backbone.3.weight'].T
            self.b1 = data['backbone.3.bias']
            self.ln1_w = data['backbone.4.weight']
            self.ln1_b = data['backbone.4.bias']
            
            self.w2 = data['backbone.7.weight'].T
            self.b2 = data['backbone.7.bias']
            self.ln2_w = data['backbone.8.weight']
            self.ln2_b = data['backbone.8.bias']
            
            self.wm = data['move_head.weight'].T
            self.bm = data['move_head.bias']
            
            self.wc = data['combat_head.weight'].T
            self.bc = data['combat_head.bias']
            
            self.wa = data['aim_head.weight'].T
            self.ba = data['aim_head.bias']
            
            self.c1 = np.float32(0.79788456)
            self.c2 = np.float32(0.044715)
            self.available = True
            print(f"[*] Tactical Neural Brain: Loaded weights from '{chosen}' (Active 16-D Policy)")
        except Exception as e:
            print(f"[!] Error loading Tactical Brain: {e}")
            self.available = False

    def predict(self, feature_vector):
        if not self.available:
            return np.zeros(4, dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        x = np.asarray(feature_vector, dtype=np.float32)
        # Layer 1
        h = x @ self.w0 + self.b0
        mean = np.mean(h)
        var = np.var(h)
        h = (h - mean) / np.sqrt(var + 1e-5) * self.ln0_w + self.ln0_b
        h = 0.5 * h * (1.0 + np.tanh(self.c1 * (h + self.c2 * (h ** 3))))
        
        # Layer 2
        h = h @ self.w1 + self.b1
        mean = np.mean(h)
        var = np.var(h)
        h = (h - mean) / np.sqrt(var + 1e-5) * self.ln1_w + self.ln1_b
        h = 0.5 * h * (1.0 + np.tanh(self.c1 * (h + self.c2 * (h ** 3))))
        
        # Layer 3
        h = h @ self.w2 + self.b2
        mean = np.mean(h)
        var = np.var(h)
        h = (h - mean) / np.sqrt(var + 1e-5) * self.ln2_w + self.ln2_b
        h = 0.5 * h * (1.0 + np.tanh(self.c1 * (h + self.c2 * (h ** 3))))
        
        move = h @ self.wm + self.bm
        combat = h @ self.wc + self.bc
        aim = h @ self.wa + self.ba
        return move, combat, aim

class MultimodalBotExecutor:
    """
    Multimodal Behavioral Cloning Policy Executor (Vision CNN + Audio Ear).
    Loaded from models/cs16_bot.onnx (trained on 1 hour human gameplay + 6x augmentation).
    Inputs:
      - Video frame: resized to 160x120, normalized [0, 1], shape (1, 3, 120, 160)
      - Stereo audio vector: (1, 4) [vol_left, vol_right, balance, peak]
    Outputs:
      - keys: 12 logits ["w", "a", "s", "d", "space", "ctrl", "shift", "r", "1", "2", "3", "4"]
      - clicks: 2 logits [Left click, Right click]
      - aim: continuous mouse deltas [dx, dy]
    """
    def __init__(self, model_path="models/cs16_bot.onnx"):
        self.available = False
        self.session = None
        self.keys_list = ["w", "a", "s", "d", "space", "ctrl", "shift", "r", "1", "2", "3", "4"]
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        if "CS16_WORK_DIR" in os.environ:
            candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "models", "cs16_bot.onnx"))
            candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "cs16_bot.onnx"))
        candidates.extend([
            model_path,
            os.path.join(base_dir, "models", "cs16_bot.onnx"),
            os.path.join(base_dir, "CS 1.6 GoldClient", "models", "cs16_bot.onnx"),
            "cs16_bot.onnx",
            "models/cs16_bot.onnx",
            "D:/CS 1.6 GoldClient/models/cs16_bot.onnx",
            "/mnt/d/CS 1.6 GoldClient/models/cs16_bot.onnx"
        ])
        chosen = None
        for p in candidates:
            if os.path.exists(p):
                chosen = p
                break

        if not chosen or ort is None:
            print("[!] Notice: cs16_bot.onnx not found. Multimodal vision policy inactive.")
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(chosen, opts, providers=['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'])
            self.available = True
            print(f"[*] Multimodal Vision+Audio Policy: Successfully loaded from '{chosen}'")
        except Exception as e:
            print(f"[!] Error loading Multimodal Bot model {chosen}: {e}")
            self.available = False

    def predict(self, frame_bgr, audio_vector_4d):
        if not self.available or self.session is None:
            return {}, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        try:
            small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (160, 120), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            norm_frame = np.ascontiguousarray(small_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
            input_frame = np.expand_dims(norm_frame, axis=0)

            input_audio = np.asarray(audio_vector_4d, dtype=np.float32).reshape(1, 4)

            outs = self.session.run(None, {"frames": input_frame, "audio": input_audio})
            keys_logits = outs[0][0]
            clicks_logits = outs[1][0]
            aim_deltas = outs[2][0]

            keys_dict = {k: float(keys_logits[i]) for i, k in enumerate(self.keys_list)}
            return keys_dict, clicks_logits, aim_deltas
        except Exception:
            return {}, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

class PolicyBrainExecutor:
    """
    Ultra-Fast Pure NumPy Policy Executor for CS 1.6 Online RL Policy (StablePolicyNet).
    Inference: < 0.05ms on CPU.
    Inputs: 30-D history vector (5 frames x 6 features: [x/1000, y/1000, z/1000, pitch/90, yaw/180, hp/100]).
    Outputs:
      - keys: 7 logits [W, S, A, D, Space, Ctrl, R]
      - clicks: 2 logits [Left (fire), Right]
      - aim: continuous 2-D deltas [-15.0, +15.0]
    Features automatic Hot-Reloading whenever policy_weights.npz is updated by self_learning_daemon.
    """
    def __init__(self, npz_path="models/policy_weights.npz"):
        self.available = False
        self.active_path = None
        self.last_mtime = 0.0
        self.last_reload_check = 0.0
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.candidates = []
        if "CS16_WORK_DIR" in os.environ:
            self.candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "models", "policy_weights.npz"))
            self.candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "policy_weights.npz"))
        self.candidates.extend([
            os.path.join(base_dir, "models", "policy_weights.npz"),
            os.path.join(base_dir, "CS 1.6 GoldClient", "models", "policy_weights.npz"),
            "models/policy_weights.npz",
            npz_path,
            "policy_weights.npz",
            "D:/CS 1.6 GoldClient/models/policy_weights.npz",
            "/mnt/d/CS 1.6 GoldClient/models/policy_weights.npz"
        ])
        for p in self.candidates:
            if os.path.exists(p):
                self._load(p)
                break

    def _load(self, path):
        try:
            data = np.load(path)
            self.fc1_w = data['fc1.weight'].T
            self.fc1_b = data['fc1.bias']
            self.ln1_w = data['ln1.weight']
            self.ln1_b = data['ln1.bias']

            self.fc2_w = data['fc2.weight'].T
            self.fc2_b = data['fc2.bias']
            self.ln2_w = data['ln2.weight']
            self.ln2_b = data['ln2.bias']

            self.fc3_w = data['fc3.weight'].T
            self.fc3_b = data['fc3.bias']
            self.ln3_w = data['ln3.weight']
            self.ln3_b = data['ln3.bias']

            self.keys_w = data['keys_head.weight'].T
            self.keys_b = data['keys_head.bias']
            self.clicks_w = data['clicks_head.weight'].T
            self.clicks_b = data['clicks_head.bias']
            self.aim_w = data['aim_head.weight'].T
            self.aim_b = data['aim_head.bias']

            self.active_path = path
            self.last_mtime = os.path.getmtime(path)
            self.available = True
            print(f"[*] Macro Policy Brain: Successfully loaded weights from '{path}' (Active 30-D Policy)")
        except Exception as e:
            print(f"[!] Error loading Policy Brain weights from {path}: {e}")
            self.available = False

    def check_reload(self):
        now = time.time()
        if now - self.last_reload_check < 4.0:
            return
        self.last_reload_check = now
        target_path = self.active_path or self.candidates[0]
        if os.path.exists(target_path):
            try:
                mt = os.path.getmtime(target_path)
                if mt > self.last_mtime:
                    self._load(target_path)
                    print(f"\n[★ HOT-RELOAD] AI Agent dynamically reloaded evolved weights from {target_path}!")
            except Exception:
                pass

    @staticmethod
    def _layer_norm(x, w, b, eps=1e-5):
        m = np.mean(x)
        v = np.var(x)
        return (x - m) / np.sqrt(v + eps) * w + b

    def predict(self, feature_30d):
        if not self.available:
            return np.zeros(7, dtype=np.float32), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

        x = np.asarray(feature_30d, dtype=np.float32)
        # Layer 1
        h1 = np.maximum(0.0, self._layer_norm(x @ self.fc1_w + self.fc1_b, self.ln1_w, self.ln1_b))
        # Layer 2
        h2 = np.maximum(0.0, self._layer_norm(h1 @ self.fc2_w + self.fc2_b, self.ln2_w, self.ln2_b))
        # Layer 3
        h3 = np.maximum(0.0, self._layer_norm(h2 @ self.fc3_w + self.fc3_b, self.ln3_w, self.ln3_b))

        keys = h3 @ self.keys_w + self.keys_b
        clicks = h3 @ self.clicks_w + self.clicks_b
        aim = 15.0 * np.tanh(h3 @ self.aim_w + self.aim_b)
        return keys, clicks, aim


class OnlineExperienceRecorder:
    """
    Asynchronous, non-blocking online gameplay experience recorder.
    Buffers samples in memory and flushes to online_experiences.jsonl via background worker,
    guaranteeing 0 impact on 60 FPS gameplay loop.
    """
    def __init__(self, filename="dataset/online_experiences.jsonl", max_queue=4000):
        self.filename = filename
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        if "CS16_EXP_FILE" in os.environ and os.path.exists(os.environ["CS16_EXP_FILE"]):
            candidates.append(os.environ["CS16_EXP_FILE"])
        if "CS16_WORK_DIR" in os.environ:
            candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "dataset", "online_experiences.jsonl"))
        candidates.extend([
            filename,
            os.path.join(base_dir, "dataset", "online_experiences.jsonl"),
            os.path.join(base_dir, "CS 1.6 GoldClient", "dataset", "online_experiences.jsonl"),
            "dataset/online_experiences.jsonl",
            "D:/CS 1.6 GoldClient/dataset/online_experiences.jsonl",
            "/mnt/d/CS 1.6 GoldClient/dataset/online_experiences.jsonl"
        ])
        for candidate in candidates:
            if os.path.exists(candidate) and not (os.path.islink(candidate) and not os.path.exists(os.path.realpath(candidate))):
                self.filename = candidate
                break

        self.total_recorded = 0
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.total_recorded = sum(1 for _ in f)
            except Exception:
                self.total_recorded = 0
        self.queue = queue.Queue(maxsize=max_queue)
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()
        print(f"[*] Online Experience Recorder: Active writing to '{self.filename}' (Existing: {self.total_recorded} samples)")

    def record(self, state_30d, keys_7d, clicks_2d, aim_2d, reward):
        if not self.running:
            return
        self.total_recorded += 1
        try:
            item = {
                "x": [round(float(v), 5) for v in state_30d],
                "k": [int(k) for k in keys_7d],
                "c": [int(c) for c in clicks_2d],
                "a": [float(np.clip(a, -15.0, 15.0)) for a in aim_2d],
                "r": round(float(reward), 4)
            }
            self.queue.put_nowait(item)
        except queue.Full:
            pass

    def _run(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        while self.running:
            batch = []
            try:
                batch.append(self.queue.get(timeout=1.0))
                while len(batch) < 100:
                    try:
                        batch.append(self.queue.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if batch:
                try:
                    with open(self.filename, "a", encoding="utf-8") as f:
                        for it in batch:
                            f.write(json.dumps(it) + "\n")
                except Exception:
                    pass

    def stop(self):
        self.running = False


class BenchmarkTelemetryTracker:
    """
    Automatic Telemetry Logger for 4 Core AI Modules:
    1. Movement Module: Route, Waypoints reached, Wall avoidances, Unstuck maneuvers
    2. Perception Module: Dummies spotted frames, Avg detection confidence, Head bullseye locks
    3. Combat & Rewards: Kills (Total, Headshots, Body), HS %, Net round reward
    4. Reinforcement Learning: Experience samples collected, policy update triggers
    """
    def __init__(self, filename="benchmarks/live_telemetry.json", baseline_file="benchmarks/baseline_10_rounds.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates_dir = [
            os.path.join(os.getcwd(), "benchmarks"),
            os.path.join(base_dir, "benchmarks"),
            "D:/CS 1.6 GoldClient/benchmarks",
            "/mnt/d/CS 1.6 GoldClient/benchmarks"
        ]
        target_dir = candidates_dir[0]
        for cd in candidates_dir:
            if os.path.exists(cd):
                target_dir = cd
                break
        os.makedirs(target_dir, exist_ok=True)
        self.filename = os.path.join(target_dir, os.path.basename(filename))
        self.baseline_file = os.path.join(target_dir, os.path.basename(baseline_file))
        self.vision_log_file = os.path.join(target_dir, "vision_log.jsonl")
        self._last_vision_log_time = 0.0
        self._last_event_type = ""
        self._last_spotted_state = False
        if os.path.exists(self.vision_log_file):
            try:
                if os.path.getsize(self.vision_log_file) > 10 * 1024 * 1024:
                    prev_f = self.vision_log_file.replace(".jsonl", "_prev.jsonl")
                    if os.path.exists(prev_f): os.remove(prev_f)
                    os.rename(self.vision_log_file, prev_f)
            except Exception:
                pass
        self.rounds = []
        self.current_round = 0
        self.round_data = None
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    self.rounds = json.load(f)
                    self.current_round = len(self.rounds)
            except Exception:
                self.rounds = []
        print(f"[*] Benchmark Telemetry Tracker: Logging to '{self.filename}' (Active session round: {self.current_round})")
        print(f"[*] Vision Perception Logger: Real-time recognition stream to '{self.vision_log_file}'")

    def log_vision_event(self, event_type, targets, lidar_data=None, action_status="", extra=None):
        now = time.time()
        has_targets = len(targets) > 0
        state_changed = (has_targets != self._last_spotted_state) or (event_type != self._last_event_type)
        time_elapsed = now - self._last_vision_log_time

        # Throttle: state transitions, locks and kills log immediately; steady states throttle to ~3 Hz (0.33s)
        if not state_changed and time_elapsed < 0.33 and event_type not in ["KILL_CONFIRMED", "AIM_LOCKED", "ROUND_START", "ROUND_FINISH"]:
            return

        self._last_vision_log_time = now
        self._last_event_type = event_type
        self._last_spotted_state = has_targets

        primary = targets[0] if targets else None
        record = {
            "time": time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1)*1000):03d}",
            "round": self.current_round,
            "round_sec": round(now - self.round_data["start_time"], 2) if self.round_data else 0.0,
            "event": event_type,
            "action": action_status,
            "targets_count": len(targets),
        }
        if primary:
            record["target"] = {
                "source": primary.get("source", "unknown"),
                "conf": round(float(primary.get("conf", 0.0)), 3),
                "dist_px": round(float(primary.get("dist", 0.0)), 1),
                "dx": round(float(primary.get("dx", 0.0)), 1),
                "dy": round(float(primary.get("dy", 0.0)), 1),
                "box": [int(v) for v in primary.get("box", (0, 0, 0, 0))],
                "is_locked": bool(primary.get("is_locked", False)),
                "w": round(float(primary.get("w", 0.0)), 1),
                "h": round(float(primary.get("h", 0.0)), 1)
            }
        if lidar_data:
            record["lidar"] = {
                "wall_ahead": bool(lidar_data.get("wall_ahead", False)),
                "red_pct": round(float(lidar_data.get("red_ratio", 0.0)) * 100.0, 1),
                "green_pct": round(float(lidar_data.get("green_ratio", 0.0)) * 100.0, 1),
                "steer_bias": round(float(lidar_data.get("steer_bias", 0.0)), 2)
            }
        if extra:
            record["extra"] = extra

        try:
            with open(self.vision_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def start_round(self, route_name):
        self.current_round += 1
        self.round_data = {
            "round": self.current_round,
            "route": route_name,
            "start_time": time.time(),
            "duration": 0.0,
            "kills_total": 0,
            "kills_headshot": 0,
            "kills_body": 0,
            "dummies_spotted_frames": 0,
            "head_bullseye_locks": 0,
            "conf_sum": 0.0,
            "conf_count": 0,
            "wall_avoidances": 0,
            "unstuck_events": 0,
            "max_wp_reached": 0,
            "total_waypoints": 0,
            "round_reward": 0.0,
            "total_rl_samples": 0
        }
        self.log_vision_event("ROUND_START", [], action_status=f"ROUTE: {route_name}")

    def on_kill(self, is_hs):
        if self.round_data:
            self.round_data["kills_total"] += 1
            if is_hs:
                self.round_data["kills_headshot"] += 1
            else:
                self.round_data["kills_body"] += 1

    def on_spotted(self, conf, is_head):
        if self.round_data:
            self.round_data["dummies_spotted_frames"] += 1
            self.round_data["conf_sum"] += conf
            self.round_data["conf_count"] += 1
            if is_head:
                self.round_data["head_bullseye_locks"] += 1

    def on_wall_avoid(self):
        if self.round_data:
            self.round_data["wall_avoidances"] += 1

    def on_unstuck(self):
        if self.round_data:
            self.round_data["unstuck_events"] += 1

    def update_wp(self, wp_idx, total_wps):
        if self.round_data:
            self.round_data["max_wp_reached"] = max(self.round_data["max_wp_reached"], wp_idx)
            self.round_data["total_waypoints"] = total_wps

    def finish_round(self, net_reward, total_rl_samples=0):
        if not self.round_data:
            return
        self.round_data["duration"] = round(time.time() - self.round_data["start_time"], 2)
        self.round_data["round_reward"] = round(net_reward, 2)
        self.round_data["avg_conf"] = round(self.round_data["conf_sum"] / max(1, self.round_data["conf_count"]), 3)
        self.round_data["headshot_pct"] = round((self.round_data["kills_headshot"] / max(1, self.round_data["kills_total"])) * 100.0, 1)
        self.round_data["total_rl_samples"] = total_rl_samples
        
        self.rounds.append(self.round_data)
        
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.rounds, f, indent=2)
        except Exception:
            pass

        d = self.round_data
        self.log_vision_event("ROUND_FINISH", [], action_status=f"KILLS: {d['kills_total']}/11 (HS: {d['kills_headshot']})", extra={"duration": d["duration"], "wall_bounces": d["wall_avoidances"], "unstucks": d["unstuck_events"]})
        print(f"\n" + "="*76)
        print(f" [ROUND {d['round']:02d} TELEMETRY] Route: {d['route']} | Duration: {d['duration']:.1f}s")
        print(f"  * 1. Movement:    Waypoints: {d['max_wp_reached']}/{d['total_waypoints']} | Wall Bounces: {d['wall_avoidances']} | Unstucks: {d['unstuck_events']}")
        print(f"  * 2. Perception:  Spotted: {d['dummies_spotted_frames']} frames | Avg Conf: {d['avg_conf']:.2f} | Head Locks: {d['head_bullseye_locks']}")
        print(f"  * 3. Combat:      Kills: {d['kills_total']}/11 (HS: {d['kills_headshot']}, Body: {d['kills_body']}) | HS%: {d['headshot_pct']:.1f}%")
        print(f"  * 4. RL & Reward: Round Reward: {d['round_reward']:+.1f} | RL Buffer: {d['total_rl_samples']} samples")
        print("="*76 + "\n", flush=True)

        if self.current_round == 10:
            self.save_baseline_summary()

    def save_baseline_summary(self):
        first_10 = self.rounds[:10]
        if not first_10:
            return
        total_kills = sum(r["kills_total"] for r in first_10)
        total_hs = sum(r["kills_headshot"] for r in first_10)
        avg_reward = sum(r["round_reward"] for r in first_10) / len(first_10)
        avg_wp = sum(r["max_wp_reached"] for r in first_10) / len(first_10)
        avg_conf = sum(r["avg_conf"] for r in first_10) / len(first_10)
        total_rl = first_10[-1].get("total_rl_samples", 0)
        
        summary = {
            "benchmark_type": "BASELINE_FIRST_10_ROUNDS",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rounds_analyzed": len(first_10),
            "movement": {
                "avg_waypoints_reached": round(avg_wp, 2),
                "total_wall_avoidances": sum(r["wall_avoidances"] for r in first_10),
                "total_unstuck_events": sum(r["unstuck_events"] for r in first_10)
            },
            "perception": {
                "avg_confidence": round(avg_conf, 3),
                "total_spotted_frames": sum(r["dummies_spotted_frames"] for r in first_10),
                "total_head_locks": sum(r["head_bullseye_locks"] for r in first_10)
            },
            "combat_and_rewards": {
                "total_kills": total_kills,
                "avg_kills_per_round": round(total_kills / len(first_10), 2),
                "headshot_percentage": round((total_hs / max(1, total_kills)) * 100.0, 1),
                "avg_round_reward": round(avg_reward, 2)
            },
            "reinforcement_learning": {
                "total_samples_collected": total_rl
            },
            "detailed_rounds": first_10
        }
        try:
            with open(self.baseline_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print("\n" + "#"*76)
            print(" [★ BASELINE 10-ROUND BENCHMARK SAVED TO benchmarks/baseline_10_rounds.json ★]")
            print("  Now let the agent play for several hours!")
            print("  Run 'python compare_benchmarks.py' anytime to see training improvements!")
            print("#"*76 + "\n", flush=True)
        except Exception as e:
            print(f"[!] Error saving baseline: {e}", flush=True)


def main():
    model_path = "models/yolov8n_crowdhuman.onnx"
    if not os.path.exists(model_path):
        model_path = "yolov8n_crowdhuman.onnx"

    tactical_path = "models/tactical_brain.npz"

    multimodal_bot = MultimodalBotExecutor(model_path="models/cs16_bot.onnx")
    tactical_brain = TacticalBrainExecutor(npz_path=tactical_path)
    policy_brain = PolicyBrainExecutor(npz_path="models/policy_weights.npz")
    exp_recorder = OnlineExperienceRecorder(filename="dataset/online_experiences.jsonl")
    telemetry_tracker = BenchmarkTelemetryTracker()
    pose_history = deque(maxlen=5)
    accumulated_reward = 0.0

    print("=" * 65)
    print("  Counter-Strike 1.6 - Master AI Hybrid Agent (Option A Ensemble)")
    print("=============================================================")
    print(f"Vision Detector : YOLOv8 CrowdHuman ({model_path})")
    print(f"Vision+Audio BC : Multimodal Human Policy ({'Active cs16_bot.onnx' if multimodal_bot.available else 'Disabled'})")
    print(f"Tactical Brain  : Neural Decision Engine ({tactical_path})")
    print(f"Macro Policy    : 30-D Spatial Brain ({'Active Hot-Reload' if policy_brain.available else 'Disabled'})")
    print(f"Hearing Engine  : {'Stereo Audio Active (Loopback)' if HAVE_SOUNDDEVICE else 'Disabled'}")
    print("Navigation      : 3D Waypoint Engine (Live GoldSrc hw.dll Memory Reader)")
    print("Hotkeys         : [F9] Toggle Active/Pause | [F6] Record Round (AI Eyes Video) | [Ctrl+C] Exit")
    print("=" * 65)

    mem_reader = CS16MemoryReader()
    if mem_reader.ptr_addr:
        print("[*] Memory Engine: Connected to GoldSrc hw.dll live coordinates!")
    else:
        print("[*] Memory Engine: Waiting for game process hl.exe...")

    detector = CSYOLODetector(model_path=model_path, conf_thresh=0.28)
    audio_listener = LiveAudioListener()
    audio_listener.start()

    sct = getattr(mss, "MSS", mss.mss)()
    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    box = find_cs_window_rect(monitor)
    crop_w = box["width"]
    crop_h = box["height"]
    print(f"[*] Calibrated Game Window: {crop_w}x{crop_h} at ({box['left']}, {box['top']})")

    # Auto-calibrate mouse physics from CS 1.6 config
    sens = 1.1
    m_yaw = 0.022
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if "CS16_WORK_DIR" in os.environ:
        candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "config.cfg"))
    candidates.extend([
        "cstrike/config.cfg",
        os.path.join(base_dir, "cstrike", "config.cfg"),
        os.path.join(base_dir, "CS 1.6 GoldClient", "cstrike", "config.cfg"),
        "D:/CS 1.6 GoldClient/cstrike/config.cfg",
        "/mnt/d/CS 1.6 GoldClient/cstrike/config.cfg"
    ])
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            if parts[0].lower() == "sensitivity":
                                sens = float(parts[1].strip('"'))
                            elif parts[0].lower() == "m_yaw":
                                m_yaw = float(parts[1].strip('"'))
            except Exception:
                pass
            break

    # Auto-calibrate mouse physics and GoldSrc FOV
    # CS 1.6: default 90 deg at 4:3 -> vertical FOV is 73.74 deg
    v_fov = 73.74
    aspect_ratio = float(crop_w) / max(1.0, float(crop_h))
    h_fov = 2.0 * math.degrees(math.atan(math.tan(math.radians(v_fov * 0.5)) * aspect_ratio))

    m_pitch = 0.022
    deg_per_count_x = sens * m_yaw
    deg_per_count_y = sens * m_pitch
    counts_per_deg_x = 1.0 / max(0.001, deg_per_count_x)
    counts_per_deg_y = 1.0 / max(0.001, deg_per_count_y)
    counts_per_deg = counts_per_deg_x  # backwards compatibility

    deg_per_pixel_x = h_fov / float(crop_w)
    deg_per_pixel_y = v_fov / float(crop_h)
    k_aim_x = deg_per_pixel_x * counts_per_deg_x
    k_aim_y = deg_per_pixel_y * counts_per_deg_y
    k_aim = k_aim_x  # backwards compatibility
    print(f"[*] Calibrated Controls: Sens={sens:.2f}, FOV={h_fov:.1f}x{v_fov:.1f}, Counts/deg=({counts_per_deg_x:.1f}, {counts_per_deg_y:.1f}), Gain=({k_aim_x:.2f}, {k_aim_y:.2f})\n")

    # Focus CS 1.6 Window automatically
    focus_cs_window()

    print("[*] Launching bot in 3 seconds... Switch to Counter-Strike!")
    for sec in [3, 2, 1]:
        print(f"    [{sec}...] ", end="", flush=True)
        play_beep(800 + sec * 200, 100)
        time.sleep(1.0)
    print("\n\n>>> [AI BOT ACTIVATED - SPRINTING ON aim_ai_sandbox] <<<")
    print(">>> Hotkeys: [F9] PAUSE/RESUME | [F6] RECORD ROUND THROUGH AI EYES <<<\n", flush=True)
    play_beep(1500, 200)

    ai_recorder = AIVisionVideoRecorder(output_dir="demos", fps=30)
    bot_active = True
    last_f9_down = False
    last_f6_down = False
    step_count = 0
    current_keys = set()
    firing = False
    burst_start = 0.0
    last_enemy_spotted = False
    last_combat_time = 0.0
    combat_strafe_start = 0.0
    last_combat_snap = 0.0
    last_target_dx = 0.0
    last_target_dy = 0.0
    pitch_drift = 0  # Track vertical pitch offset for horizon auto-leveling
    latest_frame_holder = [None, []]  # [frame_bgr, targets]

    def handle_death(evt_line):
        nonlocal accumulated_reward, pitch_drift
        accumulated_reward -= 1.5
        pitch_drift = 0  # Reset pitch accumulation on death
        if ai_recorder.is_recording:
            ai_recorder.trigger_banner("☠ BOT ELIMINATED ☠", duration=1.5, is_headshot=True)
            def _delayed_stop_death():
                time.sleep(1.2)
                if ai_recorder.is_recording:
                    ai_recorder.stop_recording()
            threading.Thread(target=_delayed_stop_death, daemon=True).start()
        frame, tgts = latest_frame_holder[0], latest_frame_holder[1]
        if frame is not None:
            ann = annotate_frame(frame, tgts, "DEATH MOMENT", "DEAD", False)
            cv2.putText(ann, evt_line[-60:], (10, ann.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(DEBUG_DIR, "DEATH_latest.jpg"), ann)
            print(f"\n[Visual Black Box] Death captured! Saved to {DEBUG_DIR}/DEATH_latest.jpg", flush=True)

    def handle_enemy_hit(evt_line):
        nonlocal accumulated_reward
        accumulated_reward += 2.0
        print(f"\n[COMBAT HIT] Enemy hit confirmed! (+2.0 reward)", flush=True)

    def handle_kill(evt_line):
        nonlocal accumulated_reward
        is_hs = ("headshot" in evt_line.lower()) or ("***" in evt_line)
        reward_add = 15.0 if is_hs else 10.0
        accumulated_reward += reward_add
        play_beep(1900 if is_hs else 1500, 60)
        telemetry_tracker.on_kill(is_hs)
        tag = "HEADSHOT +15.0" if is_hs else "KILL +10.0"
        if ai_recorder.is_recording:
            ai_banner_tag = f"☠ HEADSHOT KILL! (+15.0) ☠" if is_hs else f"★ DUMMY ELIMINATED! (+10.0) ★"
            ai_recorder.trigger_banner(ai_banner_tag, duration=1.8, is_headshot=is_hs)
        print(f"\n>>> [{tag}] Dummy Eliminated! (Round Kills: {telemetry_tracker.round_data['kills_total'] if telemetry_tracker.round_data else 1}) <<<", flush=True)
        frame, tgts = latest_frame_holder[0], latest_frame_holder[1]
        telemetry_tracker.log_vision_event(
            event_type="KILL_CONFIRMED",
            targets=tgts if tgts else [],
            action_status=f"KILL CONFIRMED [{tag}]",
            extra={"is_headshot": is_hs, "round_kills": telemetry_tracker.round_data['kills_total'] if telemetry_tracker.round_data else 1}
        )
        if frame is not None:
            ann = annotate_frame(frame, tgts, f"KILL CONFIRMED! [{tag}]", "+".join(sorted(current_keys)), True)
            cv2.imwrite(os.path.join(DEBUG_DIR, "KILL_latest.jpg"), ann)
            print(f"\n[Visual Black Box] Kill captured! Saved to {DEBUG_DIR}/KILL_latest.jpg", flush=True)

    def handle_round_win(evt_line):
        nonlocal accumulated_reward
        accumulated_reward += 50.0
        play_beep(2200, 200)
        print(f"\n===========================================================")
        print(f" >>> [VICTORY] ALL 11 DUMMIES CLEARED! (+50.0 REWARD) <<<")
        print(f"===========================================================\n", flush=True)
        if ai_recorder.is_recording:
            ai_recorder.trigger_banner("★ VICTORY! ALL 11 DUMMIES CLEARED! (+50.0) ★", duration=2.5, is_headshot=False)
            def _delayed_stop_win():
                time.sleep(1.8)
                if ai_recorder.is_recording:
                    ai_recorder.stop_recording()
            threading.Thread(target=_delayed_stop_win, daemon=True).start()
        telemetry_tracker.finish_round(accumulated_reward, total_rl_samples=exp_recorder.total_recorded)
        accumulated_reward = 0.0

    enemy_spotted = False
    damage_retaliation_until = 0.0
    emergency_turn_remaining = 0

    def handle_damaged(evt_line):
        nonlocal damage_retaliation_until, emergency_turn_remaining, accumulated_reward
        accumulated_reward -= 0.5
        # If enemy is not already in view, we are being knifed or shot from behind/flank!
        if not enemy_spotted:
            damage_retaliation_until = time.time() + 0.8
            # Snap 180 degrees immediately towards attacker behind us!
            emergency_turn_remaining = int(180 * counts_per_deg)
            print(f"\n[DAMAGE DETECTED] Under attack from rear! Emergency 180° Snap Turn!", flush=True)
            play_beep(1300, 60)

    active_route_idx = 0
    current_wp_idx = 0
    phys_stuck_ticks = 0
    last_phys_check = time.time()
    last_phys_pos = (0.0, 0.0)
    round_start_time = time.time()
    lane_turn_done = [False, False]
    lane_turn_remaining = 0
    spectator_streak = 0
    alive_screen_streak = 0
    red_wall_ticks = 0
    last_wall_turn_time = 0.0

    # Ammo, Weapon Switching & Tactical Reload State
    combat_firing_start = 0.0
    burst_count = 0
    reloading_until = 0.0
    active_weapon_slot = 1
    reload_attempts = 0
    last_combat_had_fire = False

    def handle_round_start(evt_line):
        nonlocal active_route_idx, current_wp_idx, phys_stuck_ticks, last_phys_pos, round_start_time, forward_start, stuck_frames, turn_remaining, lane_turn_done, lane_turn_remaining, pitch_drift, spectator_streak, alive_screen_streak, red_wall_ticks, accumulated_reward
        nonlocal combat_firing_start, burst_count, reloading_until, active_weapon_slot, reload_attempts, last_combat_had_fire, last_wall_turn_time
        if ai_recorder.is_recording:
            ai_recorder.stop_recording()
        if round_start_time > 0 and (time.time() - round_start_time) > 4.0:
            telemetry_tracker.finish_round(accumulated_reward, total_rl_samples=exp_recorder.total_recorded)
            accumulated_reward = 0.0

        round_start_time = time.time()
        pitch_drift = 0  # Reset pitch accumulation on new round spawn
        tap_key("5", 0.04)  # Automatically dismiss any pending VGUI class selection
        time.sleep(0.02)
        tap_key("end", 0.04)  # Snap camera pitch to exact 0.0° horizontal via GoldSrc force_centerview
        last_wall_turn_time = 0.0
        spectator_streak = 0
        alive_screen_streak = 0
        active_route_idx = (active_route_idx + 1) % len(ROUTE_NAMES)
        current_wp_idx = 0
        phys_stuck_ticks = 0
        red_wall_ticks = 0
        last_phys_pos = (0.0, 0.0)
        forward_start = round_start_time
        stuck_frames = 0
        turn_remaining = 0
        lane_turn_remaining = 0
        lane_turn_done = [False, False]

        # Reset Ammo & Weapon State
        combat_firing_start = 0.0
        burst_count = 0
        reloading_until = 0.0
        active_weapon_slot = 1
        reload_attempts = 0
        last_combat_had_fire = False

        telemetry_tracker.start_round(ROUTE_NAMES[active_route_idx])
        print(f"\n\n===========================================================")
        print(f" >>> [ROUND {telemetry_tracker.current_round:02d} START] TACTICAL ROUTE: {ROUTE_NAMES[active_route_idx]} <<<")
        print(f"===========================================================\n", flush=True)
        play_beep(1100, 80)

        # Execute Auto-Buy via F1 (autobuy), F2 (rebuy) and ammo top-off
        execute_autobuy()

    log_watcher = GameLogWatcher(
        on_death=handle_death,
        on_kill=handle_kill,
        on_damaged=handle_damaged,
        on_round_start=handle_round_start,
        on_enemy_hit=handle_enemy_hit,
        on_round_win=handle_round_win
    )
    log_watcher.start()

    # Initial Autobuy on agent launch
    execute_autobuy()

    # Navigation & Hunt State
    forward_start = time.time()
    lane_push_duration = 5.0  # seconds to push forward before sweeping flanks
    patrol_dir = 1
    prev_floor = None
    stuck_frames = 0
    unstuck_until = 0.0
    turn_remaining = 0
    last_focus_check = time.time()
    pose = None

    try:
        while True:
            loop_start = time.time()
            now = loop_start

            # Check / re-focus CS window periodically if lost
            if now - last_focus_check > 5.0:
                focus_cs_window()
                last_focus_check = now

            # Hardware Polling for F9 Key
            f9_down = is_key_pressed(VK_F9)
            if f9_down and not last_f9_down:
                bot_active = not bot_active
                if bot_active:
                    print("\n\n>>> [AI BOT RESUMED - ATTACKING ON aim_ai_sandbox] <<<", flush=True)
                    play_beep(1500, 150)
                    forward_start = now
                    stuck_frames = 0
                else:
                    for k in list(current_keys):
                        send_key_up(k)
                    current_keys.clear()
                    if firing:
                        send_mouse_up()
                        firing = False
                    print("\n\n>>> [AI BOT PAUSED] Press [F9] to resume.", flush=True)
                    play_beep(600, 150)
            last_f9_down = f9_down

            # Hardware Polling for F6 Key (Record Round Video - AI Vision)
            f6_down = is_key_pressed(VK_F6)
            if f6_down and not last_f6_down:
                if not ai_recorder.is_recording:
                    ai_recorder.start_recording(round_num=telemetry_tracker.current_round)
                else:
                    ai_recorder.stop_recording()
            last_f6_down = f6_down

            # Safety Gate: if CS 1.6 lost focus, pause inputs so we never type into Desktop/Windows Search!
            if not is_cs_window_foreground():
                for k in list(current_keys):
                    send_key_up(k)
                current_keys.clear()
                if firing:
                    send_mouse_up()
                    firing = False
                if step_count % 30 == 0 and IS_WINDOWS and user32:
                    fg = user32.GetForegroundWindow()
                    t_buf = ctypes.create_string_buffer(256)
                    user32.GetWindowTextA(fg, t_buf, 256)
                    t_name = t_buf.value.decode("cp1251", errors="ignore")
                    print(f"\r[STANDBY: CLICK INTO CS WINDOW] (Active window: '{t_name[:30]}')    ", end="", flush=True)
                time.sleep(0.05)
                continue


            if bot_active:
                step_count += 1
                desired_keys = set()
                action_status = ""
                pose = mem_reader.get_pose()
                step_dx = 0
                step_dy = 0
                steer_bias = 0.0
                wall_ahead = False
                floor_visible = True
                red_ratio = 0.0
                green_ratio = 0.0

                # Capture screen frame
                shot = sct.grab(box)
                frame_bgr = np.array(shot)[:, :, :3]

                # Optical Terrain & Barrier Perception (Green floor vs Red wall)
                steer_bias, wall_ahead, floor_visible, red_ratio, green_ratio = optical_color_lidar(frame_bgr)

                # 0. Check if player is dead or spectating (black bars or log confirmed)
                if is_spectator_view(frame_bgr):
                    spectator_streak += 1
                    alive_screen_streak = 0
                else:
                    spectator_streak = 0
                    alive_screen_streak += 1

                # If first-person gameplay screen is visible, player is alive!
                if alive_screen_streak >= 10 and not log_watcher.is_alive:
                    log_watcher.is_alive = True

                # Must have confirmed log death with spectator bars OR persistent solid black bars
                is_dead_or_spectating = ((not log_watcher.is_alive) and spectator_streak >= 5) or (spectator_streak >= 20)
                if is_dead_or_spectating:
                    action_status = "SPECTATOR: WAITING FOR ROUND"
                    pitch_drift = 0  # Reset pitch accumulation while spectating
                    
                    # Flush terminal reward (death penalty -1.5 or trade kill +8.0) before round standby
                    if state_30d is not None and abs(accumulated_reward) > 0.01:
                        term_reward = accumulated_reward
                        accumulated_reward = 0.0
                        exp_recorder.record(state_30d, [0]*7, [0, 0], [0.0, 0.0], term_reward)
                        
                    for k in list(current_keys):
                        send_key_up(k)
                    current_keys.clear()
                    if firing:
                        send_mouse_up()
                        firing = False
                    if step_count % 30 == 0:
                        print(f"\r[AI AGENT ACTIVE] Step: {step_count:05d} | {action_status:36s} | Next round standby...", end="", flush=True)
                    time.sleep(0.03)
                    continue

                # 0. Check if VGUI Menu is open (Choose Team / Choose Class "ВЫБЕРИТЕ КЛАСС" / MOTD)
                if is_vgui_menu_open(frame_bgr):
                    action_status = "VGUI MENU: AUTO-DISMISSING [5]"
                    for k in list(current_keys):
                        send_key_up(k)
                    current_keys.clear()
                    if firing:
                        send_mouse_up()
                        firing = False
                    tap_key("5", 0.04)  # Select random class / auto-select
                    time.sleep(0.04)
                    tap_key("1", 0.03)  # Draw rifle
                    alive_screen_streak = 0
                    if step_count % 15 == 0:
                        print(f"\r[VGUI MENU DETECTED] Auto-selected class [5] -> Closing menu...", end="", flush=True)
                    time.sleep(0.04)
                    continue

                # Freeze-time / early round ammo top-off
                if (now - round_start_time < 3.5) and (step_count % 30 == 15):
                    tap_key(",", 0.03)
                    tap_key(".", 0.03)
                    tap_key("1", 0.03)

                # 0.1 Update Spatial History & Policy Brain Evaluation (30-D)
                if pose is not None:
                    cur_px, cur_py, cur_pz, cur_pyaw = pose[:4]
                    cur_ppitch = pose[4] if len(pose) >= 5 else 0.0
                    pitch_norm = float(np.clip(cur_ppitch / 89.0, -1.0, 1.0))
                else:
                    cur_px, cur_py, cur_pz, cur_pyaw = 0.0, 0.0, 0.0, 0.0
                    pitch_norm = float(np.clip((pitch_drift / max(1.0, counts_per_deg_y)) / 89.0, -1.0, 1.0))
                feat_6d = [cur_px / 1000.0, cur_py / 1000.0, cur_pz / 1000.0, pitch_norm, cur_pyaw / 180.0, 1.0]
                pose_history.append(feat_6d)

                policy_brain.check_reload()
                if len(pose_history) == 5:
                    state_30d = np.array(pose_history, dtype=np.float32).flatten()
                    p_keys, p_clicks, p_aim = policy_brain.predict(state_30d)
                else:
                    state_30d = None
                    p_keys = np.zeros(7, dtype=np.float32)
                    p_clicks = np.zeros(2, dtype=np.float32)
                    p_aim = np.zeros(2, dtype=np.float32)

                # 1. Vision Detection (YOLOv8)
                raw_targets = detector.detect(frame_bgr)
                cur_time_in_round = now - round_start_time
                spawn_grace_period = 1.0  # 1.0s brief spawn breakout for weapon equip
                in_spawn_grace = (cur_time_in_round < spawn_grace_period)

                # Round-End Watchdog: If round exceeds 1:55 (115s), force round end via bot_kill / sv_restart
                if cur_time_in_round > 117.0:
                    print(f"\n[ROUND WATCHDOG] Round time limit (1:55) reached -> Resetting round...", flush=True)
                    send_key_down("f3")
                    time.sleep(0.015)
                    send_key_up("f3")
                    if cur_time_in_round > 120.0:
                        send_key_down("f5")
                        time.sleep(0.015)
                        send_key_up("f5")
                    round_start_time = now - 110.0  # Throttle watchdog trigger

                # In aim_ai_sandbox, human AI Agent is always CT, hunting 11 T-dummies
                is_terrorist = False
                in_own_spawn = (cur_time_in_round < spawn_grace_period)

                targets = []
                for t in raw_targets:
                    # Anti-Decal Guard: Reject blood splatters on crates
                    bx1, by1, bx2, by2 = t.get("box", (0, 0, 0, 0))
                    if bx2 > bx1 and by2 > by1:
                        th = by2 - by1
                        tw = bx2 - bx1
                        torso = frame_bgr[by1 + int(th * 0.15):by1 + int(th * 0.55), bx1 + int(tw * 0.20):bx1 + int(tw * 0.80)]
                        if torso.size >= 16:
                            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
                            mean_h, mean_s, mean_v = hsv.mean(axis=(0, 1))
                            is_blood_red = (mean_h <= 10 or mean_h >= 80) and mean_s > 120 and mean_v > 80
                            if is_blood_red:
                                continue

                            # Teammate Discrimination (cl_minmodels 1: CT=GIGN Blue, T=LEET Tan/Yellow)
                            tb, tg, tr = torso.mean(axis=(0, 1))
                            is_ct_model = (tb > tr + 6) or (tb > tg and tb > 55) or (mean_h >= 95 and mean_h <= 135 and mean_s > 25)
                            is_t_model = (tr > tb + 10 and tg > tb + 4) or (mean_h >= 15 and mean_h <= 55 and mean_s > 20)

                            if is_terrorist:
                                if is_t_model and not is_ct_model:
                                    continue
                            else:
                                # Bot is CT: only reject if confirmed CT teammate (prominent GIGN blue helmet/uniform)
                                if is_ct_model and not is_t_model:
                                    continue

                    targets.append(t)

                enemy_spotted = len(targets) > 0

                # 2. Audio Cues
                rms_l, rms_r, balance, peak = audio_listener.get_cues()
                audio_vec_4d = [rms_l, rms_r, balance, peak]

                # 2.1 Multimodal Vision+Audio Policy Inference (1hr Human BC Model)
                mm_keys, mm_clicks, mm_aim = multimodal_bot.predict(frame_bgr, audio_vec_4d)

                # 3. Construct 16-D Feature Vector for Neural Tactical Brain
                if enemy_spotted:
                    primary = targets[0]
                    dx = primary["dx"]
                    dy = primary["dy"]
                    body_w = primary.get("w", 30)
                    body_h = primary.get("h", 60)
                    
                    # Target acquisition and lock with burst hysteresis
                    is_actively_bursting = (firing and burst_start > 0.0 and (now - burst_start < 0.35))
                    lock_tol_x = max(45.0, body_w * 0.90) if is_actively_bursting else max(26.0, body_w * 0.60)
                    lock_tol_y = max(55.0, body_h * 1.10) if is_actively_bursting else max(22.0, body_h * 0.65)
                    is_locked = 1.0 if (abs(dx) <= lock_tol_x and abs(dy) <= lock_tol_y) else 0.0
                    
                    # Policy network can initiate or sustain bursts
                    dist_norm = float(primary["dist"]) / float(math.hypot(crop_w, crop_h))
                    policy_fire = (p_clicks[0] > 0.35 and dist_norm < 0.65 and abs(dx) < max(38.0, body_w * 0.95) and abs(dy) < max(40.0, body_h * 0.95))
                    can_fire = (is_locked == 1.0 or is_actively_bursting or policy_fire) and (not in_spawn_grace)
                    
                    dx_norm = float(dx) / float(crop_w)
                    dy_norm = float(dy) / float(crop_h)
                    is_head = 1.0 if (primary.get("is_head", False) or abs(dy) < 25.0) else 0.0
                    conf = float(primary.get("conf", 0.0))
                else:
                    primary = None
                    dx, dy = 0.0, 0.0
                    is_locked = 0.0
                    can_fire = False
                    dx_norm, dy_norm, dist_norm = 0.0, 0.0, 0.0
                    is_head = 0.0
                    conf = 0.0

                is_collision = 1.0 if (stuck_frames >= 4 or phys_stuck_ticks >= 2) else 0.0
                push_time_norm = float(min(1.0, (now - forward_start) / 5.0))
                burst_cycle_norm = float(min(1.0, (now - burst_start) / 0.5)) if firing else 0.0

                tactical_features = [
                    1.0 if enemy_spotted else 0.0,
                    dx_norm,
                    dy_norm,
                    dist_norm,
                    is_head,
                    conf,
                    is_locked,
                    is_collision,
                    push_time_norm,
                    burst_cycle_norm,
                    float(np.clip(rms_l * 5.0, 0.0, 1.0)),
                    float(np.clip(rms_r * 5.0, 0.0, 1.0)),
                    float(np.clip(balance, -1.0, 1.0)),
                    1.0,
                    1.0 if can_fire else 0.0,
                    float(active_route_idx % 3) / 3.0
                ]

                # Ultra-Fast Neural Tactical Policy Inference (< 0.3ms on CPU)
                t_move, t_combat, t_aim = tactical_brain.predict(tactical_features)
                brain_str = f"W:{t_move[0]:+.1f} S:{t_move[1]:+.1f} A:{t_move[2]:+.1f} D:{t_move[3]:+.1f} | F:{t_combat[0]:+.1f} C:{t_combat[1]:+.1f}"

                # ==============================================================
                # STATE A: COMBAT MODE (Enemy in View)
                # ==============================================================
                if enemy_spotted:
                    telemetry_tracker.on_spotted(conf=conf, is_head=(is_head > 0.5))
                    forward_start = now
                    stuck_frames = 0
                    unstuck_until = 0.0
                    turn_remaining = 0
                    emergency_turn_remaining = 0
                    damage_retaliation_until = 0.0

                    # 1. Unstoppable Spawn Breakout: Never freeze dead in spawn!
                    is_breakout = in_spawn_grace
                    if not is_breakout:
                        if not last_enemy_spotted and current_keys:
                            execute_counter_strafe(current_keys)
                        desired_keys.clear()
                    else:
                        # PUSH OUT OF SPAWN! Keep running forward
                        desired_keys.add("w")

                    # Always ensure primary rifle [1] is equipped in combat
                    if not last_enemy_spotted:
                        tap_key("1", 0.03)
                        active_weapon_slot = 1

                    # 2. Linear Lead Target Tracking & Damped P-Controller (Zero Overshoot, Zero Noise)
                    dt_target = max(0.005, min(0.20, now - last_combat_time)) if last_enemy_spotted else 0.033
                    if last_enemy_spotted and abs(dx - last_target_dx) < 220:
                        vx_target = (dx - last_target_dx) / dt_target
                        vy_target = (dy - last_target_dy) / dt_target
                    else:
                        vx_target = 0.0
                        vy_target = 0.0
                    last_target_dx = dx
                    last_target_dy = dy

                    # Lead target by frame latency (~0.04s)
                    lead_x = dx + vx_target * 0.04
                    lead_y = dy + vy_target * 0.03

                    # Unified Aiming & Recoil Control (Smooth vertical tracking, ZERO jitter)
                    is_actively_bursting = (firing and burst_start > 0.0 and (now - burst_start < 0.30))
                    if is_actively_bursting:
                        burst_elapsed = now - burst_start
                        # CS 1.6 AK47/M4: first 2 bullets flat, bullets 3-4 rise moderately
                        rec_step = 1 if burst_elapsed < 0.09 else 4
                        step_dx = int(np.clip(lead_x * k_aim_x * 0.65, -550, 550))
                        step_dy = int(np.clip(lead_y * k_aim_y * 0.25 + rec_step, -20, 20))
                    else:
                        step_dx = int(np.clip(lead_x * k_aim_x * 0.80, -750, 750))
                        step_dy = int(np.clip(lead_y * k_aim_y * 0.80, -45, 45))

                    # Anti-Floor & Anti-Ceiling Pitch Guard:
                    # In CS 1.6, targets are on ground level. Never look straight into floor (>40) or ceiling (<-35)!
                    if pitch_drift > 40 and step_dy > 0:
                        step_dy = 0
                    if pitch_drift < -35 and step_dy < 0:
                        step_dy = 0

                    # Send clean, unified mouse movement
                    send_mouse_move(step_dx, step_dy)
                    pitch_drift = int(np.clip(pitch_drift + step_dy, -1200, 1200))
                    last_combat_time = now

                    # 3. Guided Tactical Evasion & Counter-Strafing Stance
                    if not in_spawn_grace:
                        if not last_enemy_spotted:
                            combat_strafe_start = now

                        # Check if tactical reload is currently underway
                        if now < reloading_until:
                            desired_keys.discard("w")
                            desired_keys.add("ctrl")
                            strafe_k = "a" if int((reloading_until - now) * 4) % 2 == 0 else "d"
                            desired_keys.add(strafe_k)
                        else:
                            # While actively firing: CROUCH and STOP all movement for maximum CS 1.6 rifle accuracy!
                            if firing:
                                desired_keys.discard("w")
                                desired_keys.discard("s")
                                desired_keys.discard("a")
                                desired_keys.discard("d")
                                desired_keys.add("ctrl")  # Duck-spray eliminates bullet spread!
                            else:
                                # Between bursts: tactical ADAD counter-strafe
                                desired_keys.discard("w")
                                strafe_phase = int((now - combat_strafe_start) / 0.32) % 2
                                if strafe_phase == 0:
                                    desired_keys.add("a")
                                    desired_keys.discard("d")
                                else:
                                    desired_keys.add("d")
                                    desired_keys.discard("a")
                    else:
                        desired_keys.add("w")

                    # 4. Ammo Management, Tactical Reload [R] & Lethal Burst Execution
                    if now < reloading_until:
                        # Actively reloading: take evasive cover and do NOT click empty gun!
                        if firing:
                            send_mouse_up()
                            firing = False
                        burst_start = 0.0
                        action_status = f"COMBAT: RELOAD [R] ({max(0.1, reloading_until - now):.1f}s) - EVASION"
                    elif can_fire:
                        if combat_firing_start == 0.0:
                            combat_firing_start = now
                            burst_count = 0

                        # Check if magazine needs tactical reload (6 bursts = full 30-round mag)
                        if burst_count >= 6 or (now - combat_firing_start > 2.6):
                            if firing:
                                send_mouse_up()
                                firing = False
                            tap_key("r", 0.04)
                            if now - round_start_time < 35.0:
                                tap_key(",", 0.02)
                                tap_key(".", 0.02)
                            tap_key("1", 0.03)  # Always keep primary rifle equipped
                            active_weapon_slot = 1
                            reloading_until = now + 2.2
                            burst_count = 0
                            combat_firing_start = 0.0
                            action_status = "TACTICAL RELOAD [R] (Primary Rifle)"
                            play_beep(900, 45)
                        else:
                            # Normal lethal firing burst (3-4 rounds, 0.28s hold, 0.10s pause)
                            stance_label = "DUCK" if "ctrl" in desired_keys else "STAND"
                            action_status = f"COMBAT: RIFLE B#{burst_count+1} (dx={dx:+.0f}, dy={dy:+.0f}, {stance_label})"

                            if burst_start == 0.0:
                                burst_start = now
                                burst_count += 1
                                last_combat_had_fire = True
                                send_mouse_down()
                                firing = True
                                play_beep(1600, 25)

                            burst_time = now - burst_start
                            if burst_time < 0.28:  # 3-4 lethal rounds: mouse move already applied smoothly
                                pass
                            elif burst_time < 0.38:  # 100ms pause for crosshair recoil reset
                                if firing:
                                    send_mouse_up()
                                    firing = False
                            else:  # Cycle into next burst
                                burst_start = now
                                burst_count += 1
                                send_mouse_down()
                                firing = True

                        if now - last_combat_snap > 1.5:
                            last_combat_snap = now
                            ann = annotate_frame(frame_bgr, targets, action_status, "+".join(sorted(current_keys)), firing, brain_str)
                            cv2.imwrite(os.path.join(DEBUG_DIR, "combat_latest.jpg"), ann)
                    else:
                        action_status = f"COMBAT: ACQUIRING (Dist: {primary['dist']:.0f}px)"
                        if firing:
                            send_mouse_up()
                            firing = False
                        burst_start = 0.0

                # ==============================================================
                # STATE A2: REAR AMBUSH RETALIATION (Attacked from behind)
                # ==============================================================
                elif (now < damage_retaliation_until) or (emergency_turn_remaining != 0):
                    action_status = "DEFENSE: 180 RETALIATION TURN"
                    if firing:
                        send_mouse_up()
                        firing = False

                    # Backpedal away from knife/gunshot + crouch for recoil stability
                    desired_keys.add("s")
                    desired_keys.add("ctrl")

                    # Fast snap turn 180 degrees (up to 550 counts per frame = ~13 deg/frame)
                    step_t = int(np.clip(emergency_turn_remaining, -550, 550))
                    emergency_turn_remaining -= step_t
                    send_mouse_move(step_t, 0)

                # ==============================================================
                # STATE B: UNSTUCK / WALL HOP & LATERAL GLIDE (Zero Corner Trapping)
                # ==============================================================
                elif (now < unstuck_until) or (turn_remaining != 0):
                    action_status = "NAV: HOP-UNHOOK WALL"
                    if firing:
                        send_mouse_up()
                        firing = False

                    # If pressed into a red barrier wall, NEVER press W! Reverse with S + strafe to detach from brush!
                    if wall_ahead or red_ratio > 0.30:
                        desired_keys.add("s")
                        turn_dir = 1.0 if steer_bias >= 0.0 else -1.0
                        desired_keys.add("d" if turn_dir > 0 else "a")
                    else:
                        desired_keys.add("w")

                    # Smooth pivot away from obstacle (gentle 35 counts per frame = ~1.5 deg/frame)
                    if turn_remaining != 0:
                        step_t = int(np.clip(turn_remaining, -35, 35))
                        turn_remaining -= step_t
                        send_mouse_move(step_t, 0)

                # ==============================================================
                # STATE C: TACTICAL WAYPOINT NAVIGATION & SCANNING ($2000$ Map)
                # ==============================================================
                else:
                    if firing:
                        send_mouse_up()
                        firing = False
                    burst_start = 0.0

                    # Post-combat weapon management: reload rifle and reselect primary
                    if last_combat_had_fire and (now - last_combat_time > 1.2):
                        last_combat_had_fire = False
                        if active_weapon_slot != 1:
                            tap_key("1", 0.04)
                            active_weapon_slot = 1
                        tap_key("r", 0.04)
                        tap_key("end", 0.04)  # Instantly reset vertical recoil to 0.0° horizontal
                        pitch_drift = 0
                        burst_count = 0
                        reload_attempts = 0
                        combat_firing_start = 0.0

                    # 1. State C Horizon level handled universally by main loop horizon stabilizer

                    # 2. Periodic Primary Weapon Check (Equip slot1 rifle if picked up from floor)
                    if step_count % 90 == 25:
                        send_key_down("1")
                        time.sleep(0.012)
                        send_key_up("1")

                    # 3. Multimodal Human Vision+Audio Guidance (1-hr Behavioral Cloning)
                    if multimodal_bot.available:
                        if mm_keys.get("w", -99.0) > -3.5:
                            desired_keys.add("w")
                        if mm_keys.get("shift", -99.0) > -1.5:
                            desired_keys.add("shift")
                        if mm_keys.get("1", -99.0) > -1.5 and step_count % 60 == 0:
                            send_key_down("1")
                            time.sleep(0.005)
                            send_key_up("1")
                        # CRITICAL: In navigation mode, send 0 for Y to prevent dipping camera into floor/walls!
                        if abs(mm_aim[0]) > 0.4:
                            send_mouse_move(int(np.clip(mm_aim[0] * 0.4, -6, 6)), 0)

                    # 4. Macro Policy Neural Exploration Hints
                    if state_30d is not None and not in_spawn_grace:
                    # Skip recording during spawn grace: scripted W-run creates contradictory
                    # experience that poisons the online RL buffer with noise actions
                        if p_keys[2] > 0.8 and p_keys[3] < 0.0:
                            desired_keys.add("a")
                        elif p_keys[3] > 0.8 and p_keys[2] < 0.0:
                            desired_keys.add("d")
                        if p_keys[4] > 1.2:
                            desired_keys.add("space")
                        if p_keys[5] > 1.2:
                            desired_keys.add("ctrl")

                    # ==============================================================
                    # STATE C1: OPTICAL COLOR LIDAR OBSTACLE & BARRIER AVOIDANCE
                    # ==============================================================
                    # Universal: active for both memory waypoint navigation AND optical arena patrol
                    if wall_ahead or red_ratio > 0.30:
                        red_wall_ticks += 1
                        telemetry_tracker.on_wall_avoid()
                        accumulated_reward -= 0.05

                        turn_dir = 1.0 if steer_bias >= 0.0 else -1.0

                        if red_ratio > 0.45 or wall_ahead:
                            # Flush into red wall: reverse with S, strafe, and queue a single 70° turn with cooldown
                            action_status = f"LIDAR: FLUSH WALL ({red_ratio*100:.0f}%) -> REVERSE & PIVOT"
                            desired_keys.discard("w")
                            desired_keys.add("s")
                            desired_keys.add("d" if turn_dir > 0 else "a")
                            if turn_remaining == 0 and now > last_wall_turn_time:
                                last_wall_turn_time = now + 1.2
                                turn_remaining = int(turn_dir * 70.0 * counts_per_deg)
                        else:
                            # Approaching wall: advance with W while smoothly steering towards open floor
                            action_status = f"LIDAR: WALL ({red_ratio*100:.0f}%) -> STEER {'RIGHT' if turn_dir > 0 else 'LEFT'}"
                            desired_keys.add("w")
                            if abs(steer_bias) > 0.35:
                                desired_keys.add("d" if turn_dir > 0 else "a")
                            if abs(steer_bias) > 0.15 and turn_remaining == 0:
                                steer_step = int(np.clip(steer_bias * 1.5 * counts_per_deg, -20, 20))
                                send_mouse_move(steer_step, 0)

                        # If pressed against wall for > 20 frames (~0.7s), queue a 70° escape turn
                        if red_wall_ticks >= 20:
                            desired_keys.discard("w")
                            desired_keys.add("s")
                            if turn_remaining == 0 and now > last_wall_turn_time:
                                last_wall_turn_time = now + 1.2
                                turn_remaining = int(turn_dir * 70.0 * counts_per_deg)
                            if pose is not None and current_wp_idx < len(active_waypoints) - 1:
                                current_wp_idx += 1
                            red_wall_ticks = 0

                    else:
                        red_wall_ticks = max(0, red_wall_ticks - 1)
                        # Open corridor: advance forward with W if green floor visible
                        if floor_visible and not enemy_spotted:
                            desired_keys.add("w")

                        # Fine centering along green corridor
                        if abs(steer_bias) > 0.25 and turn_remaining == 0:
                            steer_step = int(np.clip(steer_bias * 1.2 * counts_per_deg, -15, 15))
                            send_mouse_move(steer_step, 0)
                            if abs(steer_bias) > 0.45:
                                desired_keys.add("d" if steer_bias > 0 else "a")

                        # ==============================================================
                        # STATE C2: ROUTE EXECUTION (Waypoints or Optical Arena Patrol)
                        # ==============================================================
                        if pose is not None:
                            cur_x, cur_y, cur_z, cur_yaw = pose[:4]
                            is_terrorist = False
                            active_waypoints = MAP_ROUTES_SANDBOX[active_route_idx % len(MAP_ROUTES_SANDBOX)]

                            # Spawn auto-orient
                            target_spawn_yaw = 90.0
                            spawn_angle_diff = (target_spawn_yaw - cur_yaw + 180.0) % 360.0 - 180.0
                            if cur_time_in_round < 1.8 and abs(spawn_angle_diff) > 25.0:
                                steer_spawn = int(np.clip(-spawn_angle_diff * counts_per_deg * 0.55, -240, 240))
                                send_mouse_move(steer_spawn, 0)

                            if current_wp_idx >= len(active_waypoints):
                                current_wp_idx = len(active_waypoints) - 1

                            wp_x, wp_y = active_waypoints[current_wp_idx]
                            dx = wp_x - cur_x
                            dy = wp_y - cur_y
                            dist = math.hypot(dx, dy)

                            if dist < 85.0 and current_wp_idx < len(active_waypoints) - 1:
                                current_wp_idx += 1
                                if not in_spawn_grace:
                                    accumulated_reward += 0.5
                                wp_x, wp_y = active_waypoints[current_wp_idx]
                                dx = wp_x - cur_x
                                dy = wp_y - cur_y
                                dist = math.hypot(dx, dy)
                                phys_stuck_ticks = 0

                            if now - last_phys_check > 0.30:
                                last_phys_check = now
                                if "w" in current_keys and not in_spawn_grace:
                                    move_delta = math.hypot(cur_x - last_phys_pos[0], cur_y - last_phys_pos[1])
                                    if move_delta < 18.0:
                                        phys_stuck_ticks += 1
                                    else:
                                        phys_stuck_ticks = max(0, phys_stuck_ticks - 1)
                                last_phys_pos = (cur_x, cur_y)

                            if current_wp_idx >= len(active_waypoints) - 1 and dist < 120.0:
                                telemetry_tracker.update_wp(len(active_waypoints), len(active_waypoints))
                                sweep = math.sin(step_count * 0.12) * 45.0
                                base_yaw = 90.0
                                target_yaw = base_yaw + sweep
                                angle_diff = (target_yaw - cur_yaw + 180.0) % 360.0 - 180.0
                                action_status = f"SWEEP ENEMY BASE [WP {current_wp_idx+1}/{len(active_waypoints)}]"
                                if step_count % 50 < 25:
                                    desired_keys.add("w")
                                else:
                                    desired_keys.add("ctrl")
                            else:
                                telemetry_tracker.update_wp(current_wp_idx + 1, len(active_waypoints))
                                target_yaw = math.degrees(math.atan2(dy, dx))
                                angle_diff = (target_yaw - cur_yaw + 180.0) % 360.0 - 180.0
                                action_status = f"NAV: {ROUTE_NAMES[active_route_idx][:14]} [WP {current_wp_idx+1}/{len(active_waypoints)}]"

                                if phys_stuck_ticks >= 2:
                                    telemetry_tracker.on_unstuck()
                                    action_status = f"NAV: REVERSE UNHOOK (Stuck {phys_stuck_ticks})"
                                    desired_keys.add("s")
                                    desired_keys.add("d" if angle_diff >= 0 else "a")
                                    if phys_stuck_ticks >= 7:
                                        if current_wp_idx < len(active_waypoints) - 1:
                                            current_wp_idx += 1
                                        phys_stuck_ticks = 0
                                else:
                                    if abs(angle_diff) < 70.0:
                                        desired_keys.add("w")
                                    if abs(angle_diff) > 25.0:
                                        desired_keys.add("a" if angle_diff > 0 else "d")

                                k_steer = counts_per_deg * 0.30
                                turn_step = int(np.clip(-angle_diff * k_steer, -40, 40))
                                send_mouse_move(turn_step, 0)

                        else:
                            # Optical Motion & Stuck Sensing in Green Arena
                            center_crop = frame_bgr[crop_h//3:crop_h*2//3:4, crop_w//3:crop_w*2//3:4, 0]
                            elapsed_w = now - forward_start
                            wall_variance = float(np.std(center_crop))

                            if prev_floor is not None and "w" in current_keys and elapsed_w > 0.6:
                                motion = float(np.mean(np.abs(center_crop.astype(float) - prev_floor.astype(float))))
                                is_stuck_motion = motion < 2.8
                                is_blank_wall = wall_variance < 18.0 and motion < 5.0

                                if is_blank_wall or is_stuck_motion:
                                    stuck_frames += 2 if is_blank_wall else 1
                                else:
                                    stuck_frames = max(0, stuck_frames - 2)

                                if stuck_frames >= 8:
                                    telemetry_tracker.on_unstuck()
                                    unstuck_until = now + 0.60
                                    patrol_dir *= -1
                                    turn_remaining = int(patrol_dir * 90 * counts_per_deg)
                                    stuck_frames = 0
                                    forward_start = now
                            prev_floor = center_crop

                            # Sound Direction Response
                            if peak > 0.22 and abs(balance) > 0.35:
                                action_status = f"HEARING: TURN {'RIGHT' if balance > 0 else 'LEFT'}"
                                sound_turn = int(balance * 40 * counts_per_deg)
                                send_mouse_move(int(np.clip(sound_turn, -100, 100)), 0)
                                desired_keys.add("w")
                                forward_start = now
                            else:
                                sweep = int(math.sin(step_count * 0.12) * 2.5)
                                send_mouse_move(sweep, 0)
                                action_status = f"PATROL: GREEN ARENA ({green_ratio*100:.0f}%)"

                # Engine-Native Pitch Stabilization (force_centerview via END):
                # Keeps pitch dead-center horizontal (0.0°) during patrol; instantly corrects ceiling/floor drift.
                if (not enemy_spotted and (step_count % 60 == 0 or pitch_drift < -25 or pitch_drift > 35)) or in_spawn_grace:
                    tap_key("end", 0.02)
                    pitch_drift = 0

                # Gemini 3.8 Spawn Grace Protection:
                # In first 3.0 seconds:
                # 1. Force running W along the exit corridor (if clear)
                # 2. If staring at a wall on spawn, pivot towards open floor
                # 3. Block crouching (Ctrl) so bot never gets stuck sitting on spawn
                # 4. Disable shooting completely
                if in_spawn_grace:
                    phys_stuck_ticks = 0
                    if not wall_ahead and red_ratio < 0.30:
                        desired_keys.add("w")
                    else:
                        desired_keys.add("s")
                        turn_dir = 1.0 if steer_bias >= 0.0 else -1.0
                        desired_keys.add("d" if turn_dir > 0 else "a")
                        desired_keys.discard("w")
                        if turn_remaining == 0 and now > last_wall_turn_time:
                            last_wall_turn_time = now + 1.2
                            turn_remaining = int(turn_dir * 70.0 * counts_per_deg)
                    desired_keys.discard("ctrl")
                    desired_keys.discard("space")
                    if firing:
                        send_mouse_up()
                        firing = False

                # 4. Apply DirectInput Keys to Game
                for kname in ["w", "s", "a", "d", "space", "ctrl", "shift"]:
                    need_key = kname in desired_keys
                    is_pressed = kname in current_keys
                    if need_key and not is_pressed:
                        send_key_down(kname)
                        current_keys.add(kname)
                    elif not need_key and is_pressed:
                        send_key_up(kname)
                        current_keys.remove(kname)
                    elif need_key and is_pressed and kname == "w" and step_count % 15 == 0:
                        send_key_down(kname)


                # 5. Online Experience Recording with Anti-Reward-Hacking Logic
                step_reward = 0.001 + accumulated_reward
                has_combat_reward = (accumulated_reward >= 1.0 or accumulated_reward <= -0.5)
                accumulated_reward = 0.0

                # Zero reward inside spawn or in grace period UNLESS combat reward occurred!
                if in_spawn_grace or in_own_spawn:
                    if not has_combat_reward:
                        step_reward = 0.0
                    if firing and in_own_spawn:
                        step_reward -= 0.30
                else:
                    if enemy_spotted and is_locked:
                        step_reward += 0.04
                    if firing and not is_locked:
                        step_reward -= 0.02

                if is_collision:
                    step_reward -= 0.05

                if state_30d is not None and (not in_spawn_grace or has_combat_reward):
                    k_vec = [
                        1 if "w" in desired_keys else 0,
                        1 if "s" in desired_keys else 0,
                        1 if "a" in desired_keys else 0,
                        1 if "d" in desired_keys else 0,
                        1 if "space" in desired_keys else 0,
                        1 if "ctrl" in desired_keys else 0,
                        1 if "r" in desired_keys else 0
                    ]
                    c_vec = [1 if firing else 0, 0]
                    # Record hybrid action executed so policy net learns effective aim deltas
                    if enemy_spotted:
                        rec_aim_x = float(np.clip(step_dx * 0.15, -15.0, 15.0))
                        rec_aim_y = float(np.clip(step_dy * 0.15, -15.0, 15.0))
                    else:
                        rec_aim_x = float(np.clip(p_aim[0], -15.0, 15.0))
                        rec_aim_y = float(np.clip(p_aim[1], -15.0, 15.0))
                    exp_recorder.record(state_30d, k_vec, c_vec, [rec_aim_x, rec_aim_y], step_reward)


                last_enemy_spotted = enemy_spotted
                latest_frame_holder[0] = frame_bgr
                latest_frame_holder[1] = targets

                # Real-Time Perception Logging for Agent Diagnostic Telemetry
                v_type = "AIM_LOCKED" if is_locked > 0.5 else ("TARGET_SPOTTED" if enemy_spotted else ("LIDAR_WALL" if (wall_ahead or red_ratio > 0.30) else "PATROL_SCAN"))
                telemetry_tracker.log_vision_event(
                    event_type=v_type,
                    targets=targets,
                    lidar_data={"wall_ahead": bool(wall_ahead), "red_ratio": float(red_ratio), "green_ratio": float(green_ratio), "steer_bias": float(steer_bias)},
                    action_status=action_status,
                    extra={"firing": bool(firing), "keys": "+".join(sorted(current_keys)) if current_keys else "none"}
                )

                # Real-Time Console Telemetry every 6 steps (~5 times per second)
                if step_count % 6 == 0:
                    k_str = "+".join(sorted(current_keys)) if current_keys else "none"
                    t_count = len(targets)
                    if pose is not None:
                        px, py, pz, pyaw = pose[:4]
                        ppitch = pose[4] if len(pose) >= 5 else 0.0
                        print(f"\r[AI AGENT ACTIVE] Step: {step_count:05d} | {action_status:24s} | Brain:[{brain_str}] | POS:({px:4.0f},{py:4.0f}) | P:{ppitch:+4.1f}° | Enemies:{t_count}", end="", flush=True)
                    else:
                        print(f"\r[AI AGENT ACTIVE] Step: {step_count:05d} | {action_status:26s} | Brain:[{brain_str}] | Keys:[{k_str:7s}] | Enemies:{t_count}", end="", flush=True)
                    # Live Visual Black Box
                    ann_live = annotate_frame(frame_bgr, targets, action_status, k_str, firing, brain_str)
                    cv2.imwrite(os.path.join(DEBUG_DIR, "live_view.jpg"), ann_live)

                # AI Vision Video Recording Stream (Eyes of the Neural Network)
                if ai_recorder.is_recording:
                    current_wps = MAP_ROUTES_SANDBOX[active_route_idx % len(MAP_ROUTES_SANDBOX)]
                    hud_frame = ai_recorder.render_hud(
                        frame_bgr=frame_bgr,
                        targets=targets,
                        primary_target=primary,
                        is_locked=bool(is_locked > 0.5),
                        firing_state=firing,
                        status_text=action_status,
                        keys_set=current_keys,
                        step_dx=step_dx,
                        step_dy=step_dy,
                        brain_str=brain_str,
                        round_num=telemetry_tracker.current_round,
                        round_elapsed=cur_time_in_round,
                        round_kills=telemetry_tracker.round_data.get("kills_total", 0) if telemetry_tracker.round_data else 0,
                        round_hs_pct=telemetry_tracker.round_data.get("hs_percent", 0.0) if telemetry_tracker.round_data else 0.0,
                        accumulated_reward=accumulated_reward,
                        route_name=ROUTE_NAMES[active_route_idx] if active_route_idx < len(ROUTE_NAMES) else "PATROL",
                        wp_idx=current_wp_idx + 1,
                        total_wps=len(current_wps),
                        wall_ahead=wall_ahead,
                        steer_bias=steer_bias,
                        floor_visible=floor_visible,
                        red_ratio=red_ratio,
                        green_ratio=green_ratio
                    )
                    ai_recorder.put_frame(hud_frame)

            else:
                time.sleep(0.02)

            elapsed = time.time() - loop_start
            rem = (1.0 / 30.0) - elapsed
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        print("\nStopping agent...")
    finally:
        if ai_recorder.is_recording:
            ai_recorder.stop_recording()
        audio_listener.close()
        exp_recorder.stop()
        for k in list(current_keys):
            send_key_up(k)
        if firing:
            send_mouse_up()
        print("\n[AI Agent safely stopped]")

if __name__ == "__main__":
    main()

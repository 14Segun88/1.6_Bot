"""
===============================================================================
Counter-Strike 1.6 - Minimalist Autonomous Hunter-Killer Agent (Option 3)
===============================================================================
A rock-solid, zero-bloat deterministic AI baseline for CS 1.6 (aim_ai_sandbox):
1. Spawn & Menu: Instant class selection (5/1/Enter), zero spawn lock.
2. Perception: YOLOv8 CrowdHuman + Yellow Crash-Test Dummy Bullseye Head Detector.
3. Navigation: Direct 3D memory waypoint navigation across arena corridors.
4. Combat: Counter-strafe dead-stop, smooth head tracking, and calibrated burst fire.
5. Telemetry: Round tracker, kill counter, and F9 pause/resume toggle.
===============================================================================
"""

import os
import sys
import time
import math
import ctypes
from ctypes import wintypes
import struct
import threading
import numpy as np
import cv2
import mss

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# =============================================================================
# 1. WIN32 INPUT & DIRECTINPUT SCAN CODES
# =============================================================================
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        psapi = ctypes.windll.psapi
    except Exception:
        psapi = None
else:
    user32 = None
    kernel32 = None
    psapi = None

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
VK_F9 = 0x78

# DirectInput Hardware Scan Codes (Recognized by GoldSrc engine)
SCAN_CODES = {
    "w": 0x11, "s": 0x1F, "a": 0x1E, "d": 0x20,
    "space": 0x39, "ctrl": 0x1D, "shift": 0x2A,
    "1": 0x02, "2": 0x03, "3": 0x04, "4": 0x05, "5": 0x06,
    "enter": 0x1C, "escape": 0x01, "r": 0x13,
    "f1": 0x3B, "f2": 0x3C, "end": 0xCF
}

# Virtual Key Codes
VK_MAP = {
    "w": 0x57, "s": 0x53, "a": 0x41, "d": 0x44,
    "space": 0x20, "ctrl": 0x11, "shift": 0x10,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
    "enter": 0x0D, "escape": 0x1B, "r": 0x52,
    "f1": 0x70, "f2": 0x71, "end": 0x23
}

if IS_WINDOWS:
    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_void_p)
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_void_p)
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("ii", INPUT_UNION)]

def send_key_down(name):
    if not (IS_WINDOWS and user32): return
    sc = SCAN_CODES.get(name)
    vk = VK_MAP.get(name, 0)
    if sc is None: return
    try:
        inp = INPUT()
        inp.type = 1
        inp.ii.ki.wVk = 0
        inp.ii.ki.wScan = sc
        inp.ii.ki.dwFlags = KEYEVENTF_SCANCODE
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    except Exception:
        pass
    user32.keybd_event(vk, sc, 0, 0)

def send_key_up(name):
    if not (IS_WINDOWS and user32): return
    sc = SCAN_CODES.get(name)
    vk = VK_MAP.get(name, 0)
    if sc is None: return
    try:
        inp = INPUT()
        inp.type = 1
        inp.ii.ki.wVk = 0
        inp.ii.ki.wScan = sc
        inp.ii.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    except Exception:
        pass
    user32.keybd_event(vk, sc, KEYEVENTF_KEYUP, 0)

def tap_key(name, hold_sec=0.06):
    send_key_down(name)
    time.sleep(hold_sec)
    send_key_up(name)

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

# =============================================================================
# 2. WINDOW FOCUS & GEOMETRY
# =============================================================================
def focus_cs_window():
    if not (IS_WINDOWS and user32): return None
    hwnds = []
    for cls in [b"SDL_app", b"Valve001"]:
        h = user32.FindWindowA(cls, None)
        if h and user32.IsWindowVisible(h):
            hwnds.append(h)

    def enum_cb(h, lparam):
        if user32.IsWindowVisible(h):
            buf = ctypes.create_string_buffer(256)
            user32.GetWindowTextA(h, buf, 256)
            if b"counter-strike" in buf.value.lower():
                hwnds.append(h)
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

    for hwnd in hwnds:
        if user32.GetForegroundWindow() == hwnd:
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
    if not (IS_WINDOWS and user32): return True
    cur_fg = user32.GetForegroundWindow()
    if not cur_fg: return False
    buf = ctypes.create_string_buffer(256)
    user32.GetWindowTextA(cur_fg, buf, 256)
    val = buf.value.lower()
    if any(k in val for k in [b"cmd.exe", b"powershell", b"terminal", b"python", b"code", b"antigravity", b"taskmgr"]):
        return False
    if any(k in val for k in [b"counter-strike", b"counter strike", b"cstrike"]):
        return True
    cls_buf = ctypes.create_string_buffer(256)
    user32.GetClassNameA(cur_fg, cls_buf, 256)
    return cls_buf.value.lower() in [b"sdl_app", b"valve001"]

def find_cs_window_rect(default_box):
    if IS_WINDOWS and user32:
        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        for cls in [b"SDL_app", b"Valve001"]:
            h = user32.FindWindowA(cls, None)
            if h and user32.IsWindowVisible(h):
                rect = RECT()
                user32.GetClientRect(h, ctypes.byref(rect))
                pt = POINT(0, 0)
                user32.ClientToScreen(h, ctypes.byref(pt))
                if rect.right > 300 and rect.bottom > 200:
                    return {"left": pt.x, "top": pt.y, "width": rect.right, "height": rect.bottom}
    return default_box

# =============================================================================
# 3. CLASS & TEAM MENU AUTO-DISMISSAL
# =============================================================================
def is_menu_open(frame_bgr):
    """Detects text menu ('Выберите класс') or VGUI menu dialog"""
    if frame_bgr is None or frame_bgr.shape[0] < 200:
        return False
    h, w = frame_bgr.shape[:2]
    # Text Menu Detection: Yellow Title ('Выберите класс') on Left 1-25%
    menu_crop = frame_bgr[int(h * 0.30):int(h * 0.75), int(w * 0.01):int(w * 0.25)]
    hsv = cv2.cvtColor(menu_crop, cv2.COLOR_BGR2HSV)
    yellow_text = cv2.inRange(hsv, np.array([18, 100, 100]), np.array([38, 255, 255]))
    white_text = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    if np.sum(yellow_text > 0) > 30 and np.sum(white_text > 0) > 500:
        return True

    # VGUI Menu Detection: Header yellow icon + dark background panel
    hdr = frame_bgr[int(h * 0.04):int(h * 0.14), int(w * 0.08):int(w * 0.35)]
    hsv_hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2HSV)
    yellow_hdr = cv2.inRange(hsv_hdr, np.array([20, 140, 140]), np.array([35, 255, 255]))
    if float(np.mean(yellow_hdr > 0)) > 0.035:
        panel = frame_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.12):int(w * 0.88)]
        if float(np.mean(panel < 35)) > 0.45:
            return True
    return False

def dismiss_menu():
    """Selects Class 5 (Random) or Class 1 (Seal Team 6) and enters game"""
    tap_key("5", 0.08)
    time.sleep(0.04)
    tap_key("1", 0.08)
    time.sleep(0.04)
    tap_key("enter", 0.08)

# =============================================================================
# 4. MEMORY POSE READER (GoldSrc hw.dll)
# =============================================================================
class MemoryReader:
    def __init__(self):
        self.hProc = None
        self.ptr_addr = None
        self.hw_base = None
        self.candidate_offsets = [0x7A834, 0x1222E8, 0x113CDC, 0x113CD0, 0x10B740, 0x150F40, 0x79834, 0x7B834]
        self.last_connect = 0.0
        self.frozen_yaw_ticks = 0
        self.last_yaw = None

    def _read_candidate_pose(self, int_base, off):
        if not self.hProc: return None
        test_ptr = int_base + off
        buf = (ctypes.c_char * 4)()
        if not kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(test_ptr), buf, 4, None):
            return None
        b_ptr = struct.unpack_from("<I", buf, 0)[0]
        if not b_ptr or b_ptr < 0x10000: return None
        buf_mem = (ctypes.c_char * 28)()
        if not kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(b_ptr + 0x3D4), buf_mem, 28, None):
            return None
        x, y, z, _, fwd_x, fwd_y, fwd_z = struct.unpack_from("<7f", buf_mem, 0)
        if math.isnan(x) or math.isnan(y) or abs(fwd_x) < 0.0001 and abs(fwd_y) < 0.0001:
            return None
        if abs(x) > 8000 or abs(y) > 8000 or (abs(x) < 1.0 and abs(y) < 1.0):
            return None
        yaw_deg = math.degrees(math.atan2(fwd_y, fwd_x))
        fwd_z_clamped = max(-1.0, min(1.0, float(fwd_z)))
        pitch_deg = -math.degrees(math.asin(fwd_z_clamped))
        return test_ptr, float(x), float(y), float(z), float(yaw_deg), float(pitch_deg)

    def connect(self):
        if not (IS_WINDOWS and kernel32): return False
        self.last_connect = time.time()
        pid = wintypes.DWORD()
        for cls in [b"SDL_app", b"Valve001"]:
            h = user32.FindWindowA(cls, None)
            if h:
                user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                if pid.value: break
        if not pid.value: return False
        self.hProc = kernel32.OpenProcess(0x1010, False, pid.value)  # PROCESS_VM_READ | PROCESS_QUERY_INFORMATION
        if not self.hProc: return False

        # Locate hw.dll / sw.dll
        class MODULEENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
                ("ProccntUsage", wintypes.DWORD), ("modBaseAddr", ctypes.c_void_p),
                ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
                ("szModule", ctypes.c_wchar * 256), ("szExePath", ctypes.c_wchar * 260)
            ]
        snap = kernel32.CreateToolhelp32Snapshot(0x08, pid.value)
        hw_base = None
        if snap and snap != -1:
            me32 = MODULEENTRY32W()
            me32.dwSize = ctypes.sizeof(MODULEENTRY32W)
            if kernel32.Module32FirstW(snap, ctypes.byref(me32)):
                while True:
                    if me32.szModule.lower() in ["hw.dll", "sw.dll"]:
                        hw_base = me32.modBaseAddr
                        break
                    if not kernel32.Module32NextW(snap, ctypes.byref(me32)): break
            kernel32.CloseHandle(snap)

        if hw_base:
            int_base = int(hw_base) if isinstance(hw_base, int) else hw_base.value
            self.hw_base = int_base
            candidates = []
            for off in self.candidate_offsets:
                res = self._read_candidate_pose(int_base, off)
                if res:
                    t_ptr, tx, ty, tz, tyaw, tpitch = res
                    # Calculate proximity to CT spawn (-772, 160)
                    spawn_dist = math.hypot(tx - (-772.0), ty - 160.0)
                    candidates.append((spawn_dist, off, t_ptr, tx, ty, tz, tyaw, tpitch))

            if candidates:
                # Sort by proximity to CT spawn so local player entity is always chosen over dummies
                candidates.sort(key=lambda c: c[0])
                best = candidates[0]
                self.ptr_addr = best[2]
                print(f"[*] Memory Engine: Connected to hw.dll (POS: {best[3]:.0f}, {best[4]:.0f}, {best[5]:.0f} | Yaw: {best[6]:.0f}°) at offset 0x{best[1]:X} [SpawnDist: {best[0]:.0f}]")
                return True
        return False

    def get_pose(self):
        if not self.hProc or not self.ptr_addr:
            if time.time() - self.last_connect > 3.0:
                self.connect()
            return None
        buf = (ctypes.c_char * 4)()
        if not kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(self.ptr_addr), buf, 4, None):
            self.hProc = None
            return None
        base_ptr = struct.unpack_from("<I", buf, 0)[0]
        if not base_ptr or base_ptr < 0x10000: return None
        buf_mem = (ctypes.c_char * 28)()
        if not kernel32.ReadProcessMemory(self.hProc, ctypes.c_void_p(base_ptr + 0x3D4), buf_mem, 28, None):
            return None
        x, y, z, _, fwd_x, fwd_y, fwd_z = struct.unpack_from("<7f", buf_mem, 0)
        if math.isnan(x) or math.isnan(y) or abs(fwd_x) < 0.0001 and abs(fwd_y) < 0.0001:
            return None
        yaw_deg = math.degrees(math.atan2(fwd_y, fwd_x))
        fwd_z_clamped = max(-1.0, min(1.0, float(fwd_z)))
        pitch_deg = -math.degrees(math.asin(fwd_z_clamped))
        return float(x), float(y), float(z), float(yaw_deg), float(pitch_deg)

# =============================================================================
# 5. TARGET DETECTOR (YOLO CrowdHuman + Crash-Test Dummy Color)
# =============================================================================
class TargetDetector:
    def __init__(self, model_path="models/yolov8n_crowdhuman.onnx"):
        self.session = None
        if ort and os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
            except Exception:
                self.session = None

    def detect(self, bgr_img):
        targets = []
        gh, gw = bgr_img.shape[:2]
        cx_game = gw // 2
        cy_game = gh // 2

        # 1. Neural YOLO Detector
        if self.session:
            resized = cv2.resize(bgr_img, (640, 640), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.transpose(rgb, (2, 0, 1))[None, ...]
            out = self.session.run(None, {self.input_name: inp})[0][0]
            if out.shape[0] < out.shape[1]: out = out.T
            boxes = out[:, :4]
            scores = out[:, 4]
            mask = scores > 0.28
            for (bx, by, bw, bh), conf in zip(boxes[mask], scores[mask]):
                real_w = float(bw * gw / 640.0)
                real_h = float(bh * gh / 640.0)
                real_cx = float(bx * gw / 640.0)
                real_cy = float(by * gh / 640.0)
                if real_h < 26 or real_w < 14 or real_cy < gh * 0.15: continue
                # Reject first-person gun model / hands in bottom-right corner
                if real_cx > gw * 0.68 and real_cy > gh * 0.60: continue
                # Head position: top 16% of box
                head_x = real_cx
                head_y = (real_cy - real_h * 0.50) + (real_h * 0.16)
                dx = head_x - cx_game
                dy = head_y - cy_game
                targets.append({
                    "dx": dx, "dy": dy, "dist": math.hypot(dx, dy),
                    "w": real_w, "h": real_h, "conf": float(conf), "source": "yolo"
                })

        # 2. Crash-Test Dummy Color Detector (Yellow Mannequin with Bullseye Head)
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([20, 85, 85]), np.array([38, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 25))
        closed = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 320: continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / max(1.0, float(bw))
            if aspect < 1.1 or aspect > 4.5 or bh < 28 or bw < 12: continue
            if by < gh * 0.15 or (by + bh) > gh * 0.95: continue
            head_x = bx + bw / 2.0
            head_y = by + bh * 0.14
            # Reject first-person gun model / hands in bottom-right corner
            if head_x > gw * 0.68 and head_y > gh * 0.60: continue
            dx = head_x - cx_game
            dy = head_y - cy_game
            dist = math.hypot(dx, dy)
            # Avoid duplicate if YOLO already caught it
            if not any(abs(t["dx"] - dx) < 25 and abs(t["dy"] - dy) < 25 for t in targets):
                targets.append({
                    "dx": dx, "dy": dy, "dist": dist,
                    "w": float(bw), "h": float(bh), "conf": 0.92, "source": "dummy_color"
                })

        targets.sort(key=lambda t: t["dist"])
        return targets

# =============================================================================
# 6. WAYPOINT MAP ROUTES (aim_ai_sandbox)
# =============================================================================
ROUTES = [
    # Route 0: MASTER CENTER ARENA CLEAR (Doorway -> Center corridor -> Courtyard sweep)
    [(-772, 160), (-772, 360), (-772, 580), (-772, 750), (-772, 950), (-772, 1200), (-950, 1300), (-550, 1300), (-772, 1350)],
    # Route 1: LEFT FLANK & TUNNEL CLEAR (Doorway -> Left arch -> Left tunnel -> Enemy Base)
    [(-772, 160), (-772, 360), (-1000, 420), (-1200, 450), (-1380, 550), (-1380, 800), (-1380, 1050), (-1100, 1300), (-772, 1350)],
    # Route 2: RIGHT FLANK & TUNNEL CLEAR (Doorway -> Right arch -> Right tunnel -> Enemy Base)
    [(-772, 160), (-772, 360), (-550, 420), (-360, 450), (-200, 550), (-200, 800), (-200, 1050), (-450, 1300), (-772, 1350)],
]
ROUTE_NAMES = ["CENTER ARENA", "LEFT FLANK", "RIGHT FLANK"]

# =============================================================================
# 7. LOG WATCHER (Round Transitions & Kills)
# =============================================================================
class LogWatcher:
    def __init__(self, log_path, on_round_start, on_kill):
        self.log_path = log_path
        self.on_round_start = on_round_start
        self.on_kill = on_kill
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        f = None
        if os.path.exists(self.log_path):
            try:
                f = open(self.log_path, "r", encoding="utf-8", errors="ignore")
                f.seek(0, os.SEEK_END)
            except Exception:
                pass

        while self.running:
            time.sleep(0.05)
            if not f or not os.path.exists(self.log_path):
                if os.path.exists(self.log_path):
                    try:
                        f = open(self.log_path, "r", encoding="utf-8", errors="ignore")
                        f.seek(0, os.SEEK_END)
                    except Exception:
                        continue
                else:
                    continue

            while self.running:
                line = f.readline()
                if not line:
                    f.seek(f.tell())
                    break
                line_str = line.strip()
                if 'World triggered "Round_Start"' in line_str:
                    self.on_round_start()
                elif 'killed "' in line_str and ("STEAM_ID_LAN" in line_str or "не моюсь" in line_str):
                    if "STEAM_ID_LAN" in line_str.split(' killed "')[0] or "не моюсь" in line_str.split(' killed "')[0]:
                        self.on_kill()

# =============================================================================
# 8. MAIN AGENT LOOP
# =============================================================================
def main():
    print("===============================================================================")
    print("  CS 1.6 - Minimalist Autonomous Hunter-Killer (Option 3)")
    print("  Focus: Rock-solid spawn, direct waypoint navigation, 100% headshot kill")
    print("  Toggle: Press [F9] at any time to pause / resume")
    print("===============================================================================\n")

    # Mouse sensitivity calibration
    sens = 2.5
    for cfg in ["cstrike/config.cfg", "cstrike_downloads/config.cfg", "../cstrike/config.cfg"]:
        if os.path.exists(cfg):
            try:
                with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("sensitivity"):
                            sens = float(line.split()[1].strip('"'))
                            break
            except Exception:
                pass
            break

    # Screen & Client Area
    sct = mss.mss()
    box = find_cs_window_rect(sct.monitors[1])
    crop_w = box["width"]
    crop_h = box["height"]

    # GoldSrc physics calibration
    m_yaw = 0.022
    m_pitch = 0.022
    deg_per_count_x = sens * m_yaw
    deg_per_count_y = sens * m_pitch
    counts_per_deg = 1.0 / max(0.001, deg_per_count_x)
    k_aim_x = (90.0 / float(crop_w)) * counts_per_deg
    k_aim_y = (73.74 / float(crop_h)) * (1.0 / max(0.001, deg_per_count_y))

    print(f"[*] Window: {crop_w}x{crop_h} | Sens: {sens:.2f} | Counts/Deg: {counts_per_deg:.1f}")

    mem = MemoryReader()
    mem.connect()
    detector = TargetDetector()

    # State variables
    active_route_idx = 0
    current_wp_idx = 0
    round_kills = 0
    total_kills = 0
    bot_active = True
    last_f9 = False
    firing = False
    burst_cooldown = 0.0
    last_stuck_check = time.time()
    round_start_time = time.time()
    last_pos = (0.0, 0.0)
    current_keys = set()

    def on_round_start():
        nonlocal current_wp_idx, round_kills, active_route_idx, burst_cooldown, round_start_time
        current_wp_idx = 0
        round_kills = 0
        burst_cooldown = 0.0
        round_start_time = time.time()
        active_route_idx = (active_route_idx + 1) % len(ROUTES)
        print(f"\n>>> [ROUND START] New Route: {ROUTE_NAMES[active_route_idx]} <<<", flush=True)
        # Snap camera horizontal, pick class, autobuy M4A1
        tap_key("end", 0.04)
        dismiss_menu()
        time.sleep(0.08)
        tap_key("f1", 0.06)  # Native autobuy
        time.sleep(0.04)
        tap_key("1", 0.05)   # Draw primary

    def on_kill():
        nonlocal round_kills, total_kills
        round_kills += 1
        total_kills += 1
        print(f"\n[★ KILL CONFIRMED] Dummies Eliminated: {round_kills}/11 (Total: {total_kills})", flush=True)

    log_path = "qconsole.log" if os.path.exists("qconsole.log") else "cstrike/qconsole.log"
    watcher = LogWatcher(log_path, on_round_start, on_kill)
    watcher.start()

    # Initial focus & spawn
    focus_cs_window()
    time.sleep(0.1)
    dismiss_menu()
    tap_key("f1", 0.06)
    tap_key("1", 0.05)

    print("\n[*] Hunter-Killer Engine Started! Watching game...\n", flush=True)
    step = 0

    try:
        while True:
            step += 1
            now = time.time()

            # F9 Pause/Resume
            f9 = is_key_pressed(VK_F9)
            if f9 and not last_f9:
                bot_active = not bot_active
                print(f"\n>>> [AI BOT {'ACTIVE' if bot_active else 'PAUSED'}] <<<", flush=True)
                for k in list(current_keys): send_key_up(k)
                current_keys.clear()
                if firing: send_mouse_up(); firing = False
            last_f9 = f9

            if not bot_active or not is_cs_window_foreground():
                time.sleep(0.05)
                continue

            # 1. Grab Frame
            frame = np.array(sct.grab(box))[:, :, :3]

            # 2. Priority 0: Handle Class/Team Menu
            if is_menu_open(frame):
                for k in list(current_keys): send_key_up(k)
                current_keys.clear()
                if firing: send_mouse_up(); firing = False
                dismiss_menu()
                if step % 15 == 0:
                    print("\r[MENU ACTIVE] Dismissing class selection...", end="", flush=True)
                time.sleep(0.05)
                continue

            # 3. Perception: Target Acquisition
            targets = detector.detect(frame)
            pose = mem.get_pose()
            cur_x, cur_y, cur_z, cur_yaw = 0.0, 0.0, 0.0, 0.0
            target_yaw, angle_diff = 0.0, 0.0
            if pose is not None:
                cur_x, cur_y, cur_z, cur_yaw, _ = pose

            # Spawn Peace: inside CT spawn room (Y < 340) or in first 1.8s, run straight through doorway into arena
            in_spawn = (pose is not None and cur_y < 340.0) or (now - round_start_time < 1.8)

            # 4. State Machine: Combat vs Navigation
            if targets and not in_spawn:
                # -------------------------------------------------------------
                # COMBAT MODE: Stop, Snap, Burst-Headshot
                # -------------------------------------------------------------
                t = targets[0]
                dx, dy = t["dx"], t["dy"]

                # Dead-stop: instantly release movement keys for 100% accuracy
                if current_keys:
                    for k in list(current_keys): send_key_up(k)
                    current_keys.clear()

                # Smoothly track to head
                move_x = int(dx * k_aim_x * 0.75)
                move_y = int(dy * k_aim_y * 0.75)
                send_mouse_move(move_x, move_y)

                # Fire when crosshair is locked onto head
                if abs(dx) <= 24 and abs(dy) <= 22:
                    if now > burst_cooldown:
                        send_mouse_down()
                        time.sleep(0.08)  # 2-3 bullet burst
                        send_mouse_up()
                        burst_cooldown = now + 0.14  # Recoil cooldown
                        action_str = f"BURST FIRE [{t['source'].upper()}]"
                    else:
                        action_str = "RECOIL SETTLE"
                else:
                    action_str = f"LOCKING HEAD ({dx:+.0f}, {dy:+.0f})"

            else:
                # -------------------------------------------------------------
                # NAVIGATION MODE: Waypoints & Corridor Patrol
                # -------------------------------------------------------------
                action_str = "PATROL"
                if pose is not None:
                    cur_x, cur_y, cur_z, cur_yaw, _ = pose
                    active_wps = ROUTES[active_route_idx]

                    # Advance waypoint if reached
                    if current_wp_idx >= len(active_wps):
                        current_wp_idx = len(active_wps) - 1

                    wp_x, wp_y = active_wps[current_wp_idx]
                    dist = math.hypot(wp_x - cur_x, wp_y - cur_y)

                    if dist < 110.0 and current_wp_idx < len(active_wps) - 1:
                        current_wp_idx += 1
                        wp_x, wp_y = active_wps[current_wp_idx]
                        dist = math.hypot(wp_x - cur_x, wp_y - cur_y)

                    # Calculate target angle
                    target_yaw = math.degrees(math.atan2(wp_y - cur_y, wp_x - cur_x))
                    angle_diff = (target_yaw - cur_yaw + 180.0) % 360.0 - 180.0

                    # Proportional steering with deadband and bounded turn rate (zero overshoot)
                    if abs(angle_diff) < 3.5:
                        steer = 0
                    else:
                        # Smooth proportional steering clamped to [-35, 35] counts (~42 deg/sec)
                        k_steer = counts_per_deg * 0.25
                        steer = int(np.clip(-angle_diff * k_steer, -35, 35))

                    if steer != 0:
                        send_mouse_move(steer, 0)

                    # Walk forward with W: keep moving forward even while adjusting heading
                    if abs(angle_diff) < 70.0:
                        if "w" not in current_keys:
                            send_key_down("w")
                            current_keys.add("w")
                    else:
                        # Large angle deviation (> 70 deg): momentarily release W to snap heading
                        if "w" in current_keys:
                            send_key_up("w")
                            current_keys.remove("w")

                    # Anti-stuck: if stuck in place for 1.8s, jump and alternate strafe
                    if now - last_stuck_check > 1.8:
                        if math.hypot(cur_x - last_pos[0], cur_y - last_pos[1]) < 8.0:
                            tap_key("space", 0.05)
                            tap_key("d" if (step % 2 == 0) else "a", 0.12)
                        last_pos = (cur_x, cur_y)
                        last_stuck_check = now

                    action_str = f"NAV: {ROUTE_NAMES[active_route_idx]} [WP {current_wp_idx+1}/{len(active_wps)}]"
                else:
                    # Waiting for memory hook to read coordinates (do not blind-walk into walls)
                    if "w" in current_keys:
                        send_key_up("w")
                        current_keys.remove("w")
                    action_str = "STANDBY (HOOKING POSE)"

            if step % 10 == 0:
                if pose is not None:
                    print(f"\r[HUNTER] Kills: {round_kills:02d}/11 | POS:({cur_x:4.0f},{cur_y:4.0f}) | YAW:{cur_yaw:4.0f}°->{target_yaw:4.0f}° (Δ{angle_diff:+3.0f}°) | {action_str:22s}", end="", flush=True)
                else:
                    print(f"\r[HUNTER] Kills: {round_kills:02d}/11 | {action_str:32s} | Target: {'YES' if targets else 'NO '}", end="", flush=True)

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    finally:
        for k in ["w", "a", "s", "d"]: send_key_up(k)
        send_mouse_up()
        watcher.running = False
        print("[*] Bot terminated cleanly.")

if __name__ == "__main__":
    main()

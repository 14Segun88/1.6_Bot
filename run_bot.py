"""
Counter-Strike 1.6 - Tactical Vision & Map Autopilot (Precision Edition)
========================================================================
Architecture:
- [CS Vision Engine]      : CrowdHuman Neural Humanoid Detector (models/yolov8n_crowdhuman.onnx)
                            Native 640x640 inference, full headshot targeting,
                            weapon rejection, aspect ratio humanoid validation.
- [Mathematical Aiming]   : Auto-calibrated from cstrike/config.cfg (sensitivity & m_yaw).
                            Exact radian-to-mouse-count ballistic conversion.
- [Tactical Navigation]   : Active 3-Lane Corridor Navigator for $2000$ & Aim Maps:
                            - 1.4s tactical forward advance with angle scanning
                            - 90° corridor corner clearing & strafe-peeking
                            - Multi-sensor wall collision & stuck recovery (reverses on S,
                              rotates 90°-120° away from walls, clears passages).
- [Combat Reflex]         : Instant counter-strafe (freezes movement for 100% first-bullet accuracy),
                            precision head snapping, and controlled 2-3 round bursts.
- [Hotkeys]               : [F9] Toggle ON/OFF  |  [F10] Save debug screenshot
"""

import os
import sys
import time
import math
import random
import numpy as np
import mss
from PIL import Image, ImageDraw

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Platform-specific Windows hardware input
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
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

    class POINT(ctypes.Structure):
        _fields_ = [
            ("x", ctypes.c_long),
            ("y", ctypes.c_long)
        ]
else:
    user32 = None
    kernel32 = None

# Virtual Key Codes:
VK_F9 = 0x78
VK_F8 = 0x77
VK_INSERT = 0x2D
VK_F10 = 0x79

VK_MAP = {
    "w": 0x57,
    "s": 0x53,
    "a": 0x41,
    "d": 0x44,
    "space": 0x20,
    "ctrl": 0x11,
    "shift": 0x10,
    "r": 0x52,
}

# DirectInput Hardware Scan Codes:
SCAN_CODES = {
    "w": 0x11,
    "s": 0x1F,
    "a": 0x1E,
    "d": 0x20,
    "space": 0x39,
    "ctrl": 0x1D,
    "shift": 0x2A,
    "r": 0x13
}

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

if IS_WINDOWS:
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

def send_key_down(name):
    if not (IS_WINDOWS and user32):
        return
    vk = VK_MAP.get(name)
    sc = SCAN_CODES.get(name)
    if vk is None or sc is None:
        return
    try:
        inp = INPUT()
        inp.type = 1
        inp.ii.ki.wVk = vk
        inp.ii.ki.wScan = sc
        inp.ii.ki.dwFlags = KEYEVENTF_SCANCODE
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    except Exception:
        pass
    user32.keybd_event(vk, sc, 0, 0)
    user32.keybd_event(0, sc, KEYEVENTF_SCANCODE, 0)

def send_key_up(name):
    if not (IS_WINDOWS and user32):
        return
    vk = VK_MAP.get(name)
    sc = SCAN_CODES.get(name)
    if vk is None or sc is None:
        return
    try:
        inp = INPUT()
        inp.type = 1
        inp.ii.ki.wVk = vk
        inp.ii.ki.wScan = sc
        inp.ii.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    except Exception:
        pass
    user32.keybd_event(vk, sc, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(0, sc, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0)

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
    if IS_WINDOWS and kernel32:
        try:
            kernel32.Beep(int(freq), int(duration_ms))
        except Exception:
            pass


# ==============================================================================
# Configuration & Sensitivity Auto-Detection
# ==============================================================================
def get_cs_sensitivity():
    """
    Reads real player sensitivity and mouse yaw from cstrike/config.cfg.
    """
    sens = 1.1
    m_yaw = 0.022
    m_pitch = 0.022

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if "CS16_WORK_DIR" in os.environ:
        candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "config.cfg"))
        candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "userconfig.cfg"))
    candidates.extend([
        os.path.join(base_dir, "cstrike", "config.cfg"),
        os.path.join(base_dir, "CS 1.6 GoldClient", "cstrike", "config.cfg"),
        os.path.join(base_dir, "cstrike", "userconfig.cfg"),
        "cstrike/config.cfg",
        "D:/CS 1.6 GoldClient/cstrike/config.cfg",
        "/mnt/d/CS 1.6 GoldClient/cstrike/config.cfg"
    ])

    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        parts = line.split()
                        if len(parts) >= 2:
                            k = parts[0].lower()
                            v = parts[1].strip('"')
                            if k == "sensitivity":
                                sens = float(v)
                            elif k == "m_yaw":
                                m_yaw = float(v)
                            elif k == "m_pitch":
                                m_pitch = float(v)
                print(f"[Config Engine] Read CS 1.6 Settings from {p}: sensitivity={sens}, m_yaw={m_yaw}")
                break
            except Exception:
                pass

    return sens, m_yaw, m_pitch


def find_cs_window():
    """
    Locates the Counter-Strike 1.6 game window across both modern SDL2 and legacy GoldSrc builds.
    """
    if not (IS_WINDOWS and user32):
        return None

    # Check SDL2 window class (used by modern Steam & GoldClient)
    for cls_name in [b"SDL_app", b"Valve001"]:
        hwnd = user32.FindWindowA(cls_name, None)
        if hwnd and user32.IsWindowVisible(hwnd):
            rect = RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            if rect.right > 300 and rect.bottom > 200:
                return hwnd

    # Check standard window titles
    for title in [b"Counter-Strike", b"Counter-Strike 1.6", b"Half-Life"]:
        hwnd = user32.FindWindowA(None, title)
        if hwnd and user32.IsWindowVisible(hwnd):
            rect = RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rect))
            if rect.right > 300 and rect.bottom > 200:
                return hwnd

    # Callback search if title has custom suffix
    found = [None]
    def enum_cb(h, _):
        if user32.IsWindowVisible(h):
            length = user32.GetWindowTextLengthA(h)
            if length > 0:
                buff = ctypes.create_string_buffer(length + 1)
                user32.GetWindowTextA(h, buff, length + 1)
                t = buff.value.lower()
                if b"counter-strike" in t or b"cstrike" in t:
                    rect = RECT()
                    user32.GetClientRect(h, ctypes.byref(rect))
                    if rect.right > 300 and rect.bottom > 200:
                        found[0] = h
                        return False
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    return found[0]


def get_cs_game_center(monitor):
    """
    Finds exact center and screen coordinates of the CS 1.6 game window.
    """
    hwnd = find_cs_window()
    if hwnd:
        rect = RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        pt = POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(pt))
        gw = rect.right
        gh = rect.bottom
        cx = pt.x + gw // 2
        cy = pt.y + gh // 2
        return cx, cy, gw, gh, (pt.x, pt.y)

    cx = monitor["left"] + monitor["width"] // 2
    cy = monitor["top"] + monitor["height"] // 2
    return cx, cy, monitor["width"], monitor["height"], (monitor["left"], monitor["top"])


def nms_boxes(boxes, scores, iou_thresh=0.45):
    """Vectorized Non-Maximum Suppression"""
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
        w_int = np.maximum(0.0, xx2 - xx1)
        h_int = np.maximum(0.0, yy2 - yy1)
        inter = w_int * h_int
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


# ==============================================================================
# Tier 1: CS Vision Engine (CrowdHuman / YOLO Neural CS Detector)
# ==============================================================================
class CSAimDetector:
    def __init__(self, conf_thresh=0.20, input_size=640):
        self.conf_thresh = conf_thresh
        self.input_size = input_size
        self.session = None
        self.is_crowdhuman = False

        if ort is None:
            raise ImportError("onnxruntime is not installed.")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "models", "yolov8n_crowdhuman.onnx"),
            os.path.join(base_dir, "yolov8n_crowdhuman.onnx"),
            os.path.join(base_dir, "models", "yolov8n_dynamic.onnx"),
            os.path.join(base_dir, "yolov8n_dynamic.onnx"),
            os.path.join(base_dir, "models", "yolov8n.onnx"),
            os.path.join(base_dir, "yolov8n.onnx"),
        ]

        model_path = None
        for path in candidates:
            if os.path.isfile(path):
                model_path = path
                break

        if not model_path:
            raise FileNotFoundError("No valid YOLO model found! Expected yolov8n_crowdhuman.onnx in models/")

        available = ort.get_available_providers()
        preferred_order = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        chosen = [p for p in preferred_order if p in available] or ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(model_path, opts, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.provider_name = self.session.get_providers()[0]

        out_shape = self.session.get_outputs()[0].shape
        self.is_crowdhuman = bool(len(out_shape) >= 2 and (out_shape[1] == 5 or out_shape[-1] == 5))

        model_title = "CrowdHuman CS Specialist" if self.is_crowdhuman else "YOLO Dynamic General"
        print(f"[Vision Engine] Model: {os.path.basename(model_path)} [{model_title}]")
        print(f"[Vision Engine] Hardware: {self.provider_name} | Resolution: {self.input_size}x{self.input_size} | Conf Thresh: {self.conf_thresh:.2f}")

    def detect(self, bgr_crop):
        gh, gw = bgr_crop.shape[:2]
        cx_game = gw // 2
        cy_game = gh // 2

        # Fast bilinear resize to native input_size
        pil_crop = Image.fromarray(bgr_crop[:, :, 2::-1])
        if (gw, gh) != (self.input_size, self.input_size):
            resized = pil_crop.resize((self.input_size, self.input_size), Image.BILINEAR)
        else:
            resized = pil_crop

        rgb = np.array(resized, dtype=np.float32) / 255.0
        inp = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)

        out = self.session.run(None, {self.input_name: inp})[0]
        preds = out[0]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T  # [anchors, channels]

        boxes = preds[:, :4]
        scores = preds[:, 4]  # class 0: person / humanoid

        mask = scores > self.conf_thresh
        if not np.any(mask):
            return []

        b_filt = boxes[mask]
        s_filt = scores[mask]

        keep_idx = nms_boxes(b_filt, s_filt, iou_thresh=0.45)
        scale_x = gw / float(self.input_size)
        scale_y = gh / float(self.input_size)
        targets = []

        for idx in keep_idx:
            bx, by, bw, bh = b_filt[idx]
            aspect = bh / max(1.0, bw)

            # Filter non-human shapes (decal marks, square icons)
            if aspect < 1.2 or bh < 12:
                continue

            y_top = by - bh / 2.0
            head_norm_x = bx
            # In CS 1.6 standing/crouching models, head is top 18% of the bounding silhouette
            head_norm_y = y_top + bh * 0.18

            # Rejection filter for first-person player gun (strictly bottom-right corner)
            if head_norm_x > (self.input_size * 0.70) and head_norm_y > (self.input_size * 0.72):
                continue

            real_head_x = head_norm_x * scale_x
            real_head_y = head_norm_y * scale_y
            real_bx = bx * scale_x
            real_by = by * scale_y
            real_bw = bw * scale_x
            real_bh = bh * scale_y

            dx = real_head_x - cx_game
            dy = real_head_y - cy_game
            dist = float(math.hypot(dx, dy))
            is_locked = (abs(dx) <= 22 and abs(dy) <= 22)

            targets.append({
                "dx": float(dx),
                "dy": float(dy),
                "head_x": float(real_head_x),
                "head_y": float(real_head_y),
                "conf": float(s_filt[idx]),
                "dist": dist,
                "is_locked": is_locked,
                "rect": (
                    float(real_bx - real_bw / 2.0),
                    float(real_by - real_bh / 2.0),
                    float(real_bx + real_bw / 2.0),
                    float(real_by + real_bh / 2.0)
                )
            })

        # Prioritize target closest to the crosshair
        targets.sort(key=lambda t: t["dist"])
        return targets


# ==============================================================================
# Main Autopilot Engine
# ==============================================================================
def main():
    print("=" * 70)
    print("   Counter-Strike 1.6 - Tactical Vision & Map Autopilot (Precision)")
    print("=" * 70)

    # 1. Calibrate ballistic physics with game configuration
    sens, m_yaw, m_pitch = get_cs_sensitivity()
    counts_per_deg = 1.0 / (sens * m_yaw)
    print(f"[Ballistics] Sensitivity: {sens:.2f} | Yaw: {m_yaw:.3f} | Mouse Units / 90°: {int(90.0 * counts_per_deg)}")

    # 2. Initialize Neural Vision
    try:
        detector = CSAimDetector(conf_thresh=0.20, input_size=640)
    except Exception as e:
        print(f"\n[FATAL ERROR] Vision Engine failed to load: {e}")
        return

    sct = getattr(mss, "MSS", mss.mss)()
    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

    cx, cy, gw, gh, (win_left, win_top) = get_cs_game_center(monitor)
    print(f"[Window] Tracking CS 1.6 window: {gw}x{gh} at ({win_left}, {win_top}), Center: ({cx}, {cy})")

    # Wide High-Resolution Combat Crop
    # Uses 800x800 around crosshair (or full window if smaller) for wide FOV without losing distant detail
    crop_w = min(gw, 800)
    crop_h = min(gh, 800)
    half_w = crop_w // 2
    half_h = crop_h // 2

    box = {
        "left": cx - half_w,
        "top": cy - half_h,
        "width": crop_w,
        "height": crop_h
    }

    # Mathematical aim scaling: calculates mouse units needed per pixel offset
    deg_per_px = 90.0 / float(crop_w)
    k_aim = deg_per_px * counts_per_deg
    print(f"[Ballistics] Crosshair Scale Factor: {k_aim:.3f} mouse units/pixel")

    current_keys = set()
    firing = False
    step_count = 0
    bot_active = False

    toggle_keys = [VK_F9, VK_F8, VK_INSERT]
    toggle_was_down = False
    f10_was_down = False

    # Navigation & Anti-Stuck state machine
    nav_state = "ADVANCE"
    forward_start = time.time()
    turn_remaining = 0
    strafe_remaining = 0
    strafe_key = "a"
    consecutive_patrol_dir = 1

    # Optical motion & collision sensing
    prev_center_crop = None
    jammed_frames = 0
    burst_start = 0.0

    target_fps = 40
    frame_interval = 1.0 / target_fps

    print("\nStatus: PAUSED. Open Counter-Strike 1.6 and press [F9] to activate!")
    print("Press [F10] at any time to capture and save an annotated debug frame.\n", flush=True)

    try:
        while True:
            t0 = time.time()

            # Hotkey: F9 Toggle
            toggle_is_down = any(is_key_pressed(vk) for vk in toggle_keys)
            if toggle_is_down and not toggle_was_down:
                bot_active = not bot_active
                if bot_active:
                    cx, cy, gw, gh, (win_left, win_top) = get_cs_game_center(monitor)
                    box["left"] = cx - half_w
                    box["top"] = cy - half_h
                    forward_start = time.time()
                    nav_state = "ADVANCE"
                    turn_remaining = 0
                    strafe_remaining = 0
                    jammed_frames = 0
                    print("\n" + "=" * 58, flush=True)
                    print(f"  >>> [AI BOT ACTIVATED] - Crosshair at ({cx}, {cy}) <<<", flush=True)
                    print("=" * 58 + "\n", flush=True)
                    play_beep(1400, 150)
                else:
                    print("\n" + "=" * 58, flush=True)
                    print("  >>> [AI BOT PAUSED] - Human in control! <<<", flush=True)
                    print("=" * 58 + "\n", flush=True)
                    play_beep(600, 150)
                    for kname in list(current_keys):
                        send_key_up(kname)
                    current_keys.clear()
                    if firing:
                        send_mouse_up()
                        firing = False

            toggle_was_down = toggle_is_down

            # Hotkey: F10 Debug Frame Capture
            f10_is_down = is_key_pressed(VK_F10)
            if f10_is_down and not f10_was_down:
                try:
                    dbg_shot = sct.grab(box)
                    dbg_bgr = np.array(dbg_shot)
                    targets = detector.detect(dbg_bgr)

                    pil_dbg = Image.fromarray(dbg_bgr[:, :, 2::-1])
                    draw = ImageDraw.Draw(pil_dbg)

                    ch_x = crop_w // 2
                    ch_y = crop_h // 2
                    draw.line([(ch_x - 14, ch_y), (ch_x + 14, ch_y)], fill="cyan", width=2)
                    draw.line([(ch_x, ch_y - 14), (ch_x, ch_y + 14)], fill="cyan", width=2)

                    for t in targets:
                        x1, y1, x2, y2 = t["rect"]
                        col = "red" if t["is_locked"] else "lime"
                        draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
                        hx, hy = t["head_x"], t["head_y"]
                        draw.ellipse([hx - 4, hy - 4, hx + 4, hy + 4], fill="red", outline="white")
                        draw.text((x1, max(0, y1 - 12)), f"Conf: {t['conf']:.2f}", fill=col)

                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    out_png = os.path.join(base_dir, "debug_yolo_view.png")
                    pil_dbg.save(out_png)
                    print(f"\n[F10 Screenshot] Saved -> {out_png} (Targets found: {len(targets)})", flush=True)
                    for idx, t in enumerate(targets):
                        print(f"  Target #{idx+1}: Conf={t['conf']:.2f}, dx={t['dx']:+.1f}, dy={t['dy']:+.1f}, locked={t['is_locked']}")
                    play_beep(1700, 100)
                except Exception as ex:
                    print(f"\n[F10 Error]: {ex}", flush=True)
            f10_was_down = f10_is_down

            # Autopilot Execution Loop
            if bot_active:
                step_count += 1
                now = time.time()

                sct_img = sct.grab(box)
                crop_bgr = np.array(sct_img)

                # 1. Neural Vision Detection
                targets = detector.detect(crop_bgr)
                enemy_spotted = len(targets) > 0
                primary = targets[0] if enemy_spotted else None

                # 2. Multi-Sensor Texture & Obstacle Collision Sensing
                # Ground texture velocity: when moving forward, floor moves rapidly.
                # If stuck on a crate, corner, or wall, floor movement drops to near-zero.
                floor_y1 = int(crop_h * 0.70)
                floor_y2 = int(crop_h * 0.90)
                floor_x1 = int(crop_w * 0.25)
                floor_x2 = int(crop_w * 0.75)
                cur_floor = crop_bgr[floor_y1:floor_y2:3, floor_x1:floor_x2:3, :2]
                is_stuck_collision = False

                if prev_center_crop is not None and "w" in current_keys and not enemy_spotted and nav_state == "ADVANCE":
                    # Velocity of the ground under player's feet
                    floor_motion = float(np.mean(np.abs(cur_floor.astype(float) - prev_center_crop.astype(float))))
                    forward_duration = now - forward_start

                    # Sensor A: Ground is static while W is held (hit wall, crate corner, or door frame)
                    if floor_motion < 7.5 and forward_duration > 0.4:
                        jammed_frames += 1
                        if jammed_frames >= 3:
                            is_stuck_collision = True
                    else:
                        jammed_frames = max(0, jammed_frames - 1)

                    # Sensor B: Corridor boundary timeout on $2000$ (end of corridor reached)
                    if forward_duration > 2.2:
                        is_stuck_collision = True

                prev_center_crop = cur_floor

                desired_keys = set()
                action_status = ""

                # ==============================================================
                # Behavior Branch A: COMBAT MODE (Enemy in View)
                # ==============================================================
                if enemy_spotted:
                    nav_state = "ADVANCE"
                    turn_remaining = 0
                    strafe_remaining = 0
                    forward_start = now
                    jammed_frames = 0

                    # 1. Instant Counter-Strafe:
                    # Release all movement keys for 100% standing accuracy!
                    desired_keys.clear()

                    # 2. Precision Head Snapping (Ballistically calibrated)
                    raw_aim_x = primary["dx"] * k_aim
                    raw_aim_y = primary["dy"] * k_aim

                    # Proportional gain for smooth snap without overshooting
                    step_dx = int(np.clip(raw_aim_x * 0.75, -150, 150))
                    step_dy = int(np.clip(raw_aim_y * 0.75, -100, 100))
                    send_mouse_move(step_dx, step_dy)

                    # 3. Controlled Burst Fire & Recoil Compensation
                    if primary["is_locked"] or (abs(primary["dx"]) <= 24 and abs(primary["dy"]) <= 24):
                        action_status = "COMBAT: FIRING [BURST]"
                        if burst_start == 0.0:
                            burst_start = now
                            send_mouse_down()
                            firing = True
                            play_beep(1500, 60)

                        burst_time = now - burst_start
                        if burst_time < 0.10:
                            # Pull down gently for recoil compensation
                            send_mouse_move(0, 2)
                        elif burst_time < 0.22:
                            if firing:
                                send_mouse_up()
                                firing = False
                        else:
                            # Reset cycle for next burst
                            burst_start = now
                            send_mouse_down()
                            firing = True
                    else:
                        action_status = "COMBAT: AIMING [HEAD]"
                        if firing:
                            send_mouse_up()
                            firing = False
                        burst_start = 0.0

                # ==============================================================
                # Behavior Branch B: GOLDSRC UNHOOK REFLEX (Texture / Obstacle Escape)
                # ==============================================================
                elif is_stuck_collision or nav_state == "UNSTUCK":
                    if nav_state != "UNSTUCK":
                        nav_state = "UNSTUCK"
                        unstuck_end = now + 0.55  # 0.55s escape sequence
                        consecutive_patrol_dir *= -1
                        # Rotate 110°-140° away from the obstruction
                        turn_deg = consecutive_patrol_dir * random.randint(110, 140)
                        turn_remaining = int(turn_deg * counts_per_deg)
                        strafe_key = "d" if consecutive_patrol_dir > 0 else "a"
                        strafe_remaining = 10
                        jump_timer = now + 0.15  # Tap space to unhook GoldSrc collision hull

                    action_status = "NAV: UNHOOK_TEXTURES"
                    if firing:
                        send_mouse_up()
                        firing = False

                    # Back up with S
                    desired_keys.add("s")

                    # Add strafe to slide off the corner
                    if strafe_remaining > 0:
                        desired_keys.add(strafe_key)
                        strafe_remaining -= 1

                    # GoldSrc Jump-unhook: tap space to break edge clipping friction
                    if now < jump_timer:
                        desired_keys.add("space")

                    # Turn away from obstacle smoothly
                    if turn_remaining != 0:
                        step_t = int(np.clip(turn_remaining, -180, 180))
                        turn_remaining -= step_t
                        send_mouse_move(step_t, 0)

                    if now > unstuck_end and turn_remaining == 0:
                        nav_state = "ADVANCE"
                        forward_start = now
                        jammed_frames = 0

                # ==============================================================
                # Behavior Branch C: CORRIDOR NAVIGATION ($2000$ Patrol)
                # ==============================================================
                else:
                    if firing:
                        send_mouse_up()
                        firing = False

                    forward_duration = now - forward_start

                    # Advance down the open corridor for 1.8s
                    if forward_duration > 1.8 and nav_state == "ADVANCE":
                        nav_state = "CORNER_TURN"
                        consecutive_patrol_dir *= -1
                        turn_deg = consecutive_patrol_dir * 90  # Clean 90-degree corridor turn
                        turn_remaining = int(turn_deg * counts_per_deg)
                        strafe_key = "d" if consecutive_patrol_dir > 0 else "a"
                        strafe_remaining = 6

                    if nav_state == "CORNER_TURN":
                        action_status = "NAV: TURNING_LANE"
                        # Pause forward run during 90° corner pivot
                        if turn_remaining != 0:
                            step_t = int(np.clip(turn_remaining, -180, 180))
                            turn_remaining -= step_t
                            send_mouse_move(step_t, 0)

                        if strafe_remaining > 0:
                            desired_keys.add(strafe_key)
                            strafe_remaining -= 1

                        if turn_remaining == 0 and strafe_remaining == 0:
                            nav_state = "ADVANCE"
                            forward_start = now
                    else:
                        action_status = "NAV: RUNNING_CORRIDOR"
                        desired_keys.add("w")
                        # Gentle horizon scan to check angles while advancing
                        scan_step = int(math.sin(step_count * 0.18) * 6.0)
                        send_mouse_move(scan_step, 0)

                # Apply Keyboard State (DirectInput)
                for kname in ["w", "s", "a", "d"]:
                    need_key = kname in desired_keys
                    is_pressed = kname in current_keys
                    if need_key and not is_pressed:
                        send_key_down(kname)
                        current_keys.add(kname)
                    elif not need_key and is_pressed:
                        send_key_up(kname)
                        current_keys.remove(kname)

                # Telemetry Reporting
                if step_count % 10 == 0:
                    k_str = "+".join(sorted(current_keys)) if current_keys else "none"
                    if primary:
                        aim_str = f"({int(primary['dx']):+03d}, {int(primary['dy']):+03d}) [Conf: {primary['conf']:.2f}]"
                    else:
                        aim_str = "(---, ---) [Scanning]"
                    print(f"\r[Precision AI] Step: {step_count:05d} | {action_status:22s} | Keys: {k_str:7s} | Aim: {aim_str} ", end="", flush=True)

            else:
                time.sleep(0.02)

            elapsed = time.time() - t0
            rem = frame_interval - elapsed
            if rem > 0:
                time.sleep(rem)

    except KeyboardInterrupt:
        print("\nStopping bot...")
    finally:
        for kname in list(current_keys):
            try:
                send_key_up(kname)
            except Exception:
                pass
        if firing:
            try:
                send_mouse_up()
            except Exception:
                pass
        print("\n[AI Bot safely stopped]")

if __name__ == "__main__":
    main()

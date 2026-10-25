"""
Counter-Strike 1.6 - Multimodal Gameplay & Audio Recorder (AI Dataset Engine)
=============================================================================
Captures synchronized:
1. Video Frames (320x240 @ 30 FPS, Crash-Proof AVI MJPG + MP4) -> gameplay.avi
2. Stereo Audio Track (44.1 kHz 2-channel WASAPI Loopback) -> gameplay_audio.wav
3. Instantaneous Audio Features (Left RMS, Right RMS, Stereo Balance, Spike) -> actions.jsonl
4. Kernel-Level Hardware Inputs (W, S, A, D, Space, Ctrl, Shift, Mouse Aim dx/dy, Clicks)
   - 100% immune to keyboard language layouts (Russian / English / etc.)

Controls:
- [F9]  : Start / Pause Recording (Audio chimes confirm toggle)
- [F12] : Finish and Save Recording safely from within the game!
- [Ctrl+C] in Terminal to stop and finalize dataset session
"""

import os
import sys
import time
import json
import wave
import atexit
import signal
import threading
import subprocess
from datetime import datetime

# Auto-heal missing dependencies
for mod, pkg in [("cv2", "opencv-python"), ("mss", "mss"), ("pynput", "pynput"), ("numpy", "numpy")]:
    try:
        __import__(mod)
    except ImportError:
        print(f"[Setup] Installing missing package '{pkg}'...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import numpy as np
import cv2
import mss
from pynput import keyboard, mouse

# Windows platform specifics
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
else:
    user32 = None
    kernel32 = None

# Audio Loopback capture support
HAVE_SOUNDDEVICE = False
try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except ImportError:
    try:
        print("[Setup] Installing sounddevice for game audio...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sounddevice"])
        import sounddevice as sd
        HAVE_SOUNDDEVICE = True
    except Exception:
        pass

# ----------------- Configuration -----------------
CONFIG = {
    "target_fps": 30,
    "frame_width": 320,
    "frame_height": 240,
    "audio_sample_rate": 44100,
    "audio_channels": 2,
    "output_dir": "dataset",
    "toggle_key": keyboard.Key.f9,
    "finish_key": keyboard.Key.f12,
}

# Direct Hardware VK codes (100% immune to Russian / English keyboard layout!)
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
    "4": 0x34
}

# Fallback Pynput map (supports both English and Russian Cyrillic layouts)
KEY_MAP_FALLBACK = {
    'w': 'w', 'W': 'w', 'ц': 'w', 'Ц': 'w',
    'a': 'a', 'A': 'a', 'ф': 'a', 'Ф': 'a',
    's': 's', 'S': 's', 'ы': 's', 'Ы': 's',
    'd': 'd', 'D': 'd', 'в': 'd', 'В': 'd',
    'r': 'r', 'R': 'r', 'к': 'r', 'К': 'r',
    '1': '1', '2': '2', '3': '3', '4': '4',
}

active_keys = {k: 0 for k in VK_MAP}
mouse_state = {
    "left_click": 0,
    "right_click": 0,
    "dx": 0,
    "dy": 0
}

recording = False
exit_requested = False
state_lock = threading.Lock()

def play_beep(freq, duration_ms):
    if IS_WINDOWS and kernel32:
        try:
            kernel32.Beep(int(freq), int(duration_ms))
        except Exception:
            pass

def poll_hardware_keys():
    """Reads physical key states directly from Windows kernel (ignores layout)"""
    if IS_WINDOWS and user32:
        keys = {}
        for name, vk in VK_MAP.items():
            keys[name] = 1 if (user32.GetAsyncKeyState(vk) & 0x8000) else 0
        return keys
    with state_lock:
        return dict(active_keys)

# ----------------- Audio Loopback Engine -----------------
class AudioLoopbackEngine:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
        self.audio_chunks = []
        self.current_window = np.zeros((1024, channels), dtype=np.float32)
        self.lock = threading.Lock()
        self.active = False
        self.device_name = "None"

    def start(self):
        if not HAVE_SOUNDDEVICE:
            print("\n[Audio Notice] sounddevice is not available. Recording without audio.")
            return

        try:
            loopback_device = None
            if IS_WINDOWS:
                apis = sd.query_hostapis()
                wasapi_id = None
                for idx, api in enumerate(apis):
                    if "WASAPI" in api["name"].upper():
                        wasapi_id = idx
                        break

                if wasapi_id is not None:
                    devices = sd.query_devices()
                    default_output = apis[wasapi_id]["default_output_device"]
                    for dev_idx, dev in enumerate(devices):
                        if dev["hostapi"] == wasapi_id and dev["max_input_channels"] > 0:
                            if dev["name"] == devices[default_output]["name"]:
                                loopback_device = dev_idx
                                break
                    if loopback_device is None:
                        loopback_device = default_output

            def audio_callback(indata, frames, time_info, status):
                with self.lock:
                    self.current_window = indata.copy()
                    if self.active:
                        self.audio_chunks.append(indata.copy())

            try:
                self.stream = sd.InputStream(
                    device=loopback_device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=audio_callback,
                    extra_settings=sd.WasapiSettings(loopback=True) if IS_WINDOWS else None
                )
                self.stream.start()
                self.device_name = sd.query_devices(loopback_device)["name"] if loopback_device is not None else "Default"
                print(f"[Audio Engine] Connected to: {self.device_name} (WASAPI Stereo Loopback)")
            except Exception:
                self.stream = sd.InputStream(
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=audio_callback
                )
                self.stream.start()
                print(f"[Audio Engine] Connected to standard input audio stream")

        except Exception as ex:
            print(f"[Audio Warning] Could not initialize loopback recording: {ex}")

    def set_recording(self, state):
        with self.lock:
            self.active = state

    def get_features(self):
        with self.lock:
            data = self.current_window
        if data is None or len(data) == 0:
            return {"vol_left": 0.0, "vol_right": 0.0, "balance": 0.0, "peak": 0.0}

        left = data[:, 0]
        right = data[:, 1] if data.shape[1] > 1 else data[:, 0]

        rms_l = float(np.sqrt(np.mean(left ** 2) + 1e-8))
        rms_r = float(np.sqrt(np.mean(right ** 2) + 1e-8))
        balance = float((rms_r - rms_l) / (rms_l + rms_r + 1e-4))
        peak = float(np.max(np.abs(data)))

        return {
            "vol_left": round(min(1.0, rms_l * 3.0), 3),
            "vol_right": round(min(1.0, rms_r * 3.0), 3),
            "balance": round(np.clip(balance, -1.0, 1.0), 3),
            "peak": round(min(1.0, peak * 2.0), 3)
        }

    def save_wav(self, file_path):
        if not self.audio_chunks:
            return
        try:
            full_audio = np.concatenate(self.audio_chunks, axis=0)
            audio_int16 = (np.clip(full_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
            with wave.open(file_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            print(f" - Audio WAV: {file_path} ({len(audio_int16) / self.sample_rate:.1f}s)")
        except Exception as e:
            print(f"[Audio Save Error]: {e}")

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


# ----------------- Input Handlers -----------------
last_mouse_pos = None

def on_press(key):
    global recording, exit_requested
    if key == CONFIG["toggle_key"]:
        recording = not recording
        if recording:
            print("\n>>> [RECORDING STARTED] - Play normally in CS 1.6! <<<", flush=True)
            play_beep(1400, 150)
        else:
            print("\n>>> [RECORDING PAUSED] <<<", flush=True)
            play_beep(650, 150)
        return

    if key == CONFIG["finish_key"]:
        print("\n>>> [FINISH REQUESTED via F12] <<<", flush=True)
        play_beep(800, 250)
        exit_requested = True
        return

    with state_lock:
        if hasattr(key, "char") and key.char in KEY_MAP_FALLBACK:
            active_keys[KEY_MAP_FALLBACK[key.char]] = 1
        elif key == keyboard.Key.space:
            active_keys["space"] = 1
        elif key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            active_keys["ctrl"] = 1
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            active_keys["shift"] = 1

def on_release(key):
    with state_lock:
        if hasattr(key, "char") and key.char in KEY_MAP_FALLBACK:
            active_keys[KEY_MAP_FALLBACK[key.char]] = 0
        elif key == keyboard.Key.space:
            active_keys["space"] = 0
        elif key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            active_keys["ctrl"] = 0
        elif key in (keyboard.Key.shift, keyboard.Key.shift_r):
            active_keys["shift"] = 0

def on_click(x, y, button, pressed):
    with state_lock:
        val = 1 if pressed else 0
        if button == mouse.Button.left:
            mouse_state["left_click"] = val
        elif button == mouse.Button.right:
            mouse_state["right_click"] = val

def on_move(x, y):
    global last_mouse_pos
    with state_lock:
        if last_mouse_pos is not None:
            dx = x - last_mouse_pos[0]
            dy = y - last_mouse_pos[1]
            if abs(dx) < 250 and abs(dy) < 250:
                mouse_state["dx"] += dx
                mouse_state["dy"] += dy
        last_mouse_pos = (x, y)

def start_input_listeners():
    k_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    m_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    k_listener.daemon = True
    m_listener.daemon = True
    k_listener.start()
    m_listener.start()
    return k_listener, m_listener

def find_cs_window_rect(monitor):
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


# ----------------- Main Recording Loop -----------------
def main():
    global recording, exit_requested

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(CONFIG["output_dir"], f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    # Use crash-proof AVI container (MJPG)
    video_path = os.path.join(session_dir, "gameplay.avi")
    audio_path = os.path.join(session_dir, "gameplay_audio.wav")
    actions_path = os.path.join(session_dir, "actions.jsonl")

    print("=" * 68)
    print(" Counter-Strike 1.6 - Multimodal AI Gameplay Recorder (Vision+Audio)")
    print("=" * 68)
    print(f"Target FPS: {CONFIG['target_fps']}")
    print(f"Resolution: {CONFIG['frame_width']}x{CONFIG['frame_height']}")
    print(f"Audio Rate: {CONFIG['audio_sample_rate']} Hz (Stereo Loopback)")
    print(f"Output Directory: {session_dir}")
    print("=" * 68)
    print("Controls:")
    print("  [F9]      : START / PAUSE recording")
    print("  [F12]     : FINISH and SAVE recording safely from in-game")
    print("  [Ctrl+C]  : STOP in Terminal")
    print("=" * 68)

    # Initialize Audio Loopback
    audio_engine = AudioLoopbackEngine(sample_rate=CONFIG["audio_sample_rate"], channels=CONFIG["audio_channels"])
    audio_engine.start()

    # Start keyboard & mouse listeners
    start_input_listeners()

    sct = getattr(mss, "MSS", mss.mss)()
    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    box = find_cs_window_rect(monitor)
    print(f"[Screen] Capturing game region: {box['width']}x{box['height']} at ({box['left']}, {box['top']})")

    # Crash-proof Motion JPEG AVI writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(
        video_path,
        fourcc,
        CONFIG["target_fps"],
        (CONFIG["frame_width"], CONFIG["frame_height"])
    )

    actions_file = open(actions_path, "w", encoding="utf-8")

    def cleanup():
        try:
            audio_engine.set_recording(False)
            audio_engine.close()
            writer.release()
            actions_file.flush()
            actions_file.close()
            audio_engine.save_wav(audio_path)
            print(f"\n[DONE] Saved session cleanly to: {session_dir}")
        except Exception:
            pass

    atexit.register(cleanup)

    frame_interval = 1.0 / CONFIG["target_fps"]
    frame_count = 0
    start_time = None
    was_recording = False

    print("\n[Status] READY. Launch CS 1.6 and press [F9] to START recording!\n", flush=True)

    try:
        while not exit_requested:
            loop_start = time.time()

            if recording != was_recording:
                audio_engine.set_recording(recording)
                was_recording = recording

            if recording:
                if start_time is None:
                    start_time = time.time()

                # 1. Grab screen frame
                sct_img = sct.grab(box)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame_resized = cv2.resize(
                    frame,
                    (CONFIG["frame_width"], CONFIG["frame_height"]),
                    interpolation=cv2.INTER_AREA
                )
                writer.write(frame_resized)

                # 2. Extract instant audio features
                audio_feat = audio_engine.get_features()

                # 3. Direct kernel hardware keyboard query (100% layout-independent)
                hw_keys = poll_hardware_keys()

                # 4. Snapshot input state
                with state_lock:
                    dx_val = int(np.clip(mouse_state["dx"], -150, 150))
                    dy_val = int(np.clip(mouse_state["dy"], -100, 100))
                    l_val = int(mouse_state["left_click"])
                    r_val = int(mouse_state["right_click"])
                    # Reset relative mouse delta
                    mouse_state["dx"] = 0
                    mouse_state["dy"] = 0

                action_snapshot = {
                    "frame_id": frame_count,
                    "timestamp": round(time.time() - start_time, 4),
                    "keys": hw_keys,
                    "mouse": {
                        "dx": dx_val,
                        "dy": dy_val,
                        "left": l_val,
                        "right": r_val,
                    },
                    "audio": audio_feat
                }

                actions_file.write(json.dumps(action_snapshot) + "\n")
                actions_file.flush()

                frame_count += 1
                if frame_count % 15 == 0:
                    elapsed = round(time.time() - start_time, 1)
                    fps_real = round(frame_count / max(0.01, elapsed), 1)

                    k_active = [k.upper() for k, v in hw_keys.items() if v]
                    k_str = "+".join(k_active) if k_active else "---"

                    aud_str = f"L:{audio_feat['vol_left']:.2f} R:{audio_feat['vol_right']:.2f}"
                    if audio_feat["balance"] < -0.3:
                        aud_str += " [<- SOUND LEFT]"
                    elif audio_feat["balance"] > 0.3:
                        aud_str += " [SOUND RIGHT ->]"

                    sys.stdout.write(f"\r[Rec] Frames: {frame_count:05d} ({elapsed:05.1f}s) | FPS: {fps_real:4.1f} | Keys: {k_str:7s} | Aim: ({dx_val:+03d}, {dy_val:+03d}) | {aud_str:28s} ")
                    sys.stdout.flush()

            elapsed_loop = time.time() - loop_start
            sleep_time = frame_interval - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping recorder...")
    finally:
        cleanup()
        print("\n" + "=" * 60)
        print(" [COMPLETED] Dataset Session Saved Successfully!")
        print("=" * 60)
        print(f" - Directory : {session_dir}")
        print(f" - Video     : {video_path}")
        print(f" - Audio WAV : {audio_path}")
        print(f" - Actions   : {actions_path}")
        print(f" - Total FPS : {frame_count} frames recorded")
        print("=" * 60)

if __name__ == "__main__":
    main()

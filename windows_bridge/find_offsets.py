"""
Automatic offset finder for CS 1.6 (GoldSrc) local player view angles and origin.

How it works (differential memory scan of hw.dll + client.dll static memory):
  1. Moves the mouse left/right  -> finds floats that behave like YAW   (degrees)
  2. Moves the mouse up/down     -> finds floats that behave like PITCH (degrees)
  3. Walks forward/back (W / S)  -> finds float triples that behave like ORIGIN (x, y, z)
  4. Cross-checks: walking direction must match the found yaw.
Results are written to offsets.json, which memory_reader.py loads automatically.

Usage (PowerShell as Administrator, CS 1.6 spawned on aim_ai_sandbox, open space in front):
    python find_offsets.py
Then click into the CS 1.6 window within the countdown and DO NOT touch mouse/keyboard.
"""
import ctypes
import json
import math
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.ReadProcessMemory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
kernel32.ReadProcessMemory.restype = wintypes.BOOL

MODULES = ("hw.dll", "client.dll")
OUT_FILE = Path(__file__).parent / "offsets.json"


# ----------------------------------------------------------------------------- process
class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD), ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256), ("szExePath", ctypes.c_wchar * 260)
    ]


def attach():
    hwnd = user32.FindWindowA(b"SDL_app", None) or user32.FindWindowA(b"Valve001", None)
    if not hwnd:
        sys.exit("CS 1.6 window not found. Start the game first.")
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    hproc = kernel32.OpenProcess(0x0010 | 0x0400, False, pid.value)
    if not hproc:
        sys.exit("OpenProcess failed. Run PowerShell as Administrator.")
    mods = {}
    snap = kernel32.CreateToolhelp32Snapshot(0x08 | 0x10, pid.value)
    me = MODULEENTRY32W()
    me.dwSize = ctypes.sizeof(MODULEENTRY32W)
    if kernel32.Module32FirstW(snap, ctypes.byref(me)):
        while True:
            name = me.szModule.lower()
            if name in MODULES:
                mods[name] = (int(me.modBaseAddr), int(me.modBaseSize))
            if not kernel32.Module32NextW(snap, ctypes.byref(me)):
                break
    kernel32.CloseHandle(snap)
    for m in MODULES:
        if m not in mods:
            sys.exit(f"Module {m} not found in process")
    return hwnd, hproc, mods


def read_module(hproc, base, size):
    out = bytearray(size)
    chunk = 0x10000
    for off in range(0, size, chunk):
        n = min(chunk, size - off)
        buf = (ctypes.c_char * n)()
        if kernel32.ReadProcessMemory(hproc, base + off, buf, n, None):
            out[off:off + n] = buf.raw
        else:
            for poff in range(off, off + n, 0x1000):
                pn = min(0x1000, size - poff)
                pbuf = (ctypes.c_char * pn)()
                if kernel32.ReadProcessMemory(hproc, base + poff, pbuf, pn, None):
                    out[poff:poff + pn] = pbuf.raw
    usable = len(out) - (len(out) % 4)
    return np.frombuffer(bytes(out[:usable]), dtype="<f4").astype(np.float64)


def snapshot(hproc, mods):
    with np.errstate(all="ignore"):
        return {m: read_module(hproc, b, s) for m, (b, s) in mods.items()}


# ----------------------------------------------------------------------------- input
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _send(inp):
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def mouse_move(dx, dy, steps=20):
    for i in range(steps):
        sx = dx * (i + 1) // steps - dx * i // steps
        sy = dy * (i + 1) // steps - dy * i // steps
        inp = INPUT(type=0)
        inp.u.mi = MOUSEINPUT(sx, sy, 0, 0x0001, 0, 0)  # MOUSEEVENTF_MOVE
        _send(inp)
        time.sleep(0.01)
    time.sleep(0.35)


def key_hold(scan, seconds):
    down = INPUT(type=1)
    down.u.ki = KEYBDINPUT(0, scan, 0x0008, 0, 0)            # KEYEVENTF_SCANCODE
    up = INPUT(type=1)
    up.u.ki = KEYBDINPUT(0, scan, 0x0008 | 0x0002, 0, 0)     # + KEYEVENTF_KEYUP
    _send(down)
    time.sleep(seconds)
    _send(up)
    time.sleep(0.6)  # let the player stop sliding


SCAN_W, SCAN_S = 0x11, 0x1F


# ----------------------------------------------------------------------------- analysis
def adiff(a, b):
    """Signed angle difference b - a in degrees, wrapped to [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0


def find_angle(hproc, mods, axis, pixels, limit, thresh):
    """axis: 'x' (yaw) or 'y' (pitch). Returns {module: (indices, d_full)}."""
    def mv(p):
        mouse_move(p, 0) if axis == "x" else mouse_move(0, p)

    s0 = snapshot(hproc, mods)
    time.sleep(0.3)
    s0b = snapshot(hproc, mods)
    mv(pixels); s1 = snapshot(hproc, mods)
    mv(-pixels); s2 = snapshot(hproc, mods)
    mv(pixels // 2); s3 = snapshot(hproc, mods)
    mv(-(pixels // 2)); s4 = snapshot(hproc, mods)
    res = {}
    with np.errstate(all="ignore"):
        for m in mods:
            a0, a0b, a1, a2, a3, a4 = s0[m], s0b[m], s1[m], s2[m], s3[m], s4[m]
            ok = np.isfinite(a0) & np.isfinite(a1) & np.isfinite(a3) & (np.abs(a0) <= limit) & (np.abs(a1) <= limit)
            ok &= (a0 == a0b)                               # static while idle
            d1, d2, d3, d4 = adiff(a0, a1), adiff(a0, a2), adiff(a0, a3), adiff(a0, a4)
            ok &= np.abs(d1) > thresh                        # moved with mouse
            ok &= np.abs(d2) < max(0.6, 0.25 * thresh)       # came back
            ok &= np.abs(d4) < max(0.6, 0.25 * thresh)
            ok &= np.sign(d3) == np.sign(d1)                 # half move -> same direction
            ok &= (np.abs(d3) > 0.2 * np.abs(d1)) & (np.abs(d3) < 0.85 * np.abs(d1))
            idx = np.nonzero(ok)[0]
            res[m] = (idx, d1[idx], a0[idx])
    return res


def find_origin(hproc, mods):
    s0 = snapshot(hproc, mods)
    time.sleep(0.3)
    s0b = snapshot(hproc, mods)
    key_hold(SCAN_W, 0.5); s1 = snapshot(hproc, mods)
    key_hold(SCAN_S, 0.5); s2 = snapshot(hproc, mods)
    key_hold(SCAN_W, 0.3); s3 = snapshot(hproc, mods)
    key_hold(SCAN_S, 0.3); s4 = snapshot(hproc, mods)
    res = {}
    with np.errstate(all="ignore"):
        for m in mods:
            a0, a0b, a1, a2, a3, a4 = s0[m], s0b[m], s1[m], s2[m], s3[m], s4[m]
            n = len(a0) - 2
            X0, Y0, Z0 = a0[:n], a0[1:n + 1], a0[2:n + 2]
            stable = (a0 == a0b)
            st = stable[:n] & stable[1:n + 1] & stable[2:n + 2]
            fin = np.isfinite(X0) & np.isfinite(Y0) & np.isfinite(Z0)
            fin &= (np.abs(X0) < 8192) & (np.abs(Y0) < 8192) & (np.abs(Z0) < 8192)
            fin &= (np.abs(X0) > 1.0) | (np.abs(Y0) > 1.0)
            dx1, dy1, dz1 = a1[:n] - X0, a1[1:n + 1] - Y0, a1[2:n + 2] - Z0
            dx3, dy3 = a3[:n] - X0, a3[1:n + 1] - Y0
            mv1 = np.hypot(dx1, dy1)
            mv3 = np.hypot(dx3, dy3)
            back2 = np.hypot(a2[:n] - X0, a2[1:n + 1] - Y0)
            back4 = np.hypot(a4[:n] - X0, a4[1:n + 1] - Y0)
            ok = st & fin & (mv1 > 15) & (mv1 < 600) & (np.abs(dz1) < 4) & (mv3 > 5)
            ok &= (back2 < 0.6 * mv1) & (back4 < 0.6 * mv1 + 10)
            # same walking direction in both W presses
            ok &= (dx1 * dx3 + dy1 * dy3) > 0
            idx = np.nonzero(ok)[0]
            res[m] = (idx, X0[idx], Y0[idx], Z0[idx], np.degrees(np.arctan2(dy1[idx], dx1[idx])), mv1[idx])
    return res


def read_dummies(hproc, mods):
    import struct
    base = mods["hw.dll"][0] + 0x288970
    pts = []
    for idx in range(5, 16):
        buf = (ctypes.c_char * 12)()
        if kernel32.ReadProcessMemory(hproc, base + idx * 0xE0, buf, 12, None):
            x, y, z = struct.unpack_from("<3f", buf, 0)
            if abs(x) > 10 and abs(y) > 10:
                pts.append((x, y, z))
    return pts


# ----------------------------------------------------------------------------- main
def main():
    hwnd, hproc, mods = attach()
    for m, (b, s) in mods.items():
        print(f"{m}: base=0x{b:X} size=0x{s:X}")

    print("\n>>> Кликните в окно CS 1.6 (вы должны стоять на карте, перед вами - свободное место).")
    print(">>> НЕ трогайте мышь и клавиатуру во время сканирования (~30 сек).")
    try:
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
    for i in range(8, 0, -1):
        fg = user32.GetForegroundWindow() == hwnd
        print(f"   старт через {i}... {'(окно CS активно)' if fg else '(ОКНО CS НЕ АКТИВНО!)'}")
        time.sleep(1)
    if user32.GetForegroundWindow() != hwnd:
        print("WARNING: окно CS 1.6 не в фокусе - ввод может не дойти до игры.")

    print("\n[1/3] Поиск YAW (мышь влево/вправо)...")
    yaw = find_angle(hproc, mods, "x", 300, 360.0, 1.0)
    print("[2/3] Поиск PITCH (мышь вверх/вниз)...")
    pitch = find_angle(hproc, mods, "y", 120, 89.5, 0.8)
    print("[3/3] Поиск ORIGIN (W / S)...")
    origin = find_origin(hproc, mods)

    for name, r in (("YAW", yaw), ("PITCH", pitch)):
        for m, (idx, d, v) in r.items():
            print(f"  {name} candidates in {m}: {len(idx)}")
            for i, dd, vv in list(zip(idx, d, v))[:8]:
                print(f"     +0x{i * 4:X}  value={vv:8.2f}  delta={dd:+.2f}")
    for m, (idx, X, Y, Z, mdir, mv) in origin.items():
        print(f"  ORIGIN candidates in {m}: {len(idx)}")
        for k in range(min(8, len(idx))):
            print(f"     +0x{idx[k] * 4:X}  pos=({X[k]:.1f}, {Y[k]:.1f}, {Z[k]:.1f}) walk_dir={mdir[k]:.1f} moved={mv[k]:.1f}")

    # ---- choose best pair (pitch, yaw) : prefer pitch directly before yaw (vec3 pitch,yaw,roll)
    angle_choice = None
    for m in MODULES:
        yset = {int(i): float(v) for i, v in zip(yaw[m][0], yaw[m][2])}
        for i in pitch[m][0]:
            if int(i) + 1 in yset:
                angle_choice = (m, int(i) * 4, (int(i) + 1) * 4, yset[int(i) + 1])
                break
        if angle_choice:
            break
    if not angle_choice:
        for m in MODULES:
            if len(yaw[m][0]) and len(pitch[m][0]):
                angle_choice = (m, int(pitch[m][0][0]) * 4, int(yaw[m][0][0]) * 4, float(yaw[m][2][0]))
                break

    origin_choice = None
    if angle_choice:
        yaw_val = angle_choice[3]
        dummies = read_dummies(hproc, mods)
        best = None
        for m in MODULES:
            idx, X, Y, Z, mdir, mv = origin[m]
            for k in range(len(idx)):
                dir_err = abs(adiff(yaw_val, float(mdir[k])))
                near = min((math.dist((X[k], Y[k]), (d[0], d[1])) for d in dummies), default=0.0)
                score = dir_err + near / 200.0 + (0 if m == "hw.dll" else 5)
                if dir_err < 25 and (best is None or score < best[0]):
                    best = (score, m, int(idx[k]) * 4, dir_err)
        if best:
            origin_choice = best[1:]

    print("\n================ RESULT ================")
    if not angle_choice or not origin_choice:
        print("Не удалось однозначно определить смещения.")
        print("Проверьте: вы заспавнены (не в меню/консоли/наблюдателе), окно CS в фокусе, впереди нет стены.")
        print("Пришлите весь вывод этого скрипта.")
        return
    am, poff, yoff, _ = angle_choice
    om, ooff, derr = origin_choice
    cfg = {
        "origin": {"module": om, "offset": hex(ooff)},
        "pitch": {"module": am, "offset": hex(poff)},
        "yaw": {"module": am, "offset": hex(yoff)},
        "angles_unit": "degrees",
        "note": f"auto-found by find_offsets.py, walk/yaw direction error {derr:.1f} deg",
    }
    OUT_FILE.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))
    print(f"\nСохранено в {OUT_FILE}. Перезапустите bridge_server.py.")


if __name__ == "__main__":
    main()


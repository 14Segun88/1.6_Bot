"""
Targeted verification of the local-player struct found by find_offsets.py.

find_offsets.py output showed one consistent struct in hw.dll:
    hw.dll + 0x288034 : origin x, y, z   (pos=(-688.8, 1485.0, -144.0), walk detected)
    hw.dll + 0x288040 : pitch (deg)      (40 copies of the same value -> real viewangle)
    hw.dll + 0x288044 : yaw?             (expected right after pitch, NOT detected - verify here)
This script moves the mouse / walks automatically, prints exactly which floats react,
detects which "dummy" table slots are actually the local player, and writes offsets.json.

Usage (PowerShell as Administrator, spawned on aim_ai_sandbox, free space in front):
    python verify_offsets.py
Click into CS 1.6 during the countdown and do NOT touch mouse/keyboard (~15 sec).
"""
import ctypes
import json
import math
import struct
import time

from find_offsets import (attach, kernel32, user32, mouse_move, key_hold, adiff,
                          SCAN_W, SCAN_S, OUT_FILE)

REGION_START = 0x288000   # dump window around the candidate struct
REGION_FLOATS = 48        # 0x288000 .. 0x2880C0
ORIGIN_OFF = 0x288034
PITCH_OFF = 0x288040
EYE_OFF = 0x42A288        # eye-position copy (z = origin z + view_ofs)
EXTRA_YAW = (0x78E8EC, 0x33C012C)
DUMMY_TABLE = 0x288970
DUMMY_STRIDE = 0xE0


def main():
    hwnd, hproc, mods = attach()
    hw = mods["hw.dll"][0]

    def rf(off, n):
        buf = (ctypes.c_char * (4 * n))()
        if not kernel32.ReadProcessMemory(hproc, hw + off, buf, 4 * n, None):
            return [float("nan")] * n
        return list(struct.unpack_from(f"<{n}f", buf, 0))

    def region():
        return rf(REGION_START, REGION_FLOATS)

    def slots():
        return [tuple(rf(DUMMY_TABLE + i * DUMMY_STRIDE, 3)) for i in range(0, 20)]

    print(">>> Кликните в окно CS 1.6 и НЕ трогайте мышь/клавиатуру (~15 сек).")
    try:
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass
    for i in range(6, 0, -1):
        fg = user32.GetForegroundWindow() == hwnd
        print(f"   старт через {i}... {'(окно CS активно)' if fg else '(ОКНО CS НЕ АКТИВНО!)'}")
        time.sleep(1)

    r0 = region()
    time.sleep(0.3)
    r0b = region()
    extra0 = [rf(o, 1)[0] for o in EXTRA_YAW]

    mouse_move(300, 0); rH = region(); extraH = [rf(o, 1)[0] for o in EXTRA_YAW]
    mouse_move(-300, 0); rH2 = region()
    mouse_move(0, 120); rV = region()
    mouse_move(0, -120); rV2 = region()

    o0 = rf(ORIGIN_OFF, 3); e0 = rf(EYE_OFF, 3); s0 = slots()
    key_hold(SCAN_W, 0.5)
    o1 = rf(ORIGIN_OFF, 3); s1 = slots()
    key_hold(SCAN_S, 0.5)

    print("\n--- Floats around hw.dll+0x288000 that react (H = mouse right 300px, V = mouse down 120px) ---")
    print("   offset      idle     dH(right)   back    dV(down)   back   idle_jitter")
    yaw_off = pitch_off = None
    for k in range(REGION_FLOATS):
        off = REGION_START + 4 * k
        a = r0[k]
        if not math.isfinite(a):
            continue
        dH, bH = adiff(a, rH[k]), adiff(a, rH2[k])
        dV, bV = adiff(a, rV[k]), adiff(a, rV2[k])
        jit = adiff(a, r0b[k])
        if max(abs(dH), abs(dV)) > 0.05 or off in (ORIGIN_OFF, ORIGIN_OFF + 4, ORIGIN_OFF + 8, PITCH_OFF, PITCH_OFF + 4):
            print(f"   +0x{off:X}  {a:9.3f}  {dH:+9.3f}  {bH:+7.3f}  {dV:+9.3f}  {bV:+7.3f}  {jit:+.3f}")
        if abs(a) <= 360 and abs(dH) > 0.8 and abs(dV) < 0.3 and abs(bH) < 0.6 and yaw_off is None:
            yaw_off = off
        if abs(a) <= 89.9 and abs(dV) > 0.8 and abs(dH) < 0.3 and abs(bV) < 0.6 and pitch_off is None:
            pitch_off = off

    for o, a, b in zip(EXTRA_YAW, extra0, extraH):
        print(f"   (old yaw candidate +0x{o:X}: {a:.3f} -> {b:.3f}, d={adiff(a, b):+.3f})")

    dx, dy, dz = o1[0] - o0[0], o1[1] - o0[1], o1[2] - o0[2]
    moved = math.hypot(dx, dy)
    walk_dir = math.degrees(math.atan2(dy, dx))
    eye_dz = e0[2] - o0[2]
    print(f"\n--- Origin hw.dll+0x{ORIGIN_OFF:X}: {tuple(round(v, 1) for v in o0)} -> {tuple(round(v, 1) for v in o1)}"
          f"  moved={moved:.1f} walk_dir={walk_dir:.1f} dz={dz:+.1f}")
    print(f"--- Eye copy hw.dll+0x{EYE_OFF:X}: {tuple(round(v, 1) for v in e0)}  eye_dz={eye_dz:+.1f}")

    print("\n--- 'Dummy' table slots (hw.dll+0x288970 + i*0xE0): before / after walking ---")
    self_slots = []
    for i, (a, b) in enumerate(zip(s0, s1)):
        if not all(math.isfinite(v) for v in a) or (abs(a[0]) < 10 and abs(a[1]) < 10):
            continue
        sm = math.hypot(b[0] - a[0], b[1] - a[1])
        d_player = math.hypot(a[0] - o0[0], a[1] - o0[1])
        tag = ""
        if sm > 5:
            tag = "  <-- MOVES WITH PLAYER (not a dummy!)"
            self_slots.append(i)
        elif d_player < 64:
            tag = "  <-- at player position"
            self_slots.append(i)
        print(f"   slot {i:2d}: ({a[0]:8.1f}, {a[1]:8.1f}, {a[2]:7.1f})  moved={sm:6.1f}  dist_to_player={d_player:7.1f}{tag}")

    print("\n================ RESULT ================")
    problems = []
    if pitch_off is None:
        problems.append("pitch не найден")
    if yaw_off is None:
        problems.append("yaw не найден (мышь по горизонтали не изменила ни одного float рядом)")
    if moved < 15:
        problems.append("origin не сдвинулся при W (стена впереди или окно не в фокусе)")
    yaw_err = None
    if yaw_off is not None and moved >= 15:
        yaw_err = abs(adiff(r0[(yaw_off - REGION_START) // 4], walk_dir))
        if yaw_err > 20:
            problems.append(f"yaw ({r0[(yaw_off - REGION_START) // 4]:.1f}) не совпадает с направлением шага ({walk_dir:.1f})")
    if problems:
        print("ПРОВЕРКА НЕ ПРОЙДЕНА: " + "; ".join(problems))
        print("offsets.json НЕ изменён. Пришлите весь вывод.")
        return

    cfg = {
        "origin": {"module": "hw.dll", "offset": hex(ORIGIN_OFF)},
        "pitch": {"module": "hw.dll", "offset": hex(pitch_off)},
        "yaw": {"module": "hw.dll", "offset": hex(yaw_off)},
        "angles_unit": "degrees",
        "eye_z_offset": round(eye_dz, 2) if 8.0 < eye_dz < 30.0 else 17.0,
        "exclude_dummy_slots": [i for i in self_slots if 5 <= i < 16],
        "note": f"verified by verify_offsets.py, walk/yaw error {yaw_err:.1f} deg",
    }
    OUT_FILE.write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))
    print(f"\nПРОВЕРКА ПРОЙДЕНА. Сохранено в {OUT_FILE}. Перезапустите bridge_server.py.")


if __name__ == "__main__":
    main()


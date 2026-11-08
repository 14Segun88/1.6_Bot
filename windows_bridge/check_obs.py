"""
Live check of what the RL agent "sees". Run on Windows while CS 1.6 is in a match:
    python check_obs.py
Point the crosshair at a yellow dummy: rel_yaw / rel_pitch must go to ~0.
Press Ctrl+C to stop.
"""
import logging
import math
import time

from memory_reader import MemoryReader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

mr = MemoryReader()
if not mr.connect():
    raise SystemExit("Не удалось подключиться к hl.exe (запустите PowerShell от имени Администратора)")

try:
    while True:
        obs = mr.get_observation()
        if obs is None:
            print("INVALID observation (игрок не прочитан)")
        else:
            s, e = obs["self"], obs["enemy"]
            n = len(mr.get_enemies())
            p, y = (math.degrees(v) for v in s["view_angles"])
            ry, rp = (math.degrees(v) for v in e["relative_angles"])
            print(f"pos=({s['origin'][0]:7.1f},{s['origin'][1]:7.1f},{s['origin'][2]:6.1f}) "
                  f"pitch={p:6.1f} yaw={y:7.1f} | targets={n} | "
                  f"rel_yaw={ry:+7.1f} rel_pitch={rp:+6.1f} dist={e['distance'] * 3000:6.0f}")
        time.sleep(0.25)
except KeyboardInterrupt:
    pass
finally:
    mr.close()


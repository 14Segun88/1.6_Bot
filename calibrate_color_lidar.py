"""
Counter-Strike 1.6 - Optical Color Lidar HSV Lighting Calibration Suite
========================================================================
Runs an empirical 10-round calibration session to analyze and tune:
1. Green Walkable Floor (Сюда МОЖНО -> press W) across bright corridors & dark shadows.
2. Red Barrier Walls (Сюда НЕЛЬЗЯ -> 45° pivot) across direct light & shaded obstacles.

Outputs:
- benchmarks/lidar_hsv_profile.json (Auto-loaded by play_agent.py)
- LIDAR_CALIBRATION_REPORT.md (Detailed lighting & statistical report)
- debug_visuals/lidar_calibration_grid.jpg (4-quadrant visual black-box)
"""

import os
import sys
import time
import math
import json
import glob
import ctypes
from datetime import datetime
import cv2
import numpy as np

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import mss
    import ctypes.wintypes as wintypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
else:
    user32 = None
    kernel32 = None
    psapi = None

# Ensure base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(SCRIPT_DIR, "debug_visuals")
BENCH_DIR = os.path.join(SCRIPT_DIR, "benchmarks")
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


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


def find_latest_log():
    candidates = [
        "cstrike/logs",
        os.path.join(SCRIPT_DIR, "cstrike", "logs"),
        "D:/CS 1.6 GoldClient/cstrike/logs",
        "/mnt/d/CS 1.6 GoldClient/cstrike/logs"
    ]
    if "CS16_WORK_DIR" in os.environ:
        candidates.insert(0, os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "logs"))
    for base in candidates:
        if os.path.exists(base):
            logs = sorted(
                [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".log")],
                key=os.path.getmtime
            )
            if logs:
                return logs[-1]
    return None


class ReservoirSampler:
    """Memory-efficient statistical reservoir sampler for large pixel streams"""
    def __init__(self, max_samples=100000):
        self.max_samples = max_samples
        self.samples = []
        self.count = 0

    def add_pixels(self, pixel_array):
        if len(pixel_array) == 0:
            return
        n = len(pixel_array)
        if len(self.samples) < self.max_samples:
            take = min(n, self.max_samples - len(self.samples))
            self.samples.extend(pixel_array[:take].tolist())
        
        # Reservoir sampling for remaining
        if len(self.samples) >= self.max_samples:
            step = max(1, n // 500)
            sub = pixel_array[::step]
            replace_idx = np.random.randint(0, self.max_samples, size=len(sub))
            for s_val, r_idx in zip(sub, replace_idx):
                self.samples[r_idx] = int(s_val)
        self.count += n

    def get_percentiles(self):
        if not self.samples:
            return {"min": 0, "p5": 0, "p25": 0, "med": 0, "p75": 0, "p95": 0, "max": 0, "count": 0}
        arr = np.array(self.samples, dtype=np.uint8)
        return {
            "min": int(np.min(arr)),
            "p5": int(np.percentile(arr, 5)),
            "p25": int(np.percentile(arr, 25)),
            "med": int(np.median(arr)),
            "p75": int(np.percentile(arr, 75)),
            "p95": int(np.percentile(arr, 95)),
            "max": int(np.max(arr)),
            "count": self.count
        }


def create_calibration_grid(frame_bgr, green_mask, red_mask, current_stats, target_round, total_rounds=10):
    """Generates an aesthetic 4-quadrant diagnostic black box visualization"""
    h, w = frame_bgr.shape[:2]
    out_w, out_h = 640, 360
    
    # Panel 1: Original In-Game View with ROIs
    p1 = cv2.resize(frame_bgr, (out_w, out_h))
    # Draw ROIs on P1
    cv2.rectangle(p1, (0, int(out_h * 0.50)), (out_w, int(out_h * 0.95)), (0, 255, 120), 2)
    cv2.putText(p1, "SCAN ROI: FLOOR TERRAIN (50-95%)", (10, int(out_h * 0.50) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1)
    cv2.rectangle(p1, (int(out_w * 0.25), int(out_h * 0.25)), (int(out_w * 0.75), int(out_h * 0.75)), (0, 0, 255), 2)
    cv2.putText(p1, "SCAN ROI: BARRIER WALL (25-75%)", (int(out_w * 0.25) + 5, int(out_h * 0.25) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    
    # Panel 2: Green Walkable Floor Mask with Overlay
    p2 = cv2.resize(frame_bgr, (out_w, out_h))
    g_res = cv2.resize(green_mask, (out_w, out_h))
    green_overlay = np.zeros_like(p2)
    green_overlay[:, :, 1] = 220
    p2 = np.where(g_res[:, :, None] > 0, cv2.addWeighted(p2, 0.45, green_overlay, 0.55, 0), p2)
    cv2.putText(p2, "LIDAR: GREEN FLOOR DETECTED (WALKABLE)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 2)
    
    # Panel 3: Red Barrier Wall Mask with Overlay
    p3 = cv2.resize(frame_bgr, (out_w, out_h))
    r_res = cv2.resize(red_mask, (out_w, out_h))
    red_overlay = np.zeros_like(p3)
    red_overlay[:, :, 2] = 220
    p3 = np.where(r_res[:, :, None] > 0, cv2.addWeighted(p3, 0.45, red_overlay, 0.55, 0), p3)
    cv2.putText(p3, "LIDAR: RED WALL DETECTED (NO-GO BARRIER)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Panel 4: Statistical Telemetry & HSV Calibration Monitor
    p4 = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    p4[:] = (20, 22, 28)
    
    cv2.putText(p4, f"COLOR LIDAR CALIBRATOR: ROUND {target_round:02d} / {total_rounds:02d}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 240), 2)
    cv2.line(p4, (15, 42), (out_w - 15, 42), (80, 80, 90), 1)
    
    # Green Stats
    g_h = current_stats["g_h"].get_percentiles()
    g_s = current_stats["g_s"].get_percentiles()
    g_v = current_stats["g_v"].get_percentiles()
    cv2.putText(p4, "[GREEN FLOOR LIGHTING SPECTRUM]", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 100), 1)
    cv2.putText(p4, f"HUE (Color):       [{g_h['p5']:02d} .. {g_h['p95']:02d}]  (Med: {g_h['med']:02d})", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 200), 1)
    cv2.putText(p4, f"SAT (Saturation):  [{g_s['p5']:02d} .. {g_s['p95']:02d}]  (Med: {g_s['med']:02d})", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 200), 1)
    cv2.putText(p4, f"VAL (Brightness):  [{g_v['p5']:02d} .. {g_v['p95']:02d}]  (Shadow min: {g_v['min']:02d}, Max: {g_v['max']:02d})", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 200), 1)
    
    # Red Stats
    r_h = current_stats["r_h"].get_percentiles()
    r_s = current_stats["r_s"].get_percentiles()
    r_v = current_stats["r_v"].get_percentiles()
    cv2.putText(p4, "[RED WALL BARRIER SPECTRUM]", (15, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1)
    cv2.putText(p4, f"HUE (Color):       [{r_h['p5']:02d} .. {r_h['p95']:02d}]  (Med: {r_h['med']:02d})", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1)
    cv2.putText(p4, f"SAT (Saturation):  [{r_s['p5']:02d} .. {r_s['p95']:02d}]  (Med: {r_s['med']:02d})", (25, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1)
    cv2.putText(p4, f"VAL (Brightness):  [{r_v['p5']:02d} .. {r_v['p95']:02d}]  (Shadow min: {r_v['min']:02d}, Max: {r_v['max']:02d})", (25, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1)
    
    # Status footer
    tot_samples = g_h["count"] + r_h["count"]
    cv2.line(p4, (15, 275), (out_w - 15, 275), (80, 80, 90), 1)
    cv2.putText(p4, f"Total Analyzed Pixels: {tot_samples:,} | Sampling 5 FPS", (15, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)
    cv2.putText(p4, "Calibrating real-time map lighting across all 10 rounds...", (15, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)
    
    # Combine 4 panels (2x2 grid)
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])
    return grid


def main():
    print("=" * 78)
    print("   CS 1.6 - OPTICAL COLOR LIDAR: 10-ROUND HSV LIGHTING CALIBRATOR")
    print("=" * 78)
    print("Objective: Calibrate HSV thresholds for green floor and red barrier walls")
    print("under realistic map lighting (direct sunlight, shaded corridors, dark corners).")
    print("=" * 78)

    # Initialize Screen Capture
    sct = getattr(mss, "MSS", mss.mss)() if 'mss' in globals() else None
    if not sct:
        print("[!] mss library not found. Please run with Windows python environment.")
        return

    monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
    box = find_cs_window_rect(monitor)
    print(f"[*] Attached to Game Window: {box['width']}x{box['height']} at ({box['left']}, {box['top']})")

    # Initialize Samplers
    samplers = {
        "g_h": ReservoirSampler(),
        "g_s": ReservoirSampler(),
        "g_v": ReservoirSampler(),
        "r_h": ReservoirSampler(),
        "r_s": ReservoirSampler(),
        "r_v": ReservoirSampler(),
    }

    # Watchdog for game log
    log_file = find_latest_log()
    last_log_size = os.path.getsize(log_file) if (log_file and os.path.exists(log_file)) else 0
    print(f"[*] Monitoring Game Engine Log: {log_file if log_file else 'Polling screen timer'}")

    current_round = 1
    total_rounds = 10
    round_start_time = time.time()
    last_sample_time = 0.0
    last_snap_time = 0.0
    frames_processed = 0

    print(f"\n>>> [STARTING 10-ROUND CALIBRATION SESSION] <<<")
    print("Please ensure Counter-Strike 1.6 is running with bot on aim_ai_sandbox / $2000$!")
    print("Press Ctrl+C at any time to finish and save report early.\n")

    try:
        while current_round <= total_rounds:
            now = time.time()

            # 1. Check for Round Transitions via Game Engine Logs
            if log_file and os.path.exists(log_file):
                try:
                    curr_size = os.path.getsize(log_file)
                    if curr_size > last_log_size:
                        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                            f.seek(last_log_size)
                            new_lines = f.readlines()
                        last_log_size = curr_size
                        for line in new_lines:
                            if 'World triggered "Round_Start"' in line:
                                current_round += 1
                                round_start_time = now
                                print(f"\n[+] ROUND TRANSITION -> Round {current_round:02d} / {total_rounds:02d} Started!")
                                if current_round > total_rounds:
                                    break
                except Exception:
                    pass

            # Fallback timeout if logs are stationary (e.g. 120s round limit)
            if now - round_start_time > 130.0:
                current_round += 1
                round_start_time = now
                print(f"\n[+] ROUND WATCHDOG -> Transition to Round {current_round:02d} / {total_rounds:02d}")
                if current_round > total_rounds:
                    break

            # 2. Sample Frame at ~5 FPS (every 0.20s)
            if now - last_sample_time >= 0.18:
                last_sample_time = now
                raw = np.array(sct.grab(box))
                frame_bgr = raw[:, :, :3]
                fh, fw = frame_bgr.shape[:2]

                if fh < 100 or fw < 100:
                    time.sleep(0.05)
                    continue

                frames_processed += 1

                # ROIs
                # Floor ROI: bottom 50% to 95%. Mask out bottom-right weapon area (to avoid weapon textures)
                floor_roi = frame_bgr[int(fh * 0.50):int(fh * 0.95), :]
                weapon_mask = np.ones(floor_roi.shape[:2], dtype=bool)
                fl_h, fl_w = floor_roi.shape[:2]
                weapon_mask[int(fl_h * 0.40):, int(fl_w * 0.65):] = False  # Mask weapon/hands

                # Center ROI: middle 50%x50% rectangle directly ahead
                center_roi = frame_bgr[int(fh * 0.25):int(fh * 0.75), int(fw * 0.25):int(fw * 0.75)]

                hsv_f = cv2.cvtColor(floor_roi, cv2.COLOR_BGR2HSV)
                hsv_c = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

                # Chromatic filters to extract true map colors under all lighting:
                # 1. Green floor candidates: green is dominant color (G > R+12 and G > B+12)
                b_f, g_f, r_f = cv2.split(floor_roi)
                green_cand = (g_f > r_f + 12) & (g_f > b_f + 12) & weapon_mask
                if np.sum(green_cand) > 10:
                    samplers["g_h"].add_pixels(hsv_f[:, :, 0][green_cand])
                    samplers["g_s"].add_pixels(hsv_f[:, :, 1][green_cand])
                    samplers["g_v"].add_pixels(hsv_f[:, :, 2][green_cand])

                # 2. Red barrier wall candidates: red is dominant color (R > G+16 and R > B+16)
                b_c, g_c, r_c = cv2.split(center_roi)
                red_cand = (r_c > g_c + 16) & (r_c > b_c + 16)
                if np.sum(red_cand) > 10:
                    samplers["r_h"].add_pixels(hsv_c[:, :, 0][red_cand])
                    samplers["r_s"].add_pixels(hsv_c[:, :, 1][red_cand])
                    samplers["r_v"].add_pixels(hsv_c[:, :, 2][red_cand])

                # Mask creation for visual diagnostics
                g_mask_full = np.zeros((fh, fw), dtype=np.uint8)
                r_mask_full = np.zeros((fh, fw), dtype=np.uint8)

                g_submask = ((g_f > r_f + 12) & (g_f > b_f + 12) & weapon_mask).astype(np.uint8) * 255
                r_submask = ((r_c > g_c + 16) & (r_c > b_c + 16)).astype(np.uint8) * 255

                g_mask_full[int(fh * 0.50):int(fh * 0.95), :] = g_submask
                r_mask_full[int(fh * 0.25):int(fh * 0.75), int(fw * 0.25):int(fw * 0.75)] = r_submask

                # Update Visual Grid every 1.2 seconds
                if now - last_snap_time >= 1.2:
                    last_snap_time = now
                    grid_img = create_calibration_grid(frame_bgr, g_mask_full, r_mask_full, samplers, current_round, total_rounds)
                    grid_path = os.path.join(DEBUG_DIR, "lidar_calibration_grid.jpg")
                    cv2.imwrite(grid_path, grid_img)

                    # Console Telemetry
                    gh = samplers["g_h"].get_percentiles()
                    gv = samplers["g_v"].get_percentiles()
                    rh = samplers["r_h"].get_percentiles()
                    rv = samplers["r_v"].get_percentiles()
                    print(f"\r[CALIBRATING] Rnd {current_round:02d}/{total_rounds:02d} | Frames: {frames_processed:04d} | Green H:[{gh['p5']:02d}-{gh['p95']:02d}] V:[{gv['p5']:02d}-{gv['p95']:02d}] | Red H:[{rh['p5']:02d}-{rh['p95']:02d}] V:[{rv['p5']:02d}-{rv['p95']:02d}]", end="", flush=True)

            time.sleep(0.015)

    except KeyboardInterrupt:
        print("\n\n[*] Early termination requested by user. Finalizing calibration with collected samples...")

    print("\n\n" + "=" * 78)
    print("             CALIBRATION ANALYSIS COMPLETE - COMPUTING OPTIMAL HSV")
    print("=" * 78)

    # Compute Statistical Boundaries
    gh = samplers["g_h"].get_percentiles()
    gs = samplers["g_s"].get_percentiles()
    gv = samplers["g_v"].get_percentiles()

    rh = samplers["r_h"].get_percentiles()
    rs = samplers["r_s"].get_percentiles()
    rv = samplers["r_v"].get_percentiles()

    # Optimal tuned boundaries with safety margins for shadows and ambient lighting
    tuned_green_h_min = max(25, gh["p5"] - 5) if gh["count"] > 100 else 32
    tuned_green_h_max = min(95, gh["p95"] + 8) if gh["count"] > 100 else 88
    tuned_green_s_min = max(20, gs["p5"] - 15) if gs["count"] > 100 else 35
    tuned_green_v_min = max(18, gv["min"] if gv["min"] > 15 else (gv["p5"] - 10)) if gv["count"] > 100 else 25

    # Red wraps around 0 and 180 in HSV
    tuned_red_h_max1 = min(18, rh["p95"] + 4) if rh["count"] > 100 else 15
    tuned_red_h_min2 = 165
    tuned_red_s_min = max(30, rs["p5"] - 15) if rs["count"] > 100 else 45
    tuned_red_v_min = max(18, rv["min"] if rv["min"] > 15 else (rv["p5"] - 10)) if rv["count"] > 100 else 30

    hsv_profile = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_frames_analyzed": frames_processed,
        "green_floor": {
            "h_range": [tuned_green_h_min, tuned_green_h_max],
            "s_min": tuned_green_s_min,
            "v_min": tuned_green_v_min,
            "raw_stats": {
                "h": gh, "s": gs, "v": gv
            }
        },
        "red_wall": {
            "h_range_1": [0, tuned_red_h_max1],
            "h_range_2": [tuned_red_h_min2, 180],
            "s_min": tuned_red_s_min,
            "v_min": tuned_red_v_min,
            "raw_stats": {
                "h": rh, "s": rs, "v": rv
            }
        }
    }

    # Save JSON Profile
    profile_path = os.path.join(BENCH_DIR, "lidar_hsv_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(hsv_profile, f, indent=2)
    print(f"[✓] Tuned HSV Profile saved to: {profile_path}")

    # Sync to client folder if running in dual-mount
    for alt in ["/mnt/d/CS 1.6 GoldClient/benchmarks", "D:/CS 1.6 GoldClient/benchmarks"]:
        if os.path.exists(alt):
            try:
                with open(os.path.join(alt, "lidar_hsv_profile.json"), "w", encoding="utf-8") as f:
                    json.dump(hsv_profile, f, indent=2)
                print(f"[✓] Synced profile to: {alt}/lidar_hsv_profile.json")
            except Exception:
                pass

    # Generate Markdown Report
    report_md = [
        "# Optical Color Lidar - 10-Round HSV Lighting Calibration Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Frames Analyzed:** {frames_processed:,}",
        f"**Map Tested:** `aim_ai_sandbox` / `$2000$` (Vibrant Green Floor + Red Barriers)",
        "",
        "## 1. Optimal Calibrated HSV Thresholds",
        "",
        "### 🟢 Green Walkable Floor (Сюда МОЖНО -> Press W)",
        "| Parameter | Old Static Value | **Calibrated Optimal Value** | Empirical Lighting Range |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Hue (Color Shade)** | `[32 .. 88]` | **`[{tuned_green_h_min} .. {tuned_green_h_max}]`** | 5th-95th percentile: `{gh['p5']} .. {gh['p95']}` (Median: {gh['med']}) |",
        f"| **Saturation (Vibrancy)** | `> 45` | **`> {tuned_green_s_min}`** | Shadow min: `{gs['min']}`, Median: `{gs['med']}` |",
        f"| **Value (Brightness)** | `> 35` | **`> {tuned_green_v_min}`** | Dark corner min: `{gv['min']}`, Bright sunlight: `{gv['max']}` |",
        "",
        "### 🔴 Red Barrier Walls (Сюда НЕЛЬЗЯ -> 45° Pivot)",
        "| Parameter | Old Static Value | **Calibrated Optimal Value** | Empirical Lighting Range |",
        "| :--- | :---: | :---: | :--- |",
        f"| **Hue Sector 1** | `[0 .. 14]` | **`[0 .. {tuned_red_h_max1}]`** | Pure red peak: `{rh['med']}` |",
        f"| **Hue Sector 2** | `[165 .. 180]` | **`[{tuned_red_h_min2} .. 180]`** | Wrap-around red spectrum |",
        f"| **Saturation (Vibrancy)** | `> 50` | **`> {tuned_red_s_min}`** | Shadow min: `{rs['min']}`, Median: `{rs['med']}` |",
        f"| **Value (Brightness)** | `> 40` | **`> {tuned_red_v_min}`** | Shaded obstacle min: `{rv['min']}`, Highlight: `{rv['max']}` |",
        "",
        "---",
        "## 2. Visual Diagnostic Black-Box",
        "- [4-Quadrant Calibration Preview (Frame / Masks / Radar)](file://debug_visuals/lidar_calibration_grid.jpg)",
        "",
        "---",
        "## 3. Ready-To-Use Code Implementation for `play_agent.py`",
        "```python",
        "# Calibrated Optical Color Lidar for aim_ai_sandbox under all lighting conditions:",
        f"green_f = cv2.inRange(hsv_f, np.array([{tuned_green_h_min}, {tuned_green_s_min}, {tuned_green_v_min}]), np.array([{tuned_green_h_max}, 255, 255]))",
        "red_f = cv2.bitwise_or(",
        f"    cv2.inRange(hsv_f, np.array([0, {tuned_red_s_min}, {tuned_red_v_min}]), np.array([{tuned_red_h_max1}, 255, 255])),",
        f"    cv2.inRange(hsv_f, np.array([{tuned_red_h_min2}, {tuned_red_s_min}, {tuned_red_v_min}]), np.array([180, 255, 255]))",
        ")",
        "```",
        ""
    ]

    report_path = os.path.join(SCRIPT_DIR, "LIDAR_CALIBRATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md) + "\n")
    print(f"[✓] Detailed Markdown Report generated: {report_path}")

    for alt in ["/mnt/d/CS 1.6 GoldClient", "D:/CS 1.6 GoldClient"]:
        if os.path.exists(alt):
            try:
                with open(os.path.join(alt, "LIDAR_CALIBRATION_REPORT.md"), "w", encoding="utf-8") as f:
                    f.write("\n".join(report_md) + "\n")
                print(f"[✓] Synced report to: {alt}/LIDAR_CALIBRATION_REPORT.md")
            except Exception:
                pass

    print("\n" + "=" * 78)
    print("                     CALIBRATION READY & SAVED")
    print("=" * 78)
    print(f"Recommended Green Floor HSV:  H:[{tuned_green_h_min}..{tuned_green_h_max}], S>{tuned_green_s_min}, V>{tuned_green_v_min}")
    print(f"Recommended Red Wall HSV:     H:[0..{tuned_red_h_max1} & {tuned_red_h_min2}..180], S>{tuned_red_s_min}, V>{tuned_red_v_min}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

"""
Counter-Strike 1.6 - Master AI Self-Learning Arena Orchestrator
================================================================
Single entry point that:
1. Sets working directory to game / project folder.
2. Checks dependencies (torch, onnxruntime, cv2, mss, pynput, sounddevice, numpy).
3. Verifies models (YOLOv8, cs16_bot.onnx, tactical_brain, policy_weights).
4. Launches CS 1.6 on map $2000$ with 10 bots (if not already running).
5. Spawns the 3 AI Arena engines (Daemon, Arbitrator, Agent) in dedicated console windows.
6. Keeps master window open with live telemetry status.
"""

import os
import sys
import time
import subprocess
import traceback

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("onnxruntime", "onnxruntime"),
    ("cv2", "opencv-python"),
    ("mss", "mss"),
    ("pynput", "pynput"),
    ("sounddevice", "sounddevice"),
    ("numpy", "numpy")
]

def setup_working_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    
    # 1. Environment variable override
    if "CS16_DIR" in os.environ and os.path.isdir(os.environ["CS16_DIR"]):
        candidates.append(os.environ["CS16_DIR"])
    if "CS16_PATH" in os.environ and os.path.isdir(os.environ["CS16_PATH"]):
        candidates.append(os.environ["CS16_PATH"])

    # 2. Bundled game in script directory or current folder
    candidates.append(os.path.join(script_dir, "CS 1.6 GoldClient"))
    candidates.append(script_dir)
    candidates.append(os.getcwd())

    # 3. Known drive locations (Windows D: and WSL /mnt/d)
    candidates.append(r"D:\CS 1.6 GoldClient")
    candidates.append("/mnt/d/CS 1.6 GoldClient")

    # 4. Standard installation locations
    candidates.append(r"C:\CS 1.6 GoldClient")
    candidates.append(r"C:\Games\CS 1.6 GoldClient")
    candidates.append(r"C:\Games\Counter-Strike 1.6")
    candidates.append(r"C:\Program Files (x86)\Counter-Strike 1.6")

    for d in candidates:
        if os.path.exists(d) and (os.path.exists(os.path.join(d, "hl.exe")) or os.path.exists(os.path.join(d, "cstrike.exe"))):
            try:
                os.chdir(d)
            except Exception:
                pass
            print(f"[+] Target Game Directory detected: {d}", flush=True)
            return d

    # Fallback to script directory
    os.chdir(script_dir)
    print(f"[+] Working Directory set: {script_dir}", flush=True)
    return script_dir

def check_and_install_dependencies():
    print("\n[*] Step 1/4: Checking AI dependencies...", flush=True)
    missing = []
    for mod_name, pkg_name in REQUIRED_PACKAGES:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        print(f"    [!] Installing missing packages via pip: {missing}...", flush=True)
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        subprocess.check_call(cmd)
        print("    [+] Packages installed successfully!", flush=True)
    else:
        print("    [+] All AI packages are installed and verified (PyTorch, ONNX, OpenCV, MSS, etc.)", flush=True)

def verify_assets(work_dir):
    print("\n[*] Step 2/4: Verifying directories and model assets...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for base in [work_dir, script_dir]:
        os.makedirs(os.path.join(base, "dataset"), exist_ok=True)
        os.makedirs(os.path.join(base, "models"), exist_ok=True)
        os.makedirs(os.path.join(base, "debug_visuals"), exist_ok=True)

    # --- Auto-deploy: copy Python scripts from script dir to game dir if running separately ---
    if os.path.abspath(script_dir) != os.path.abspath(work_dir):
        import shutil
        py_scripts = [
            "play_agent.py", "self_learning_daemon.py", "arena_arbitrator.py",
            "run_arena_master.py", "eval_benchmark.py", "run_bot.py",
            "recorder.py", "stable_train.py", "train_tactical_brain.py"
        ]
        for fname in py_scripts:
            src = os.path.join(script_dir, fname)
            dst = os.path.join(work_dir, fname)
            if os.path.exists(src):
                try:
                    if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                        shutil.copy2(src, dst)
                        print(f"    [+] Deployed: {fname} -> {work_dir}", flush=True)
                except Exception as e:
                    print(f"    [!] Notice deploying {fname}: {e}", flush=True)

        # Deploy all model files from script_dir/models to work_dir/models
        script_models = os.path.join(script_dir, "models")
        work_models = os.path.join(work_dir, "models")
        if os.path.isdir(script_models):
            for mfile in os.listdir(script_models):
                msrc = os.path.join(script_models, mfile)
                mdst = os.path.join(work_models, mfile)
                if os.path.isfile(msrc):
                    if not os.path.exists(mdst) or os.path.getsize(msrc) != os.path.getsize(mdst):
                        try:
                            shutil.copy2(msrc, mdst)
                            print(f"    [+] Deployed Model: {mfile} -> {work_models}", flush=True)
                        except Exception as e:
                            print(f"    [!] Notice deploying model {mfile}: {e}", flush=True)

    # --- Copy ai_arena.cfg to cstrike/ in game dir ---
    cfg_src_candidates = [
        os.path.join(script_dir, "cstrike", "ai_arena.cfg"),
        os.path.join(script_dir, "ai_arena.cfg"),
        os.path.join(work_dir, "cstrike", "ai_arena.cfg")
    ]
    cfg_dst = os.path.join(work_dir, "cstrike", "ai_arena.cfg")
    os.makedirs(os.path.dirname(cfg_dst), exist_ok=True)
    import shutil
    for cfg_src in cfg_src_candidates:
        if os.path.exists(cfg_src) and os.path.abspath(cfg_src) != os.path.abspath(cfg_dst):
            try:
                shutil.copy2(cfg_src, cfg_dst)
                print(f"    [+] Deployed: ai_arena.cfg -> {cfg_dst}", flush=True)
                break
            except Exception as e:
                print(f"    [!] Notice deploying ai_arena.cfg: {e}", flush=True)

    # --- Experience buffer: initialize clean experience buffer in all active locations ---
    clean_sample = '{"x": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], "k": [1,0,0,0,0,0,0], "c": [0,0], "a": [0.0,0.0], "r": 0.0}\n'
    for base in [work_dir, script_dir]:
        exp_file = os.path.join(base, "dataset", "online_experiences.jsonl")
        if os.path.islink(exp_file) and not os.path.exists(exp_file):
            try:
                os.remove(exp_file)
                print(f"    [!] Removed dead symlink: {exp_file}", flush=True)
            except Exception:
                pass
        if not os.path.exists(exp_file) or os.path.getsize(exp_file) == 0:
            try:
                with open(exp_file, "w", encoding="utf-8") as f:
                    f.write(clean_sample)
                print(f"    [+] Initialized clean experience buffer: {exp_file}", flush=True)
            except Exception as e:
                print(f"    [!] Notice creating {exp_file}: {e}", flush=True)

    target_exp = os.path.join(work_dir, "dataset", "online_experiences.jsonl")
    if os.path.exists(target_exp):
        size_kb = os.path.getsize(target_exp) // 1024
        print(f"    [+] Active experience buffer: {target_exp} ({size_kb} KB)", flush=True)
        os.environ["CS16_EXP_FILE"] = target_exp
    os.environ["CS16_WORK_DIR"] = work_dir

    # Check model weights
    models = ["yolov8n_crowdhuman.onnx", "cs16_bot.onnx", "tactical_brain.npz", "policy_weights.npz"]
    for m in models:
        candidates = [
            os.path.join(work_dir, "models", m),
            os.path.join(script_dir, "models", m),
            os.path.join(work_dir, m),
            os.path.join(script_dir, m)
        ]
        found = any(os.path.exists(p) for p in candidates)
        if found:
            p = next(p for p in candidates if os.path.exists(p))
            print(f"    [+] Model verified: {m} ({os.path.getsize(p):,} bytes)", flush=True)
        else:
            print(f"    [!] Warning: Model {m} not found in models/", flush=True)

def is_hl_running():
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output('tasklist /FI "IMAGENAME eq hl.exe"', shell=True).decode("cp866", errors="ignore")
            if "hl.exe" in out.lower():
                return True
            out2 = subprocess.check_output('tasklist /FI "IMAGENAME eq cstrike.exe"', shell=True).decode("cp866", errors="ignore")
            return "cstrike.exe" in out2.lower()
        except Exception:
            return False
    return False

def find_game_executable(work_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        (os.path.join(work_dir, "hl.exe"), work_dir),
        (os.path.join(work_dir, "cstrike.exe"), work_dir),
        (os.path.join(script_dir, "CS 1.6 GoldClient", "hl.exe"), os.path.join(script_dir, "CS 1.6 GoldClient")),
        (os.path.join(script_dir, "CS 1.6 GoldClient", "cstrike.exe"), os.path.join(script_dir, "CS 1.6 GoldClient")),
        (r"D:\CS 1.6 GoldClient\hl.exe", r"D:\CS 1.6 GoldClient"),
        (r"C:\CS 1.6 GoldClient\hl.exe", r"C:\CS 1.6 GoldClient"),
    ]
    for exe, d in candidates:
        if os.path.exists(exe):
            return exe, d
    return None, work_dir

def launch_game_if_needed(work_dir, force_restart=True):
    print("\n[*] Step 3/4: Checking Counter-Strike 1.6 status...", flush=True)
    if is_hl_running() and force_restart:
        print("    [*] Restarting CS 1.6 to apply fresh 1:55 round settings...", flush=True)
        if sys.platform.startswith("win"):
            subprocess.run('taskkill /F /IM hl.exe >nul 2>&1', shell=True)
            subprocess.run('taskkill /F /IM cstrike.exe >nul 2>&1', shell=True)
            time.sleep(1.0)
    elif is_hl_running():
        print("    [+] CS 1.6 (hl.exe) is ALREADY RUNNING!", flush=True)
        return
    
    exe_path, run_dir = find_game_executable(work_dir)
    if exe_path and sys.platform.startswith("win"):
        print(f"    [*] Launching CS 1.6 on map aim_ai_sandbox with 11 dummies ({exe_path})...", flush=True)
        cmd = f'start "" "{exe_path}" -game cstrike -nomaster -novid -steam -noforcemparms -condebug +maxplayers 24 +map aim_ai_sandbox +exec listenserver.cfg +exec maps/aim_ai_sandbox.cfg +mp_roundtime 1.92 +mp_round_restart_delay 1.55'
        subprocess.Popen(cmd, cwd=run_dir, shell=True)
        print("    [*] Waiting 5 seconds for CS 1.6 engine to initialize...", flush=True)
        time.sleep(5.0)
    elif exe_path:
        print(f"    [*] Game binary located at: {exe_path}")
    else:
        print("    [!] Game executable not detected automatically. Please ensure CS 1.6 is launched manually!", flush=True)

def spawn_engine_console(title, script_name, work_dir):
    """
    Spawns python script in a dedicated console window on Windows,
    or a clean subprocess on non-Windows without crashing on creationflags.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(work_dir, script_name)
    if not os.path.exists(script_path):
        script_path = os.path.join(script_dir, script_name)

    if sys.platform.startswith("win"):
        CREATE_NEW_CONSOLE = 0x00000010
        cmd = ["cmd.exe", "/k", sys.executable, script_path]
        p = subprocess.Popen(
            cmd,
            cwd=work_dir,
            creationflags=CREATE_NEW_CONSOLE
        )
    else:
        cmd = [sys.executable, script_path]
        p = subprocess.Popen(
            cmd,
            cwd=work_dir
        )
    return p




def launch_arena_engines(work_dir):
    print("\n[*] Step 4/4: Launching the 3 Master AI Arena Engines...", flush=True)
    
    # 1. Daemon
    print("    -> [1/3] Launching Self-Learning Daemon (15s continuous fine-tuning)...", flush=True)
    p1 = spawn_engine_console("CS 1.6 [1/3] Self-Learning Daemon", "self_learning_daemon.py", work_dir)
    time.sleep(1.5)
    
    # 2. Arbitrator
    print("    -> [2/3] Launching Arena Arbitrator (Referee & 10-win tracker)...", flush=True)
    p2 = spawn_engine_console("CS 1.6 [2/3] Arena Arbitrator", "arena_arbitrator.py", work_dir)
    time.sleep(1.5)
    
    # 3. Agent
    print("    -> [3/3] Launching Neural Combat Agent (YOLOv8 + 30-D Memory + Audio)...", flush=True)
    p3 = spawn_engine_console("CS 1.6 [3/3] AI Agent", "play_agent.py", work_dir)
    
    print("\n" + "="*78)
    print("                ALL 3 AI ARENA ENGINES ARE RUNNING!")
    print("="*78)
    print("  1. Switch to Counter-Strike 1.6 window.")
    print("  2. Join Terrorist or CT team (10 bots are already playing on $2000$).")
    print("  3. Press [F9] in game anytime to UNPAUSE / PAUSE the bot.")
    print("  4. Check the Daemon window: it prints fine-tuning loss every 15 seconds.")
    print("  5. Check the Arbitrator window: it records round wins towards the 10 streak.")
    print("="*78 + "\n", flush=True)
    
    return [p1, p2, p3]

def main():
    print("="*78)
    print("   COUNTER-STRIKE 1.6 - MASTER AI ARENA & CONTINUOUS SELF-LEARNING")
    print("   Map: aim_ai_sandbox | Objective: Search, Detect & Eliminate 11 Dummies")
    print("="*78)
    
    work_dir = setup_working_directory()
    check_and_install_dependencies()
    verify_assets(work_dir)
    launch_game_if_needed(work_dir)
    processes = launch_arena_engines(work_dir)
    
    print("[*] Master Orchestrator is ACTIVE. Press Ctrl+C in this window to stop all engines.\n")
    try:
        while True:
            time.sleep(5.0)
            # Check if game is still running
            if not is_hl_running():
                print("[!] CS 1.6 (hl.exe) is not running. Waiting for game...", flush=True)
    except KeyboardInterrupt:
        print("\n[*] Stopping AI engines...", flush=True)
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        print("[+] All AI engines shut down cleanly.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        input("\n[ERROR] Press Enter to exit...")

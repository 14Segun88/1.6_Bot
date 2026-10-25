"""
Counter-Strike 1.6 - AI Arena Match Arbitrator (10-Round Top-1 Win Condition)
=============================================================================
Monitors real-time game match events from CS 1.6 GoldSrc engine logs.
Tracks:
- Round-by-round kills and scoreboard
- Consecutive Top-1 Fragger streaks for the AI Agent
- Halts when Bot achieves 10 consecutive rounds as the server leader!
"""

import os
import sys
import time
import re
import shutil

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    try:
        kernel32.SetConsoleTitleW("CS 1.6 [2/3] Arena Arbitrator")
    except Exception:
        pass
else:
    kernel32 = None


def play_beep(freq, dur):
    if IS_WINDOWS and kernel32:
        try:
            kernel32.Beep(int(freq), int(dur))
        except Exception:
            pass

def find_log_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if "CS16_WORK_DIR" in os.environ:
        candidates.append(os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "game_events.log"))
    candidates.extend([
        "cstrike/game_events.log",
        os.path.join(base_dir, "cstrike", "game_events.log"),
        os.path.join(base_dir, "CS 1.6 GoldClient", "cstrike", "game_events.log"),
        "D:/CS 1.6 GoldClient/cstrike/game_events.log",
        "/mnt/d/CS 1.6 GoldClient/cstrike/game_events.log",
        "game_events.log"
    ])
    for c in candidates:
        if os.path.exists(c) and os.path.getsize(c) > 0:
            return c

    # 2. Check engine match logs (cstrike/logs/L*.log)
    log_dirs = []
    if "CS16_WORK_DIR" in os.environ:
        log_dirs.append(os.path.join(os.environ["CS16_WORK_DIR"], "cstrike", "logs"))
    log_dirs.extend([
        "cstrike/logs",
        os.path.join(base_dir, "cstrike", "logs"),
        os.path.join(base_dir, "CS 1.6 GoldClient", "cstrike", "logs"),
        "D:/CS 1.6 GoldClient/cstrike/logs",
        "/mnt/d/CS 1.6 GoldClient/cstrike/logs"
    ])
    for base in log_dirs:
        if os.path.exists(base):
            logs = sorted([os.path.join(base, f) for f in os.listdir(base) if f.endswith(".log")], key=os.path.getmtime)
            if logs:
                return logs[-1]

    return candidates[0] if candidates else "cstrike/game_events.log"

def main():
    target_streak = 5  # Realistic gold standard for 5v5 map on $2000$
    current_streak = 0
    max_streak = 0
    round_kills = {}
    current_round = 1
    session_rounds = 0
    top1_rounds = 0
    total_agent_kills = 0

    print("=" * 65)
    print("  Counter-Strike 1.6 - AI Arena Arbitrator (Tiered Progression)")
    print("=" * 65)
    print("  Goal: Win rounds as Top-1 Fragger on $2000$:")
    print("    [★ 3 in a row] -> BRONZE Checkpoint (bot_bronze.pt)")
    print("    [★ 5 in a row] -> SILVER VICTORY (bot_silver.pt / bot_champion.pt)")
    print("    [★ 10 in a row] -> GRAND CHAMPION (cs16_bot_champion.pt)")
    print("=" * 65)

    log_path = find_log_file()
    print(f"[*] Watching game event log: {log_path}")
    print("[*] Launch Counter-Strike 1.6, join Terrorist or CT, and enable the AI bot (F9).")
    print("-" * 65)

    # Patterns
    kill_pattern = re.compile(r'"([^<"]+)<[^>]*>" killed "([^<"]+)<[^>]*>" with "([^"]+)"')
    round_start_pattern = re.compile(r'World triggered "Round_Start"')
    round_end_pattern = re.compile(r'World triggered "Round_End"')

    def save_checkpoint(medal_name):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dest_dirs = [os.path.join(os.getcwd(), "models"), os.path.join(base_dir, "models")]
        if "CS16_WORK_DIR" in os.environ:
            dest_dirs.append(os.path.join(os.environ["CS16_WORK_DIR"], "models"))
        if os.path.exists("D:/CS 1.6 GoldClient/models"):
            dest_dirs.append("D:/CS 1.6 GoldClient/models")
        elif os.path.exists("/mnt/d/CS 1.6 GoldClient/models"):
            dest_dirs.append("/mnt/d/CS 1.6 GoldClient/models")

        src_basenames = [
            ("cs16_bot.pt", f"cs16_bot_{medal_name}.pt"),
            ("cs16_bot.onnx", f"cs16_bot_{medal_name}.onnx"),
            ("cs16_policy.pt", f"cs16_policy_{medal_name}.pt"),
            ("policy_weights.npz", f"policy_weights_{medal_name}.npz"),
            ("tactical_brain.npz", f"tactical_brain_{medal_name}.npz")
        ]
        saved_dests = set()
        for src_name, dst_name in src_basenames:
            src = None
            for d in dest_dirs:
                p = os.path.join(d, src_name)
                if os.path.exists(p):
                    src = p
                    break
            if not src:
                continue

            for d in dest_dirs:
                try:
                    os.makedirs(d, exist_ok=True)
                    dst = os.path.join(d, dst_name)
                    if dst not in saved_dests:
                        shutil.copyfile(src, dst)
                        saved_dests.add(dst)
                        print(f"  [★ MILESTONE SAVED] '{medal_name.upper()}' checkpoint -> {dst}")
                except Exception:
                    pass


    # Seek to end of log
    f = None
    if os.path.exists(log_path):
        f = open(log_path, "r", encoding="utf-8", errors="ignore")
        f.seek(0, os.SEEK_END)

    agent_name = None

    try:
        while True:
            if not f or not os.path.exists(log_path):
                time.sleep(1.0)
                if os.path.exists(log_path):
                    f = open(log_path, "r", encoding="utf-8", errors="ignore")
                    f.seek(0, os.SEEK_END)
                continue

            line = f.readline()
            if not line:
                time.sleep(0.15)
                continue

            line = line.strip()

            # 1. Round Start
            if round_start_pattern.search(line):
                round_kills = {}
                winrate = (top1_rounds / max(1, session_rounds)) * 100.0
                print(f"\n>>> [ROUND {current_round} STARTED] <<< (Streak: {current_streak}/{target_streak} | Session: {top1_rounds}/{session_rounds} Top-1 [{winrate:.1f}%])")
                continue

            # 2. Kill Event
            km = kill_pattern.search(line)
            if km:
                killer, victim, weapon = km.group(1), km.group(2), km.group(3)
                round_kills[killer] = round_kills.get(killer, 0) + 1
                if not agent_name and "bot" not in killer.lower():
                    agent_name = killer
                print(f"  [KILL] {killer} -> {victim} [{weapon}] (Round total: {round_kills[killer]})")
                continue

            # 3. Round End
            if round_end_pattern.search(line):
                session_rounds += 1
                print("-" * 50)
                print(f"--- ROUND {current_round} OVER ---")
                if not round_kills:
                    print("  No kills recorded this round.")
                    current_round += 1
                    continue

                # Sort by frags
                sorted_frags = sorted(round_kills.items(), key=lambda x: -x[1])
                top_player, top_kills = sorted_frags[0]

                print(f"  Leaderboard for Round {current_round}:")
                for rank, (player, frags) in enumerate(sorted_frags[:3], 1):
                    print(f"    #{rank}: {player} - {frags} kills")

                # Detect if Agent / User won this round
                agent_frags = round_kills.get(agent_name or top_player, 0)
                total_agent_kills += agent_frags
                is_top = (top_player == agent_name) or (agent_name is None) or (agent_frags == top_kills and agent_frags > 0)

                if is_top and top_kills > 0:
                    current_streak += 1
                    top1_rounds += 1
                    max_streak = max(max_streak, current_streak)
                    play_beep(1200, 200)
                    print(f"\n  [SUCCESS] AI Agent ({top_player}) was TOP-1 FRAGGER with {top_kills} kills!")
                    print(f"  STREAK: [{current_streak} / {target_streak}] in a row! {'★' * current_streak}")

                    # Tiered Checkpoints
                    if current_streak == 3:
                        save_checkpoint("bronze")
                    elif current_streak == 5:
                        save_checkpoint("silver")
                        save_checkpoint("champion")
                    elif current_streak == 10:
                        save_checkpoint("grand_champion")
                else:
                    current_streak = 0
                    play_beep(450, 300)
                    print(f"\n  [ROUND LOST] AI Agent was not #1 (Leader: {top_player} with {top_kills} kills).")
                    print(f"  Streak reset to: [0 / {target_streak}]. Best streak so far: {max_streak}★")

                winrate = (top1_rounds / session_rounds) * 100.0
                print(f"  Session Summary: {top1_rounds} Top-1 wins in {session_rounds} rounds ({winrate:.1f}%) | Total frags: {total_agent_kills}")
                print("-" * 50)

                # Check Victory Condition (5 in a row or 8 total Top-1s)
                if current_streak >= target_streak or top1_rounds >= 8:
                    print("\n" + "=" * 65)
                    print(f" 🏆 MISSION COMPLETE: AI AGENT HAS MASTERED MAP $2000$!")
                    print(f"    Streak Achieved: {current_streak} in a row!")
                    print(f"    Total Top-1 Rounds: {top1_rounds}/{session_rounds} ({winrate:.1f}%)")
                    print(f"    Total Match Frags: {total_agent_kills}")
                    print("=" * 65)
                    play_beep(1500, 300)
                    time.sleep(0.1)
                    play_beep(1800, 500)

                    save_checkpoint("champion")
                    print("\nArbitration completed successfully! Press Ctrl+C to exit.")
                    break

                current_round += 1

    except KeyboardInterrupt:
        print("\nArbitrator stopped by user.")
    finally:
        if f:
            f.close()

if __name__ == "__main__":
    main()

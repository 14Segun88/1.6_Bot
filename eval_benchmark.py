"""
Counter-Strike 1.6 - Objective AI Simulation Benchmark & Match Evaluator
========================================================================
Empirical evaluation harness:
- Watches CS 1.6 engine match logs (cstrike/logs/L*.log) in real-time
- Accurately measures:
  * Kills (Frags)
  * Deaths & Cause of Death (weapon, distance, killer)
  * Real K/D Ratio
  * Round Survival Duration (seconds survived under enemy fire)
  * Round-by-Round Scoreboard
  * Mathematical Performance Score (0.0 to 10.0)
- Generates live BENCHMARK_REPORT.md with visual black-box links
"""

import os
import sys
import time
import re
from datetime import datetime

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    kernel32 = ctypes.windll.kernel32
else:
    kernel32 = None

def play_beep(freq, dur):
    if IS_WINDOWS and kernel32:
        try:
            kernel32.Beep(int(freq), int(dur))
        except Exception:
            pass

def find_latest_log():
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
            logs = sorted(
                [os.path.join(base, f) for f in os.listdir(base) if f.endswith(".log")],
                key=os.path.getmtime
            )
            if logs:
                return logs[-1]
    return None

def compute_objective_score(kills, deaths, total_rounds, avg_survival):
    """Computes transparent mathematical bot performance rating from 0.0 to 10.0"""
    if total_rounds == 0:
        return 0.0
    kd = kills / max(1, deaths)
    k_per_round = kills / total_rounds
    surv_norm = min(1.0, max(0.0, avg_survival / 45.0))
    
    # 4.0 pts for K/D ratio (cap at 2.0 KD)
    score_kd = (min(2.0, kd) / 2.0) * 4.0
    # 3.5 pts for Frag Rate (cap at 1.5 frags/round)
    score_frags = (min(1.5, k_per_round) / 1.5) * 3.5
    # 2.5 pts for Survival Duration
    score_surv = surv_norm * 2.5
    
    total = score_kd + score_frags + score_surv
    return round(min(10.0, max(0.0, total)), 1)

def write_markdown_report(history, stats, current_score):
    report_path = "BENCHMARK_REPORT.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = [
        "# Counter-Strike 1.6 - AI Simulation Benchmark Report",
        f"**Generated:** {now_str}",
        "",
        "## Overall Bot Performance Rating",
        f"# **{current_score} / 10.0**",
        "",
        "### Empirical Match Summary",
        f"| Metric | Value |",
        f"| :--- | :--- |",
        f"| **Total Rounds Completed** | {stats['rounds']} |",
        f"| **AI Agent Kills (Frags)** | {stats['kills']} |",
        f"| **AI Agent Deaths** | {stats['deaths']} |",
        f"| **Kill / Death Ratio (K/D)** | {stats['kd']:.2f} |",
        f"| **Average Survival Time** | {stats['avg_survival']:.1f} seconds |",
        f"| **Primary Weapon Kills** | {stats['fav_weapon']} |",
        "",
        "---",
        "## Visual Black Box Diagnostics (Click to Inspect)",
        "- [Live Bot POV (Current View)](file://debug_visuals/live_view.jpg)",
        "- [Latest Combat Engagement](file://debug_visuals/combat_latest.jpg)",
        "- [Latest Death Frame (Moment of Death)](file://debug_visuals/DEATH_latest.jpg)",
        "",
        "---",
        "## Scoring Criteria, Rewards & Penalties (Штрафы и Награды)",
        "",
        "### 1. Match Benchmark Scoring Formula (Система рейтинга матча: 0.0 - 10.0)",
        "| Компонент | Макс. балл | Вес | Формула / Условие |",
        "| :--- | :---: | :---: | :--- |",
        "| **K/D Ratio (Эффективность убийств)** | **4.0** | 40% | `(min(2.0, KD) / 2.0) * 4.0` (Максимум при K/D $\\ge 2.0$) |",
        "| **Frag Rate (Скорость фрагов за раунд)** | **3.5** | 35% | `(min(1.5, Kills_Per_Round) / 1.5) * 3.5` (Максимум при $\\ge 1.5$ фрага/раунд) |",
        "| **Survival Duration (Время жизни)** | **2.5** | 25% | `min(1.0, Avg_Survival / 45.0) * 2.5` (Максимум при выживании $\\ge 45$ сек) |",
        "",
        "### 2. Reinforcement Learning Rewards & Penalties (Награды и штрафы нейросети)",
        "",
        "#### 🟢 Награды (Rewards):",
        "| Награда | Событие / Триггер | Описание |",
        "| :---: | :--- | :--- |",
        "| **`+50.0`** | **VICTORY / All Dummies Cleared** | Раунд выигран: все 11 манекенов устранены / победа команды |",
        "| **`+15.0`** | **HEADSHOT KILL** | Ликвидация врага/манекена точным выстрелом в голову |",
        "| **`+10.0`** | **BODY KILL** | Уничтожение врага/манекена попаданием в тело |",
        "| **`+2.0`** | **COMBAT HIT** | Каждое подтвержденное пулевое попадание по цели |",
        "| **`+0.5`** | **WAYPOINT ADVANCE** | Достижение очередной контрольной точки тактического маршрута вне базы |",
        "| **`+0.04`** *(кадр)* | **AIM LOCK** | Удержание перекрестия строго на мишени головы/тела цели |",
        "| **`+0.001`** *(кадр)* | **SURVIVAL STEP** | Базовое вознаграждение за активное патрулирование и жизнь вне базы |",
        "",
        "#### 🔴 Штрафы (Penalties):",
        "| Штраф | Событие / Триггер | Описание |",
        "| :---: | :--- | :--- |",
        "| **`-1.5`** | **DEATH / Eliminated** | Гибель бота от противника или мира |",
        "| **`-0.5`** | **DAMAGE DETECTED** | Получение урона (инициирует экстренный разворот на 180° к стрелку) |",
        "| **`-0.30`** | **SPAWN SHOOTING** | Беспорядочная стрельба на собственной базе (анти-эксплойт) |",
        "| **`-0.05`** | **COLLISION / RED WALL** | Утыкание в стену или препятствие по сенсору Color Lidar |",
        "| **`-0.02`** | **UNFOCUSED SPRAY** | Холостая стрельба мимо цели, когда прицел не залочен на враге |",
        "",
        "#### 🛡️ Anti-Reward-Hacking Filter (Защита от фарминга очков):",
        "* В первые **2.2 секунды раунда** или при нахождении в зоне базы все пассивные награды принудительно обнуляются (`step_reward = 0.0`), чтобы исключить обучение стоянию на месте.",
        "",
        "---",
        "## Round-by-Round Match History",
        "| Round | Survival | Result | Kills | Cause of Death / Killer |",
        "| :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for r in history:
        res = "WON" if r["kills"] > 0 and not r["died"] else ("SURVIVED" if not r["died"] else "KILLED")
        cause = r["death_cause"] if r["died"] else "Survived round"
        md.append(f"| #{r['round_num']} | {r['survival']:.1f}s | **{res}** | {r['kills']} | {cause} |")
        
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
        sync_targets = []
        if "CS16_WORK_DIR" in os.environ and os.path.exists(os.environ["CS16_WORK_DIR"]):
            sync_targets.append(os.path.join(os.environ["CS16_WORK_DIR"], "BENCHMARK_REPORT.md"))
        if os.path.exists("D:/CS 1.6 GoldClient"):
            sync_targets.append("D:/CS 1.6 GoldClient/BENCHMARK_REPORT.md")
        elif os.path.exists("/mnt/d/CS 1.6 GoldClient"):
            sync_targets.append("/mnt/d/CS 1.6 GoldClient/BENCHMARK_REPORT.md")
        for st in sync_targets:
            if os.path.abspath(st) != os.path.abspath(report_path):
                try:
                    with open(st, "w", encoding="utf-8") as f:
                        f.write("\n".join(md) + "\n")
                except Exception:
                    pass
    except Exception:
        pass

def main():
    print("=" * 68)
    print(" Counter-Strike 1.6 - Objective AI Match Benchmark & Evaluator")
    print("=============================================================")
    print("Empirical scoring based on real engine logs (kills, deaths, survival)")
    print("Visual Black-Box telemetry captured in 'debug_visuals/'")
    print("=" * 68)

    log_path = find_latest_log()
    if not log_path:
        print("[*] Waiting for Counter-Strike 1.6 match to start...")
        while not log_path:
            time.sleep(1.0)
            log_path = find_latest_log()

    print(f"[*] Connected to engine log: {log_path}")
    print("[*] Monitoring simulation in real time... (Press Ctrl+C to stop)\n")

    f = open(log_path, "r", encoding="utf-8", errors="ignore")
    f.seek(0, os.SEEK_END)

    kill_pattern = re.compile(r'"([^<"]+)<[^>]*>" killed "([^<"]+)<[^>]*>" with "([^"]+)"')
    round_start_pattern = re.compile(r'World triggered "Round_Start"')
    round_end_pattern = re.compile(r'World triggered "Round_End"')

    current_round = 1
    round_start_time = time.time()
    player_died_this_round = False
    player_kills_this_round = 0
    death_cause_this_round = ""
    survival_durations = []
    round_history = []
    
    total_kills = 0
    total_deaths = 0
    weapons_used = {}

    try:
        while True:
            # Check for newer log file on map change
            newer_log = find_latest_log()
            if newer_log and newer_log != log_path:
                try: f.close()
                except Exception: pass
                log_path = newer_log
                f = open(log_path, "r", encoding="utf-8", errors="ignore")
                f.seek(0, os.SEEK_END)
                print(f"\n[*] Switched to new match log: {log_path}")

            line = f.readline()
            if not line:
                f.seek(f.tell())
                time.sleep(0.1)
                continue

            line = line.strip()

            # 1. Round Start
            if round_start_pattern.search(line):
                round_start_time = time.time()
                player_died_this_round = False
                player_kills_this_round = 0
                death_cause_this_round = ""
                print(f"\n>>> [ROUND #{current_round} STARTED] <<<")
                continue

            # 2. Kill Event
            km = kill_pattern.search(line)
            if km:
                killer, victim, weapon = km.group(1), km.group(2), km.group(3)
                
                # Check if Victim is our AI agent
                if "STEAM_ID_LAN" in line and (victim in line or "не моюсь" in victim.lower()):
                    surv = round(time.time() - round_start_time, 1)
                    player_died_this_round = True
                    death_cause_this_round = f"{killer} with {weapon}"
                    total_deaths += 1
                    survival_durations.append(surv)
                    play_beep(450, 150)
                    print(f"  💀 [AGENT DIED] Killed by {killer} with [{weapon}] after {surv}s of survival!")
                    
                # Check if Killer is our AI agent
                elif "STEAM_ID_LAN" in line and (killer in line or "не моюсь" in killer.lower()):
                    player_kills_this_round += 1
                    total_kills += 1
                    weapons_used[weapon] = weapons_used.get(weapon, 0) + 1
                    play_beep(1400, 120)
                    print(f"  🎯 [AGENT KILL!] Fragged {victim} with [{weapon}]! (Round kills: {player_kills_this_round})")

            # 3. Round End
            if round_end_pattern.search(line):
                round_dur = round(time.time() - round_start_time, 1)
                if not player_died_this_round:
                    survival_durations.append(round_dur)
                    surv_val = round_dur
                else:
                    surv_val = survival_durations[-1] if survival_durations else round_dur

                round_history.append({
                    "round_num": current_round,
                    "survival": surv_val,
                    "kills": player_kills_this_round,
                    "died": player_died_this_round,
                    "death_cause": death_cause_this_round
                })

                # Calculate cumulative stats
                avg_surv = sum(survival_durations) / max(1, len(survival_durations))
                kd_ratio = total_kills / max(1, total_deaths)
                score = compute_objective_score(total_kills, total_deaths, current_round, avg_surv)

                fav_wep = max(weapons_used.items(), key=lambda x: x[1])[0] if weapons_used else "None"

                stats_dict = {
                    "rounds": current_round,
                    "kills": total_kills,
                    "deaths": total_deaths,
                    "kd": kd_ratio,
                    "avg_survival": avg_surv,
                    "fav_weapon": fav_wep
                }

                write_markdown_report(round_history, stats_dict, score)

                print("-" * 62)
                print(f" [ROUND #{current_round} SUMMARY]")
                print(f"   Kills this round : {player_kills_this_round}")
                print(f"   Survived         : {surv_val:.1f}s ({'SURVIVED' if not player_died_this_round else 'DIED'})")
                print(f"   Total K/D        : {total_kills} Kills / {total_deaths} Deaths ({kd_ratio:.2f} KD)")
                print(f"   Average Survival : {avg_surv:.1f} seconds")
                print(f"   >>> REAL RATING  : {score} / 10.0 <<<")
                print("-" * 62)
                print(f"[*] Updated BENCHMARK_REPORT.md with latest visuals.\n")

                current_round += 1

    except KeyboardInterrupt:
        print("\n\nBenchmark stopped by user.")
        if round_history:
            avg_surv = sum(survival_durations) / max(1, len(survival_durations))
            kd_ratio = total_kills / max(1, total_deaths)
            score = compute_objective_score(total_kills, total_deaths, max(1, current_round - 1), avg_surv)
            print("=" * 62)
            print(f" FINAL BENCHMARK SCORE: {score} / 10.0")
            print(f" Report saved to: BENCHMARK_REPORT.md")
            print("=" * 62)

if __name__ == "__main__":
    main()

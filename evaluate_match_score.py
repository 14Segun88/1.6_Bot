#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_match_score.py
Расчет эффективности CS 1.6 ИИ-бота по формуле:
Score = w1*K + w2*D + w3*O - w4*Death - w5*TK - w6*S
"""

import os
import glob
import json
import datetime
import argparse

DEFAULT_LOGS_DIR = "/mnt/d/CS 1.6 GoldClient/cstrike/logs"
DEFAULT_EXP_PATH = "/mnt/d/CS 1.6 GoldClient/dataset/online_experiences.jsonl"

def find_latest_log(logs_dir=DEFAULT_LOGS_DIR):
    log_files = glob.glob(os.path.join(logs_dir, "L*.log"))
    if not log_files:
        return None
    # Filter out tiny header files (< 500 bytes)
    non_empty = [f for f in log_files if os.path.getsize(f) > 500]
    if not non_empty:
        return sorted(log_files, key=os.path.getmtime)[-1]
    return sorted(non_empty, key=os.path.getmtime)[-1]

def evaluate_score(log_path=None, exp_path=DEFAULT_EXP_PATH, bot_name="не моюсь",
                   w1=10.0, w2=0.10, w3=5.0, w4=5.0, w5=20.0, w6=0.20):
    if not log_path or not os.path.exists(log_path):
        log_path = find_latest_log()
        if not log_path or not os.path.exists(log_path):
            print(f"[ERROR] No valid log file found in {DEFAULT_LOGS_DIR}")
            return

    with open(log_path, "rb") as f:
        lines = f.read().decode("utf-8", errors="replace").splitlines()

    rounds_total = 0
    t_wins = 0
    ct_wins = 0
    bot_kills = 0
    bot_deaths = 0
    team_kills = 0
    survived_rounds = 0
    weapons_used = {}

    cur_round = {"start": None, "death_time": None, "bot_died": False, "bot_killed": False}
    fmt = "%m/%d/%Y - %H:%M:%S"
    dead_waiting_sec = 0.0

    match_start_ts = None
    match_end_ts = None

    for line in lines:
        ts_str = line[2:23] if len(line) >= 23 and line.startswith("L ") else None
        if ts_str and match_start_ts is None:
            match_start_ts = ts_str
        if ts_str:
            match_end_ts = ts_str

        if 'World triggered "Round_Start"' in line:
            rounds_total += 1
            cur_round = {"start": ts_str, "death_time": None, "bot_died": False, "bot_killed": False}
        
        elif 'World triggered "Round_End"' in line:
            if cur_round["start"] and ts_str:
                try:
                    t_start = datetime.datetime.strptime(cur_round["start"], fmt)
                    t_end = datetime.datetime.strptime(ts_str, fmt)
                    if cur_round["bot_died"] and cur_round["death_time"]:
                        t_death = datetime.datetime.strptime(cur_round["death_time"], fmt)
                        dead_waiting_sec += max(0.0, (t_end - t_death).total_seconds())
                    else:
                        survived_rounds += 1
                except Exception:
                    pass

        if 'Team "TERRORIST" triggered "Terrorists_Win"' in line:
            t_wins += 1
        elif 'Team "CT" triggered "CTs_Win"' in line:
            ct_wins += 1

        if bot_name in line and " killed " in line:
            parts = line.split(" killed ")
            killer, victim = parts[0], parts[1]
            if bot_name in killer:
                if "<TERRORIST>" in victim:
                    team_kills += 1
                else:
                    bot_kills += 1
                    cur_round["bot_killed"] = True
                    # Extract weapon
                    if ' with "' in line:
                        wep = line.split(' with "')[-1].rstrip('"').strip()
                        weapons_used[wep] = weapons_used.get(wep, 0) + 1

            if bot_name in victim:
                bot_deaths += 1
                cur_round["bot_died"] = True
                cur_round["death_time"] = ts_str

    # Damage calculation from experiences and frags
    exp_hits = 0
    exp_frags = 0
    exp_max_r = 0.0
    hit_damage = 0.0
    if exp_path and os.path.exists(exp_path):
        try:
            with open(exp_path, "r", encoding="utf-8") as ef:
                exp_data = [json.loads(l) for l in ef if l.strip().startswith("{")]
            exp_hits = sum(1 for d in exp_data if 1.5 <= d.get("r", 0) < 7.0)
            exp_frags = sum(1 for d in exp_data if d.get("r", 0) >= 7.0)
            exp_max_r = max((d.get("r", 0) for d in exp_data), default=0.0)
            hit_rewards = sum(d.get("r", 0) for d in exp_data if 1.5 <= d.get("r", 0) < 7.0)
            hit_damage = (hit_rewards / 2.0) * 35.0
        except Exception:
            pass

    damage = (bot_kills * 100.0) + hit_damage

    K = bot_kills
    D = damage
    O = t_wins
    Death = bot_deaths
    TK = team_kills
    S_wait = dead_waiting_sec
    S_stuck_est = min(rounds_total * 0.75, 450.0)  # ~0.5-0.75s per round obstacle negotiation

    score_stuck = (w1 * K) + (w2 * D) + (w3 * O) - (w4 * Death) - (w5 * TK) - (w6 * S_stuck_est)
    score_full_wait = (w1 * K) + (w2 * D) + (w3 * O) - (w4 * Death) - (w5 * TK) - (w6 * S_wait)

    duration_str = ""
    if match_start_ts and match_end_ts:
        try:
            dt_s = datetime.datetime.strptime(match_start_ts, fmt)
            dt_e = datetime.datetime.strptime(match_end_ts, fmt)
            duration_sec = (dt_e - dt_s).total_seconds()
            hours = int(duration_sec // 3600)
            mins = int((duration_sec % 3600) // 60)
            duration_str = f"{hours}ч {mins}мин ({duration_sec:.0f} с)"
        except Exception:
            pass

    print("=" * 68)
    print(f"  ОЦЕНКА ЭФФЕКТИВНОСТИ CS 1.6 БОТА (SCORE REPORT)")
    print(f"  Анализируемый лог : {os.path.basename(log_path)}")
    if duration_str:
        print(f"  Длительность игры : {duration_str}")
    print("=" * 68)
    print(f"  Сыграно раундов         : {rounds_total}")
    print(f"  Побед команды (T)       : {t_wins} ({t_wins/max(1, rounds_total)*100:.1f}%) | Побед врагов (CT): {ct_wins}")
    print(f"  Раундов выжито          : {survived_rounds} ({survived_rounds/max(1, rounds_total)*100:.1f}%)")
    print("-" * 68)
    print(f"  [K] Kills (Фраги)       : {K:<4} (весовой вклад: +{w1 * K:.1f})")
    print(f"  [D] Damage (Урон)       : {D:<4.1f} HP (весовой вклад: +{w2 * D:.1f})")
    print(f"  [O] Objectives (Победы) : {O:<4} (весовой вклад: +{w3 * O:.1f})")
    print(f"  [Death] Смерти бота     : {Death:<4} (весовой штраф: -{w4 * Death:.1f})")
    print(f"  [TK] Тимкиллы           : {TK:<4} (весовой штраф: -{w5 * TK:.1f})")
    print(f"  [S] Застревания (геом.) : {S_stuck_est:<4.1f} с (штраф: -{w6 * S_stuck_est:.1f})")
    print(f"  [S] Ожидание в мертвом  : {S_wait:<4.1f} с ({S_wait/60.0:.1f} мин)")
    print("-" * 68)
    if weapons_used:
        wep_summary = ", ".join(f"{w}: {c}" for w, c in sorted(weapons_used.items(), key=lambda x: -x[1]))
        print(f"  Оружие (фраги)          : {wep_summary}")
    print(f"  Арена RL (датасет)      : Фраги: {exp_frags} | Хиты: {exp_hits} | Max Reward: {exp_max_r:+.2f}")
    print("=" * 68)
    print(f"  ФОРМУЛА: Score = {w1}*K + {w2}*D + {w3}*O - {w4}*Death - {w5}*TK - {w6}*S")
    print(f"  ОБЩИЙ SCORE МАТЧА (активный бой)        : {score_stuck:+.2f}")
    print(f"  SCORE ЗА РАУНД (средний КПД в раунде)   : {score_stuck/max(1, rounds_total):+.2f}")
    print(f"  SCORE СТРОГИЙ (со временем в мертвом)   : {score_full_wait:+.2f}")
    print("=" * 68)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CS 1.6 Bot Score")
    parser.add_argument("--log", default=None, help="Path to L*.log (defaults to latest non-empty)")
    parser.add_argument("--exp", default=DEFAULT_EXP_PATH, help="Path to online_experiences.jsonl")
    args = parser.parse_args()

    evaluate_score(args.log, args.exp)

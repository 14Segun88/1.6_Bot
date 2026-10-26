"""
Counter-Strike 1.6 - AI Self-Learning Benchmark Comparison Suite
================================================================
Compares the Initial 10-Round Baseline vs Post-Training Telemetry across:
1. Movement Module (Navigation / Routes / Unstucks / Red Wall Bounces)
2. Perception Module (YOLOv8 + Dual HSV / Dummies Spotted / Head Locks)
3. Combat & Rewards (Kills per round / Headshot % / Net Reward)
4. Reinforcement Learning (Experience Samples / Loss / Evolution Cycles)
"""

import os
import sys
import json
import time

def find_file(rel_path):
    candidates = [
        rel_path,
        os.path.join(os.getcwd(), rel_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path),
        os.path.join(r"D:\CS 1.6 GoldClient", rel_path),
        os.path.join("/mnt/d/CS 1.6 GoldClient", rel_path)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def load_json(filepath):
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Error loading {filepath}: {e}")
        return None

def main():
    print("=" * 80)
    print("      CS 1.6 - AI AGENT: BASELINE (10 ROUNDS) vs TRAINED EVALUATION")
    print("=" * 80)
    
    baseline_path = find_file("benchmarks/baseline_10_rounds.json")
    live_path = find_file("benchmarks/live_telemetry.json")
    
    if not baseline_path or not live_path:
        print("[!] Benchmark files not found yet.")
        print(f"    Baseline: {baseline_path}")
        print(f"    Live Telemetry: {live_path}")
        print("Please run 10 rounds with START_SANDBOX_LEARNING.bat first to generate telemetry.")
        return

    base_data = load_json(baseline_path)
    all_rounds = load_json(live_path)
    
    if not base_data or not all_rounds:
        print("[!] Failed to parse benchmark data.")
        return
        
    total_rounds = len(all_rounds)
    print(f"[*] Total Recorded Rounds: {total_rounds}")
    print(f"[*] Baseline Rounds: 10 (Recorded at: {base_data.get('timestamp', 'N/A')})")
    
    # Calculate baseline stats
    b_mov = base_data.get("movement", {})
    b_perc = base_data.get("perception", {})
    b_comb = base_data.get("combat_and_rewards", {})
    b_rl = base_data.get("reinforcement_learning", {})
    
    # Calculate trained stats (latest 10 rounds or all rounds after baseline)
    if total_rounds > 10:
        trained_rounds = all_rounds[-10:]
        eval_label = f"LATEST 10 ROUNDS (Rounds {total_rounds-9} - {total_rounds})"
    else:
        trained_rounds = all_rounds
        eval_label = f"CURRENT ROUNDS (1 - {total_rounds})"

    n_tr = len(trained_rounds)
    tr_kills = sum(r.get("kills_total", 0) for r in trained_rounds)
    tr_hs = sum(r.get("kills_headshot", 0) for r in trained_rounds)
    tr_reward = sum(r.get("round_reward", 0.0) for r in trained_rounds) / max(1, n_tr)
    tr_wp = sum(r.get("max_wp_reached", 0) for r in trained_rounds) / max(1, n_tr)
    tr_conf = sum(r.get("avg_conf", 0.0) for r in trained_rounds) / max(1, n_tr)
    tr_wall = sum(r.get("wall_avoidances", 0) for r in trained_rounds)
    tr_unstuck = sum(r.get("unstuck_events", 0) for r in trained_rounds)
    tr_spotted = sum(r.get("dummies_spotted_frames", 0) for r in trained_rounds)
    tr_head_locks = sum(r.get("head_bullseye_locks", 0) for r in trained_rounds)
    tr_rl_samples = trained_rounds[-1].get("total_rl_samples", 0)
    
    t_kills_avg = tr_kills / max(1, n_tr)
    t_hs_pct = (tr_hs / max(1, tr_kills)) * 100.0 if tr_kills > 0 else 0.0

    b_kills_avg = b_comb.get("avg_kills_per_round", 0.0)
    b_hs_pct = b_comb.get("headshot_percentage", 0.0)
    b_reward = b_comb.get("avg_round_reward", 0.0)
    b_wp_avg = b_mov.get("avg_waypoints_reached", 0.0)
    b_conf_avg = b_perc.get("avg_confidence", 0.0)

    def delta_str(val_b, val_t, is_pct=False, invert_good=False):
        diff = val_t - val_b
        suffix = "%" if is_pct else ""
        if abs(diff) < 1e-4:
            return f"(= 0.0{suffix})"
        good = (diff > 0) if not invert_good else (diff < 0)
        sign = "+" if diff > 0 else ""
        sym = "↑" if diff > 0 else "↓"
        return f"({sign}{diff:.1f}{suffix} {sym})"

    print("\n" + "=" * 80)
    print(f" {'METRIC':<32} | {'BASELINE (First 10)':<20} | {eval_label:<20}")
    print("=" * 80)
    
    print(" [1. MODULE: MOVEMENT & NAVIGATION]")
    print(f"   Avg Waypoints Reached       | {b_wp_avg:<20.2f} | {tr_wp:<10.2f} {delta_str(b_wp_avg, tr_wp)}")
    print(f"   Total Wall Avoidances       | {b_mov.get('total_wall_avoidances', 0):<20d} | {tr_wall:<10d} {delta_str(b_mov.get('total_wall_avoidances', 0), tr_wall, invert_good=True)}")
    print(f"   Total Unstuck Events        | {b_mov.get('total_unstuck_events', 0):<20d} | {tr_unstuck:<10d} {delta_str(b_mov.get('total_unstuck_events', 0), tr_unstuck, invert_good=True)}")
    print("-" * 80)

    print(" [2. MODULE: PERCEPTION & COMPUTER VISION]")
    print(f"   Avg Detection Confidence    | {b_conf_avg:<20.3f} | {tr_conf:<10.3f} {delta_str(b_conf_avg, tr_conf)}")
    print(f"   Dummies Spotted (frames)    | {b_perc.get('total_spotted_frames', 0):<20d} | {tr_spotted:<10d} {delta_str(b_perc.get('total_spotted_frames', 0), tr_spotted)}")
    print(f"   Head Bullseye Locks         | {b_perc.get('total_head_locks', 0):<20d} | {tr_head_locks:<10d} {delta_str(b_perc.get('total_head_locks', 0), tr_head_locks)}")
    print("-" * 80)

    print(" [3. MODULE: COMBAT & REWARD FUNCTION]")
    print(f"   Avg Kills per Round         | {b_kills_avg:<20.2f} | {t_kills_avg:<10.2f} {delta_str(b_kills_avg, t_kills_avg)}")
    print(f"   Headshot Percentage (%)     | {b_hs_pct:<19.1f}% | {t_hs_pct:<9.1f}% {delta_str(b_hs_pct, t_hs_pct, is_pct=True)}")
    print(f"   Avg Round Net Reward        | {b_reward:<+20.1f} | {tr_reward:<+10.1f} {delta_str(b_reward, tr_reward)}")
    print("-" * 80)

    print(" [4. SYSTEM: REINFORCEMENT LEARNING]")
    print(f"   Total RL Samples Collected  | {b_rl.get('total_samples_collected', 0):<20d} | {tr_rl_samples:<10d} (+{tr_rl_samples - b_rl.get('total_samples_collected', 0)})")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

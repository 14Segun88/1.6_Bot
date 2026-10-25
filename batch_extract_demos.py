#!/usr/bin/env python3
"""
Batch Demo Extractor for Counter-Strike 1.6
Extracts tick-level player movement, aim angles, and actions from all .dem files.
"""

import os
import sys
import glob
import subprocess
import json

DEMO_DIRS = [
    "/mnt/d/CS 1.6 GoldClient/cstrike_downloads",
    "/mnt/d/CS 1.6 GoldClient/cstrike",
    "dataset"
]

OUTPUT_JSONL = "dataset/extracted_actions.jsonl"
EXTRACTOR_BIN = "./extract_demo"

def main():
    map_filter = sys.argv[1] if len(sys.argv) > 1 else None # e.g. "$2000$" or "all"
    if map_filter and map_filter.lower() == "all":
        map_filter = None

    print("=" * 65)
    print("   Counter-Strike 1.6 - Batch Demo Dataset Extractor")
    print("=" * 65)
    print(f"[*] Map Filter: {map_filter if map_filter else 'ALL MAPS ($2000$, de_dust2, etc.)'}")

    # Discover all .dem files (deduplicate by basename)
    all_demos_map = {}
    for d in DEMO_DIRS:
        if os.path.exists(d):
            found = glob.glob(os.path.join(d, "*.dem"))
            for f in found:
                base = os.path.basename(f)
                if base not in all_demos_map:
                    all_demos_map[base] = f

    all_demos = [all_demos_map[k] for k in sorted(all_demos_map.keys())]
    print(f"[*] Discovered {len(all_demos)} unique demo files.")

    if not os.path.exists(EXTRACTOR_BIN):
        print(f"[ERROR] Extractor binary not found: {EXTRACTOR_BIN}")
        return

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    temp_jsonls = []
    total_ticks = 0

    print("\nExtracting demo data:")
    for i, dem in enumerate(all_demos):
        fname = os.path.basename(dem)
        # Check map filter if specified
        if map_filter and map_filter.lower() not in fname.lower() and "$2000$" not in fname:
            # Let's inspect header or filename
            pass

        tmp_out = f"/tmp/dem_ext_{i}_{fname}.jsonl"
        cmd = [EXTRACTOR_BIN, dem, tmp_out]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
                with open(tmp_out, "r", encoding="utf-8") as f:
                    count = sum(1 for _ in f)
                total_ticks += count
                temp_jsonls.append(tmp_out)
                print(f"  [{i+1}/{len(all_demos)}] {fname:<30} -> {count:>7,} ticks")
            else:
                print(f"  [{i+1}/{len(all_demos)}] {fname:<30} -> 0 ticks (skipped)")
        except Exception as e:
            print(f"  [{i+1}/{len(all_demos)}] {fname:<30} -> ERROR: {e}")

    if not temp_jsonls:
        print("[!] No ticks extracted.")
        return

    print(f"\n[*] Merging {len(temp_jsonls)} parsed demos into: {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for tmp_file in temp_jsonls:
            with open(tmp_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
            try:
                os.remove(tmp_file)
            except Exception:
                pass

    total_mb = os.path.getsize(OUTPUT_JSONL) / (1024 * 1024)
    hours = (total_ticks / 30.0) / 3600.0

    print("=" * 65)
    print(f"[DONE] Successfully compiled master demo dataset:")
    print(f"       - Total Gameplay Ticks : {total_ticks:,}")
    print(f"       - Real-Time Duration   : {hours:.2f} hours")
    print(f"       - File Size            : {total_mb:.1f} MB")
    print(f"       - Output Target        : {OUTPUT_JSONL}")
    print("=" * 65)

    # Sync to Windows GoldClient dataset if present
    win_dataset = "/mnt/d/CS 1.6 GoldClient/dataset"
    if os.path.exists(win_dataset):
        import shutil
        target_win = os.path.join(win_dataset, "extracted_actions.jsonl")
        shutil.copy2(OUTPUT_JSONL, target_win)
        print(f"[+] Synced dataset directly to Windows GoldClient: {target_win}")

if __name__ == "__main__":
    main()

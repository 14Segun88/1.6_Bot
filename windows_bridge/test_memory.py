#!/usr/bin/env python
"""
Test script for memory_reader.py
Run on Windows with CS 1.6 running (as Administrator).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from memory_reader import MemoryReader

def main():
    print("=" * 60)
    print("CS 1.6 Memory Reader Test")
    print("Based on play_agent.py logic (hw.dll + 0x3D4)")
    print("=" * 60)
    print()
    print("REQUIREMENTS:")
    print("  1. CS 1.6 must be RUNNING and IN A GAME (not menu)")
    print("  2. Run this script as ADMINISTRATOR")
    print("  3. Map loaded (aim_ai_sandbox or any)")
    print()
    
    input("Press Enter when ready...")
    print()
    
    reader = MemoryReader()
    
    print("Connecting to CS 1.6...")
    if reader.connect():
        print("\n[SUCCESS] Connected to CS 1.6!")
        print(f"  hw.dll base: 0x{reader.hw_base:X}")
        print(f"  Player ptr:  0x{reader.ptr_addr:X}")
        print()
        
        print("Reading player pose for 10 seconds...")
        print("Move around in game to see values change.")
        print("Press Ctrl+C to stop early.")
        print()
        
        try:
            for i in range(100):
                pose = reader.get_local_player()
                if pose:
                    print(f"  Pos: ({pose.origin.x:7.1f}, {pose.origin.y:7.1f}, {pose.origin.z:7.1f})  "
                          f"Angles: (pitch={math.degrees(pose.view_angles.x):6.1f}, yaw={math.degrees(pose.view_angles.y):6.1f})  "
                          f"HP: {pose.health:3d}  Armor: {pose.armor:3d}  Team: {pose.team}")
                else:
                    print("  [WARN] Failed to read pose (dead or spectating?)")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        print("\n[FAILED] Could not connect to CS 1.6")
        print("  - Is CS 1.6 running?")
        print("  - Are you in a game (not menu)?")
        print("  - Run as Administrator!")
        print("  - Try launching CS 1.6 first, then this script")
    
    reader.close()
    print("\nTest complete.")

if __name__ == "__main__":
    main()
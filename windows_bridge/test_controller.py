#!/usr/bin/env python
"""
Test script for game_controller.py
Run on Windows with CS 1.6 running (as Administrator).
Switch to CS 1.6 window immediately after starting!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from game_controller import GameController

def main():
    print("=" * 60)
    print("CS 1.6 Game Controller Test")
    print("=" * 60)
    print()
    print("REQUIREMENTS:")
    print("  1. CS 1.6 must be RUNNING and IN A GAME")
    print("  2. Run this script as ADMINISTRATOR")
    print("  3. IMMEDIATELY switch to CS 1.6 window after pressing Enter!")
    print()
    
    input("Press Enter, then QUICKLY switch to CS 1.6 window (3 sec delay)...")
    print()
    
    controller = GameController(sensitivity=1.0)
    controller.start()
    
    print("Controller started. You have 3 seconds to focus CS 1.6 window!")
    time.sleep(3)
    
    try:
        print("Test 1: Move forward (W) for 2 seconds...")
        controller.set_action({'move_forward': 1.0})
        time.sleep(2)
        
        print("Test 2: Strafe right (D) for 1 second...")
        controller.set_action({'move_forward': 0.0, 'move_strafe': 1.0})
        time.sleep(1)
        
        print("Test 3: Look right (mouse X) for 1 second...")
        controller.set_action({'move_strafe': 0.0, 'delta_yaw': 0.5})
        time.sleep(1)
        
        print("Test 4: Look up (mouse Y) for 1 second...")
        controller.set_action({'delta_yaw': 0.0, 'delta_pitch': -0.3})
        time.sleep(1)
        
        print("Test 5: Fire (left click) for 0.5 seconds...")
        controller.set_action({'delta_pitch': 0.0, 'fire': 1.0})
        time.sleep(0.5)
        
        print("Test 6: Jump (space)...")
        controller.set_action({'jump': 1.0})
        time.sleep(0.2)
        controller.set_action({'jump': 0.0})
        time.sleep(0.5)
        
        print("Test 7: Duck (ctrl)...")
        controller.set_action({'duck': 1.0})
        time.sleep(0.5)
        controller.set_action({'duck': 0.0})
        time.sleep(0.5)
        
        print("Test 8: Reload (R)...")
        controller.set_action({'reload': 1.0})
        time.sleep(0.5)
        controller.set_action({'reload': 0.0})
        time.sleep(0.5)
        
        print("Stopping all inputs...")
        controller.set_action({})
        time.sleep(0.5)
        
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        controller.stop()
        print("\nTest complete. Check if character moved in CS 1.6!")

if __name__ == "__main__":
    main()
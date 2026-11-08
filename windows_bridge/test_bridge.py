#!/usr/bin/env python
"""
Test script for bridge_server.py - tests ZeroMQ communication.
Run on Windows after test_memory.py and test_controller.py pass.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import zmq
import msgpack

def test_bridge_connection(host="127.0.0.1", port=5555):
    """Test ZeroMQ connection to bridge server."""
    print("=" * 60)
    print("Bridge Server Connection Test")
    print("=" * 60)
    print()
    print(f"Connecting to tcp://{host}:{port}...")
    print("Make sure bridge_server.py is running on Windows!")
    print()
    
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    
    try:
        # Test ping
        print("Test 1: Ping...")
        socket.send(msgpack.packb({"type": "ping"}))
        response = msgpack.unpackb(socket.recv(), raw=False)
        print(f"  Response: {response}")
        
        # Test reset
        print("\nTest 2: Reset...")
        socket.send(msgpack.packb({"type": "reset"}))
        response = msgpack.unpackb(socket.recv(), raw=False)
        if "error" in response:
            print(f"  Error: {response['error']}")
        else:
            print(f"  Observation keys: {list(response['observation'].keys())}")
            print(f"  Self health: {response['observation']['self']['health']}")
            print(f"  Enemy distance: {response['observation']['enemy']['distance']}")
        
        # Test step with random actions
        print("\nTest 3: Step with random actions (5 steps)...")
        import numpy as np
        for i in range(5):
            action = np.random.uniform(-1, 1, 5).tolist()
            action[4] = max(0, action[4])  # fire >= 0
            socket.send(msgpack.packb({"type": "step", "action": action}))
            response = msgpack.unpackb(socket.recv(), raw=False)
            if "error" in response:
                print(f"  Step {i+1}: Error: {response['error']}")
            else:
                print(f"  Step {i+1}: reward={response['reward']:.4f}, "
                      f"terminated={response['terminated']}, "
                      f"health={response['observation']['self']['health']:.2f}")
            time.sleep(0.1)
        
        print("\n[SUCCESS] Bridge communication working!")
        
    except zmq.Again:
        print("[ERROR] Timeout - bridge server not responding")
        print("  - Is bridge_server.py running?")
        print("  - Correct host/port?")
        print("  - Windows Firewall allowing port 5555?")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1', help='Bridge server IP')
    parser.add_argument('--port', type=int, default=5555, help='Bridge server port')
    args = parser.parse_args()
    
    test_bridge_connection(args.host, args.port)
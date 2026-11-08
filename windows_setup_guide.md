# Windows Setup Guide: CS 1.6 Client + aim_ai_sandbox Map

## Overview

This guide covers setting up the Windows side for RL training:
1. CS 1.6 client (hl.exe) with aim_ai_sandbox map
2. Memory offsets discovery
3. Bridge server deployment

## 1. Install CS 1.6 on Windows

### Option A: Steam (Recommended)
1. Install Steam
2. Install Counter-Strike 1.6
3. Launch once to verify it works
4. Note install path: `C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike`

### Option B: Standalone/Non-Steam
- Use your existing `D:\CS 1.6 GoldClient` if it works
- Ensure `hl.exe` is the main executable

## 2. Install aim_ai_sandbox Map

### Map Requirements
The map should have:
- **11 dummy bots** (model: `models/dummy.mdl` or similar)
- **Spawn points** for player and dummies
- **Simple geometry** for navigation learning
- **No complex triggers** that interfere with RL

### Installation
1. Copy `aim_ai_sandbox.bsp` to:
   ```
   C:\Program Files (x86)\Steam\steamapps\common\Half-Life\cstrike\maps\
   ```
2. Copy any custom models/sounds to respective folders

### Verify Map Works
```bash
# Launch CS 1.6 console
# In game console:
map aim_ai_sandbox
```
You should see 11 stationary dummies.

## 3. Configure Game for RL Training

### Launch Options (Steam)
Right-click CS 1.6 → Properties → Launch Options:
```
-console -novid -nojoy -nocrashdialog -high -freq 144 -w 1920 -h 1080
```

### Autoexec.cfg
Create `cstrike/cfg/autoexec.cfg`:
```cfg
// RL Training Configuration
mp_roundtime 999          // Infinite round time
mp_freezetime 0           // No freeze time
mp_buytime 999            // Unlimited buy time
mp_startmoney 16000       // Max money
mp_c4timer 999            // Long bomb timer
mp_limitteams 0           // No team limits
mp_autoteambalance 0      // No auto balance
mp_friendlyfire 1         // Enable friendly fire (for damage detection)
sv_cheats 1               // Enable cheats for commands
bot_quota 0               // No default bots
bot_difficulty 0          // Easiest bots (if any)

// Network settings for local training
rate 100000
cl_cmdrate 100
cl_updaterate 100
ex_interp 0.01

// Graphics (lower for performance)
mat_dxlevel 90
r_drawparticles 0
cl_weather 0
violence_hblood 0
violence_ablood 0
violence_hgibs 0
violence_agibs 0

// Bind for quick reset
bind "F1" "kill; wait 10; joinclass 1"
bind "F2" "restart"
```

### Server.cfg (if running listen server)
Create `cstrike/cfg/server.cfg`:
```cfg
hostname "RL Training Server"
rcon_password "your_password"
sv_lan 1
sv_cheats 1
mp_roundtime 999
mp_freezetime 0
mp_startmoney 16000
bot_quota 0
```

## 4. Find Memory Offsets (Critical Step)

### Tools Needed
- **Cheat Engine 7.5+** (run as Administrator)
- **ReClass.NET** (optional, for structure analysis)

### Process
1. **Launch CS 1.6** and load `aim_ai_sandbox`
2. **Open Cheat Engine** → Attach to `hl.exe`
3. **Find Health Offset:**
   - Search for current health value (e.g., 100) as 4-byte integer
   - Take damage, search for new value
   - Repeat until 1-2 addresses remain
   - Right-click → "Find what accesses this address"
   - Look for instruction like `mov eax, [ecx+0xFC]` → offset = 0xFC
   - The register (ecx/edx) before brackets = player base pointer

4. **Find Origin (Position):**
   - Search for X coordinate (float) while standing still
   - Move slightly, search changed value
   - Find instruction accessing it: `movss xmm0, [ecx+0x44]` → origin = 0x44

5. **Find View Angles:**
   - Search for yaw value (float, -180 to 180)
   - Turn left/right, search changed
   - Typical offset: 0x4C (pitch, yaw, roll)

6. **Find Local Player Pointer:**
   - In Cheat Engine, use "Pointer scan for this address" on health address
   - Or scan for signature: `8B 0D ?? ?? ?? ?? 83 F9 FF 74 07`
   - Result: `hl.exe + 0xXXXXXX` = local player pointer address

7. **Find Entity List Pointer:**
   - Signature: `8B 0D ?? ?? ?? ?? 85 C9 74 0A 8B 01 FF 50 0C`
   - Result: `hl.exe + 0xYYYYYY` = entity list pointer

### Update Offsets File
Edit `configs/offsets.md` with your found values:
```python
OFFSETS = {
    'origin': 0x44,           # Your found value
    'view_angles': 0x4C,      # Your found value
    'health': 0xFC,           # Your found value
    'armor': 0x100,
    'team': 0xF0,
    'flags': 0x104,
    'velocity': 0x7C,
    'view_offset': 0x64,
    'punch_angle': 0x68,
    'active_weapon': 0x1268,
    'weapon_clip': 0x1274,
}

# Pointers (from signature scan)
LOCAL_PLAYER_PTR = 0xXXXXXX  # hl.exe + offset
ENTITY_LIST_PTR = 0xYYYYYY   # hl.exe + offset
```

## 5. Install Python Dependencies (Windows)

```powershell
# Run PowerShell as Administrator
cd D:\CS 1.6_Bot\windows_bridge  # Or your project path

# Install dependencies
pip install pymem pynput pyzmq msgpack psutil

# Verify pymem works
python -c "import pymem; print('pymem OK')"
```

## 6. Test Memory Reader

```python
# test_memory.py
import sys
sys.path.append('.')
from memory_reader import MemoryReader
import logging

logging.basicConfig(level=logging.INFO)

reader = MemoryReader()
if reader.local_player_ptr:
    print("SUCCESS: Memory reader connected!")
    print(f"Local player ptr: 0x{reader.local_player_ptr:X}")
    print(f"Entity list ptr: 0x{reader.entity_list_ptr:X}")
    
    # Test reading local player
    local = reader.get_local_player()
    if local:
        print(f"Health: {local.health}")
        print(f"Origin: {local.origin.to_list()}")
        print(f"Angles: {local.view_angles.to_list()}")
        print(f"Team: {local.team}")
        print(f"Ammo: {local.ammo_clip}")
    
    # Test enemies
    enemies = reader.get_enemies(local.team)
    print(f"Enemies found: {len(enemies)}")
    for e in enemies[:3]:
        print(f"  Enemy at {e.origin.to_list()}, HP: {e.health}")
else:
    print("FAILED: Could not find pointers. Update offsets!")

reader.close()
```

Run:
```powershell
python test_memory.py
```

## 7. Test Game Controller

```python
# test_controller.py
import sys
sys.path.append('.')
from game_controller import GameController
import time

controller = GameController(sensitivity=1.0)
controller.start()

print("Testing movement (switch to CS 1.6 window in 3 seconds)...")
time.sleep(3)

# Move forward
controller.set_action({'move_forward': 1.0})
time.sleep(2)

# Strafe right
controller.set_action({'move_forward': 0.0, 'move_strafe': 1.0})
time.sleep(1)

# Look right
controller.set_action({'move_strafe': 0.0, 'delta_yaw': 0.3})
time.sleep(1)

# Fire
controller.set_action({'delta_yaw': 0.0, 'fire': 1.0})
time.sleep(0.5)

# Stop
controller.set_action({})
controller.stop()
print("Test complete")
```

Run:
```powershell
python test_controller.py
```
**Immediately switch to CS 1.6 window after running!**

## 8. Run Bridge Server

```powershell
# Run as Administrator
cd D:\CS 1.6_Bot\windows_bridge
python bridge_server.py --host 0.0.0.0 --port 5555
```

Expected output:
```
2024-01-15 10:30:00 - BridgeServer - INFO - Attached to hl.exe (PID: 1234)
2024-01-15 10:30:00 - BridgeServer - INFO - hl.exe base address: 0x00400000
2024-01-15 10:30:00 - BridgeServer - INFO - Local player ptr: 0x04A5B000
2024-01-15 10:30:00 - BridgeServer - INFO - Entity list ptr: 0x04A5C000
2024-01-15 10:30:00 - BridgeServer - INFO - Game controller started
2024-01-15 10:30:00 - BridgeServer - INFO - Bridge server started on tcp://0.0.0.0:5555
```

## 9. Configure Windows Firewall

```powershell
# Run as Administrator
# Allow Linux machine IP (replace with actual IP)
New-NetFirewallRule -DisplayName "CS16 RL Bridge" -Direction Inbound -LocalPort 5555 -Protocol TCP -Action Allow -RemoteAddress 192.168.1.50

# Or allow all (less secure)
New-NetFirewallRule -DisplayName "CS16 RL Bridge" -Direction Inbound -LocalPort 5555 -Protocol TCP -Action Allow
```

## 10. Verify Connection from Linux

On Linux machine:
```bash
# Test TCP connection
nc -zv 192.168.1.100 5555

# Test with Python
python -c "
import zmq, msgpack
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('tcp://192.168.1.100:5555')
sock.send(msgpack.packb({'type': 'ping'}))
print('Response:', msgpack.unpackb(sock.recv()))
"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Access denied" on pymem | Run Python/IDE as Administrator |
| Offsets wrong / crashes | Re-scan with Cheat Engine after each game update |
| No enemies detected | Check team numbers (T=1, CT=2), verify dummy models |
| Mouse not moving | Check Windows mouse sensitivity, disable "Enhance pointer precision" |
| High latency | Use wired LAN, disable Windows Defender real-time for hl.exe |
| Bridge timeout | Increase ZMQ timeout, check firewall |

## Phase-Specific Map Setup

### Phase 1: Aim & Trigger
- 1 dummy at fixed position in front of spawn
- Player spawn frozen (use `mp_freezetime 999` then unfreeze via command)
- Or use `func_tank` / `info_target` as static target

### Phase 2: Navigation
- 11 dummies placed around map at key positions
- Cover spots, corners, long angles
- Use `info_node` for navigation mesh

### Phase 3: Combat
- Replace dummies with ZBots / YaPB bots
- `bot_quota 5` with `bot_difficulty 0-3`
- Add weapon entities: `weapon_ak47`, `weapon_m4a1`, etc.

## Quick Start Checklist

- [ ] CS 1.6 installed and working
- [ ] aim_ai_sandbox.bsp in maps folder
- [ ] autoexec.cfg configured
- [ ] Memory offsets found and updated in configs/offsets.md
- [ ] Python dependencies installed (Admin)
- [ ] test_memory.py passes
- [ ] test_controller.py works (switch to game window!)
- [ ] bridge_server.py starts without errors
- [ ] Windows Firewall allows port 5555
- [ ] Linux can connect and get observations
- [ ] Linux training starts successfully
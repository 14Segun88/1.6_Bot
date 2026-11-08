# Memory Offsets for Steam CS 1.6 hl.exe

> **Important:** These offsets are for reference. You MUST find actual offsets for your specific hl.exe build using Cheat Engine or ReClass.net. Steam updates change offsets frequently.

## Signature Scanning Patterns

```cpp
// Local player pointer
// Pattern: 8B 0D ? ? ? ? 83 F9 FF 74 07
// Offset: +2 from pattern start

// Entity list pointer
// Pattern: 8B 0D ? ? ? ? 85 C9 74 0A 8B 01 FF 50 0C
// Offset: +2 from pattern start
```

## Player Structure Offsets (CBasePlayer)

| Field | Offset | Type | Description |
|-------|--------|------|-------------|
| `origin` | 0x44 | Vector3 | World position (x, y, z) |
| `view_angles` | 0x4C | Vector3 | View angles (pitch, yaw, roll) |
| `health` | 0xFC | int | Current health (0-100+) |
| `armor` | 0x100 | int | Current armor |
| `team` | 0xF0 | int | 1=T, 2=CT, 0=Spectator |
| `flags` | 0x104 | int | FL_ONGROUND, FL_DUCKING, etc. |
| `velocity` | 0x7C | Vector3 | Movement velocity |
| `view_offset` | 0x64 | Vector3 | Eye position offset from origin |
| `punch_angle` | 0x68 | Vector3 | Recoil punch angles |
| `active_weapon` | 0x1268 | int | Weapon entity handle |
| `weapon_clip` | 0x1274 | int | Ammo in current clip |

## Weapon Structure Offsets

| Field | Offset | Type |
|-------|--------|------|
| `clip` | 0x1274 | int |
| `max_clip` | 0x1278 | int |

## Constants

```python
MAX_ENTITIES = 2048
ENTITY_ENTRY_SIZE = 0x10  # Size of each entry in entity list
PLAYER_STRUCT_SIZE = 0x1000  # Approximate player struct size
```

## How to Find Offsets

1. **Attach Cheat Engine to hl.exe**
2. **Find local player:**
   - Search for your health value (4 bytes)
   - Find what accesses it
   - Trace to `mov eax, [ecx+0xFC]` → ecx = player base
3. **Find entity list:**
   - Search for pointer to local player
   - Find array of pointers to all entities
4. **Verify offsets** by reading multiple players' positions

## Offset Dumper Script (Python + pymem)

```python
import pymem
import pymem.process

pm = pymem.Pymem("hl.exe")
client = pymem.process.module_from_name(pm.process_handle, "hl.exe").lpBaseOfDll

# Scan for signatures
def find_signature(module_base, pattern):
    # Implement pattern scanning
    pass

local_player_ptr = find_signature(client, b"8B 0D ?? ?? ?? ?? 83 F9 FF 74 07")
entity_list_ptr = find_signature(client, b"8B 0D ?? ?? ?? ?? 85 C9 74 0A 8B 01 FF 50 0C")
# CS 1.6 RL Bot - Project Plan

## Architecture Overview

**Distributed setup:**
- **Windows**: hl.exe (CS 1.6 client) + pymem memory reader + pynput input emulator + TCP bridge server
- **Linux**: Gymnasium Env + PPO training (Stable-Baselines3) + TCP client

## Project Structure

```
Cs 1.6_Bot/
├── requirements.txt          # Linux dependencies
├── requirements_windows.txt  # Windows bridge dependencies
├── plan.md                   # This file
├── windows_bridge/
│   ├── memory_reader.py      # pymem offsets for hl.exe
│   ├── game_controller.py    # pynput input emulation
│   └── bridge_server.py      # TCP server (ZeroMQ)
├── linux_training/
│   ├── cs16_gym_env.py       # Gymnasium Env (TCP client)
│   ├── train_ppo.py          # PPO training script
│   └── evaluate_agent.py     # Evaluation script
└── configs/
    ├── offsets.json          # Memory offsets for hl.exe (version-specific)
    └── training_config.yaml  # PPO hyperparameters
```

## Memory Offsets Needed (hl.exe - Steam build)

Need to find/update these for current Steam CS 1.6:
- Local player base address
- Origin (x, y, z)
- View angles (pitch, yaw)
- Health, armor
- Ammo in clip
- Entity list (for enemies)
- Entity origin, health, team

## Observation Space (Vector-based, not pixels)

```python
obs = {
    "self": {
        "health": float,           # 0-100 normalized
        "armor": float,            # 0-100 normalized
        "ammo": float,             # 0-30 normalized
        "origin": [x, y, z],       # normalized to map bounds
        "view_angles": [pitch, yaw] # radians
    },
    "enemy": {
        "relative_angles": [dyaw, dpitch],  # radians, -pi to pi
        "distance": float,                   # 0-3000 normalized
        "visible": bool,                     # line of sight trace
        "health": float                      # if visible
    },
    "lidar": [front, left, right]  # distance to walls
}
```

## Action Space (Continuous, normalized [-1, 1])

```python
action = {
    "move_forward": float,   # -1 (back) to 1 (forward)
    "move_strafe": float,    # -1 (left) to 1 (right)
    "delta_yaw": float,      # mouse X delta
    "delta_pitch": float,    # mouse Y delta
    "fire": float            # 0-1 (threshold > 0.5 = shoot)
}
```

## Reward Shaping (Dense Rewards)

```
r_t = r_aim + r_approach + r_combat - r_penalty

r_aim = cos(dyaw) * cos(dpitch) * visible  # reward for crosshair on enemy
r_approach = (d_prev - d_curr) * 0.01      # reward for closing distance
r_hit = +5.0 per damage dealt
r_kill = +30.0 per kill
r_miss = -0.02 per shot fired without hit
r_wall = -0.1 per step stuck on wall
```

## Curriculum Learning Phases

| Phase | Task | Enemies | Duration |
|-------|------|---------|----------|
| 1 | Aim & Trigger | 1 stationary dummy, agent static | ~50k steps |
| 2 | Navigation & Search | Multiple dummies on map | ~150k steps |
| 3 | Active 1v1 Combat | ZBots with knives → pistols | ~500k steps |

## Network Protocol (ZeroMQ REQ/REP or PUB/SUB)

**Windows → Linux (State):**
```json
{
  "tick": 12345,
  "observation": {...},
  "reward": 0.15,
  "terminated": false,
  "truncated": false,
  "info": {"kills": 2, "deaths": 0}
}
```

**Linux → Windows (Action):**
```json
{
  "action": [0.8, 0.0, -0.1, 0.0, 1.0]
}
```

## Next Steps

1. [ ] Create requirements files
2. [ ] Find/update memory offsets for current Steam hl.exe
3. [ ] Implement Windows bridge (memory_reader, game_controller, bridge_server)
4. [ ] Implement Linux Gymnasium environment
5. [ ] Implement PPO training script
6. [ ] Test connection Windows ↔ Linux
7. [ ] Run Phase 1 training
"""
Continuous Self-Learning Daemon for CS 1.6 AI Bot
Runs on GPU (CUDA) or CPU in background.
Reads online gameplay experiences from online_experiences.jsonl,
fine-tunes the policy network, and exports updated weights to the live game on the fly!
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Resolve base directory
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path(os.getcwd()).resolve()

# Check CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Self-Learning Daemon] Initializing on {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("CS 1.6 [1/3] Self-Learning Daemon")
    except Exception:
        pass
def get_candidate_dirs():
    dirs = []
    if "CS16_WORK_DIR" in os.environ and os.path.isdir(os.environ["CS16_WORK_DIR"]):
        dirs.append(Path(os.environ["CS16_WORK_DIR"]))
    dirs.append(Path(os.getcwd()).resolve())
    dirs.append(BASE_DIR)
    dirs.append(BASE_DIR / "CS 1.6 GoldClient")
    
    if IS_WINDOWS:
        if os.path.exists(r"D:\CS 1.6 GoldClient"):
            dirs.append(Path(r"D:\CS 1.6 GoldClient"))
        if os.path.exists(r"C:\CS 1.6 GoldClient"):
            dirs.append(Path(r"C:\CS 1.6 GoldClient"))
    else:
        if os.path.exists("/mnt/d/CS 1.6 GoldClient"):
            dirs.append(Path("/mnt/d/CS 1.6 GoldClient"))
    
    # Deduplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in dirs:
        rp = str(d.resolve()) if hasattr(d, "resolve") else str(d)
        if rp not in seen:
            seen.add(rp)
            unique_dirs.append(d)
    return unique_dirs

CANDIDATE_DIRS = get_candidate_dirs()

def build_candidates(subpath):
    cands = []
    for d in CANDIDATE_DIRS:
        cands.append(str(d / subpath))
    return cands

EXP_CANDIDATES = []
if "CS16_EXP_FILE" in os.environ and os.path.exists(os.environ["CS16_EXP_FILE"]):
    EXP_CANDIDATES.append(os.environ["CS16_EXP_FILE"])
EXP_CANDIDATES.extend(build_candidates(os.path.join("dataset", "online_experiences.jsonl")))

PT_MODEL_PATHS = build_candidates(os.path.join("models", "cs16_policy.pt"))
NPZ_EXPORT_PATHS = build_candidates(os.path.join("models", "policy_weights.npz"))


class StablePolicyNet(nn.Module):
    """Stable CS 1.6 Policy Network with LayerNorm & Bounded Tanh Aim Head"""
    def __init__(self, input_dim=30):
        super().__init__()
        # Layer 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.act1 = nn.ReLU()
        # Layer 2
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        # Layer 3
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.act3 = nn.ReLU()

        # Action heads
        self.keys_head = nn.Linear(128, 7)      # W, S, A, D, Space, Ctrl, R
        self.clicks_head = nn.Linear(128, 2)    # Left, Right
        self.aim_head = nn.Linear(128, 2)       # Aim deltas dx, dy

    def forward(self, x):
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.act2(self.ln2(self.fc2(h)))
        h = self.drop(h)
        h = self.act3(self.ln3(self.fc3(h)))

        keys_logits = self.keys_head(h)
        clicks_logits = self.clicks_head(h)
        # Bounded Tanh aim: strictly [-15, +15] pixels
        aim_deltas = 15.0 * torch.tanh(self.aim_head(h))
        return keys_logits, clicks_logits, aim_deltas


def find_exp_file():
    for p in EXP_CANDIDATES:
        if os.path.exists(p):
            return p
    # Fallback to creating local path
    default_p = str(BASE_DIR / "dataset" / "online_experiences.jsonl")
    os.makedirs(os.path.dirname(default_p), exist_ok=True)
    return default_p


def load_model():
    model = StablePolicyNet(input_dim=30)
    start_version = 1
    loaded_from = None
    for p in PT_MODEL_PATHS:
        if os.path.exists(p):
            try:
                chk = torch.load(p, map_location=DEVICE)
                state = chk["model_state"] if (isinstance(chk, dict) and "model_state" in chk) else chk
                if isinstance(chk, dict) and "version" in chk:
                    start_version = int(chk["version"]) + 1
                model.load_state_dict(state)
                loaded_from = p
                print(f"[Self-Learning Daemon] Loaded base model from {p} (Next version: v{start_version})")
                break
            except Exception as e:
                print(f"[Self-Learning Daemon] Notice: Could not load {p}: {e}")
    model = model.to(DEVICE)
    return model, start_version, loaded_from


def export_npz(model):
    """Atomically export weights to avoid reading partial files during game inference"""
    model.eval()
    state = model.state_dict()
    weights = {}
    for k, v in state.items():
        weights[k] = v.cpu().numpy()

    for path in NPZ_EXPORT_PATHS:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp_path = path + ".tmp.npz"
            np.savez(tmp_path, **weights)
            shutil.move(tmp_path, path)
            print(f"[Self-Learning Daemon] Successfully exported weights to {path}")
        except Exception as e:
            pass


def run_daemon():
    model, evolution_round, loaded_from = load_model()
    export_npz(model)  # Ensure weights exist for live bot immediately
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_bce_raw = nn.BCEWithLogitsLoss(reduction='none')
    loss_aim_raw = nn.SmoothL1Loss(beta=1.0, reduction='none')

    last_file_pos = 0
    last_speed_check = time.time()
    last_sample_count = 0

    exp_file = find_exp_file()
    print(f"[Self-Learning Daemon] Monitoring online experiences from: {exp_file}")
    print("[Self-Learning Daemon] Waiting for gameplay data...")

    accumulated_inputs = []
    accumulated_keys = []
    accumulated_clicks = []
    accumulated_aims = []
    accumulated_rewards = []

    while True:
        time.sleep(10)  # Check every 10 seconds

        if not os.path.exists(exp_file):
            exp_file = find_exp_file()
            if not os.path.exists(exp_file):
                continue

        try:
            curr_size = os.path.getsize(exp_file)
            if curr_size > last_file_pos:
                with open(exp_file, "r", encoding="utf-8") as f:
                    f.seek(last_file_pos)
                    lines = f.readlines()
                    last_file_pos = f.tell()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        x = d["x"]
                        k = d["k"]
                        c = d["c"]
                        a = d["a"]
                        r = float(d.get("r", 0.0))

                        # Strict Sanity Validation
                        if len(x) != 30 or len(k) != 7 or len(c) != 2 or len(a) != 2:
                            continue
                        if abs(a[0]) > 40.0 or abs(a[1]) > 40.0:
                            continue
                        if any(np.isnan(v) or np.isinf(v) for v in x + a):
                            continue

                        # Check for corrupted pitch drift feature (feat 3, 9, 15, 21, 27)
                        # Clamping extreme state features to [-1.0, 1.0]
                        x_clean = [float(np.clip(v, -3.0, 3.0)) for v in x]

                        # Clamp aim targets to [-15, 15]
                        a_clamped = [
                            float(np.clip(a[0], -15.0, 15.0)),
                            float(np.clip(a[1], -15.0, 15.0))
                        ]

                        accumulated_inputs.append(x_clean)
                        accumulated_keys.append([float(val) for val in k])
                        accumulated_clicks.append([float(val) for val in c])
                        accumulated_aims.append(a_clamped)
                        accumulated_rewards.append(r)
                    except Exception:
                        pass

            n_samples = len(accumulated_rewards)
            if n_samples < 24:
                continue

            # -------------------------------------------------------------
            # REINFORCEMENT LEARNING: Discounted Return G_t & Advantage Filtering
            # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} ... (gamma = 0.95)
            # -------------------------------------------------------------
            gamma = 0.95
            returns = [0.0] * n_samples
            running_g = 0.0
            for t in reversed(range(n_samples)):
                r_t = accumulated_rewards[t]
                if r_t <= -1.0:  # Terminal death event: reset running return
                    running_g = r_t
                else:
                    running_g = r_t + gamma * running_g
                returns[t] = running_g

            # Filter out negative experiences (collisions, deaths, useless wall-running)
            # Prioritize experiences that led to damage (+2.0), kills (+8.0), or combat locks (+0.04)
            pos_indices = [i for i in range(n_samples) if returns[i] > 0.025]
            if len(pos_indices) < 16:
                # Still building positive combat trajectories, wait for more experience
                continue

            # Select recent positive experiences (up to 512)
            selected_idx = pos_indices[-512:]
            train_inputs = [accumulated_inputs[i] for i in selected_idx]
            train_keys = [accumulated_keys[i] for i in selected_idx]
            train_clicks = [accumulated_clicks[i] for i in selected_idx]
            train_aims = [accumulated_aims[i] for i in selected_idx]
            train_weights = [float(np.clip(returns[i], 0.1, 5.0)) for i in selected_idx]

            print(f"\n[Evolution v{evolution_round}] RL Reward-Weighted Optimization on {len(train_inputs)} positive experiences (Buffer: {n_samples}) on {DEVICE}...")

            x_t = torch.tensor(train_inputs, dtype=torch.float32, device=DEVICE)
            k_t = torch.tensor(train_keys, dtype=torch.float32, device=DEVICE)
            c_t = torch.tensor(train_clicks, dtype=torch.float32, device=DEVICE)
            a_t = torch.tensor(train_aims, dtype=torch.float32, device=DEVICE)
            w_t = torch.tensor(train_weights, dtype=torch.float32, device=DEVICE)

            dataset = TensorDataset(x_t, k_t, c_t, a_t, w_t)
            batch_sz = min(64, max(16, len(train_inputs) // 4))
            loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)

            model.train()
            total_loss = 0.0
            epochs = 3

            for epoch in range(epochs):
                epoch_loss = 0.0
                for bx, bk, bc, ba, bw in loader:
                    optimizer.zero_grad()
                    out_k, out_c, out_a = model(bx)
                    
                    # Advantage-Weighted loss: only reinforce trajectories that dealt damage and scored kills!
                    lk = (loss_bce_raw(out_k, bk).mean(dim=-1) * bw).mean()
                    lc = (loss_bce_raw(out_c, bc).mean(dim=-1) * bw).mean()
                    la = (loss_aim_raw(out_a, ba).mean(dim=-1) * bw).mean()
                    loss = lk + 1.5 * lc + 0.3 * la
                    
                    loss.backward()
                    # Gradient clipping to prevent gradient explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                total_loss += epoch_loss / max(1, len(loader))
            total_loss = total_loss / epochs

            now_t = time.time()
            elapsed_t = max(1.0, now_t - last_speed_check)
            new_samples = len(accumulated_inputs) - last_sample_count
            train_speed = max(0.0, float(new_samples) / elapsed_t) if new_samples > 0 else float(len(train_inputs)) / elapsed_t
            last_speed_check = now_t
            last_sample_count = len(accumulated_inputs)

            print(f"[Evolution v{evolution_round}] Complete! Loss: {total_loss:.4f} | Speed: {train_speed:.1f} exp/s | Buffer: {len(accumulated_inputs)}")

            # Save checkpoint
            save_pt = loaded_from or str(BASE_DIR / "models" / "cs16_policy.pt")
            try:
                os.makedirs(os.path.dirname(save_pt), exist_ok=True)
                torch.save({"model_state": model.state_dict(), "version": evolution_round, "loss": total_loss}, save_pt)
            except Exception as e:
                print(f"[Self-Learning Daemon] Notice saving checkpoint: {e}")

            # Export weights to game NPZ
            export_npz(model)
            evolution_round += 1

            # Retain recent 512 samples for experience replay
            accumulated_inputs = accumulated_inputs[-512:]
            accumulated_keys = accumulated_keys[-512:]
            accumulated_clicks = accumulated_clicks[-512:]
            accumulated_aims = accumulated_aims[-512:]
            accumulated_rewards = accumulated_rewards[-512:]

        except Exception as e:
            print(f"[Error in daemon]: {e}")


if __name__ == "__main__":
    run_daemon()

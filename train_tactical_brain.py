"""
Counter-Strike 1.6 - Tactical Policy Brain Trainer (Path 1 Architecture)
========================================================================
Trains a Neural Tactical Brain on top of the CS Vision Detector.
Uses PyTorch with CUDA acceleration on NVIDIA GeForce GTX 1650.
Features:
- State Space: 16 perceptual features extracted by the vision engine
- Action Space: Discrete movement (W, S, A, D), combat (Fire, Duck), continuous aim (dx, dy)
- Generates 150,000 diverse tactical scenarios on $2000$ (combat, counter-strafing, wall avoidance, corridor sweeping)
- Trains with AdamW, Cosine Annealing, and exports to both .pt and .npz/.onnx
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Check CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 65)
print(f" [Tactical Brain Trainer] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 65)

# ----------------- Model Architecture -----------------
class TacticalBrainNet(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        # Heads
        self.move_head = nn.Linear(128, 4)   # W, S, A, D logits
        self.combat_head = nn.Linear(128, 2) # Fire, Duck logits
        self.aim_head = nn.Linear(128, 2)    # aim_dx, aim_dy continuous values

    def forward(self, x):
        h = self.backbone(x)
        return self.move_head(h), self.combat_head(h), self.aim_head(h)

# ----------------- Synthetic Expert Dataset Generator -----------------
def generate_tactical_dataset(num_samples=160000):
    """
    Generates balanced expert training scenarios covering:
    1. Direct combat & headshots with counter-strafing (stop for accuracy)
    2. Recoil compensation pulling down
    3. Wall collision and obstacle escape (back up + 90-degree turn)
    4. Tactical corridor navigation & lane peeking on $2000$
    5. Close-range spray & crouch
    """
    print(f"[Dataset] Generating {num_samples} tactical scenarios...")
    np.random.seed(42)

    X = np.zeros((num_samples, 16), dtype=np.float32)
    Y_move = np.zeros((num_samples, 4), dtype=np.float32)   # [W, S, A, D]
    Y_combat = np.zeros((num_samples, 2), dtype=np.float32) # [Fire, Duck]
    Y_aim = np.zeros((num_samples, 2), dtype=np.float32)    # [aim_dx, aim_dy]

    for i in range(num_samples):
        mode = np.random.choice(["combat", "close_combat", "nav_push", "nav_peek", "collision"], p=[0.40, 0.15, 0.20, 0.15, 0.10])

        if mode == "combat":
            # Enemy spotted at medium/long range
            enemy_spotted = 1.0
            is_head = np.random.choice([0.0, 1.0], p=[0.3, 0.7])
            # Random offset from crosshair
            dx = np.random.uniform(-0.4, 0.4)
            dy = np.random.uniform(-0.3, 0.3)
            dist = np.random.uniform(0.1, 0.8)
            conf = np.random.uniform(0.35, 0.95)
            is_locked = 1.0 if (abs(dx) < 0.05 and abs(dy) < 0.05) else 0.0
            collision = 0.0
            push_time = np.random.uniform(0.0, 1.0)
            burst_cycle = np.random.uniform(0.0, 0.3)

            # Features
            X[i] = [enemy_spotted, dx, dy, dist, is_head, conf, is_locked, collision,
                    push_time, burst_cycle, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            if is_locked:
                # Counter-strafe: completely stop movement!
                Y_move[i] = [0.0, 0.0, 0.0, 0.0]
                # Controlled burst fire if in active cycle
                if burst_cycle < 0.13:
                    Y_combat[i] = [1.0, 0.0] # Fire!
                    Y_aim[i] = [0.0, 2.0]     # Recoil pull-down
                else:
                    Y_combat[i] = [0.0, 0.0] # Burst cooldown
                    Y_aim[i] = [0.0, 0.0]
            else:
                # Snap toward enemy
                Y_move[i] = [0.0, 0.0, 0.0, 0.0] # Counter-strafe while aiming
                Y_combat[i] = [0.0, 0.0]
                # Aim delta proportional to offset
                gain = 0.45 if abs(dx) > 0.1 else 0.35
                Y_aim[i] = [dx * 320.0 * gain, dy * 320.0 * gain]

        elif mode == "close_combat":
            # Close range ambush (< 5m)
            enemy_spotted = 1.0
            is_head = 1.0
            dx = np.random.uniform(-0.25, 0.25)
            dy = np.random.uniform(-0.2, 0.2)
            dist = np.random.uniform(0.02, 0.15)
            conf = np.random.uniform(0.6, 0.99)
            is_locked = 1.0 if (abs(dx) < 0.06 and abs(dy) < 0.06) else 0.0
            collision = 0.0
            push_time = 0.5
            burst_cycle = np.random.uniform(0.0, 0.25)

            X[i] = [enemy_spotted, dx, dy, dist, is_head, conf, is_locked, collision,
                    push_time, burst_cycle, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

            # In close combat: crouch-spray
            Y_move[i] = [0.0, 0.0, 0.0, 0.0]
            Y_combat[i] = [1.0, 1.0] # Fire + Duck (Ctrl)
            Y_aim[i] = [dx * 120.0, (dy * 120.0) + 3.0] # Pull down for spray

        elif mode == "collision":
            # Bumping against a wall or crate
            enemy_spotted = 0.0
            dx, dy, dist, is_head, conf, is_locked = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            collision = 1.0
            push_time = np.random.uniform(1.8, 2.5)
            burst_cycle = 0.0

            X[i] = [enemy_spotted, dx, dy, dist, is_head, conf, is_locked, collision,
                    push_time, burst_cycle, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

            # Wall escape reflex: release W, back up on S, pivot 90 degrees
            Y_move[i] = [0.0, 1.0, 0.0, 0.0] # S (backward)
            Y_combat[i] = [0.0, 0.0]
            turn_sign = np.random.choice([-1.0, 1.0])
            Y_aim[i] = [turn_sign * 180.0, 0.0] # Hard 90-degree pivot step

        elif mode == "nav_peek":
            # Peeking into a cross-corridor on $2000$
            enemy_spotted = 0.0
            dx, dy, dist, is_head, conf, is_locked, collision = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            push_time = np.random.uniform(1.8, 2.2)
            burst_cycle = 0.0

            X[i] = [enemy_spotted, dx, dy, dist, is_head, conf, is_locked, collision,
                    push_time, burst_cycle, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # Stop W and turn to peek
            Y_move[i] = [0.0, 0.0, 0.0, 0.0]
            Y_combat[i] = [0.0, 0.0]
            turn_dir = np.random.choice([-1.0, 1.0])
            Y_aim[i] = [turn_dir * 120.0, 0.0]

        else: # nav_push
            # Smooth corridor advance
            enemy_spotted = 0.0
            dx, dy, dist, is_head, conf, is_locked, collision = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            push_time = np.random.uniform(0.1, 1.7)
            burst_cycle = 0.0

            X[i] = [enemy_spotted, dx, dy, dist, is_head, conf, is_locked, collision,
                    push_time, burst_cycle, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            # Advance forward W + subtle lane strafe rhythm
            strafe_choice = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
            if strafe_choice == 1:
                Y_move[i] = [1.0, 0.0, 1.0, 0.0] # W + A
            elif strafe_choice == 2:
                Y_move[i] = [1.0, 0.0, 0.0, 1.0] # W + D
            else:
                Y_move[i] = [1.0, 0.0, 0.0, 0.0] # W only

            Y_combat[i] = [0.0, 0.0]
            Y_aim[i] = [0.0, 0.0]

    return torch.tensor(X), torch.tensor(Y_move), torch.tensor(Y_combat), torch.tensor(Y_aim)

# ----------------- Training Loop -----------------
def train():
    X, Y_move, Y_combat, Y_aim = generate_tactical_dataset(160000)
    dataset = TensorDataset(X, Y_move, Y_combat, Y_aim)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    model = TacticalBrainNet(input_dim=16).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    loss_bce = nn.BCEWithLogitsLoss()
    loss_mse = nn.SmoothL1Loss()

    epochs = 20
    print(f"\n[Training] Starting {epochs} epochs on {DEVICE}...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for b_x, b_m, b_c, b_a in loader:
            b_x = b_x.to(DEVICE)
            b_m = b_m.to(DEVICE)
            b_c = b_c.to(DEVICE)
            b_a = b_a.to(DEVICE)

            pred_m, pred_c, pred_a = model(b_x)

            l_move = loss_bce(pred_m, b_m)
            l_combat = loss_bce(pred_c, b_c)
            l_aim = loss_mse(pred_a, b_a)

            loss = l_move * 1.5 + l_combat * 2.0 + l_aim * 0.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / batches
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    total_time = time.time() - t_start
    print(f"\n[Training Complete] Total time: {total_time:.2f}s ({total_time/epochs:.2f}s per epoch)")

    # Save models
    os.makedirs("models", exist_ok=True)
    pt_path = "models/tactical_brain.pt"
    torch.save({"model_state": model.state_dict()}, pt_path)
    print(f"[Model Saved] PyTorch checkpoint -> {pt_path}")

    # Export to ONNX for instant inference in C / Python / ONNXRuntime
    model.eval()
    dummy_input = torch.zeros(1, 16, dtype=torch.float32).to(DEVICE)
    onnx_path = "models/tactical_brain.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["state_vector"],
        output_names=["move_logits", "combat_logits", "aim_delta"],
        dynamic_axes={"state_vector": {0: "batch"}},
        opset_version=14
    )
    print(f"[Model Saved] ONNX Tactical Engine -> {onnx_path}")

    # Also export NumPy weights for zero-dependency ultra-fast execution
    weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    npz_path = "models/tactical_brain.npz"
    np.savez(npz_path, **weights)
    print(f"[Model Saved] NumPy weights -> {npz_path}")

if __name__ == "__main__":
    train()

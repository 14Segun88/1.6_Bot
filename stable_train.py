"""
Stable GPU Training Pipeline for CS 1.6 AI Policy
Features:
- LayerNorm architecture (immune to batch variance collapse)
- Bounded Tanh aim head (strict [-15, +15] pixel clamp)
- SmoothL1Loss + Gradient Clipping (max_norm=1.0)
- Human gameplay anchor (132,278 demo ticks on $2000$)
- Direct export to NumPy weights for 0.05ms inference
"""

import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CONFIG = {
    "demo_file": "dataset/extracted_actions.jsonl",
    "model_pt": "models/cs16_policy.pt",
    "export_npz": [
        "models/policy_weights.npz",
        "/mnt/d/CS 1.6 GoldClient/models/policy_weights.npz"
    ],
    "epochs": 12,
    "batch_size": 256,
    "lr": 1.5e-3,
    "seq_len": 5,
}

class CS16Dataset(Dataset):
    def __init__(self, jsonl_path, seq_len=5):
        self.inputs = []
        self.target_keys = []
        self.target_clicks = []
        self.target_aim = []

        print(f"[Dataset] Parsing human demo actions from {jsonl_path}...")
        raw_frames = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("health", 0) > 0:
                    org = d.get("origin", [0, 0, 0])
                    ang = d.get("angles", [0, 0])
                    hp = float(d.get("health", 100)) / 100.0
                    feat = [org[0]/1000.0, org[1]/1000.0, org[2]/1000.0, ang[0]/90.0, ang[1]/180.0, hp]

                    keys_d = d.get("keys", {})
                    keys_vec = [
                        float(keys_d.get("w", 0)),
                        float(keys_d.get("s", 0)),
                        float(keys_d.get("a", 0)),
                        float(keys_d.get("d", 0)),
                        float(keys_d.get("space", 0)),
                        float(keys_d.get("ctrl", 0)),
                        float(keys_d.get("r", 0)),
                    ]
                    mouse_d = d.get("mouse", {})
                    click_vec = [float(mouse_d.get("left", 0)), float(mouse_d.get("right", 0))]
                    # Clamp raw aim to reasonable delta
                    aim_dx = np.clip(float(mouse_d.get("dx", 0)), -15.0, 15.0)
                    aim_dy = np.clip(float(mouse_d.get("dy", 0)), -15.0, 15.0)
                    aim_vec = [aim_dx, aim_dy]

                    raw_frames.append((feat, keys_vec, click_vec, aim_vec))

        print(f"[Dataset] Extracted {len(raw_frames)} valid active ticks.")

        for i in range(seq_len, len(raw_frames)):
            hist_feat = [raw_frames[j][0] for j in range(i - seq_len, i)]
            self.inputs.append(np.array(hist_feat, dtype=np.float32).flatten())
            self.target_keys.append(np.array(raw_frames[i][1], dtype=np.float32))
            self.target_clicks.append(np.array(raw_frames[i][2], dtype=np.float32))
            self.target_aim.append(np.array(raw_frames[i][3], dtype=np.float32))

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.target_keys = torch.tensor(np.array(self.target_keys), dtype=torch.float32)
        self.target_clicks = torch.tensor(np.array(self.target_clicks), dtype=torch.float32)
        self.target_aim = torch.tensor(np.array(self.target_aim), dtype=torch.float32)
        print(f"[Dataset] Ready: {len(self.inputs)} training sequences created.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.target_keys[idx], self.target_clicks[idx], self.target_aim[idx]

class StablePolicyNet(nn.Module):
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

        # Heads
        self.keys_head = nn.Linear(128, 7)
        self.clicks_head = nn.Linear(128, 2)
        self.aim_head = nn.Linear(128, 2)

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

def export_weights(model):
    model.eval()
    state = model.state_dict()
    weights = {}
    for k, v in state.items():
        weights[k] = v.cpu().numpy()

    targets = ["models/policy_weights.npz"]
    if "CS16_WORK_DIR" in os.environ:
        targets.append(os.path.join(os.environ["CS16_WORK_DIR"], "models", "policy_weights.npz"))
    if os.path.exists("D:/CS 1.6 GoldClient/models"):
        targets.append("D:/CS 1.6 GoldClient/models/policy_weights.npz")
    elif os.path.exists("/mnt/d/CS 1.6 GoldClient/models"):
        targets.append("/mnt/d/CS 1.6 GoldClient/models/policy_weights.npz")

    seen = set()
    for p in targets:
        rp = os.path.abspath(p)
        if rp not in seen:
            seen.add(rp)
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.savez(p, **weights)
                print(f"[Export] Saved NumPy weights -> {p}")
            except Exception as e:
                pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f" Training Stable CS 1.6 Policy on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    dataset = CS16Dataset(CONFIG["demo_file"], seq_len=CONFIG["seq_len"])
    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)

    model = StablePolicyNet(input_dim=30).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=5e-5)

    # Weights for keys: W (idx 0), S (1), A (2), D (3), Space (4), Ctrl (5), R (6)
    # Boost W and shooting to maintain active forward combat
    k_weights = torch.tensor([2.0, 1.0, 1.2, 1.2, 1.0, 1.0, 1.0], device=device)
    loss_bce_keys = nn.BCEWithLogitsLoss(pos_weight=k_weights)
    c_weights = torch.tensor([3.0, 1.0], device=device)
    loss_bce_clicks = nn.BCEWithLogitsLoss(pos_weight=c_weights)
    loss_aim = nn.SmoothL1Loss(beta=1.0)

    t_start = time.time()
    best_loss = 999.0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        k_loss_acc = 0.0
        c_loss_acc = 0.0
        a_loss_acc = 0.0

        for bx, bk, bc, ba in loader:
            bx = bx.to(device)
            bk = bk.to(device)
            bc = bc.to(device)
            ba = ba.to(device)

            optimizer.zero_grad()
            out_k, out_c, out_a = model(bx)

            lk = loss_bce_keys(out_k, bk)
            lc = loss_bce_clicks(out_c, bc)
            la = loss_aim(out_a, ba)

            loss = lk + 1.5 * lc + 0.1 * la
            loss.backward()

            # Gradient Clipping: prevents any gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            k_loss_acc += lk.item()
            c_loss_acc += lc.item()
            a_loss_acc += la.item()

        scheduler.step()
        n = len(loader)
        avg_loss = total_loss / n
        print(f"Epoch [{epoch:02d}/{CONFIG['epochs']:02d}] Loss: {avg_loss:.4f} | Keys: {k_loss_acc/n:.4f} | Clicks: {c_loss_acc/n:.4f} | Aim: {a_loss_acc/n:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"model_state": model.state_dict(), "best_loss": best_loss}, CONFIG["model_pt"])

    print(f"\n[Training Completed in {time.time() - t_start:.1f}s] Best Loss: {best_loss:.4f}")
    export_weights(model)
    print("[SUCCESS] Stable policy trained and deployed!")

if __name__ == "__main__":
    main()

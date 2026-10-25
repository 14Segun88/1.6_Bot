"""
Train CS 1.6 Policy Model from Extracted Demo Actions
Uses PyTorch with CUDA acceleration on NVIDIA GeForce GTX 1650.
"""

import json
import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CONFIG = {
    "actions_file": "dataset/extracted_actions.jsonl",
    "model_path": "models/cs16_policy.pt",
    "epochs": 15,
    "batch_size": 128,
    "lr": 1e-3,
    "seq_len": 5, # 5 historical frames for temporal dynamics
}

class CS16DemoDataset(Dataset):
    def __init__(self, jsonl_path, seq_len=5):
        self.seq_len = seq_len
        self.inputs = []
        self.target_keys = []
        self.target_clicks = []
        self.target_aim = []

        print(f"[Dataset] Loading data from {jsonl_path}...")
        raw_frames = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                # Keep active rounds where health > 0 or alive
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
                    aim_vec = [float(mouse_d.get("dx", 0)), float(mouse_d.get("dy", 0))]

                    raw_frames.append((feat, keys_vec, click_vec, aim_vec))

        print(f"[Dataset] Filtered {len(raw_frames)} active gameplay ticks from demo.")
        if len(raw_frames) <= seq_len:
            print("[Warning] Not enough samples.")
            return

        for i in range(seq_len, len(raw_frames)):
            # Sequence of features
            hist_feat = [raw_frames[j][0] for j in range(i - seq_len, i)]
            target_k = raw_frames[i][1]
            target_c = raw_frames[i][2]
            target_a = raw_frames[i][3]

            self.inputs.append(np.array(hist_feat, dtype=np.float32).flatten())
            self.target_keys.append(np.array(target_k, dtype=np.float32))
            self.target_clicks.append(np.array(target_c, dtype=np.float32))
            self.target_aim.append(np.array(target_a, dtype=np.float32))

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.target_keys = torch.tensor(np.array(self.target_keys), dtype=torch.float32)
        self.target_clicks = torch.tensor(np.array(self.target_clicks), dtype=torch.float32)
        self.target_aim = torch.tensor(np.array(self.target_aim), dtype=torch.float32)

        print(f"[Dataset] Ready: {len(self.inputs)} training sequences created.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.target_keys[idx],
            self.target_clicks[idx],
            self.target_aim[idx],
        )

class CS16PolicyNetwork(nn.Module):
    def __init__(self, input_dim=30): # 5 frames * 6 features = 30
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 1. Movement keys: W, S, A, D, Space, Ctrl, R (7)
        self.keys_head = nn.Linear(128, 7)
        # 2. Mouse buttons: Left, Right (2)
        self.clicks_head = nn.Linear(128, 2)
        # 3. Mouse aim: dx, dy (2)
        self.aim_head = nn.Linear(128, 2)

    def forward(self, x):
        h = self.backbone(x)
        keys_logits = self.keys_head(h)
        clicks_logits = self.clicks_head(h)
        aim_deltas = self.aim_head(h)
        return keys_logits, clicks_logits, aim_deltas

def train_policy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f" Training CS 1.6 Bot Policy on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"==========================================")

    dataset = CS16DemoDataset(CONFIG["actions_file"], seq_len=CONFIG["seq_len"])
    if len(dataset) == 0:
        print("[ERROR] Dataset is empty.")
        return

    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    input_dim = CONFIG["seq_len"] * 6
    model = CS16PolicyNetwork(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.SmoothL1Loss()

    start_time = time.time()
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0.0
        key_loss_acc = 0.0
        click_loss_acc = 0.0
        aim_loss_acc = 0.0

        for inp, t_keys, t_clicks, t_aim in dataloader:
            inp = inp.to(device)
            t_keys = t_keys.to(device)
            t_clicks = t_clicks.to(device)
            t_aim = t_aim.to(device)

            optimizer.zero_grad()
            p_keys, p_clicks, p_aim = model(inp)

            loss_k = bce_loss(p_keys, t_keys)
            loss_c = bce_loss(p_clicks, t_clicks)
            loss_a = l1_loss(p_aim, t_aim)

            loss = loss_k + 3.0 * loss_c + 0.5 * loss_a
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            key_loss_acc += loss_k.item()
            click_loss_acc += loss_c.item()
            aim_loss_acc += loss_a.item()

        n = len(dataloader)
        print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']}] | Total Loss: {total_loss/n:.4f} | Keys Loss: {key_loss_acc/n:.4f} | Fire Loss: {click_loss_acc/n:.4f} | Aim Loss: {aim_loss_acc/n:.4f}")

    elapsed = round(time.time() - start_time, 2)
    os.makedirs(os.path.dirname(CONFIG["model_path"]), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": input_dim,
        "seq_len": CONFIG["seq_len"],
        "keys_names": ["w", "s", "a", "d", "space", "ctrl", "r"],
    }, CONFIG["model_path"])

    print(f"\n[DONE] Policy training finished in {elapsed}s!")
    print(f"[Model Saved] -> {CONFIG['model_path']}")

if __name__ == "__main__":
    train_policy()

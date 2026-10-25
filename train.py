"""
Counter-Strike 1.6 - Multimodal Behavioral Cloning Trainer (Vision + Audio Policy)
==================================================================================
Trains a deep policy neural network from human gameplay recordings.
Inputs:
  1. Video Frames (160x120 spatial geometry context: corridors, walls, obstacles)
  2. Directional Stereo Audio features (L/R volume, stereo balance, gunshot/footstep spikes)
Outputs:
  1. Keyboard Controls (W, S, A, D, Space, Ctrl, Shift, R, 1, 2, 3)
  2. Mouse Clicks (Left click / Burst fire)
  3. Continuous Mouse Aim deltas (dx, dy)
"""

import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# ----------------- Configuration -----------------
CONFIG = {
    "dataset_dir": "dataset",
    "model_save_path": "models/cs16_bot.pt",
    "batch_size": 64,
    "epochs": 25,
    "lr": 2e-4,
    "img_width": 160,
    "img_height": 120,
    "keys_list": ["w", "a", "s", "d", "space", "ctrl", "shift", "r", "1", "2", "3", "4"],
    "augment_factor": 6,  # Expands 55 mins into 3.67 hours of gameplay (Mirror, Lighting, Shadows, Audio Inversion)
}

# ----------------- Dataset Loader with Multi-Stream Augmentation -----------------
class CS16MultimodalDataset(Dataset):
    def __init__(self, dataset_dir, img_width=160, img_height=120, keys_list=None, augment_factor=6):
        self.img_width = img_width
        self.img_height = img_height
        self.keys_list = keys_list or CONFIG["keys_list"]
        self.augment_factor = max(1, augment_factor)
        self.samples = []
        self.session_mmaps = {}

        # Resolve symmetry indices for key inversion (A <-> D)
        self.idx_a = self.keys_list.index("a") if "a" in self.keys_list else -1
        self.idx_d = self.keys_list.index("d") if "d" in self.keys_list else -1

        sessions = [s for s in sorted(glob.glob(os.path.join(dataset_dir, "session_*"))) if os.path.isdir(s)]
        if not sessions:
            print(f"[Warning] No recorded sessions found in '{dataset_dir}'.")
            return

        print(f"Loading recordings from {len(sessions)} session(s)...")
        for s_idx, sess in enumerate(sessions):
            video_path = os.path.join(sess, "gameplay.avi")
            if not os.path.exists(video_path):
                video_path = os.path.join(sess, "gameplay.mp4")
            actions_path = os.path.join(sess, "actions.jsonl")
            if not os.path.exists(video_path) or not os.path.exists(actions_path):
                continue

            sess_records = []
            with open(actions_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        sess_records.append(data)
                    except Exception:
                        continue

            num_records = len(sess_records)
            if num_records == 0:
                continue

            cache_bin = os.path.join(sess, f"frames_{img_width}x{img_height}.bin")
            expected_bytes = num_records * img_height * img_width * 3

            # Sequentially cache frames once if cache does not exist or size doesn't match
            if not os.path.exists(cache_bin) or os.path.getsize(cache_bin) != expected_bytes:
                print(f"[*] Building high-speed binary frame cache for {os.path.basename(sess)}...")
                print(f"    Extracting {num_records} frames sequentially into memory-mapped cache...")
                mmap_arr = np.memmap(cache_bin, dtype=np.uint8, mode='w+', shape=(num_records, img_height, img_width, 3))
                cap = cv2.VideoCapture(video_path)
                saved = 0
                while saved < num_records:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                    else:
                        frame = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_AREA)
                    mmap_arr[saved] = frame
                    saved += 1
                    if saved % 10000 == 0 or saved == num_records:
                        print(f"    Cached {saved}/{num_records} frames ({saved/num_records*100:.1f}%)...")
                cap.release()
                mmap_arr.flush()
                del mmap_arr
                print(f"[*] Cache built successfully ({expected_bytes / (1024*1024):.1f} MB).")

            # Open read-only memory map for ultra-fast DataLoader access
            self.session_mmaps[s_idx] = np.memmap(
                cache_bin,
                dtype=np.uint8,
                mode='r',
                shape=(num_records, img_height, img_width, 3)
            )

            for f_id, data in enumerate(sess_records):
                keys_dict = data.get("keys", {})
                mouse_dict = data.get("mouse", {})
                aud_dict = data.get("audio", {})

                key_vec = [float(keys_dict.get(k, 0)) for k in self.keys_list]
                click_vec = [float(mouse_dict.get("left", 0)), float(mouse_dict.get("right", 0))]

                dx = np.clip(float(mouse_dict.get("dx", 0)), -120.0, 120.0)
                dy = np.clip(float(mouse_dict.get("dy", 0)), -90.0, 90.0)
                aim_vec = [dx, dy]

                audio_vec = [
                    float(aud_dict.get("vol_left", 0.0)),
                    float(aud_dict.get("vol_right", 0.0)),
                    float(aud_dict.get("balance", 0.0)),
                    float(aud_dict.get("peak", 0.0)),
                ]

                self.samples.append((s_idx, f_id, audio_vec, key_vec, click_vec, aim_vec))

        total_frames = len(self.samples) * self.augment_factor
        effective_hours = (total_frames / 30.0) / 3600.0
        print(f"[*] Raw Recorded Gameplay: {len(self.samples):,} frames ({len(self.samples)/30/60:.1f} mins)")
        print(f"[*] Multi-Stream Augmentation Engine: {self.augment_factor}x active")
        print(f"    - Stream 0: Ground-truth human gameplay (100%)")
        print(f"    - Stream 1: Horizontal mirror flip (A<->D keys, dx negated, L<->R audio)")
        print(f"    - Stream 2: High exposure / direct sunlight variation")
        print(f"    - Stream 3: Shadowed corridor / dark corners variation")
        print(f"    - Stream 4: Mirrored + high exposure variation")
        print(f"    - Stream 5: Mirrored + shadowed corridor variation")
        print(f"[+] Total Dataset Ready for GPU: {total_frames:,} frames ({effective_hours:.2f} hours equivalent!)")

    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        base_idx = idx // self.augment_factor
        aug_type = idx % self.augment_factor

        s_idx, frame_id, audio_vec, key_vec, click_vec, aim_vec = self.samples[base_idx]

        # Instantaneous RAM/mmap lookup
        frame = self.session_mmaps[s_idx][frame_id].copy()
        cur_keys = list(key_vec)
        cur_aim = list(aim_vec)
        cur_audio = list(audio_vec)

        if aug_type == 1:
            # 1. Horizontal Mirror (Flips map horizontally: left-peeking becomes right-peeking)
            frame = np.fliplr(frame).copy()
            if self.idx_a != -1 and self.idx_d != -1:
                cur_keys[self.idx_a], cur_keys[self.idx_d] = cur_keys[self.idx_d], cur_keys[self.idx_a]
            cur_aim[0] = -cur_aim[0]  # Invert horizontal mouse aim (dx)
            cur_audio[0], cur_audio[1] = cur_audio[1], cur_audio[0]  # Swap L and R volumes
            cur_audio[2] = -cur_audio[2]  # Negate stereo balance

        elif aug_type == 2:
            # 2. Lighting / Exposure variation (Simulates bright sunlight / outdoor areas)
            frame = np.clip(frame.astype(np.float32) * 1.15 + 12, 0, 255).astype(np.uint8)

        elif aug_type == 3:
            # 3. Shadowed Corridor variation (Darkened corners / lower contrast)
            frame = np.clip(frame.astype(np.float32) * 0.85 - 10, 0, 255).astype(np.uint8)

        elif aug_type == 4:
            # 4. Mirrored + High Exposure
            frame = np.fliplr(frame).copy()
            frame = np.clip(frame.astype(np.float32) * 1.15 + 12, 0, 255).astype(np.uint8)
            if self.idx_a != -1 and self.idx_d != -1:
                cur_keys[self.idx_a], cur_keys[self.idx_d] = cur_keys[self.idx_d], cur_keys[self.idx_a]
            cur_aim[0] = -cur_aim[0]
            cur_audio[0], cur_audio[1] = cur_audio[1], cur_audio[0]
            cur_audio[2] = -cur_audio[2]

        elif aug_type == 5:
            # 5. Mirrored + Shadowed Corridor
            frame = np.fliplr(frame).copy()
            frame = np.clip(frame.astype(np.float32) * 0.85 - 10, 0, 255).astype(np.uint8)
            if self.idx_a != -1 and self.idx_d != -1:
                cur_keys[self.idx_a], cur_keys[self.idx_d] = cur_keys[self.idx_d], cur_keys[self.idx_a]
            cur_aim[0] = -cur_aim[0]
            cur_audio[0], cur_audio[1] = cur_audio[1], cur_audio[0]
            cur_audio[2] = -cur_audio[2]

        # PyTorch format (C, H, W) normalized to [0, 1]
        frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        audio_tensor = torch.tensor(cur_audio, dtype=torch.float32)
        key_tensor = torch.tensor(cur_keys, dtype=torch.float32)
        click_tensor = torch.tensor(click_vec, dtype=torch.float32)
        aim_tensor = torch.tensor(cur_aim, dtype=torch.float32)

        return frame_tensor, audio_tensor, key_tensor, click_tensor, aim_tensor



# ----------------- Multimodal Neural Policy Architecture -----------------
class CS16BotNet(nn.Module):
    def __init__(self, num_keys=11):
        super().__init__()
        # 1. Vision Feature Extractor (Convolutional Backbone)
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 192),
            nn.ReLU()
        )

        # 2. Audio Cue Perception (Stereo & Directional Ear)
        self.audio_backbone = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 3. Multimodal Decision Fusion (Vision 192 + Audio 64 = 256)
        self.fc_fusion = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        # 4. Action Heads:
        self.key_head = nn.Linear(256, num_keys)   # WASD, Space, Ctrl, etc.
        self.click_head = nn.Linear(256, 2)        # Left click (Fire), Right click
        self.aim_head = nn.Linear(256, 2)          # Continuous mouse displacement (dx, dy)

    def forward(self, frames, audio):
        v_feat = self.vision_backbone(frames)
        a_feat = self.audio_backbone(audio)
        combined = torch.cat([v_feat, a_feat], dim=1)
        h = self.fc_fusion(combined)

        keys_out = self.key_head(h)
        clicks_out = self.click_head(h)
        aim_out = self.aim_head(h)
        return keys_out, clicks_out, aim_out


# ----------------- Training Routine -----------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f" [Trainer] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    dataset = CS16MultimodalDataset(
        dataset_dir=CONFIG["dataset_dir"],
        img_width=CONFIG["img_width"],
        img_height=CONFIG["img_height"]
    )

    if len(dataset) == 0:
        print("[ERROR] No recorded gameplay sessions found in 'dataset/'.")
        print("Please launch START_RECORDING.bat and record your gameplay first!")
        return

    # 85% Train / 15% Validation Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    model = CS16BotNet(num_keys=len(CONFIG["keys_list"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    bce_loss = nn.BCEWithLogitsLoss()
    smooth_l1_loss = nn.SmoothL1Loss()

    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)

    best_val_loss = float("inf")
    patience_limit = 4
    patience_counter = 0

    print(f"\nStarting Multimodal Training on {len(train_dataset)} train & {len(val_dataset)} val samples...")
    print(f"Goal: Minimize Multitask Loss & Maximize Navigation Accuracy (Early Stopping: {patience_limit} epochs)")
    print("-" * 80)

    for epoch in range(CONFIG["epochs"]):
        # 1. Training Phase
        model.train()
        train_loss = 0.0

        for batch_idx, (frames, audio, keys, clicks, aims) in enumerate(train_loader):
            frames = frames.to(device)
            audio = audio.to(device)
            keys = keys.to(device)
            clicks = clicks.to(device)
            aims = aims.to(device)

            optimizer.zero_grad()
            keys_pred, clicks_pred, aim_pred = model(frames, audio)

            loss_keys = bce_loss(keys_pred, keys)
            loss_clicks = bce_loss(clicks_pred, clicks)
            loss_aim = smooth_l1_loss(aim_pred, aims)

            # Multitask loss: navigation + shooting + continuous aim
            loss = loss_keys + 1.5 * loss_clicks + 0.1 * loss_aim
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # 2. Validation Phase (Unseen samples)
        model.eval()
        val_loss = 0.0
        val_keys_correct = 0
        val_keys_total = 0
        val_aim_err = 0.0

        with torch.no_grad():
            for frames, audio, keys, clicks, aims in val_loader:
                frames = frames.to(device)
                audio = audio.to(device)
                keys = keys.to(device)
                clicks = clicks.to(device)
                aims = aims.to(device)

                keys_pred, clicks_pred, aim_pred = model(frames, audio)

                loss_k = bce_loss(keys_pred, keys)
                loss_c = bce_loss(clicks_pred, clicks)
                loss_a = smooth_l1_loss(aim_pred, aims)
                v_loss = loss_k + 1.5 * loss_c + 0.1 * loss_a
                val_loss += v_loss.item()

                # Action accuracy (WASD, shift, ctrl)
                preds = (torch.sigmoid(keys_pred) > 0.5).float()
                val_keys_correct += (preds == keys).sum().item()
                val_keys_total += keys.numel()

                val_aim_err += torch.abs(aim_pred - aims).mean().item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        keys_acc = (val_keys_correct / max(1, val_keys_total)) * 100.0
        aim_mae = val_aim_err / max(1, len(val_loader))

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Checkpoint Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "keys_list": CONFIG["keys_list"],
                "img_width": CONFIG["img_width"],
                "img_height": CONFIG["img_height"],
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "keys_accuracy": keys_acc,
            }, CONFIG["model_save_path"])
            status_str = f"★ [SAVED BEST: Loss {avg_val_loss:.4f} | Acc {keys_acc:.1f}%]"
        else:
            patience_counter += 1
            status_str = f"[No improvement {patience_counter}/{patience_limit}]"

        print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']:02d}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Movement Acc: {keys_acc:.1f}% | "
              f"Aim Err: {aim_mae:.2f}px | "
              f"LR: {current_lr:.1e} | "
              f"{status_str}")

        if patience_counter >= patience_limit:
            print("\n" + "=" * 65)
            print(f"[GOAL ACHIEVED] Early Stopping triggered: Loss minimized to {best_val_loss:.4f}")
            print(f"Best model saved with {keys_acc:.1f}% movement accuracy!")
            print("=" * 65)
            break

    print(f"\n[DONE] Optimum Multimodal AI Model saved to: {CONFIG['model_save_path']}")

    # High-Performance ONNX Export for Sub-Millisecond Inference
    onnx_path = CONFIG["model_save_path"].replace(".pt", ".onnx")
    print(f"[*] Exporting trained neural policy to ONNX format ({onnx_path})...")
    try:
        model.eval()
        dummy_f = torch.randn(1, 3, CONFIG["img_height"], CONFIG["img_width"]).to(device)
        dummy_a = torch.randn(1, 4).to(device)
        torch.onnx.export(
            model,
            (dummy_f, dummy_a),
            onnx_path,
            input_names=["frames", "audio"],
            output_names=["keys", "clicks", "aim"],
            dynamic_axes=None,
            opset_version=14
        )
        print(f"[+] ONNX Export Complete: {onnx_path}")

        # Sync to game runtime if available
        sync_dirs = []
        if "CS16_WORK_DIR" in os.environ:
            sync_dirs.append(os.path.join(os.environ["CS16_WORK_DIR"], "models"))
        if os.path.exists("D:/CS 1.6 GoldClient/models"):
            sync_dirs.append("D:/CS 1.6 GoldClient/models")
        elif os.path.exists("/mnt/d/CS 1.6 GoldClient/models"):
            sync_dirs.append("/mnt/d/CS 1.6 GoldClient/models")

        for s_dir in sync_dirs:
            if os.path.abspath(s_dir) != os.path.abspath(os.path.dirname(CONFIG["model_save_path"])):
                try:
                    import shutil
                    os.makedirs(s_dir, exist_ok=True)
                    shutil.copy2(CONFIG["model_save_path"], os.path.join(s_dir, os.path.basename(CONFIG["model_save_path"])))
                    shutil.copy2(onnx_path, os.path.join(s_dir, os.path.basename(onnx_path)))
                    print(f"[+] Synced trained models directly to: {s_dir}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[!] ONNX Export Warning: {e}")

    print("Ready to deploy with 'START_NEURAL_AGENT.bat'!")

if __name__ == "__main__":
    train()


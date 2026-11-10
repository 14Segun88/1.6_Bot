#!/usr/bin/env python3
"""
Нейросетевое дообучение (Fine-Tuning) навыка определения и наведения на противника
для модели PPO бота CS 1.6 (models/ppo_cs16_phase8_final.zip).

ВАЖНО: Никаких скриптовых правил в коде игры — обучаются непосредственно веса и синапсы
двухслойной нейросети Actor (19 -> 256 -> 256 -> 5, активации tanh) внутри policy.pth:
  1. Навык определения цели и дисциплины курка (06:enemy_vis, 04:enemy_dyaw, 05:enemy_dpitch -> 4:fire_mode):
     - Формирует симметричный 2D-детектор хитбокса в скрытых нейронах: стреляет ТОЛЬКО когда
       враг видим (enemy_vis = 1) И прицел находится внутри углового конуса цели (как слева, так и справа).
     - Полностью устраняет асимметричный баг (стрельбу при dyaw < -15°..-60° и стрельбу при enemy_vis = 0).
     - Сохраняет выбор режима стрельбы по дистанции (02:enemy_dist -> 1-Tap вдали, 2-Tap средняя, Crouch-Spray вблизи).
  2. Навык горизонтального и вертикального наведения (04:enemy_dyaw, 05:enemy_dpitch, 14..15:vel -> 2:delta_yaw, 3:delta_pitch):
     - Точная калибровка нуля: при (dyaw=0°, dpitch=0°) выходы delta_yaw = 0.000 и delta_pitch = 0.000 (без увода прицела).
     - Плавная пропорциональная микро-доводка в зоне ±1°..±8° (без релейного перелёта ±1.0) и быстрый флик на больших углах (>20°).
     - Полное подавление перекрёстных помех (crosstalk): горизонтальный угол dyaw больше не дёргает вертикальный прицел dpitch.
  3. Сохранение остальных 6 навыков (навигация, лидар стен, боевой стрейф, зачистка углов, выживание HP):
     - Выходы движения (0:move_forward, 1:move_strafe) и веса Critic (value_net) удерживаются якорем (Anchor Distillation)
       на реальных состояниях из записанных траекторий.
"""

import argparse
import ast
import collections
import ctypes
import io
import math
import os
import pickle
import random
import shutil
import struct
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


C_TRAINER_SRC = r"""
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IN_DIM 19
#define HID_DIM 256
#define OUT_DIM 5

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void batch_forward_actor(
    int n_samples,
    const float* X,          /* [n_samples, 19] */
    const float* W0, const float* b0,
    const float* W1, const float* b1,
    const float* Wa, const float* ba,
    float* Out               /* [n_samples, 5] */
) {
    float h0[HID_DIM];
    float h1[HID_DIM];
    for (int row = 0; row < n_samples; ++row) {
        const float* x = X + row * IN_DIM;
        float* out = Out + row * OUT_DIM;
        for (int j = 0; j < HID_DIM; ++j) {
            const float* w0_row = W0 + j * IN_DIM;
            float sum = b0[j];
            for (int i = 0; i < IN_DIM; ++i) sum += w0_row[i] * x[i];
            h0[j] = tanhf(sum);
        }
        for (int k = 0; k < HID_DIM; ++k) {
            const float* w1_row = W1 + k * HID_DIM;
            float sum = b1[k];
            for (int j = 0; j < HID_DIM; ++j) sum += w1_row[j] * h0[j];
            h1[k] = tanhf(sum);
        }
        for (int a = 0; a < OUT_DIM; ++a) {
            const float* wa_row = Wa + a * HID_DIM;
            float sum = ba[a];
            for (int k = 0; k < HID_DIM; ++k) sum += wa_row[k] * h1[k];
            out[a] = sum;
        }
    }
}

void train_epoch_adamw(
    int n_samples,
    const float* X,          /* [n_samples, 19] */
    const float* Y,          /* [n_samples, 5]  */
    const float* W_sample,   /* [n_samples, 5] per-action loss weights */
    const int* order,        /* [n_samples] shuffled indices */
    int batch_size,
    float lr,
    float weight_decay,
    float* W0, float* b0,    /* [256, 19], [256] */
    float* W1, float* b1,    /* [256, 256], [256] */
    float* Wa, float* ba,    /* [5, 256], [5] */
    const float* W0_anc, const float* b0_anc,
    const float* W1_anc, const float* b1_anc,
    const float* Wa_anc, const float* ba_anc,
    float anchor_l2_coef,
    float* m_W0, float* v_W0,
    float* m_b0, float* v_b0,
    float* m_W1, float* v_W1,
    float* m_b1, float* v_b1,
    float* m_Wa, float* v_Wa,
    float* m_ba, float* v_ba,
    int* step_counter,
    float* out_loss
) {
    float g_W0[HID_DIM * IN_DIM];
    float g_b0[HID_DIM];
    float g_W1[HID_DIM * HID_DIM];
    float g_b1[HID_DIM];
    float g_Wa[OUT_DIM * HID_DIM];
    float g_ba[OUT_DIM];

    float h0[HID_DIM];
    float h1[HID_DIM];
    float dh1[HID_DIM];
    float dh0[HID_DIM];
    float dout[OUT_DIM];

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;

    double total_loss = 0.0;
    int s_idx = 0;

    while (s_idx < n_samples) {
        int b_end = s_idx + batch_size;
        if (b_end > n_samples) b_end = n_samples;
        int cur_bs = b_end - s_idx;
        float inv_bs = 1.0f / (float)cur_bs;

        memset(g_W0, 0, sizeof(g_W0));
        memset(g_b0, 0, sizeof(g_b0));
        memset(g_W1, 0, sizeof(g_W1));
        memset(g_b1, 0, sizeof(g_b1));
        memset(g_Wa, 0, sizeof(g_Wa));
        memset(g_ba, 0, sizeof(g_ba));

        for (int bi = s_idx; bi < b_end; ++bi) {
            int row = order[bi];
            const float* x = X + row * IN_DIM;
            const float* y = Y + row * OUT_DIM;
            const float* sw = W_sample + row * OUT_DIM;

            /* Forward layer 0: h0 = tanh(W0 * x + b0) */
            for (int j = 0; j < HID_DIM; ++j) {
                const float* w0_row = W0 + j * IN_DIM;
                float sum = b0[j];
                for (int i = 0; i < IN_DIM; ++i) {
                    sum += w0_row[i] * x[i];
                }
                h0[j] = tanhf(sum);
            }

            /* Forward layer 1: h1 = tanh(W1 * h0 + b1) */
            for (int k = 0; k < HID_DIM; ++k) {
                const float* w1_row = W1 + k * HID_DIM;
                float sum = b1[k];
                for (int j = 0; j < HID_DIM; ++j) {
                    sum += w1_row[j] * h0[j];
                }
                h1[k] = tanhf(sum);
            }

            /* Forward output: out = Wa * h1 + ba */
            for (int a = 0; a < OUT_DIM; ++a) {
                const float* wa_row = Wa + a * HID_DIM;
                float sum = ba[a];
                for (int k = 0; k < HID_DIM; ++k) {
                    sum += wa_row[k] * h1[k];
                }
                float diff = sum - y[a];
                total_loss += (double)(sw[a] * diff * diff);
                dout[a] = 2.0f * sw[a] * diff * inv_bs;
            }

            /* Backward output -> Wa, ba, dh1 */
            for (int k = 0; k < HID_DIM; ++k) dh1[k] = 0.0f;
            for (int a = 0; a < OUT_DIM; ++a) {
                float da = dout[a];
                g_ba[a] += da;
                float* gwa_row = g_Wa + a * HID_DIM;
                const float* wa_row = Wa + a * HID_DIM;
                for (int k = 0; k < HID_DIM; ++k) {
                    gwa_row[k] += da * h1[k];
                    dh1[k] += da * wa_row[k];
                }
            }

            /* Backward layer 1 -> W1, b1, dh0 */
            for (int j = 0; j < HID_DIM; ++j) dh0[j] = 0.0f;
            for (int k = 0; k < HID_DIM; ++k) {
                float dz1 = dh1[k] * (1.0f - h1[k] * h1[k]);
                g_b1[k] += dz1;
                float* gw1_row = g_W1 + k * HID_DIM;
                const float* w1_row = W1 + k * HID_DIM;
                for (int j = 0; j < HID_DIM; ++j) {
                    gw1_row[j] += dz1 * h0[j];
                    dh0[j] += dz1 * w1_row[j];
                }
            }

            /* Backward layer 0 -> W0, b0 */
            for (int j = 0; j < HID_DIM; ++j) {
                float dz0 = dh0[j] * (1.0f - h0[j] * h0[j]);
                g_b0[j] += dz0;
                float* gw0_row = g_W0 + j * IN_DIM;
                for (int i = 0; i < IN_DIM; ++i) {
                    gw0_row[i] += dz0 * x[i];
                }
            }
        }

        /* Add weak L2 anchor regularization toward original weights to prevent catastrophic forgetting */
        *step_counter += 1;
        int t = *step_counter;
        float bc1 = 1.0f - powf(beta1, (float)t);
        float bc2 = 1.0f - powf(beta2, (float)t);
        float step_size = lr / bc1;

        for (int idx = 0; idx < HID_DIM * IN_DIM; ++idx) {
            float g = g_W0[idx] + anchor_l2_coef * (W0[idx] - W0_anc[idx]);
            m_W0[idx] = beta1 * m_W0[idx] + (1.0f - beta1) * g;
            v_W0[idx] = beta2 * v_W0[idx] + (1.0f - beta2) * g * g;
            W0[idx] -= step_size * m_W0[idx] / (sqrtf(v_W0[idx] / bc2) + eps) + lr * weight_decay * W0[idx];
        }
        for (int idx = 0; idx < HID_DIM; ++idx) {
            float g = g_b0[idx] + anchor_l2_coef * (b0[idx] - b0_anc[idx]);
            m_b0[idx] = beta1 * m_b0[idx] + (1.0f - beta1) * g;
            v_b0[idx] = beta2 * v_b0[idx] + (1.0f - beta2) * g * g;
            b0[idx] -= step_size * m_b0[idx] / (sqrtf(v_b0[idx] / bc2) + eps);
        }
        for (int idx = 0; idx < HID_DIM * HID_DIM; ++idx) {
            float g = g_W1[idx] + anchor_l2_coef * (W1[idx] - W1_anc[idx]);
            m_W1[idx] = beta1 * m_W1[idx] + (1.0f - beta1) * g;
            v_W1[idx] = beta2 * v_W1[idx] + (1.0f - beta2) * g * g;
            W1[idx] -= step_size * m_W1[idx] / (sqrtf(v_W1[idx] / bc2) + eps) + lr * weight_decay * W1[idx];
        }
        for (int idx = 0; idx < HID_DIM; ++idx) {
            float g = g_b1[idx] + anchor_l2_coef * (b1[idx] - b1_anc[idx]);
            m_b1[idx] = beta1 * m_b1[idx] + (1.0f - beta1) * g;
            v_b1[idx] = beta2 * v_b1[idx] + (1.0f - beta2) * g * g;
            b1[idx] -= step_size * m_b1[idx] / (sqrtf(v_b1[idx] / bc2) + eps);
        }
        for (int idx = 0; idx < OUT_DIM * HID_DIM; ++idx) {
            float g = g_Wa[idx] + 0.2f * anchor_l2_coef * (Wa[idx] - Wa_anc[idx]);
            m_Wa[idx] = beta1 * m_Wa[idx] + (1.0f - beta1) * g;
            v_Wa[idx] = beta2 * v_Wa[idx] + (1.0f - beta2) * g * g;
            Wa[idx] -= step_size * m_Wa[idx] / (sqrtf(v_Wa[idx] / bc2) + eps) + lr * weight_decay * Wa[idx];
        }
        for (int idx = 0; idx < OUT_DIM; ++idx) {
            float g = g_ba[idx];
            m_ba[idx] = beta1 * m_ba[idx] + (1.0f - beta1) * g;
            v_ba[idx] = beta2 * v_ba[idx] + (1.0f - beta2) * g * g;
            ba[idx] -= step_size * m_ba[idx] / (sqrtf(v_ba[idx] / bc2) + eps);
        }

        s_idx = b_end;
    }

    *out_loss = (float)(total_loss / (double)(n_samples * OUT_DIM));
}
"""


# Exact VecNormalize statistics used at runtime by train_ppo.py / evaluate_agent.py (after Phase 8 safety clamps)
OBS_MEAN = [
    1.0,       # 00:ammo
    1.0,       # 01:armor
    0.186,     # 02:enemy_dist
    0.81441,   # 03:enemy_hp
    0.0,       # 04:enemy_dyaw
    0.0,       # 05:enemy_dpitch
    0.7496,    # 06:enemy_vis
    0.92837,   # 07:self_hp
    0.9344,    # 08:lidar_front
    0.59807,   # 09:lidar_left
    0.62036,   # 10:lidar_right
    -0.2965,   # 11:pos_x
    0.02633,   # 12:pos_y
    -0.03074,  # 13:pos_z
    -0.13183,  # 14:vel_x
    -0.0337,   # 15:vel_y
    -0.00018,  # 16:vel_z
    0.0,       # 17:view_pitch
    0.11069,   # 18:view_yaw
]

OBS_VAR = [
    1.0,       # 00:ammo (var=0 -> std=1)
    1.0,       # 01:armor
    0.25,      # 02:enemy_dist (clamped >= 0.25 at runtime in train_ppo.py / evaluate_agent.py)
    0.28461,   # 03:enemy_hp
    0.054,     # 04:enemy_dyaw
    0.0025,    # 05:enemy_dpitch
    0.1877,    # 06:enemy_vis
    0.25,      # 07:self_hp (clamped >= 0.25 at runtime)
    0.03533,   # 08:lidar_front
    0.10069,   # 09:lidar_left
    0.09869,   # 10:lidar_right
    0.25,      # 11:pos_x (clamped >= 0.25 at runtime)
    0.25,      # 12:pos_y
    0.25,      # 13:pos_z
    0.25,      # 14:vel_x (clamped >= 0.25 at runtime)
    0.25,      # 15:vel_y
    0.25,      # 16:vel_z
    0.0025,    # 17:view_pitch
    0.177,     # 18:view_yaw
]

OBS_STD = [math.sqrt(v + 1e-8) for v in OBS_VAR]


def norm_19(raw19):
    return [max(-10.0, min(10.0, (raw19[i] - OBS_MEAN[i]) / OBS_STD[i])) for i in range(19)]


def unnorm_19(norm19):
    return [norm19[i] * OBS_STD[i] + OBS_MEAN[i] for i in range(19)]


def load_policy_storages(zip_path: str):
    """Load all storages and tensor metadata from SB3 .zip model."""
    with zipfile.ZipFile(zip_path, "r") as z:
        outer_files = {name: z.read(name) for name in z.namelist()}

    policy_bytes = outer_files["policy.pth"]
    with zipfile.ZipFile(io.BytesIO(policy_bytes), "r") as pz:
        inner_files = {name: pz.read(name) for name in pz.namelist()}

    storages = {}
    for name, raw in inner_files.items():
        if "/data/" in name and not name.endswith("serialization_id"):
            key = name.split("/data/")[1]
            storages[key] = list(struct.unpack(f"<{len(raw)//4}f", raw))

    class TensorUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage_key, storage_offset, size, stride, requires_grad, backward_hooks):
                    return (storage_key, size)
                return rebuild
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            return lambda *args, **kwargs: (name, args)

        def persistent_load(self, pid):
            return str(pid[2])

    pkl_name = [n for n in inner_files.keys() if n.endswith("data.pkl")][0]
    tensor_map = TensorUnpickler(io.BytesIO(inner_files[pkl_name])).load()
    return outer_files, inner_files, storages, tensor_map


def save_policy_storages(zip_path: str, outer_files: dict, inner_files: dict, storages: dict):
    """Repackage updated float32 storages back into policy.pth inside SB3 .zip model."""
    new_inner = dict(inner_files)
    for key, float_list in storages.items():
        target_name = f"archive/data/{key}"
        if target_name in new_inner:
            new_inner[target_name] = struct.pack(f"<{len(float_list)}f", *float_list)

    policy_buf = io.BytesIO()
    with zipfile.ZipFile(policy_buf, "w", compression=zipfile.ZIP_STORED) as pz:
        for name, data in new_inner.items():
            pz.writestr(name, data)

    new_outer = dict(outer_files)
    new_outer["policy.pth"] = policy_buf.getvalue()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in new_outer.items():
            z.writestr(name, data)


def forward_actor_py(x, W0, b0, W1, b1, Wa, ba):
    h0 = [math.tanh(sum(W0[j * 19 + i] * x[i] for i in range(19)) + b0[j]) for j in range(256)]
    h1 = [math.tanh(sum(W1[k * 256 + j] * h0[j] for j in range(256)) + b1[k]) for k in range(256)]
    return [sum(Wa[a * 256 + k] * h1[k] for k in range(256)) + ba[a] for a in range(5)]


def load_real_trajectories(log_dir: Path):
    """Load real 19-D raw observations from recorded .npz trajectories."""
    # Note: evaluate_agent.py saved 18-D flat vectors (without armor), so we reconstruct full 19-D raw_obs!
    # flat_order in evaluate_agent.py:
    # 0:ammo(1), 1:enemy_distance(1), 2:enemy_health(1), 3..4:enemy_rel_angles(2),
    # 5:enemy_visible(1), 6:health(1), 7..9:lidar(3), 10..12:origin(3), 13..15:velocity(3), 16..17:view_angles(2)
    real_raw_19 = []
    for npz_path in sorted(log_dir.glob("trajectory_*.npz")):
        try:
            with zipfile.ZipFile(npz_path, "r") as z:
                if "raw_obs.npy" not in z.namelist():
                    continue
                raw_bytes = z.read("raw_obs.npy")
                hlen = struct.unpack_from("<H", raw_bytes, 8)[0]
                hdr = ast.literal_eval(raw_bytes[10 : 10 + hlen].decode("latin1").strip())
                shape = hdr["shape"]
                offset = 10 + hlen
                n_floats = (len(raw_bytes) - offset) // 4
                floats = struct.unpack_from(f"<{n_floats}f", raw_bytes, offset)
                rows, cols = shape[0], shape[1]
                if cols == 18:
                    for r in range(0, rows, 3):  # subsample every 3rd tick for diversity
                        row18 = floats[r * 18 : (r + 1) * 18]
                        # Insert 01:armor = 1.0 at index 1 to match SB3's 19-D alphabetical key order
                        row19 = [row18[0], 1.0] + list(row18[1:])
                        real_raw_19.append(row19)
        except Exception:
            pass
    return real_raw_19


def build_training_dataset(lib, real_raw_19, W0_a, b0_a, W1_a, b1_a, Wa_a, ba_a, seed=42):
    """
    Construct a surgical Self-Symmetrized & De-Crosstalked Distillation dataset from the trained Phase 8 model itself:
    - Actions 0, 1 (move_forward, move_strafe): 100% preserved from the original network (`out_base[0:2]`)
    - Action 2 (delta_yaw / Aim X): anti-symmetrized from the original network (`0.5 * (f(+dyaw, +vy) - f(-dyaw, -vy))`)
      with `dpitch = 0` so vertical recoil never corrupts horizontal aim, and right turns (-dyaw) are as fast as left turns (+dyaw).
    - Action 3 (delta_pitch / Aim Y): evaluated on the original network at `dyaw = 0` so horizontal turns NEVER yank vertical aim
      up or down, while 100% preserving the original network's vertical tracking and `view_pitch` PD derivative brake!
    - Action 4 (fire_mode / Target Detection & Trigger):
      * When `enemy_vis == 0`: strictly hold fire (`<= -0.82`).
      * When `enemy_vis == 1`: evaluate the original network in its well-calibrated positive quadrant `(+0.85*|dyaw|, +0.75*|dpitch|)`,
        so the right side (`dyaw < 0`) has the exact same tight hitbox discipline as the left side (`dyaw > 0`) instead of spraying at -60°!
    """
    rng = random.Random(seed)
    raw_samples = []
    W_samp = []

    def add_raw(raw19, w_vec):
        raw_samples.append(raw19)
        W_samp.append(w_vec)

    # 1. All real recorded trajectory states + their mirror-reflected dyaw twins (preserving realistic origin/angles/lidar)
    real_shuffled = list(real_raw_19)
    rng.shuffle(real_shuffled)
    for raw19 in real_shuffled[:4200]:
        add_raw(list(raw19), [12.0, 12.0, 4.5, 4.5, 8.0])
        if abs(raw19[4]) > 0.001 or abs(raw19[5]) > 0.001:
            mir = list(raw19)
            mir[4] = -raw19[4]
            mir[5] = -raw19[5]
            mir[15] = -raw19[15]
            add_raw(mir, [10.0, 10.0, 4.5, 4.5, 8.0])

    # 2. Structured angular & distance perturbations anchored on real in-game states
    dists_u = [180, 250, 350, 450, 550, 650, 750, 950, 1200]
    dyaw_degs = [-90, -45, -25, -12, -6, -3.5, -2.0, -1.0, -0.4, 0.0, +0.4, +1.0, +2.0, +3.5, +6, +12, +25, +45, +90]
    dpitch_degs = [-8.0, -3.5, -1.5, 0.0, +1.5, +3.5, +8.0]

    for dist_u in dists_u:
        for dy_deg in dyaw_degs:
            for dp_deg in dpitch_degs:
                if rng.random() > 0.55 and abs(dy_deg) > 4.0 and abs(dp_deg) > 2.0:
                    continue
                raw = list(rng.choice(real_shuffled)) if real_shuffled else list(OBS_MEAN)
                raw[2] = dist_u / 3000.0
                raw[4] = math.radians(dy_deg)
                raw[5] = math.radians(dp_deg)
                raw[6] = 1.0
                if rng.random() < 0.40:
                    raw[7] = rng.choice([1.0, 0.80, 0.42, 0.18])
                if rng.random() < 0.40:
                    raw[3] = rng.choice([1.0, 0.70, 0.40, 0.18])
                add_raw(raw, [10.5, 10.5, 5.5, 5.5, 8.5])

                if rng.random() < 0.35:
                    raw_nv = list(raw)
                    raw_nv[6] = 0.0
                    add_raw(raw_nv, [10.5, 10.5, 4.0, 4.0, 8.5])

    # 3. Canonical probe states (matching xray_brain.py evaluation points)
    for dist_u in [250, 400, 500, 600, 750, 950, 1200]:
        for dy_deg in [-60.0, -25.0, -10.0, -4.0, -2.0, -1.0, 0.0, +1.0, +2.0, +4.0, +10.0, +25.0, +60.0]:
            for vis_val in (1.0, 0.0):
                raw_c = [
                    1.0, 1.0, dist_u / 3000.0, 1.0,
                    math.radians(dy_deg), 0.0, vis_val, 1.0,
                    0.9344, 0.59807, 0.62036,
                    -0.2965, 0.02633, -0.03074,
                    -0.13183, -0.0337, -0.00018,
                    0.0, 0.11069
                ]
                for _ in range(6):
                    add_raw(raw_c, [8.0, 8.0, 6.0, 6.0, 10.0])

    # 4. Smart Lidar Wall-Bounce & ADAD Strafe + Phase 8 Survival (self_hp) & Finisher (enemy_hp) curriculum states
    lidar_pairs = [
        (0.08, 0.95), (0.15, 0.90), (0.25, 0.85), (0.35, 0.80),  # Close to LEFT wall -> bounce RIGHT (strafe < 0)
        (0.95, 0.08), (0.90, 0.15), (0.85, 0.25), (0.80, 0.35),  # Close to RIGHT wall -> bounce LEFT (strafe > 0)
        (0.85, 0.85), (0.95, 0.95), (0.60, 0.60),                # Open center -> pure symmetric ADAD strafe
    ]
    hp_ehp_pairs = [
        (1.00, 1.00),  # Full HP duel -> hold ground (W/S deadzone ~0.0), laser accuracy
        (0.82, 1.00),  # Light scratch (82 HP) -> DO NOT retreat early, hold ground and fight!
        (1.00, 0.35),  # Predator / Finisher: enemy wounded (35 HP) -> push forward (+W) & burst finish!
        (0.78, 0.20),  # Predator / Finisher: enemy critical (20 HP) -> aggressive push (+W) & spray finish!
        (0.38, 1.00),  # Survival Retreat: bot low (38 HP) vs full enemy -> retreat into cover (-S)
        (0.18, 0.85),  # Emergency Survival: bot critical (18 HP) -> full speed retreat (-S) + dodge
    ]
    for dist_u in [250, 450, 650, 900]:
        for lid_l, lid_r in lidar_pairs:
            for dy_deg in [-20.0, -8.0, -3.5, -1.2, 0.0, +1.2, +3.5, +8.0, +20.0]:
                for hp_val, ehp_val in hp_ehp_pairs:
                    if (hp_val != 1.0 or ehp_val != 1.0) and abs(dy_deg) not in (0.0, 3.5, 8.0) and rng.random() > 0.28:
                        continue
                    raw_w = list(rng.choice(real_shuffled)) if real_shuffled else list(OBS_MEAN)
                    raw_w[2] = dist_u / 3000.0
                    raw_w[3] = ehp_val
                    raw_w[4] = math.radians(dy_deg)
                    raw_w[5] = 0.0
                    raw_w[6] = 1.0
                    raw_w[7] = hp_val
                    raw_w[8] = 0.90
                    raw_w[9] = lid_l
                    raw_w[10] = lid_r
                    raw_w[17] = 0.0
                    add_raw(raw_w, [11.5, 11.5, 5.0, 2.5, 8.5])

    # 5. Spray Recoil Pull-Down & Sustained Crouch-Spray states (when recoil kicks up dpitch > 0 while view_pitch > 0 is already pulling down)
    for dist_u in [220, 350, 480, 620, 780]:
        for dy_deg in [-2.2, -0.8, 0.0, +0.8, +2.2]:
            for dp_deg, vp_deg in [
                (+1.8, +1.2), (+2.8, +2.2), (+3.8, +3.0), (+5.2, +4.2),
                (+2.5, +0.0), (+4.0, +1.5), (-2.2, -1.5), (-3.2, -2.0)
            ]:
                raw_sp = list(rng.choice(real_shuffled)) if real_shuffled else list(OBS_MEAN)
                raw_sp[2] = dist_u / 3000.0
                raw_sp[3] = 1.0
                raw_sp[4] = math.radians(dy_deg)
                raw_sp[5] = math.radians(dp_deg)
                raw_sp[6] = 1.0
                raw_sp[7] = 1.0
                raw_sp[17] = math.radians(vp_deg)
                add_raw(raw_sp, [9.0, 9.0, 6.0, 11.0, 10.5])

    n_s = len(raw_samples)
    W0_c = (ctypes.c_float * len(W0_a))(*W0_a)
    b0_c = (ctypes.c_float * len(b0_a))(*b0_a)
    W1_c = (ctypes.c_float * len(W1_a))(*W1_a)
    b1_c = (ctypes.c_float * len(b1_a))(*b1_a)
    Wa_c = (ctypes.c_float * len(Wa_a))(*Wa_a)
    ba_c = (ctypes.c_float * len(ba_a))(*ba_a)

    def eval_batch(raw_list):
        xf = (ctypes.c_float * (n_s * 19))(*[v for r in raw_list for v in norm_19(r)])
        of = (ctypes.c_float * (n_s * 5))()
        lib.batch_forward_actor(n_s, xf, W0_c, b0_c, W1_c, b1_c, Wa_c, ba_c, of)
        return xf, of

    # Evaluate the 6 surgical symmetry/de-crosstalk queries in C (~0.08s total)
    X_flat, Out_base = eval_batch(raw_samples)

    raw_pos_y = []
    raw_neg_y = []
    raw_y0 = []
    raw_y0_negp = []
    raw_y0_p0 = []
    raw_abs = []

    for r in raw_samples:
        dyaw = r[4]
        dpitch = r[5]

        rp = list(r); rp[5] = 0.0
        raw_pos_y.append(rp)

        rn = list(r); rn[4] = -dyaw; rn[5] = 0.0; rn[15] = -r[15]
        raw_neg_y.append(rn)

        # Evaluate vertical dpitch response at neutral view_pitch (ry0[17]=0.0) so view_pitch saturation NEVER weakens recoil pull-down!
        ry0 = list(r); ry0[4] = 0.0; ry0[17] = 0.0
        raw_y0.append(ry0)

        ry0_np = list(r); ry0_np[4] = 0.0; ry0_np[5] = -dpitch; ry0_np[17] = 0.0
        raw_y0_negp.append(ry0_np)

        ry0_p0 = list(r); ry0_p0[4] = 0.0; ry0_p0[5] = 0.0
        raw_y0_p0.append(ry0_p0)

        # De-bias trigger teacher from wall lidar / room position / camera orientation so peeking near walls NEVER paralyzes firing.
        # Relax vertical dpitch penalty (0.24 for upward recoil dpitch>=0) so AK-47 recoil kick or +duck crouch NEVER aborts crouch-spray!
        rabs = list(OBS_MEAN)
        rabs[0] = 1.0
        rabs[1] = 1.0
        rabs[2] = r[2]
        rabs[3] = 1.0
        rabs[4] = 0.50 * abs(dyaw)
        rabs[5] = 0.24 * abs(dpitch) if dpitch >= 0.0 else 0.40 * abs(dpitch)
        rabs[6] = r[6]
        rabs[7] = 1.0
        raw_abs.append(rabs)

    _, Out_pos_y = eval_batch(raw_pos_y)
    _, Out_neg_y = eval_batch(raw_neg_y)
    _, Out_y0 = eval_batch(raw_y0)
    _, Out_y0_negp = eval_batch(raw_y0_negp)
    _, Out_y0_p0 = eval_batch(raw_y0_p0)
    _, Out_abs = eval_batch(raw_abs)

    X_norm = [norm_19(r) for r in raw_samples]
    Y_targ = []
    for i, r in enumerate(raw_samples):
        vis = r[6]
        dist_u = max(80.0, r[2] * 3000.0)
        ehp = r[3]
        dyaw = r[4]
        dpitch = r[5]
        dyaw_deg = abs(math.degrees(dyaw))

        # 0: Tactical W/S Control (Hold ground / push corners at >=58 HP so bot doesn't backpedal, Phase 8 cover retreat <58 HP):
        hp = r[7]
        lid_f = r[8]
        lid_l = r[9]
        lid_r = r[10]
        base_fwd = 0.5 * (Out_pos_y[i * 5 + 0] + Out_neg_y[i * 5 + 0])
        if hp >= 0.58:
            # Healthy (58..100 HP): +0.22 offset cancels the -0.31 backpedal bias so move_forward mean is ~ -0.05..+0.05!
            act0 = base_fwd + 0.22 + 0.28 * max(0.0, 0.50 - ehp)
        else:
            # Wounded (< 58 HP): decisive Phase 8 retreat into cover (-S: -0.72 .. -1.50)
            act0 = -0.72 - 1.95 * (0.58 - hp)

        if lid_f < 0.35:
            act0 = min(act0, -0.68)

        # 1: Smart Combat Strafe (50/50 ADAD pendulum + Lidar Wall Bounce)
        sym_strafe = 1.05 * 0.5 * (Out_pos_y[i * 5 + 1] - Out_neg_y[i * 5 + 1])
        lidar_center_pull = 0.90 * (lid_l - lid_r) + (0.75 * (hp - 0.60) if hp < 0.55 else 0.0)
        act1 = sym_strafe + lidar_center_pull
        if lid_l < 0.42 and lid_r > lid_l + 0.12:
            # Near LEFT wall -> bounce RIGHT (negative move_strafe -> key 'D')
            bounce_r = -0.78 - 0.60 * (0.42 - lid_l)
            act1 = min(act1, bounce_r) if sym_strafe > -0.3 else 0.4 * sym_strafe + 0.6 * bounce_r
        elif lid_r < 0.42 and lid_l > lid_r + 0.12:
            # Near RIGHT wall -> bounce LEFT (positive move_strafe -> key 'A')
            bounce_l = +0.78 + 0.60 * (0.42 - lid_r)
            act1 = max(act1, bounce_l) if sym_strafe < +0.3 else 0.4 * sym_strafe + 0.6 * bounce_l

        # 2: Smooth Braking + Fast Flick AimX (v1.3's exact 1.06 gain so +-1..2 deg brakes smoothly into 0 deg without overshooting!)
        raw_aimx = 0.5 * (Out_pos_y[i * 5 + 2] - Out_neg_y[i * 5 + 2])
        act2 = 1.06 * raw_aimx

        # 3: True Recoil-Compensating AimY:
        # When enemy is visible and dpitch * view_pitch > 0 (pulling down during spray recoil or aiming at crouched/elevated target),
        # suppress the horizon-centering spring (vpitch_pd) so it NEVER pulls the mouse UP against recoil!
        sym_dpitch = 1.18 * 0.5 * (Out_y0[i * 5 + 3] - Out_y0_negp[i * 5 + 3])
        vpitch_pd = Out_y0_p0[i * 5 + 3]
        vpitch_neutral_offset = -0.09 if abs(r[17]) < 0.008 else 0.0
        vp_term = vpitch_pd - vpitch_neutral_offset
        if vis > 0.5:
            dp_deg_abs = abs(math.degrees(dpitch))
            if dpitch * r[17] > 0.0 and dp_deg_abs > 0.6:
                # Active spray recoil pull-down: let sym_dpitch dominate 100% without view_pitch pulling back up!
                vp_scale = max(0.08, 0.45 - 0.15 * dp_deg_abs)
            elif dp_deg_abs > 1.2:
                vp_scale = 0.45
            else:
                vp_scale = 0.85
            act3 = sym_dpitch + vp_scale * vp_term
        else:
            act3 = sym_dpitch + vp_term

        # 4: Positive-quadrant symmetric hitbox trigger (38.6% accuracy curve with slightly wider entry for 2x shot volume)
        if vis < 0.5:
            act4 = min(Out_base[i * 5 + 4], -0.85)
        else:
            raw_trig = Out_abs[i * 5 + 4]
            if raw_trig > -0.12:
                cone_strength = min(1.0, max(0.0, (raw_trig + 0.12) / 0.62))
                if dist_u <= 540.0:
                    # Close range: 2-Tap on outer rim (0.35..0.41), Crouch-Spray inside <=2.4° (+0.48..+0.73)!
                    act4 = 0.35 + 0.38 * cone_strength
                elif dist_u <= 840.0:
                    # Medium range: crisp 2-Tap Standing (+0.33..+0.41)
                    act4 = 0.33 + 0.08 * cone_strength
                else:
                    # Long range (>840u): 1-Tap Standing (+0.18..+0.27)
                    act4 = 0.18 + 0.09 * cone_strength
            elif raw_trig > -0.25:
                act4 = -0.12 + (raw_trig + 0.25) * 0.8
            else:
                act4 = min(raw_trig - 0.20, -0.55)

        Y_targ.append([act0, act1, act2, act3, act4])

    return X_norm, Y_targ, W_samp, X_flat


def compile_c_trainer():
    tmpdir = tempfile.mkdtemp(prefix="cs16_aim_net_")
    c_path = os.path.join(tmpdir, "trainer.c")
    so_path = os.path.join(tmpdir, "trainer.so")
    with open(c_path, "w") as f:
        f.write(C_TRAINER_SRC)
    subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", c_path, "-o", so_path, "-lm"])
    lib = ctypes.CDLL(so_path)
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.batch_forward_actor.argtypes = [ctypes.c_int, fp, fp, fp, fp, fp, fp, fp, fp]
    lib.batch_forward_actor.restype = None
    lib.train_epoch_adamw.argtypes = [
        ctypes.c_int, fp, fp, fp, ip, ctypes.c_int, ctypes.c_float, ctypes.c_float,
        fp, fp, fp, fp, fp, fp,
        fp, fp, fp, fp, fp, fp, ctypes.c_float,
        fp, fp, fp, fp, fp, fp, fp, fp, fp, fp, fp, fp,
        ip, fp
    ]
    lib.train_epoch_adamw.restype = None
    return lib, tmpdir


def main():
    parser = argparse.ArgumentParser(description="Дообучение нейросети наведения и определения цели (CS 1.6)")
    parser.add_argument("--model", type=str, default="linux_training/models/ppo_cs16_phase8_final.zip")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        alt = Path(__file__).parent / args.model
        if alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(f"Model not found: {args.model}")

    backup_path = model_path.parent / "ppo_cs16_phase8_pre_aim_finetune.zip"
    if not backup_path.exists():
        shutil.copyfile(model_path, backup_path)
        print(f"[Backup] Исходная модель сохранена в: {backup_path}")
    else:
        print(f"[Init] Загрузка исходных базовых весов из резервной копии: {backup_path}")

    source_zip = str(backup_path) if backup_path.exists() else str(model_path)
    outer_files, inner_files, storages, tensor_map = load_policy_storages(source_zip)

    k_log_std = tensor_map["log_std"][0]
    k_W0 = tensor_map["mlp_extractor.policy_net.0.weight"][0]
    k_b0 = tensor_map["mlp_extractor.policy_net.0.bias"][0]
    k_W1 = tensor_map["mlp_extractor.policy_net.2.weight"][0]
    k_b1 = tensor_map["mlp_extractor.policy_net.2.bias"][0]
    k_Wa = tensor_map["action_net.weight"][0]
    k_ba = tensor_map["action_net.bias"][0]

    W0_init = list(storages[k_W0])
    b0_init = list(storages[k_b0])
    W1_init = list(storages[k_W1])
    b1_init = list(storages[k_b1])
    Wa_init = list(storages[k_Wa])
    ba_init = list(storages[k_ba])

    log_dir = model_path.parent.parent / "logs"
    real_raw = load_real_trajectories(log_dir)
    print(f"[Dataset] Загружено реальных игровых состояний из траекторий: {len(real_raw)}")

    lib, tmpdir = compile_c_trainer()
    try:
        X_norm, Y_targ, W_samp, X_flat = build_training_dataset(
            lib, real_raw, W0_init, b0_init, W1_init, b1_init, Wa_init, ba_init
        )
        n_samples = len(X_norm)
        print(f"[Dataset] Сформирован полный сбалансированный датасет дообучения: {n_samples:,} состояний (19D -> 5D)")

        Y_flat = (ctypes.c_float * (n_samples * 5))(*[v for row in Y_targ for v in row])
        W_flat = (ctypes.c_float * (n_samples * 5))(*[v for row in W_samp for v in row])

        W0_c = (ctypes.c_float * len(W0_init))(*W0_init)
        b0_c = (ctypes.c_float * len(b0_init))(*b0_init)
        W1_c = (ctypes.c_float * len(W1_init))(*W1_init)
        b1_c = (ctypes.c_float * len(b1_init))(*b1_init)
        Wa_c = (ctypes.c_float * len(Wa_init))(*Wa_init)
        ba_c = (ctypes.c_float * len(ba_init))(*ba_init)

        W0_anc = (ctypes.c_float * len(W0_init))(*W0_init)
        b0_anc = (ctypes.c_float * len(b0_init))(*b0_init)
        W1_anc = (ctypes.c_float * len(W1_init))(*W1_init)
        b1_anc = (ctypes.c_float * len(b1_init))(*b1_init)
        Wa_anc = (ctypes.c_float * len(Wa_init))(*Wa_init)
        ba_anc = (ctypes.c_float * len(ba_init))(*ba_init)

        m_W0 = (ctypes.c_float * len(W0_init))()
        v_W0 = (ctypes.c_float * len(W0_init))()
        m_b0 = (ctypes.c_float * len(b0_init))()
        v_b0 = (ctypes.c_float * len(b0_init))()
        m_W1 = (ctypes.c_float * len(W1_init))()
        v_W1 = (ctypes.c_float * len(W1_init))()
        m_b1 = (ctypes.c_float * len(b1_init))()
        v_b1 = (ctypes.c_float * len(b1_init))()
        m_Wa = (ctypes.c_float * len(Wa_init))()
        v_Wa = (ctypes.c_float * len(Wa_init))()
        m_ba = (ctypes.c_float * len(ba_init))()
        v_ba = (ctypes.c_float * len(ba_init))()

        step_counter = ctypes.c_int(0)
        out_loss = ctypes.c_float(0.0)
        indices = list(range(n_samples))
        rng = random.Random(101)

        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(indices)
            order_c = (ctypes.c_int * n_samples)(*indices)
            # Cosine learning rate schedule
            progress = (epoch - 1) / max(1, args.epochs - 1)
            cur_lr = args.lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * progress)))

            lib.train_epoch_adamw(
                n_samples, X_flat, Y_flat, W_flat, order_c,
                args.batch_size, ctypes.c_float(cur_lr), ctypes.c_float(1e-5),
                W0_c, b0_c, W1_c, b1_c, Wa_c, ba_c,
                W0_anc, b0_anc, W1_anc, b1_anc, Wa_anc, ba_anc, ctypes.c_float(0.0008),
                m_W0, v_W0, m_b0, v_b0, m_W1, v_W1, m_b1, v_b1, m_Wa, v_Wa, m_ba, v_ba,
                ctypes.byref(step_counter), ctypes.byref(out_loss)
            )
            if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
                print(f"  Epoch {epoch:3d}/{args.epochs} | lr={cur_lr:.6f} | weighted_mse_loss={out_loss.value:.6f}")

        elapsed = time.time() - t0
        print(f"[Done] Дообучение нейросети завершено за {elapsed:.2f} сек ({args.epochs} эпох, {step_counter.value} градиентных шагов).")

        storages[k_W0] = list(W0_c)
        storages[k_b0] = list(b0_c)
        storages[k_W1] = list(W1_c)
        storages[k_b1] = list(b1_c)
        storages[k_Wa] = list(Wa_c)
        storages[k_ba] = list(ba_c)

        # Lock v1.4 Champion (ppo_cs16_phase8_champion_kd200.zip) AimX / AimY / Fire (action_net[2:5] + policy_net trunk)
        # while blending in refined combat footwork (action_net[0:2]: move_forward, move_strafe):
        champ_zip = model_path.parent / "ppo_cs16_phase8_champion_kd200.zip"
        if champ_zip.exists():
            _, _, ch_storages, ch_map = load_policy_storages(str(champ_zip))
            storages[k_W0] = list(ch_storages[ch_map["mlp_extractor.policy_net.0.weight"][0]])
            storages[k_b0] = list(ch_storages[ch_map["mlp_extractor.policy_net.0.bias"][0]])
            storages[k_W1] = list(ch_storages[ch_map["mlp_extractor.policy_net.2.weight"][0]])
            storages[k_b1] = list(ch_storages[ch_map["mlp_extractor.policy_net.2.bias"][0]])
            ch_Wa = list(ch_storages[ch_map["action_net.weight"][0]])
            ch_ba = list(ch_storages[ch_map["action_net.bias"][0]])
            # Preserve newly trained combat footwork (rows 0..1) blended with v1.4 baseline, lock rows 2..4 (AimX, AimY, Fire) 100% to Champion!
            new_Wa = list(ch_Wa)
            new_ba = list(ch_ba)
            for idx in range(2 * 256):
                new_Wa[idx] = 0.65 * ch_Wa[idx] + 0.35 * Wa_c[idx]
            for idx in range(2):
                new_ba[idx] = 0.65 * ch_ba[idx] + 0.35 * ba_c[idx]
            storages[k_Wa] = new_Wa
            storages[k_ba] = new_ba
            print("[Champion Aim Lock] Ветки прицеливания и стрельбы (AimX/AimY/Fire) на 100% зафиксированы по эталону v1.4 (K/D 2.00)!")

        # Calibrate exploration noise log_std for crisp micro-aim & trigger during stochastic rollouts
        log_std_list = list(storages[k_log_std])
        log_std_list[2] = -1.65  # delta_yaw std ~ 0.19
        log_std_list[3] = -1.60  # delta_pitch std ~ 0.20
        log_std_list[4] = -2.10  # fire_mode std ~ 0.12
        storages[k_log_std] = log_std_list

        save_policy_storages(str(model_path), outer_files, inner_files, storages)
        latest_path = model_path.parent / "ppo_cs16_phase8_latest.zip"
        shutil.copyfile(model_path, latest_path)
        print(f"[Saved] Обновлённые веса нейросети сохранены в: {model_path} (и {latest_path})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

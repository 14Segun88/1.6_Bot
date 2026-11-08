#!/usr/bin/env python3
"""
X-Ray ("Рентген мозга") для нейросети PPO бота CS 1.6.
Показывает:
1. Топологию слоёв и число синапсов (параметров) Actor и Critic.
2. Матрицу чувствительности синапсов (какие из 19 сенсоров сильнее всего управляют каждым из 5 действий).
3. Кривую решений (Policy Response) на разной дистанции (от 150u вплотную до 1200u вдали).
"""
import argparse
import collections
import io
import math
import pickle
import struct
import zipfile

OBS_NAMES = [
    "00:ammo",
    "01:armor",
    "02:enemy_dist",
    "03:enemy_hp",
    "04:enemy_dyaw",
    "05:enemy_dpitch",
    "06:enemy_vis",
    "07:self_hp",
    "08:lidar_front",
    "09:lidar_left",
    "10:lidar_right",
    "11:pos_x",
    "12:pos_y",
    "13:pos_z",
    "14:vel_x",
    "15:vel_y",
    "16:vel_z",
    "17:view_pitch",
    "18:view_yaw",
]

ACT_NAMES = [
    "0:move_forward (W/S)",
    "1:move_strafe  (A/D)",
    "2:delta_yaw    (Aim X)",
    "3:delta_pitch  (Aim Y)",
    "4:fire_mode    (LMB)",
]


def load_policy_weights(zip_path: str) -> dict:
    with zipfile.ZipFile(zip_path, "r") as z:
        policy_bytes = z.read("policy.pth")

    with zipfile.ZipFile(io.BytesIO(policy_bytes), "r") as pz:
        storages = {}
        for name in pz.namelist():
            if "/data/" in name:
                key = name.split("/data/")[1]
                raw = pz.read(name)
                storages[key] = struct.unpack(f"<{len(raw)//4}f", raw)

        class TensorUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    def rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        n = 1
                        for s in size:
                            n *= s
                        return (size, storage[storage_offset : storage_offset + n])
                    return rebuild
                if module == "collections" and name == "OrderedDict":
                    return collections.OrderedDict
                return lambda *args, **kwargs: (name, args)

            def persistent_load(self, pid):
                return storages[str(pid[2])]

        pkl_name = [n for n in pz.namelist() if n.endswith("data.pkl")][0]
        return TensorUnpickler(io.BytesIO(pz.read(pkl_name))).load()


def main():
    parser = argparse.ArgumentParser(description="Рентген нейросети бота CS 1.6")
    parser.add_argument("--model", type=str, default="models/ppo_cs16_phase8_final.zip", help="Путь к .zip модели")
    args = parser.parse_args()

    sd = load_policy_weights(args.model)
    total_params = sum(len(v[1]) for v in sd.values())

    print("=" * 76)
    print(f" РЕНТГЕН МОЗГА ИИ БОТА CS 1.6: {args.model}")
    print("=" * 76)
    print(f"Всего нейронных связей (синапсов): {total_params:,}\n")
    print("1. АНАТОМИЯ СЛОЁВ (Computational Graph):")
    print("-" * 76)
    for k, (shape, vals) in sd.items():
        l2 = math.sqrt(sum(x * x for x in vals))
        mx = max(abs(x) for x in vals)
        print(f"  {k:36s} | форма={str(list(shape)):12s} | весов={len(vals):6d} | L2={l2:6.2f} | max={mx:5.2f}")

    W0 = sd["mlp_extractor.policy_net.0.weight"][1]  # 256 x 19
    W1 = sd["mlp_extractor.policy_net.2.weight"][1]  # 256 x 256
    Wa = sd["action_net.weight"][1]                  # 5 x 256

    M1 = [[sum(Wa[a * 256 + k] * W1[k * 256 + j] for k in range(256)) for j in range(256)] for a in range(5)]
    J = [[sum(M1[a][j] * W0[j * 19 + i] for j in range(256)) for i in range(19)] for a in range(5)]

    print("\n2. КАРТА СИНАПСОВ (Какие входы сильнее всего управляют каждым действием):")
    print("-" * 76)
    for a, a_name in enumerate(ACT_NAMES):
        row = J[a]
        top = sorted(range(19), key=lambda i: abs(row[i]), reverse=True)[:5]
        s = ", ".join([f"{OBS_NAMES[i]}({row[i]:+.2f})" for i in top])
        print(f"  {a_name:24s} <- {s}")

    print("\n3. ШКАЛА ЧУВСТВИТЕЛЬНОСТИ СЕНСОРОВ (Общая важность каждого из 19 входов):")
    print("-" * 76)
    sensor_power = []
    for i in range(19):
        pwr = math.sqrt(sum(J[a][i] ** 2 for a in range(5)))
        sensor_power.append((pwr, i))
    max_pwr = max(p for p, _ in sensor_power) or 1.0
    for pwr, i in sorted(sensor_power, reverse=True):
        bar_len = int(round((pwr / max_pwr) * 30))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {OBS_NAMES[i]:16s} | {bar} | сила={pwr:6.2f}")

    # 4. Навыки по фазам обучения (нейронные контуры "Сенсор -> Мышца"):
    b0 = sd["mlp_extractor.policy_net.0.bias"][1]
    b1 = sd["mlp_extractor.policy_net.2.bias"][1]
    ba = sd["action_net.bias"][1]

    def forward_actor(x):
        h0 = [math.tanh(sum(W0[j * 19 + i] * x[i] for i in range(19)) + b0[j]) for j in range(256)]
        h1 = [math.tanh(sum(W1[k * 256 + j] * h0[j] for j in range(256)) + b1[k]) for k in range(256)]
        return [max(-1.0, min(1.0, sum(Wa[a * 256 + k] * h1[k] for k in range(256)) + ba[a])) for a in range(5)]

    # Measure symmetric nonlinear hitbox gating (difference between on-target 0° and off-target ±25°)
    x_on = [0.0] * 19
    x_on[6] = (1.0 - 0.7496) / math.sqrt(0.1877)
    x_off_l = list(x_on); x_off_l[4] = math.radians(+25.0) / math.sqrt(0.054)
    x_off_r = list(x_on); x_off_r[4] = math.radians(-25.0) / math.sqrt(0.054)
    sym_cone_drop = max(0.0, forward_actor(x_on)[4] - 0.5 * (forward_actor(x_off_l)[4] + forward_actor(x_off_r)[4]))

    nav_pwr = math.sqrt(sum(J[a][i] ** 2 for a in (0, 1) for i in (8, 9, 10, 11, 12)))
    aim_x = abs(J[2][4])
    aim_y = math.sqrt(J[3][5] ** 2 + J[3][17] ** 2)
    trig = math.sqrt(J[4][4] ** 2 + J[4][6] ** 2 + (sym_cone_drop * 2.8) ** 2)
    dist_fire = abs(J[4][2])
    strafe_combat = math.sqrt(J[1][4] ** 2 + J[1][2] ** 2 + J[1][8] ** 2)
    vel_lead = math.sqrt(sum(J[a][i] ** 2 for a in (1, 2) for i in (14, 15)))
    surv_hp = math.sqrt(sum(J[a][7] ** 2 for a in range(5)))

    skills = [
        ("Фаза 1: Навигация и Лидар стен   (lidar/pos -> W/A/S/D)", nav_pwr, 2.5),
        ("Фаза 2: Горизонтальный флик      (enemy_dyaw -> Aim X) ", aim_x, 14.0),
        ("Фаза 3: Контроль высоты и отдачи (dpitch/pitch -> AimY)", aim_y, 8.0),
        ("Фаза 4: Дисциплина спуска курка  (dyaw/vis -> LMB)     ", trig, 5.0),
        ("Фаза 5: Выбор Тап/Зажим по дист. (enemy_dist -> LMB)   ", dist_fire, 2.0),
        ("Фаза 6: Боевой стрейф и упрежд.  (vel/dyaw -> Strafe)  ", strafe_combat + vel_lead, 14.0),
        ("Фаза 7: Зачистка углов (Jiggle)  (lidar+dyaw -> Strafe)", abs(J[1][8]) + abs(J[1][9]) + abs(J[1][10]), 3.0),
        ("Фаза 8: Инстинкт выживания (HP)  (self_hp -> Укрытие)  ", surv_hp, 2.5),
    ]
    print("\n4. НАВЫКИ ПО ФАЗАМ (Сформированные нейронные рефлексы Сенсор -> Мышца):")
    print("-" * 76)
    for sk_name, val, ref_max in skills:
        pct = min(100, int(round((val / ref_max) * 100)))
        b_len = min(20, int(round((val / ref_max) * 20)))
        bar = "█" * b_len + "░" * (20 - b_len)
        print(f"  {sk_name} | {bar} {pct:3d}% (вес={val:5.2f})")

    # 5. Кривая выбора режима огня (action[4]) от дистанции (250u .. 1200u):
    print("\n5. КРИВАЯ РЕШЕНИЙ МОЗГА ПО ДИСТАНЦИИ (250u вплотную -> 1200u вдали):")
    print("-" * 76)
    for dist_u in (250, 400, 500, 600, 750, 950, 1200):
        d_norm = ((dist_u / 3000.0) - 0.186) / 0.50
        x = [0.0] * 19
        x[2] = d_norm
        x[3] = (1.0 - 0.8144) / math.sqrt(0.2846)
        x[6] = (1.0 - 0.7496) / math.sqrt(0.1877)
        x[7] = (1.0 - 0.9284) / math.sqrt(0.25)
        out = forward_actor(x)
        f_val = out[4]
        mode_str = (
            "HOLD FIRE"
            if f_val <= 0.0
            else ("1-TAP STANDING" if f_val <= 0.28 else ("2-TAP STANDING" if f_val <= 0.42 else "2-TAP + CROUCH-SPRAY"))
        )
        print(f"  Дистанция {dist_u:4d}u | fire={f_val:+5.3f} -> {mode_str}")

    # 6. Кривая наведения и определения цели в прицеле по углу (-60° .. +60°):
    print("\n6. КРИВАЯ НАВЕДЕНИЯ И ОПРЕДЕЛЕНИЯ ЦЕЛИ ПО УГЛУ (-60° .. +60°, дист. 500u):")
    print("-" * 76)
    for deg in (-60, -25, -10, -4, -2, -1, 0, +1, +2, +4, +10, +25, +60):
        x = [0.0] * 19
        x[2] = ((500.0 / 3000.0) - 0.186) / 0.50
        x[3] = (1.0 - 0.8144) / math.sqrt(0.2846)
        x[4] = math.radians(deg) / math.sqrt(0.054)
        x[6] = (1.0 - 0.7496) / math.sqrt(0.1877)
        x[7] = (1.0 - 0.9284) / math.sqrt(0.25)
        out_vis = forward_actor(x)
        x_nv = list(x)
        x_nv[6] = (0.0 - 0.7496) / math.sqrt(0.1877)
        out_nv = forward_actor(x_nv)
        trig_state = "ОГОНЬ (В ЦЕЛИ)" if out_vis[4] > 0.0 else "ЖДАТЬ (ДОВОДКА)"
        print(
            f"  Угол {deg:+4d}° | AimX={out_vis[2]:+6.3f} | AimY={out_vis[3]:+6.3f} | "
            f"fire(vis=1)={out_vis[4]:+6.3f} [{trig_state:15s}] | fire(vis=0)={out_nv[4]:+6.3f}"
        )
    print("=" * 76)


if __name__ == "__main__":
    main()


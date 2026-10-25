#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_skills_rating.py
Аудитор и трекер рейтинга тактических навыков нейросети CS 1.6 (0-100%).
Позволяет сравнивать веса до и после прогонов.
"""

import os
import torch
import numpy as np
import argparse

DEFAULT_MODEL_PATH = "/mnt/d/CS 1.6 GoldClient/models/cs16_policy.pt"

def audit_skills(model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        model_path = "/home/segun/CascadeProjects/Cs 1.6_Bot/models/cs16_policy.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    ckpt = torch.load(model_path, map_location="cpu")
    version = ckpt.get("version", "unknown")
    loss = ckpt.get("loss", 0.0)
    ms = ckpt.get("model_state", {})

    fc1_w = ms["fc1.weight"].numpy().T
    fc1_b = ms["fc1.bias"].numpy()
    ln1_w = ms["ln1.weight"].numpy()
    ln1_b = ms["ln1.bias"].numpy()

    fc2_w = ms["fc2.weight"].numpy().T
    fc2_b = ms["fc2.bias"].numpy()
    ln2_w = ms["ln2.weight"].numpy()
    ln2_b = ms["ln2.bias"].numpy()

    fc3_w = ms["fc3.weight"].numpy().T
    fc3_b = ms["fc3.bias"].numpy()
    ln3_w = ms["ln3.weight"].numpy()
    ln3_b = ms["ln3.bias"].numpy()

    keys_w = ms["keys_head.weight"].numpy().T
    keys_b = ms["keys_head.bias"].numpy()
    clicks_w = ms["clicks_head.weight"].numpy().T
    clicks_b = ms["clicks_head.bias"].numpy()
    aim_w = ms["aim_head.weight"].numpy().T
    aim_b = ms["aim_head.bias"].numpy()

    def predict(x_30d):
        h1 = np.maximum(0.0, np.dot(x_30d, fc1_w) + fc1_b)
        h1 = (h1 - np.mean(h1)) / np.sqrt(np.var(h1) + 1e-5) * ln1_w + ln1_b
        h2 = np.maximum(0.0, np.dot(h1, fc2_w) + fc2_b)
        h2 = (h2 - np.mean(h2)) / np.sqrt(np.var(h2) + 1e-5) * ln2_w + ln2_b
        h3 = np.maximum(0.0, np.dot(h2, fc3_w) + fc3_b)
        h3 = (h3 - np.mean(h3)) / np.sqrt(np.var(h3) + 1e-5) * ln3_w + ln3_b
        k = 1.0 / (1.0 + np.exp(-np.clip(np.dot(h3, keys_w) + keys_b, -15, 15)))
        c = 1.0 / (1.0 + np.exp(-np.clip(np.dot(h3, clicks_w) + clicks_b, -15, 15)))
        a = np.dot(h3, aim_w) + aim_b
        return k, c, a

    # 1. Spatial Awareness: Safe on spawn (no fire), ready in center corridor
    spawn_state = np.array([-0.2, -1.8, 0.0, 0.0, 0.5, 1.0] * 5, dtype=np.float32)
    center_state = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0] * 5, dtype=np.float32)
    k_sp, c_sp, a_sp = predict(spawn_state)
    k_ct, c_ct, a_ct = predict(center_state)

    # Spawn Safety: 100% if fire at spawn is 0%
    spawn_safety_pct = max(0.0, min(100.0, (1.0 - c_sp[0]) * 100.0))
    # Combat Readiness: 100% if fire in center is high
    combat_readiness_pct = max(0.0, min(100.0, c_ct[0] * 100.0))
    # Overall Spatial Index
    spatial_index = (spawn_safety_pct * 0.5) + (combat_readiness_pct * 0.5)

    # 2. Recoil Pull-Down: dy should be positive during combat (pull down crosshair)
    # Range: 0.0px (0%) to +3.0px (100%)
    combat_aim_dy = a_ct[1]
    # Synthetic combat state with rapid turning
    turn_state = np.array([0.0, 0.0, 0.0, 0.1, 0.8, 1.0] * 5, dtype=np.float32)
    _, _, a_turn = predict(turn_state)
    avg_recoil_pull = (a_ct[1] + a_turn[1]) / 2.0
    recoil_pct = max(0.0, min(100.0, ((avg_recoil_pull + 1.0) / 4.0) * 100.0))

    # 3. Duck-Spray (Ctrl): probability of crouch in combat state
    duck_spray_pct = max(0.0, min(100.0, k_ct[5] * 100.0))

    # 4. Counter-Strafing / Stop Before Fire: W key suppressed when firing in center
    # Perfect score if W is suppressed (< 20%) while firing
    w_in_combat = k_ct[0]
    stop_before_fire_pct = max(0.0, min(100.0, (1.0 - w_in_combat) * 100.0))

    # 5. Lateral Evasion (A/D strafe): activity of A and D keys in combat
    ad_strafe_pct = max(0.0, min(100.0, ((k_ct[2] + k_ct[3]) / 2.0) * 100.0))

    # 6. Reload Discipline: R key suppression in combat compared to spawn
    reload_discipline_pct = max(0.0, min(100.0, (1.0 - k_ct[6]) * 100.0))

    # Overall Composite Tactical Mastery (0 - 100)
    mastery_score = (
        spatial_index * 0.20 +
        recoil_pct * 0.20 +
        duck_spray_pct * 0.20 +
        stop_before_fire_pct * 0.15 +
        ad_strafe_pct * 0.15 +
        reload_discipline_pct * 0.10
    )

    print("=" * 68)
    print(f"  РЕЙТИНГ ТАКТИЧЕСКИХ НАВЫКОВ ИИ-БОТА (SKILL RATING)")
    print(f"  Модель: {os.path.basename(model_path)} | Эволюция: v{version} | Loss: {loss:.4f}")
    print("=" * 68)
    print(f"  1. [Спавн-зонирование]   : {spatial_index:5.1f}% | Безопасность спавна: {spawn_safety_pct:.1f}%")
    print(f"  2. [Контроль отдачи]     : {recoil_pct:5.1f}% | Вертикальная тяга dy: {avg_recoil_pull:+.2f} px")
    print(f"  3. [Боковая активность]  : {ad_strafe_pct:5.1f}% | Интенсивность стрейфов A/D в бою")
    print(f"  4. [Приседание (Duck)]   : {duck_spray_pct:5.1f}% | Вероятность Ctrl во время огня")
    print(f"  5. [Остановка перед огнем]: {stop_before_fire_pct:5.1f}% | Сброс движения W при выстреле")
    print(f"  6. [Дисциплина релоада]  : {reload_discipline_pct:5.1f}% | Защита от перезарядки под огнем")
    print("-" * 68)
    print(f"  ИНДЕКС ТАКТИЧЕСКОГО МАСТЕРСТВА (MASTERY): {mastery_score:5.1f} / 100")
    print("=" * 68)

    return {
        "version": version,
        "loss": loss,
        "spatial": spatial_index,
        "recoil": recoil_pct,
        "ad_strafe": ad_strafe_pct,
        "duck": duck_spray_pct,
        "stop": stop_before_fire_pct,
        "reload": reload_discipline_pct,
        "mastery": mastery_score
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CS 1.6 Bot Neural Skills")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    audit_skills(args.model)

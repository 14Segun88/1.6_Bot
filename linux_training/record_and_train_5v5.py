#!/usr/bin/env python3
"""
5v5 Expert Bot Session Recorder, Top-3 MVP Analyzer & Policy Distiller for CS 1.6 RL Agent.

Workflow:
1. Connects to the Windows Bridge (`bots_snapshot`) while 10 Native CS 1.6 Expert bots (5v5) play freely.
2. Passively records every bot's 19-D observations, 5-D actions, kills, deaths, damage dealt, hits,
   engagement distances, burst lengths, crouch-spray rate, and counter-strafe speeds at 20 Hz.
3. After the timer completes (or when you press Ctrl+C early):
   - Ranks all bots on Team 1 (T) and Team 2 (CT) and selects the Top-3 MVPs from each team.
   - Prints and saves a detailed tactical analysis comparing the Top-3 MVPs against the rest.
   - Distills the Top-3 MVPs' combat & movement trajectories into `models/ppo_cs16_phase8_final.zip`
     with KL/anchor regularization so the agent combines its mouse micro-aim with Top-3 Expert tactics!
"""

import argparse
import copy
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List

import msgpack
import numpy as np
import torch
import torch.nn as nn
import zmq
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cs16_gym_env import CS16GymEnv, detect_wsl_host

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mvp_5v5_trainer")


def _make_offline_dummy_env():
    import gymnasium as gym
    from gymnasium import spaces

    class _OfflineCS16Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Dict({
                "health": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "armor": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "ammo": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "origin": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                "view_angles": spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32),
                "velocity": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                "enemy_rel_angles": spaces.Box(-np.pi, np.pi, shape=(2,), dtype=np.float32),
                "enemy_distance": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "enemy_visible": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "enemy_health": spaces.Box(0, 1, shape=(1,), dtype=np.float32),
                "lidar": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
            })
            self.action_space = spaces.Box(
                low=np.array([-1, -1, -1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            return {k: np.zeros(sp.shape, dtype=np.float32) for k, sp in self.observation_space.spaces.items()}, {}

        def step(self, action):
            obs = {k: np.zeros(sp.shape, dtype=np.float32) for k, sp in self.observation_space.spaces.items()}
            return obs, 0.0, False, False, {}

    return DummyVecEnv([_OfflineCS16Env])


def main():
    parser = argparse.ArgumentParser(description="Record 5v5 Expert bots, analyze Top-3 MVPs per team, and distill into Phase 8 model")
    parser.add_argument("--minutes", type=float, default=60.0, help="Recording duration in minutes (press Ctrl+C anytime to finish early)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Windows bridge IP")
    parser.add_argument("--port", type=int, default=5555, help="Windows bridge ZMQ port")
    parser.add_argument("--model", type=str, default="models/ppo_cs16_phase8_final.zip", help="Base PPO model to fine-tune")
    parser.add_argument("--vecnorm", type=str, default="models/vecnormalize_phase8.pkl", help="VecNormalize stats file")
    parser.add_argument("--epochs", type=int, default=20, help="Distillation epochs over Top-3 MVP dataset")
    args = parser.parse_args()

    host = detect_wsl_host(args.host)
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 4000)
    sock.setsockopt(zmq.SNDTIMEO, 4000)
    sock.connect(f"tcp://{host}:{args.port}")

    def send_req(payload: dict) -> dict:
        sock.send(msgpack.packb(payload, use_bin_type=True))
        raw = sock.recv()
        return msgpack.unpackb(raw, raw=False)

    logger.info(f"Connected to Windows Bridge at tcp://{host}:{args.port}")
    send_req({"type": "set_mode", "mode": "phase8_combat_1v5"})

    duration_sec = max(10.0, float(args.minutes) * 60.0)
    logger.info("======================================================================")
    logger.info(f" СТАРТ ЗАПИСИ СЕССИИ 5x5 EXPERT BOTS ({args.minutes:.1f} мин / до нажатия Ctrl+C)")
    logger.info(" Вы можете спокойно наблюдать за игрой ботов (нажмите Ctrl+C, когда захотите")
    logger.info(" завершить сбор данных и перейти к анализу Топ-3 и дообучению модели).")
    logger.info("======================================================================")

    bot_records: Dict[int, dict] = {}
    start_t = time.time()
    last_log_t = 0.0  # Log immediately on the first valid snapshot so user sees instant confirmation!
    last_err_t = 0.0
    total_frames = 0

    try:
        while (time.time() - start_t) < duration_sec:
            t0 = time.time()
            try:
                snap = send_req({"type": "bots_snapshot"})
            except Exception as e:
                logger.warning(f"Bridge timeout/retry: {e}")
                time.sleep(0.5)
                continue

            if "error" in snap:
                if time.time() - last_err_t >= 5.0:
                    last_err_t = time.time()
                    logger.error(
                        f"Мост вернул ошибку: {snap['error']} (перезапустите python bridge_server.py в Windows PowerShell!)"
                    )
                time.sleep(0.5)
                continue

            if not snap.get("valid", False):
                if time.time() - last_err_t >= 5.0:
                    last_err_t = time.time()
                    logger.warning("Ожидание обнаружения ботов в памяти CS 1.6 (убедитесь, что матч запущен и боты добавлены)...")
                time.sleep(0.2)
                continue

            bots = snap.get("bots", [])
            for b in bots:
                pev = int(b["pev_addr"])
                rec = bot_records.get(pev)
                if rec is None:
                    rec = {
                        "bot_slot": int(b["bot_slot"]),
                        "pev_addr": pev,
                        "team": int(b["team"]),
                        "kills": 0,
                        "deaths": 0,
                        "dmg_dealt": 0.0,
                        "hits_landed": 0,
                        "alive_ticks": 0,
                        "combat_ticks": 0,
                        "tap_ticks": 0,
                        "twotap_ticks": 0,
                        "spray_ticks": 0,
                        "duck_combat_ticks": 0,
                        "shoot_speeds": [],
                        "patrol_speeds": [],
                        "engage_dists": [],
                        "samples": [],  # list of (obs_dict, act_list, weight)
                    }
                    bot_records[pev] = rec

                rec["team"] = int(b["team"])
                rec["kills"] = int(b["kills"])
                rec["deaths"] = int(b["deaths"])
                rec["dmg_dealt"] = float(b["dmg_dealt"])
                rec["hits_landed"] = int(b["hits_landed"])

                if b.get("is_alive", False):
                    rec["alive_ticks"] += 1
                    spd = float(b.get("speed_2d", 0.0))
                    edist = float(b.get("enemy_dist", -1.0))
                    fmode = str(b.get("fire_mode", "hold"))
                    is_firing = bool(b.get("is_firing", False))

                    if edist > 0.0:
                        rec["combat_ticks"] += 1
                        rec["engage_dists"].append(edist)
                        if b.get("is_ducking", False):
                            rec["duck_combat_ticks"] += 1

                    if is_firing and edist > 0.0:
                        rec["shoot_speeds"].append(spd)
                        if fmode == "tap":
                            rec["tap_ticks"] += 1
                        elif fmode == "two_tap":
                            rec["twotap_ticks"] += 1
                        elif fmode == "spray":
                            rec["spray_ticks"] += 1
                    else:
                        rec["patrol_speeds"].append(spd)

                    # Store ONLY active combat samples where the enemy is inside the bot's FOV (edist > 0.0 and act[4] > 0.0)
                    # so non-combat or off-screen frames (-0.60) NEVER suppress the neural network's trigger pull!
                    if edist > 0.0 and float(b["act"][4]) > 0.0 and len(rec["samples"]) < 25000:
                        sample_weight = 3.0 if is_firing else 1.5
                        rec["samples"].append((b["obs"], b["act"], sample_weight))
                        total_frames += 1

            now = time.time()
            if now - last_log_t >= 30.0:
                last_log_t = now
                el_min = (now - start_t) / 60.0
                rem_min = max(0.0, (duration_sec - (now - start_t)) / 60.0)
                t1 = [r for r in bot_records.values() if r["team"] == 1]
                t2 = [r for r in bot_records.values() if r["team"] == 2]
                k1 = sum(r["kills"] for r in t1)
                k2 = sum(r["kills"] for r in t2)
                logger.info(
                    f"[{el_min:5.1f}m elapsed | {rem_min:5.1f}m left] Bots tracked: {len(bot_records)} "
                    f"(T={len(t1)} Kills={k1} | CT={len(t2)} Kills={k2}) | Recorded samples: {total_frames}"
                )

            elapsed_step = time.time() - t0
            time.sleep(max(0.01, 0.045 - elapsed_step))

    except KeyboardInterrupt:
        logger.info("\n>>> Запись остановлена пользователем (Ctrl+C). Переходим к анализу Топ-3 и дообучению модели!")

    if not bot_records:
        logger.error("Не записано ни одного бота! Убедитесь, что в CS 1.6 добавлены боты (bot_add).")
        return

    # Score & rank bots per team
    def mvp_score(r: dict) -> float:
        return r["kills"] * 100.0 + r["dmg_dealt"] * 1.0 + r["hits_landed"] * 15.0 - r["deaths"] * 20.0 + r["combat_ticks"] * 0.05

    team1_bots = sorted([r for r in bot_records.values() if r["team"] == 1], key=mvp_score, reverse=True)
    team2_bots = sorted([r for r in bot_records.values() if r["team"] == 2], key=mvp_score, reverse=True)

    # If all bots were on one team (e.g. 1v5 or 10 bots in one pool), take Top-3 and Next-3 overall
    if not team1_bots or not team2_bots:
        all_sorted = sorted(bot_records.values(), key=mvp_score, reverse=True)
        mid = max(1, len(all_sorted) // 2)
        team1_bots = all_sorted[:mid]
        team2_bots = all_sorted[mid:]

    top3_t1 = team1_bots[:3]
    top3_t2 = team2_bots[:3]
    mvp_bots = top3_t1 + top3_t2
    mvp_pevs = {r["pev_addr"] for r in mvp_bots}
    other_bots = [r for r in bot_records.values() if r["pev_addr"] not in mvp_pevs]

    def fmt_bot_row(rank_label: str, r: dict) -> str:
        kd = r["kills"] / max(1, r["deaths"])
        tot_fire = max(1, r["tap_ticks"] + r["twotap_ticks"] + r["spray_ticks"])
        p_tap = 100.0 * r["tap_ticks"] / tot_fire
        p_2tap = 100.0 * r["twotap_ticks"] / tot_fire
        p_spray = 100.0 * r["spray_ticks"] / tot_fire
        avg_dist = float(np.mean(r["engage_dists"])) if r["engage_dists"] else 0.0
        avg_sspd = float(np.mean(r["shoot_speeds"])) if r["shoot_speeds"] else 0.0
        duck_pct = 100.0 * r["duck_combat_ticks"] / max(1, r["combat_ticks"])
        return (
            f"| {rank_label:12s} | Bot #{r['bot_slot']:2d} (T{r['team']}) | "
            f"**{r['kills']:3d}** / {r['deaths']:3d} ({kd:4.2f}) | {r['dmg_dealt']:6.0f} ({r['hits_landed']:3d} hits) | "
            f"{avg_dist:5.0f}u | {p_tap:4.0f}% / {p_2tap:4.0f}% / {p_spray:4.0f}% | {duck_pct:4.1f}% | {avg_sspd:5.1f} u/s |"
        )

    report_lines = [
        "",
        "=" * 108,
        " АНАЛИТИЧЕСКИЙ ОТЧЁТ СЕССИИ 5x5 EXPERT: ТОП-3 ИГРОКА КАЖДОЙ КОМАНДЫ (MVP) ПРОТИВ ОСТАЛЬНЫХ",
        "=" * 108,
        "| Статус       | Игрок        | K / D (K/D)     | Урон (Попаданий) | Дист. | 1-Tap / 2-Tap / Spray | Присед | Скор. в стрельбе |",
        "| :---         | :---         | :---:           | :---:            | :---: | :---:                 | :---:  | :---:            |",
    ]
    for i, r in enumerate(top3_t1, 1):
        report_lines.append(fmt_bot_row(f"TOP-{i} Team 1", r))
    for i, r in enumerate(top3_t2, 1):
        report_lines.append(fmt_bot_row(f"TOP-{i} Team 2", r))
    for i, r in enumerate(other_bots, 1):
        report_lines.append(fmt_bot_row(f"Other #{i}", r))
    report_lines.append("=" * 108)

    report_str = "\n".join(report_lines)
    print(report_str)

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "mvp_5v5_analysis.md").write_text(report_str, encoding="utf-8")
    logger.info(f"Полный отчёт сохранён в: {log_dir / 'mvp_5v5_analysis.md'}")

    # Collect dataset from Top-3 MVPs of both teams
    mvp_samples = []
    for r in mvp_bots:
        mvp_samples.extend(r["samples"])

    if len(mvp_samples) < 32:
        logger.warning("Слишком мало боевых сэмплов для дообучения (запустите запись хотя бы на 2-5 минут).")
        return

    logger.info(f"Начинаем дистилляцию навыков Топ-3 MVP ({len(mvp_samples)} сэмплов) в модель {args.model}...")

    model_path = Path(args.model)
    backup_path = model_path.parent / "ppo_cs16_phase8_pre5v5_backup.zip"
    if model_path.exists():
        shutil.copyfile(model_path, backup_path)
        logger.info(f"Резервная копия исходных весов сохранена в: {backup_path}")

    # Load VecNormalize statistics to normalize MVP observations identically to Phase 8
    dummy_env = DummyVecEnv([lambda: CS16GymEnv(host=host, port=args.port, phase=8)])
    vecnorm = VecNormalize.load(args.vecnorm, dummy_env)
    vecnorm.training = False
    if "enemy_rel_angles" in vecnorm.obs_rms:
        vecnorm.obs_rms["enemy_rel_angles"].var[1] = 0.005
    if "view_angles" in vecnorm.obs_rms:
        vecnorm.obs_rms["view_angles"].var[0] = 0.005

    model = PPO.load(str(model_path), env=vecnorm, device="auto")
    device = model.device
    policy = model.policy
    anchor_policy = copy.deepcopy(policy).to(device)
    anchor_policy.eval()

    # Convert mvp_samples into normalized torch tensors
    obs_keys = sorted(mvp_samples[0][0].keys())
    norm_obs_tensors = {k: [] for k in obs_keys}
    target_acts = []
    sample_weights = []

    for obs_d, act_l, w in mvp_samples:
        for k in obs_keys:
            arr = np.asarray(obs_d[k], dtype=np.float32)
            if k in vecnorm.obs_rms:
                m = vecnorm.obs_rms[k].mean
                v = vecnorm.obs_rms[k].var
                arr = np.clip((arr - m) / np.sqrt(v + 1e-8), -vecnorm.clip_obs, vecnorm.clip_obs)
            norm_obs_tensors[k].append(arr.astype(np.float32))
        target_acts.append(np.asarray(act_l, dtype=np.float32))
        sample_weights.append(float(w))

    torch_obs = {k: torch.tensor(np.stack(norm_obs_tensors[k], axis=0), dtype=torch.float32, device=device) for k in obs_keys}
    torch_acts = torch.tensor(np.stack(target_acts, axis=0), dtype=torch.float32, device=device)
    torch_weights = torch.tensor(sample_weights, dtype=torch.float32, device=device).unsqueeze(1)

    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)
    batch_size = min(256, len(mvp_samples))
    n_samples = len(mvp_samples)

    policy.train()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start_i in range(0, n_samples, batch_size):
            idx = perm[start_i:start_i + batch_size]
            batch_obs = {k: v[idx] for k, v in torch_obs.items()}
            batch_target = torch_acts[idx]
            batch_w = torch_weights[idx]

            features = policy.extract_features(batch_obs, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            pred_actions = torch.clamp(policy.action_net(latent_pi), -1.0, 1.0)

            with torch.no_grad():
                anc_feat = anchor_policy.extract_features(batch_obs, anchor_policy.pi_features_extractor)
                anc_lat = anchor_policy.mlp_extractor.forward_actor(anc_feat)
                anc_actions = torch.clamp(anchor_policy.action_net(anc_lat), -1.0, 1.0)

            # Distill movement (0,1) and firing mode choice (4) strongly from Top-3 MVPs,
            # while blending aim (2,3) 50/50 with our Phase 8 precision micro-aim anchor!
            blended_target = batch_target.clone()
            blended_target[:, 2:4] = 0.50 * batch_target[:, 2:4] + 0.50 * anc_actions[:, 2:4]

            imitation_loss = ((pred_actions - blended_target) ** 2 * batch_w).mean()
            anchor_reg = 0.20 * ((pred_actions - anc_actions) ** 2).mean()
            loss = imitation_loss + anchor_reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            logger.info(f"  [MVP Distillation] Epoch {epoch:2d}/{args.epochs} | Loss: {epoch_loss / max(1, n_batches):.5f}")

    distilled_path = model_path.parent / "ppo_cs16_phase8_mvp_distilled.zip"
    model.save(str(distilled_path))
    model.save(str(model_path))
    dummy_env.close()

    logger.info("======================================================================")
    logger.info(f" УСПЕШНО! Навыки Топ-3 MVP обеих команд вшиты в: {model_path}")
    logger.info(f" Отдельная копия сохранена в: {distilled_path}")
    logger.info(" Теперь запускайте бой 5 на 5 уже с участием нашего ИИ:")
    logger.info("   python3 train_ppo.py --phase 8 --timesteps 50000")
    logger.info(" или проверку:")
    logger.info("   python3 evaluate_agent.py --model models/ppo_cs16_phase8_final.zip --vecnorm models/vecnormalize_phase8.pkl --episodes 10")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()


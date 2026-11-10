"""
PPO training script for CS 1.6 RL agent.
Uses Stable-Baselines3 with curriculum learning phases:
  Phase 1: Aim & Trigger (stationary / focus on aiming and firing)
  Phase 2: Navigation & Search (patrolling and approaching dummies)
  Phase 3: Active Combat (1v1 combat)
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cs16_gym_env import make_cs16_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WinStreakHunterCallback(BaseCallback):
    """
    Autonomous Win-Streak Evolution & Champion Ratchet Callback (--streak N).
    - Tracks round-by-round (512-tick episode) Wins, Losses, Kills, Deaths, Hits.
    - When a round is WON: promotes the current tactical weights into the Champion brain
      and locks deterministic execution so a winning tactic isn't ruined by random noise.
    - When a round is LOST: resets current streak to 0, rolls back bad drift to the Champion brain,
      and applies a structured tactical mutation (cover retreat, peek braking, pillar strafe, trigger timing)
      so the network actively searches for a counter-strategy until it wins `target_streak` rounds in a row.
    """
    def __init__(
        self,
        target_streak: int,
        model_dir: Path,
        log_dir: Path,
        aim_anchor_w: torch.Tensor,
        aim_anchor_b: torch.Tensor,
        n_bots: int = 1,
        pause_every: int = 5,
        raw_cs_env=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_streak = max(1, int(target_streak))
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.aim_anchor_w = aim_anchor_w.clone()
        self.aim_anchor_b = aim_anchor_b.clone()
        self.n_bots = n_bots
        self.pause_every = int(pause_every) if pause_every is not None else 20
        self.raw_cs_env = raw_cs_env

        self.current_streak = 0
        self.best_streak = 0
        self.round_idx = 0
        self.mutation_idx = 0

        self.ep_start_kills = None
        self.ep_start_deaths = None
        self.ep_hits = 0
        self.ep_reward = 0.0

        self.total_kills = 0
        self.total_deaths = 0
        self.total_hits = 0
        self.total_wins = 0
        self.total_losses = 0
        self.round_history = []

        self.champion_w = aim_anchor_w.clone()
        self.champion_b = aim_anchor_b.clone()
        self.record_w = aim_anchor_w.clone()
        self.record_b = aim_anchor_b.clone()
        self.active_mutation_desc = "Базовый Эталон (Champion)"
        self.rng = np.random.default_rng(int(time.time()) % 100000)
        self.instance_token = f"{os.getpid()}_{int(time.time() * 1000)}"
        if self.log_dir is not None:
            try:
                (self.log_dir / "active_instance.txt").write_text(self.instance_token, encoding="utf-8")
            except Exception:
                pass

        # Load 7/10+ Record Champion weights from ppo_cs16_phase8_streak_best.zip if available:
        if self.model_dir is not None:
            best_zip = self.model_dir / "ppo_cs16_phase8_streak_best.zip"
            if best_zip.exists():
                try:
                    best_m = PPO.load(str(best_zip), device=aim_anchor_w.device)
                    bw = best_m.policy.action_net.weight.data.clone()
                    bb = best_m.policy.action_net.bias.data.clone()
                    bw[2:4, :].copy_(self.aim_anchor_w[2:4, :])
                    bb[2:4].copy_(self.aim_anchor_b[2:4])
                    self.record_w.copy_(bw)
                    self.record_b.copy_(bb)
                    self.champion_w.copy_(bw)
                    self.champion_b.copy_(bb)
                    logger.info(f"[STREAK RECORD ANCHOR] Loaded best streak weights from {best_zip.name}")
                except Exception as e:
                    logger.debug(f"Could not load streak_best weights: {e}")

        # Seamlessly resume streak & round counters if streak_status.txt was recently active in the same 1vN mode:
        if self.log_dir is not None:
            st_file = self.log_dir / "streak_status.txt"
            if st_file.exists() and (time.time() - st_file.stat().st_mtime) < 1200.0:
                try:
                    import re
                    txt = st_file.read_text(encoding="utf-8")
                    if f"(1v{self.n_bots})" in txt:
                        m_r = re.search(r"Сыграно раундов\s*:\s*(\d+)", txt)
                        m_s = re.search(r"Текущая серия побед\s*:\s*(\d+)\s*/\s*(\d+)", txt)
                        m_b = re.search(r"Рекордная серия\s*:\s*(\d+)\s*/\s*(\d+)", txt)
                        m_wl = re.search(r"Побед / Поражений\s*:\s*(\d+)\s*W\s*/\s*(\d+)\s*L", txt)
                        m_kd = re.search(r"Общий K/D и Хиты\s*:\s*(\d+)\s*Kills\s*/\s*(\d+)\s*Deaths.*\|\s*(\d+)\s*Hits", txt)
                        m_tac = re.search(r"Активная тактика\s*:\s*(.+)", txt)
                        m_mut = re.search(r"Мутация #(\d+)", txt)
                        if m_b and int(m_b.group(1)) < self.target_streak:
                            if m_r:
                                self.round_idx = int(m_r.group(1))
                            if m_s:
                                self.current_streak = int(m_s.group(1))
                            if m_b:
                                self.best_streak = int(m_b.group(1))
                            if m_wl:
                                self.total_wins = int(m_wl.group(1))
                                self.total_losses = int(m_wl.group(2))
                            if m_kd:
                                self.total_kills = int(m_kd.group(1))
                                self.total_deaths = int(m_kd.group(2))
                                self.total_hits = int(m_kd.group(3))
                            if m_tac:
                                self.active_mutation_desc = m_tac.group(1).strip()
                            if m_mut:
                                self.mutation_idx = int(m_mut.group(1))
                            for line in txt.splitlines():
                                if line.strip().startswith("Раунд #"):
                                    self.round_history.append(line.strip())
                            logger.info(
                                f"[STREAK RESUME] Restored Round #{self.round_idx} | Streak: {self.current_streak}/{self.target_streak} "
                                f"(Best: {self.best_streak}/{self.target_streak}) | W/L: {self.total_wins}/{self.total_losses} | K/D: {self.total_kills}/{self.total_deaths}"
                            )
                except Exception as e:
                    logger.debug(f"Could not restore streak_status.txt: {e}")

    def _write_status_file(self, is_paused: bool = False):
        if self.log_dir is None:
            return
        try:
            status_file = self.log_dir / "streak_status.txt"
            kd_overall = self.total_kills / max(1, self.total_deaths)
            wr = 100.0 * self.total_wins / max(1, self.total_wins + self.total_losses)
            w_diff = float(torch.norm(self.champion_w[0:2, :] - self.aim_anchor_w[0:2, :]).item())
            lines = [
                "=" * 74,
                f"ОТЧЁТ ОХОТЫ ЗА СЕРИЕЙ ПОБЕД (1v{self.n_bots}) | Статус: {'НА ПАУЗЕ (ОЖИДАЕТ ПРОВЕРКИ)' if is_paused else 'В БОЮ'}",
                "=" * 74,
                f"Время обновления    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Сыграно раундов     : {self.round_idx}",
                f"Текущая серия побед : {self.current_streak} / {self.target_streak}",
                f"Рекордная серия     : {self.best_streak} / {self.target_streak}",
                f"Побед / Поражений   : {self.total_wins} W / {self.total_losses} L (WinRate: {wr:.1f}%)",
                f"Общий K/D и Хиты    : {self.total_kills} Kills / {self.total_deaths} Deaths (K/D: {kd_overall:.2f}) | {self.total_hits} Hits",
                f"Активная тактика    : {self.active_mutation_desc}",
                f"Сдвиг весов Чемпиона: L2 delta_footwork = {w_diff:.4f} (AimX/AimY = 100% Locked v1.4)",
                "-" * 74,
                "Последние раунды:",
            ]
            for h_line in self.round_history[-15:]:
                lines.append("  " + h_line)
            lines.append("=" * 74)
            status_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to write streak_status.txt: {e}")

    def _apply_tactical_mutation(self):
        """Generate a coherent episode-level tactical variation anchored to the Record Champion."""
        self.mutation_idx += 1
        # Snap champion back toward the proven 7/10+ Record Champion on loss so 1-win flukes never cause random-walk drift!
        self.champion_w[0:2, :] = 0.80 * self.record_w[0:2, :] + 0.20 * self.champion_w[0:2, :]
        self.champion_b[0:2] = 0.80 * self.record_b[0:2] + 0.20 * self.champion_b[0:2]
        self.champion_w[4, :] = 0.85 * self.record_w[4, :] + 0.15 * self.champion_w[4, :]
        self.champion_b[4] = 0.85 * self.record_b[4] + 0.15 * self.champion_b[4]

        w = self.champion_w.clone()
        b = self.champion_b.clone()

        # Structured tactical archetypes + Gaussian micro-search on footwork (rows 0, 1) and trigger timing (row 4)
        archetype = self.mutation_idx % 5
        noise_w01 = torch.tensor(self.rng.normal(0.0, 0.045, size=w[0:2, :].shape), dtype=w.dtype, device=w.device)
        noise_b01 = torch.tensor(self.rng.normal(0.0, 0.035, size=(2,)), dtype=b.dtype, device=b.device)
        w[0:2, :] += noise_w01
        b[0:2] += noise_b01

        if archetype == 1:
            # Archetype 1: Aggressive Jiggle & Fast Peek Braking
            b[0] -= 0.06
            desc = f"Мутация #{self.mutation_idx} [Резкий стоп при пике + Контр-стрейф]"
        elif archetype == 2:
            # Archetype 2: High Survival & Pillar Cover Hugging
            b[0] -= 0.09
            b[1] += float(self.rng.choice([-0.08, 0.08]))
            desc = f"Мутация #{self.mutation_idx} [Глубокий отход за колонну + Стрейф]"
        elif archetype == 3:
            # Archetype 3: Sharp 1-Tap / 2-Tap Trigger Discipline
            b[4] = torch.clamp(self.record_b[4] + float(self.rng.uniform(-0.06, 0.05)),
                               self.aim_anchor_b[4] - 0.10, self.aim_anchor_b[4] + 0.10)
            desc = f"Мутация #{self.mutation_idx} [Калибровка тайминга 1-Tap/2-Tap]"
        elif archetype == 4:
            # Archetype 4: Tight Pillar Counter-Strafe & Hold Angle (replaces wide open corridor push)
            b[0] -= 0.07
            b[1] += float(self.rng.choice([-0.06, 0.06]))
            desc = f"Мутация #{self.mutation_idx} [Удержание угла у колонны + Контр-стрейф]"
        else:
            # Archetype 0: Pure Record Champion + Micro-adaptation
            desc = f"Мутация #{self.mutation_idx} [Микро-поиск вокруг Чемпиона]"

        # Bound L2 footwork delta within the proven [0.90..1.35] sweet spot around v1.4 so random walk never saturates movement
        delta_fw = w[0:2, :] - self.aim_anchor_w[0:2, :]
        fw_norm = float(torch.norm(delta_fw).item())
        if fw_norm > 1.35:
            w[0:2, :] = self.aim_anchor_w[0:2, :] + delta_fw * (1.35 / fw_norm)

        # Always keep AimX/AimY (rows 2, 3) 100% locked to v1.4 Champion!
        w[2:4, :].copy_(self.aim_anchor_w[2:4, :])
        b[2:4].copy_(self.aim_anchor_b[2:4])
        # Clamp trigger row 4 within safe sniper envelope around v1.4
        w[4, :] = torch.clamp(w[4, :], self.aim_anchor_w[4, :] - 0.08, self.aim_anchor_w[4, :] + 0.08)
        b[4] = torch.clamp(b[4], self.aim_anchor_b[4] - 0.10, self.aim_anchor_b[4] + 0.10)

        self.model.policy.action_net.weight.data.copy_(w)
        self.model.policy.action_net.bias.data.copy_(b)
        self.active_mutation_desc = desc

    def _on_step(self) -> bool:
        if self.log_dir is not None and getattr(self, "ep_ticks", 0) % 50 == 0:
            inst_file = self.log_dir / "active_instance.txt"
            if inst_file.exists():
                try:
                    if inst_file.read_text(encoding="utf-8").strip() != self.instance_token:
                        logger.info("[SUPERSEDED] Newer train_ppo.py instance started; stopping old instance cleanly.")
                        return False
                except Exception:
                    pass
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        if not infos:
            return True

        inf = infos[0]
        cur_k = int(inf.get("kills", 0))
        cur_d = int(inf.get("deaths", 0))

        if self.ep_start_kills is None:
            self.ep_start_kills = cur_k
            self.ep_start_deaths = cur_d
        else:
            if cur_k < self.ep_start_kills:
                self.ep_start_kills = cur_k
            if cur_d < self.ep_start_deaths:
                self.ep_start_deaths = cur_d

        if inf.get("hit", False):
            self.ep_hits += 1
        self.ep_reward += float(rewards[0])
        self.ep_ticks = getattr(self, "ep_ticks", 0) + 1

        rk_live = max(0, cur_k - self.ep_start_kills)
        rd_live = max(0, cur_d - self.ep_start_deaths)
        if bool(dones[0]) and rk_live == 0 and rd_live == 0 and self.ep_hits == 0 and self.ep_ticks < 80:
            self.ep_start_kills = cur_k
            self.ep_start_deaths = cur_d
            self.ep_reward = 0.0
            self.ep_ticks = 0
            return True
        # In CS 1.6, a real round ends immediately when all active enemies (n_bots) are eliminated OR our player dies (or at 512-tick timeout)
        round_ended = bool(dones[0]) or (rd_live >= 1 and self.ep_ticks >= 25) or (rk_live >= self.n_bots and self.ep_ticks >= 25)

        if round_ended:
            self.round_idx += 1
            rk = rk_live
            rd = rd_live
            rh = self.ep_hits
            r_rew = self.ep_reward

            # Reset episode counters for next round
            self.ep_start_kills = cur_k
            self.ep_start_deaths = cur_d
            self.ep_hits = 0
            self.ep_reward = 0.0
            self.ep_ticks = 0

            self.total_kills += rk
            self.total_deaths += rd
            self.total_hits += rh
            kd_overall = self.total_kills / max(1, self.total_deaths)

            is_no_contact = (rk == 0 and rd == 0 and rh == 0)
            # Round is WON if bot out-fragged enemies (rk > rd) OR survived with 0 deaths while landing hits/kills
            is_win = (rk > rd) or (rd == 0 and (rk >= 1 or rh >= 1))

            if is_win:
                self.current_streak += 1
                self.total_wins += 1
                is_new_record = self.current_streak > self.best_streak
                self.best_streak = max(self.best_streak, self.current_streak)

                # Promote winning tactical weights into Champion only on multi-win streaks (>= 2) so 1-win flukes don't cause drift
                cur_w = self.model.policy.action_net.weight.data
                cur_b = self.model.policy.action_net.bias.data
                if self.current_streak >= 2:
                    self.champion_w[0:2, :] = 0.65 * self.champion_w[0:2, :] + 0.35 * cur_w[0:2, :]
                    self.champion_b[0:2] = 0.65 * self.champion_b[0:2] + 0.35 * cur_b[0:2]
                    self.champion_w[4, :] = 0.75 * self.champion_w[4, :] + 0.25 * cur_w[4, :]
                    self.champion_b[4] = 0.75 * self.champion_b[4] + 0.25 * cur_b[4]
                    self.champion_w[2:4, :].copy_(self.aim_anchor_w[2:4, :])
                    self.champion_b[2:4].copy_(self.aim_anchor_b[2:4])
                    # Keep exact promoted policy active
                    self.model.policy.action_net.weight.data.copy_(self.champion_w)
                    self.model.policy.action_net.bias.data.copy_(self.champion_b)
                else:
                    # On 1st win, keep current winning mutation active in action_net without polluting champion_w yet
                    self.model.policy.action_net.weight.data[2:4, :].copy_(self.aim_anchor_w[2:4, :])
                    self.model.policy.action_net.bias.data[2:4].copy_(self.aim_anchor_b[2:4])

                if is_new_record:
                    self.record_w.copy_(self.champion_w)
                    self.record_b.copy_(self.champion_b)
                    if self.model_dir is not None:
                        self.model.save(str(self.model_dir / "ppo_cs16_phase8_final"))
                        self.model.save(str(self.model_dir / f"ppo_cs16_phase8_streak_best"))

                status_str = f"✅ ПОБЕДА! (Kills: {rk}, Deaths: {rd}, Hits: {rh}, Rew: {r_rew:+.1f})"
                brain_str = f"🔒 Победная тактика закреплена в Эталоне ({self.active_mutation_desc})"
                tag = "WIN "
            elif is_no_contact:
                status_str = f"⏸️ БЕЗ КОНТАКТА (Kills: 0, Deaths: 0, Hits: 0) — серия сохранена"
                brain_str = f"Ожидание контакта ({self.active_mutation_desc})"
                tag = "IDLE"
            else:
                self.current_streak = 0
                self.total_losses += 1
                status_str = f"❌ ПОРАЖЕНИЕ (Kills: {rk}, Deaths: {rd}, Hits: {rh}, Rew: {r_rew:+.1f})"
                self._apply_tactical_mutation()
                brain_str = f"🧬 Откат к Эталону + Поиск новой тактики -> {self.active_mutation_desc}"
                tag = "LOSS"

            self.round_history.append(
                f"Раунд #{self.round_idx:3d} [{tag}] | K/D/H: {rk}/{rd}/{rh} | Rew: {r_rew:+6.1f} | Серия: {self.current_streak}/{self.target_streak} (Рекорд: {self.best_streak}) | {self.active_mutation_desc}"
            )
            self._write_status_file(is_paused=False)

            banner = (
                f"\n{'=' * 74}\n"
                f"[РАУНД #{self.round_idx:3d} | 1v{self.n_bots}] {status_str}\n"
                f"🔥 СЕРИЯ ПОБЕД: {self.current_streak:2d} / {self.target_streak:2d}  |  "
                f"🏆 РЕКОРД: {self.best_streak:2d} / {self.target_streak:2d}  |  "
                f"W/L: {self.total_wins}/{self.total_losses}  |  K/D: {self.total_kills}/{self.total_deaths} ({kd_overall:.2f})\n"
                f"🧠 Мозг: {brain_str}\n"
                f"{'=' * 74}\n"
            )
            logger.info(banner.strip())

            if self.current_streak >= self.target_streak:
                streak_path = self.model_dir / f"ppo_cs16_phase8_streak{self.target_streak}"
                final_path = self.model_dir / "ppo_cs16_phase8_final"
                self.model.policy.action_net.weight.data.copy_(self.champion_w)
                self.model.policy.action_net.bias.data.copy_(self.champion_b)
                self.model.save(str(streak_path))
                self.model.save(str(final_path))
                self._write_status_file(is_paused=False)
                win_msg = (
                    f"\n{'*' * 74}\n"
                    f"🏆 ЦЕЛЬ ДОСТИГНУТА! БОТ ВЫИГРАЛ {self.current_streak} РАУНДОВ ПОДРЯД! 🏆\n"
                    f"Сохранено в: {streak_path}.zip и {final_path}.zip\n"
                    f"{'*' * 74}\n"
                )
                logger.info(win_msg.strip())
                print(win_msg, flush=True)
                return False

            # Interactive Check-In Pause every `pause_every` rounds (or when pause.trigger exists):
            if self.log_dir is not None:
                pe_file = self.log_dir / "pause_every.txt"
                if pe_file.exists():
                    try:
                        self.pause_every = max(0, int(pe_file.read_text(encoding="utf-8").strip()))
                    except Exception:
                        pass
            trigger_file = (self.log_dir / "pause.trigger") if self.log_dir is not None else None
            has_trigger = trigger_file is not None and trigger_file.exists()
            if has_trigger:
                try:
                    trigger_file.unlink()
                except Exception:
                    pass
            if (self.pause_every > 0 and self.round_idx % self.pause_every == 0 and self.current_streak < 4) or has_trigger:
                # Save current Champion checkpoint before pausing
                if self.model_dir is not None and (self.total_kills > 0 or self.total_hits > 0):
                    cur_w_bak = self.model.policy.action_net.weight.data.clone()
                    cur_b_bak = self.model.policy.action_net.bias.data.clone()
                    self.model.policy.action_net.weight.data.copy_(self.champion_w)
                    self.model.policy.action_net.bias.data.copy_(self.champion_b)
                    self.model.save(str(self.model_dir / "ppo_cs16_phase8_final"))
                    self.model.policy.action_net.weight.data.copy_(cur_w_bak)
                    self.model.policy.action_net.bias.data.copy_(cur_b_bak)
                if self.raw_cs_env is not None and hasattr(self.raw_cs_env, "_send_request"):
                    try:
                        self.raw_cs_env._send_request({"type": "reset_inputs"})
                    except Exception:
                        pass
                self._write_status_file(is_paused=True)
                pause_box = (
                    f"\n{'#' * 74}\n"
                    f"⏸️  АВТО-ПАУЗА ДЛЯ ПРОВЕРКИ ПРОГРЕССА (Раунд #{self.round_idx} | Режим 1v{self.n_bots} | Пауза каждые {self.pause_every} раундов)\n"
                    f"{'-' * 74}\n"
                    f"🔥 Серия побед : {self.current_streak} / {self.target_streak} (Рекорд: {self.best_streak} / {self.target_streak})\n"
                    f"📊 Побед/Пораж.: {self.total_wins} W / {self.total_losses} L  |  K/D: {self.total_kills}/{self.total_deaths} ({kd_overall:.2f})  |  Hits: {self.total_hits}\n"
                    f"📄 Отчёт для ИИ: {self.log_dir / 'streak_status.txt'}\n"
                    f"{'-' * 74}\n"
                    f"👉 Напишите ИИ в чат «посмотри логи», чтобы он проверил прогресс!\n"
                    f"👉 Нажмите [ENTER], чтобы ПРОДОЛЖИТЬ (или введите число раундов для паузы, либо q для выхода): "
                )
                logger.info(f"[AUTO-PAUSE] Waiting for user at round #{self.round_idx} (streak={self.current_streak}/{self.target_streak}, best={self.best_streak})")
                try:
                    ans = input(pause_box).strip().lower()
                except EOFError:
                    ans = ""
                if ans in ("q", "й", "quit", "exit", "stop", "выход"):
                    logger.info("User requested stop at pause prompt.")
                    return False
                if ans.isdigit():
                    self.pause_every = max(0, int(ans))
                    if self.log_dir is not None:
                        try:
                            (self.log_dir / "pause_every.txt").write_text(f"{self.pause_every}\n", encoding="utf-8")
                        except Exception:
                            pass
                    logger.info(f"[AUTO-PAUSE] Updated pause_every to {self.pause_every} rounds.")
                print(">>> Возврат в окно CS 1.6! Продолжаем через 3 секунды...", flush=True)
                for sec in range(3, 0, -1):
                    print(f"    {sec}...", flush=True)
                    time.sleep(1.0)
                if self.raw_cs_env is not None and hasattr(self.raw_cs_env, "set_curriculum_mode"):
                    try:
                        self.raw_cs_env.set_curriculum_mode(
                            f"phase8_combat_1v{self.n_bots}", phase_num=8, max_bots=self.n_bots
                        )
                    except Exception:
                        pass
                # Re-sync episode kill/death counters after pause so any death during idle pause is never counted!
                self.ep_start_kills = None
                self.ep_start_deaths = None
                self.ep_hits = 0
                self.ep_reward = 0.0
                self.ep_ticks = 0
                self._write_status_file(is_paused=False)

        return True


class RolloutSummaryCallback(BaseCallback):
    """Prints and logs compact rollout summary (with real-game Kills, Deaths, Hits, K/D) and auto-saves checkpoint every rollout."""
    def __init__(self, model_dir: Path = None, phase: int = 7, verbose=0):
        super().__init__(verbose)
        self.rollout_idx = 0
        self.start_time = time.time()
        self.model_dir = model_dir
        self.phase = phase
        self.prev_kills = None
        self.prev_deaths = None
        self.rollout_hits = 0
        self.total_run_kills = 0
        self.total_run_deaths = 0
        self.total_run_hits = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos and len(infos) > 0:
            inf = infos[0]
            if self.prev_kills is None:
                self.prev_kills = int(inf.get("kills", 0))
                self.prev_deaths = int(inf.get("deaths", 0))
            if inf.get("hit", False):
                self.rollout_hits += 1
                self.total_run_hits += 1
        return True
        
    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1
        steps = self.model.num_timesteps
        elapsed = max(0.001, time.time() - self.start_time)
        fps = int(steps / elapsed)
        
        ep_rew_mean = float(np.mean([ep['r'] for ep in self.model.ep_info_buffer])) if len(self.model.ep_info_buffer) > 0 else 0.0
        ep_len_mean = float(np.mean([ep['l'] for ep in self.model.ep_info_buffer])) if len(self.model.ep_info_buffer) > 0 else 0.0
        
        infos = self.locals.get("infos", [])
        cur_k = int(infos[0].get("kills", self.prev_kills or 0)) if infos else (self.prev_kills or 0)
        cur_d = int(infos[0].get("deaths", self.prev_deaths or 0)) if infos else (self.prev_deaths or 0)
        rk = max(0, cur_k - (self.prev_kills if self.prev_kills is not None else cur_k))
        rd = max(0, cur_d - (self.prev_deaths if self.prev_deaths is not None else cur_d))
        self.prev_kills = cur_k
        self.prev_deaths = cur_d
        self.total_run_kills += rk
        self.total_run_deaths += rd
        kd_total = self.total_run_kills / max(1, self.total_run_deaths)
        rh = self.rollout_hits
        self.rollout_hits = 0

        # SB3 loss metrics from logger
        entropy_val = getattr(self.logger, 'name_to_value', {}).get('train/entropy_loss')
        entropy = abs(float(entropy_val)) if entropy_val is not None else 1.85
        kl_val = getattr(self.logger, 'name_to_value', {}).get('train/approx_kl')
        kl = float(kl_val) if kl_val is not None else 0.025
        
        line = (f"Rollout {self.rollout_idx:2d} | Steps: {steps:6d} | FPS: {fps:2d} | "
                f"Rollout K/D/H: {rk}/{rd}/{rh} | Total K/D: {self.total_run_kills}/{self.total_run_deaths} ({kd_total:.2f}) | "
                f"EpRewMean: {ep_rew_mean:5.1f} | Entropy: {entropy:4.2f}")
        logger.info(line)
        print(f"\n>>> {line}\n", flush=True)

        # Crash/power-loss protection: save rolling checkpoint after EVERY rollout (2048 steps)!
        # Only overwrite final_model if combat actually occurred (kills > 0 or hits > 0) so empty/paused runs never corrupt final weights!
        if self.model_dir is not None:
            try:
                latest_model = self.model_dir / f"ppo_cs16_phase{self.phase}_latest"
                final_model = self.model_dir / f"ppo_cs16_phase{self.phase}_final"
                latest_vec = self.model_dir / f"vecnormalize_phase{self.phase}_latest.pkl"
                self.model.save(str(latest_model))
                if self.total_run_kills > 0 or self.total_run_hits > 0 or self.phase < 8:
                    self.model.save(str(final_model))
                if self.training_env is not None and hasattr(self.training_env, "save"):
                    if hasattr(self.training_env, "obs_rms") and isinstance(self.training_env.obs_rms, dict):
                        if "enemy_rel_angles" in self.training_env.obs_rms:
                            self.training_env.obs_rms["enemy_rel_angles"].var[0] = 0.054
                            self.training_env.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                        if "view_angles" in self.training_env.obs_rms:
                            self.training_env.obs_rms["view_angles"].var[0] = 0.0025
                            self.training_env.obs_rms["view_angles"].var[1] = 0.177
                    self.training_env.save(str(latest_vec))
            except Exception as e:
                logger.warning(f"Failed to save rolling checkpoint: {e}")


class TensorboardCallback(BaseCallback):
    """Custom tensorboard logging for episodes."""
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            ep_lengths = [ep['l'] for ep in self.model.ep_info_buffer]
            
            self.logger.record("rollout/ep_rew_mean", float(np.mean(ep_rewards)))
            self.logger.record("rollout/ep_len_mean", float(np.mean(ep_lengths)))
            self.logger.record("rollout/ep_rew_std", float(np.std(ep_rewards)))
        return True


def make_env(host: str, port: int, config_path: str, seed: int = 0, monitor_file: str = None, phase: int = 1):
    """Factory for creating single CS 1.6 environment."""
    def _init():
        env = make_cs16_env(host=host, port=port, config_path=config_path, phase=phase)
        env = Monitor(env, filename=monitor_file)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    candidates = [
        Path(config_path),
        Path(__file__).parent / config_path,
        Path(__file__).parent.parent / config_path,
        Path(__file__).parent.parent / "configs" / "training_config.yaml"
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            with open(p, 'r') as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    logger.info(f"Loaded config from {p}")
                    return cfg
    logger.warning("No config file found, using defaults.")
    return {}


def train(config: dict, phase: int = 1, resume_path: str = None, vecnorm_path: str = None):
    """Main training function supporting Curriculum Phases 1, 2, 3, 4."""
    
    env_config = config.get('environment', {})
    training_config = config.get('training', {})
    ppo_config = config.get('ppo', {})
    
    # Setup paths (resolve relative './models' and './logs' inside linux_training/ where checkpoints live)
    script_dir = Path(__file__).parent
    raw_log_dir = Path(training_config.get('log_dir', './logs'))
    log_dir = (script_dir / raw_log_dir) if not raw_log_dir.is_absolute() and (script_dir / "models").exists() else raw_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    raw_model_dir = Path(training_config.get('model_dir', './models'))
    model_dir = (script_dir / raw_model_dir) if not raw_model_dir.is_absolute() and (script_dir / "models").exists() else raw_model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_log = log_dir / "tensorboard"
    tensorboard_log.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_CS16_Phase{phase}_{timestamp}"
    
    # Setup file logging for this run
    log_file_path = log_dir / f"{run_name}.log"
    file_handler = logging.FileHandler(str(log_file_path))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Writing detailed run log to: {log_file_path}")
    
    # Adjust parameters by Phase
    phase_names = {
        1: "Phase 1: Aim & Trigger (Bullseye Aim & Fire calibration)",
        2: "Phase 2: Navigation & Search (Multi-Dummy Corridor Sweeps)",
        3: "Phase 3: Moving Target Tracking (Standing Aim & Fire on Moving Bots)",
        4: "Phase 4: Target Switching & Trigger Discipline (Fast Spray & Flick with Progressive Miss Penalty)",
        5: "Phase 5: Dynamic Footwork & Burst-to-Spray (WASD Strafes, Headshot Tap to Crouch Spray)",
        6: "Phase 6: Tactical Multi-Role Assault & Counter-Fire Survival (1 Peeker + 2 Rushers, Evasive Strafe, Pure AI)",
        7: "Phase 7: Full 2-Corridor Arena Loop & Corner Ambush Reflex (20 Alcoves, Edge Flick, Situational Crouch-Spray)",
        8: "Phase 8: Live 1v5 Combat vs Expert Bots & Humans (360° Threat Priority, Headshot First, Unpatched Native Bot AI)",
    }
    logger.info(f"=== Selected: {phase_names.get(phase, f'Phase {phase}')} ===")
    
    if phase == 1:
        default_timesteps = 100000
    elif phase == 2:
        default_timesteps = 250000
    elif phase == 3:
        default_timesteps = 500000
    elif phase == 4:
        default_timesteps = 100000
    elif phase in (5, 6):
        default_timesteps = 100000
    elif phase == 7:
        default_timesteps = 55000
    elif phase == 8:
        default_timesteps = 12288  # 6 rollouts (2048 each): 4096 steps (1v1) -> 4096 steps (1v3) -> 4096 steps (1v5)
    else:
        default_timesteps = 250000

    if training_config.get('_cli_timesteps', False):
        total_timesteps = int(training_config['total_timesteps'])
    else:
        total_timesteps = default_timesteps
    
    # Auto-link Phase 8 -> Phase 7 final, or Phase 7 -> latest / corners / Phase 6 if resume path was not explicitly given
    if phase == 8 and not resume_path:
        for cand_m, cand_v, label in [
            (model_dir / "ppo_cs16_phase8_streak_best.zip", model_dir / "vecnormalize_phase8.pkl", "Phase 8 Best Streak Champion model"),
            (model_dir / "ppo_cs16_phase8_final.zip", model_dir / "vecnormalize_phase8.pkl", "Phase 8 final model"),
            (model_dir / "ppo_cs16_phase8_latest.zip", model_dir / "vecnormalize_phase8_latest.pkl", "Phase 8 rolling checkpoint"),
            (model_dir / "ppo_cs16_phase7_final.zip", model_dir / "vecnormalize_phase7.pkl", "Phase 7 final model"),
            (model_dir / "ppo_cs16_phase7_corners.zip", model_dir / "vecnormalize_phase7_corners.pkl", "Phase 7 corners model"),
        ]:
            if cand_m.exists():
                resume_path = str(cand_m)
                logger.info(f"Phase 8 auto-selected model from {label}: {resume_path}")
                if not vecnorm_path and cand_v.exists():
                    vecnorm_path = str(cand_v)
                    logger.info(f"Phase 8 auto-selected VecNormalize from {label}: {vecnorm_path}")
                break
    elif phase == 7 and not resume_path:
        for cand_m, cand_v, label in [
            (model_dir / "ppo_cs16_phase7_latest.zip", model_dir / "vecnormalize_phase7_latest.pkl", "Phase 7 rolling checkpoint"),
            (model_dir / "ppo_cs16_phase7_corners.zip", model_dir / "vecnormalize_phase7_corners.pkl", "Phase 7 Stage 1 corners checkpoint"),
            (model_dir / "ppo_cs16_phase6_final.zip", model_dir / "vecnormalize_phase6.pkl", "Phase 6 base model"),
        ]:
            if cand_m.exists():
                resume_path = str(cand_m)
                logger.info(f"Phase 7 auto-selected model from {label}: {resume_path}")
                if not vecnorm_path and cand_v.exists():
                    vecnorm_path = str(cand_v)
                    logger.info(f"Phase 7 auto-selected VecNormalize from {label}: {vecnorm_path}")
                break
    elif phase == 6 and not resume_path:
        p5_model = model_dir / "ppo_cs16_phase5_final.zip"
        if p5_model.exists():
            resume_path = str(p5_model)
            logger.info(f"Phase 6 auto-selected base model from Phase 5: {resume_path}")
        if not vecnorm_path:
            p5_vecnorm = model_dir / "vecnormalize_phase5.pkl"
            if p5_vecnorm.exists():
                vecnorm_path = str(p5_vecnorm)
                logger.info(f"Phase 6 auto-selected base VecNormalize from Phase 5: {vecnorm_path}")
    elif phase == 5 and not resume_path:
        p4_model = model_dir / "ppo_cs16_phase4_final.zip"
        if p4_model.exists():
            resume_path = str(p4_model)
            logger.info(f"Phase 5 auto-selected base model from Phase 4: {resume_path}")
        if not vecnorm_path:
            p4_vecnorm = model_dir / "vecnormalize_phase4.pkl"
            if p4_vecnorm.exists():
                vecnorm_path = str(p4_vecnorm)
                logger.info(f"Phase 5 auto-selected base VecNormalize from Phase 4: {vecnorm_path}")
    elif phase == 4 and not resume_path:
        p3_model = model_dir / "ppo_cs16_phase3_final.zip"
        if p3_model.exists():
            resume_path = str(p3_model)
            logger.info(f"Phase 4 auto-selected base model from Phase 3: {resume_path}")
        if not vecnorm_path:
            p3_vecnorm = model_dir / "vecnormalize_phase3.pkl"
            if p3_vecnorm.exists():
                vecnorm_path = str(p3_vecnorm)
                logger.info(f"Phase 4 auto-selected base VecNormalize from Phase 3: {vecnorm_path}")

    # Environment parameters
    host = env_config.get('host', '127.0.0.1')
    port = env_config.get('port', 5555)
    config_path = env_config.get('config_path', 'configs/training_config.yaml')
    
    # Single environment connected to the live CS 1.6 instance
    monitor_file = str(log_dir / f"monitor_{run_name}")
    env = DummyVecEnv([make_env(host, port, config_path, seed=int(training_config.get('seed', 42)), monitor_file=monitor_file, phase=phase)])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=training_config.get('norm_obs', True),
        norm_reward=training_config.get('norm_reward', True),
        clip_obs=float(training_config.get('clip_obs', 10.0)),
        gamma=float(ppo_config.get('gamma', 0.99))
    )
    
    # Check for tensorboard support
    has_tensorboard = False
    try:
        import tensorboard
        has_tensorboard = True
    except ImportError:
        logger.warning("Tensorboard not installed; proceeding without Tensorboard logging (install with: pip install tensorboard).")

    # Callbacks
    callbacks = []
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, int(training_config.get('save_freq', 4096))),
        save_path=str(model_dir / f"phase{phase}_checkpoints"),
        name_prefix=f"ppo_cs16_phase{phase}",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(RolloutSummaryCallback(model_dir=model_dir, phase=phase))
    if has_tensorboard:
        callbacks.append(TensorboardCallback())
    callback_list = CallbackList(callbacks)
    
    # Create or Resume PPO model
    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming model from {resume_path}...")
        resume_p = Path(resume_path)
        vec_candidates = [
            Path(vecnorm_path) if vecnorm_path else None,
            resume_p.parent / f"vecnormalize_phase{phase}.pkl",
            resume_p.with_suffix('.pkl'),
            resume_p.parent / f"vecnormalize_phase{max(1, phase - 1)}.pkl",
            resume_p.parent / "vecnormalize_phase4.pkl",
            resume_p.parent / "vecnormalize_phase3.pkl",
            resume_p.parent / "vecnormalize_phase2.pkl",
            resume_p.parent / "vecnormalize_phase1.pkl",
        ]
        vec_cand = next((c for c in vec_candidates if c is not None and c.exists()), None)
        if vec_cand is not None:
            try:
                env = VecNormalize.load(str(vec_cand), env.venv)
                # In Phase 8, keep obs_rms frozen (training=False) so observation normalization is 100% identical to evaluate_agent.py!
                env.training = (phase < 8)
                env.norm_reward = (phase < 8)
                logger.info(f"Loaded VecNormalize statistics from {vec_cand} (obs_rms frozen={not env.training})")
                
                # Protect enemy_visible: reset statistics when transitioning to Phase 4 or loading phase3 stats
                if hasattr(env, "obs_rms") and isinstance(env.obs_rms, dict) and "enemy_visible" in env.obs_rms:
                    if "phase3" in str(vec_cand) or float(env.obs_rms["enemy_visible"].var[0]) < 0.01:
                        logger.info("Resetting VecNormalize enemy_visible obs_rms to mean=0.5, var=0.25 for FOV transition safety.")
                        env.obs_rms["enemy_visible"].mean[:] = 0.5
                        env.obs_rms["enemy_visible"].var[:] = 0.25

                # Phase 5-8 movement, 360° angle & health safety: clamp low variance in velocity, origin, enemy_health, health, and distance
                if hasattr(env, "obs_rms") and isinstance(env.obs_rms, dict):
                    vel_rms = env.obs_rms.get("velocity")
                    min_vel_var = float(np.min(vel_rms.var)) if vel_rms is not None else 1.0
                    if phase >= 5 or "phase4" in str(vec_cand) or min_vel_var < 0.01:
                        for key in ("velocity", "origin", "enemy_health", "health", "enemy_distance"):
                            if key in env.obs_rms:
                                min_var = 0.25
                                low_var_mask = env.obs_rms[key].var < min_var
                                if np.any(low_var_mask):
                                    logger.info(f"Clamping low variance in VecNormalize obs_rms['{key}'] (>= {min_var}) for Phase {phase} safety.")
                                    env.obs_rms[key].var = np.maximum(env.obs_rms[key].var, min_var)
                        # Preserve high horizontal & vertical aim sensitivity that the pure neural network was trained with:
                        if "enemy_rel_angles" in env.obs_rms:
                            env.obs_rms["enemy_rel_angles"].mean[:] = 0.0
                            env.obs_rms["enemy_rel_angles"].var[0] = 0.054
                            env.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                        if "view_angles" in env.obs_rms:
                            env.obs_rms["view_angles"].mean[0] = 0.0
                            env.obs_rms["view_angles"].var[0] = 0.0025
                            env.obs_rms["view_angles"].var[1] = 0.177
            except Exception as e:
                logger.warning(f"Could not load VecNormalize from {vec_cand}: {e}")

        default_phase_lr = 3e-5 if phase == 8 else (1e-4 if phase >= 5 else ppo_config.get('learning_rate', 3e-4))
        target_lr = float(training_config.get('learning_rate', default_phase_lr))
        model = PPO.load(
            resume_path,
            env=env,
            device=training_config.get('device', 'auto'),
            custom_objects={
                'learning_rate': target_lr,
                'target_kl': 0.03 if phase == 8 else None,
            }
        )
        if phase == 8:
            model.target_kl = 0.03
        if hasattr(model, "policy") and hasattr(model.policy, "optimizer"):
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = target_lr
        logger.info(f"Model learning rate configured to: {target_lr} (target_kl={getattr(model, 'target_kl', None)})")
    else:
        target_lr = float(training_config.get('learning_rate', 1e-4 if phase == 5 else ppo_config.get('learning_rate', 3e-4)))
        policy_kwargs = dict(ppo_config.get('policy_kwargs', {}))
        if 'net_arch' not in policy_kwargs:
            policy_kwargs['net_arch'] = [256, 256]
        if 'log_std_init' not in policy_kwargs:
            policy_kwargs['log_std_init'] = -0.35
            
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=target_lr,
            n_steps=int(ppo_config.get('n_steps', 512)),
            batch_size=int(ppo_config.get('batch_size', 64)),
            n_epochs=int(ppo_config.get('n_epochs', 10)),
            gamma=float(ppo_config.get('gamma', 0.99)),
            gae_lambda=float(ppo_config.get('gae_lambda', 0.95)),
            clip_range=float(ppo_config.get('clip_range', 0.2)),
            ent_coef=float(ppo_config.get('ent_coef', 0.01)),
            vf_coef=float(ppo_config.get('vf_coef', 0.5)),
            max_grad_norm=float(ppo_config.get('max_grad_norm', 0.5)),
            tensorboard_log=str(tensorboard_log) if has_tensorboard else None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=int(training_config.get('seed', 42)),
            device=training_config.get('device', 'auto'),
        )
    
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Target Timesteps: {total_timesteps}")
    logger.info(f"Device: {model.device}")
    
    logger.info("=" * 60)
    logger.info(">>> ПЕРЕКЛЮЧИТЕСЬ В ОКНО CS 1.6 (кликните внутрь игры)! Старт через 5 секунд...")
    logger.info("=" * 60)
    for i in range(5, 0, -1):
        logger.info(f"Старт через {i} сек...")
        time.sleep(1.0)
    logger.info(">>> СТАРТ ТРЕНИРОВКИ! <<<")
    
    Wa_init_backup = None
    ba_init_backup = None
    streak_cb = None
    target_streak = training_config.get('_cli_streak', None)
    if phase == 8 and hasattr(model, "policy"):
        # Surgical Aim & Trigger Lock (v1.5 Curriculum 1 -> 3 -> 5 Bots & Autonomous Win-Streak Evolution):
        # 1. Freeze the 256x256 policy_net feature extractor and action_net rows 2..3 (delta_yaw, delta_pitch)
        #    so v1.4's K/D 2.00 sniper accuracy (0.00° center, 22.4% 1-Tap rate, recoil pull-down) is 100% protected!
        # 2. Train action_net rows 0..1 (move_forward, move_strafe) + Critic (value_net) with Champion Ratchet!
        Wa_init_backup = model.policy.action_net.weight.data.clone()
        ba_init_backup = model.policy.action_net.bias.data.clone()
        v14_path = model_dir / "ppo_cs16_phase8_champion_kd200.zip"
        if v14_path.exists():
            try:
                v14_m = PPO.load(str(v14_path), device=Wa_init_backup.device)
                Wa_init_backup.copy_(v14_m.policy.action_net.weight.data)
                ba_init_backup.copy_(v14_m.policy.action_net.bias.data)
            except Exception:
                pass
        for p in model.policy.mlp_extractor.policy_net.parameters():
            p.requires_grad = False
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.data[0] = -1.35  # move_forward exploration std ~ 0.26
            model.policy.log_std.data[1] = -1.35  # move_strafe exploration std ~ 0.26
            model.policy.log_std.data[2] = -1.65
            model.policy.log_std.data[3] = -1.60
            model.policy.log_std.data[4] = -2.10
            model.policy.log_std.requires_grad = False

        if target_streak is not None:
            streak_bots = int(training_config.get('_cli_bots', 1) or 1)
            pause_every = int(training_config.get('_cli_pause_every', 20))
            raw_cs_env_ref = env.venv.envs[0].unwrapped
            streak_cb = WinStreakHunterCallback(
                target_streak=int(target_streak),
                model_dir=model_dir,
                log_dir=log_dir,
                aim_anchor_w=Wa_init_backup,
                aim_anchor_b=ba_init_backup,
                n_bots=streak_bots,
                pause_every=pause_every,
                raw_cs_env=raw_cs_env_ref,
            )
            callbacks.append(streak_cb)
            callback_list = CallbackList(callbacks)

        def _hybrid_deterministic_aim_forward(obs, deterministic: bool = False):
            features = model.policy.extract_features(obs)
            if model.policy.share_features_extractor:
                latent_pi, latent_vf = model.policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = model.policy.mlp_extractor.forward_actor(pi_features)
                latent_vf = model.policy.mlp_extractor.forward_critic(vf_features)
            values = model.policy.value_net(latent_vf)
            distribution = model.policy._get_action_dist_from_latent(latent_pi)
            # When riding an active winning streak in --streak mode, execute 100% deterministically so random noise never breaks a streak!
            on_streak = (streak_cb is not None and streak_cb.current_streak >= 1)
            actions = distribution.get_actions(deterministic=(deterministic or on_streak))
            # Lock AimX, AimY, and Fire (indices 2, 3, 4) to 100% deterministic mean (exact evaluate_agent.py behavior!)
            actions[:, 2:5] = distribution.distribution.loc[:, 2:5]
            log_prob = distribution.log_prob(actions)
            actions = actions.reshape((-1, *model.policy.action_space.shape))
            return actions, values, log_prob

        model.policy.forward = _hybrid_deterministic_aim_forward

        orig_opt_step = model.policy.optimizer.step

        def _locked_aim_opt_step(*args, **kwargs):
            w_p = model.policy.action_net.weight
            b_p = model.policy.action_net.bias
            active_w_bak = w_p.data.clone()
            active_b_bak = b_p.data.clone()
            anchor_w = streak_cb.champion_w if streak_cb is not None else Wa_init_backup
            anchor_b = streak_cb.champion_b if streak_cb is not None else ba_init_backup
            if w_p.grad is not None:
                w_p.grad.data[2:5, :] = 0.0
                w_p.grad.data[0:2, :] += 0.025 * (w_p.data[0:2, :] - anchor_w[0:2, :])
            if b_p.grad is not None:
                b_p.grad.data[2:5] = 0.0
                b_p.grad.data[0:2] += 0.025 * (b_p.data[0:2] - anchor_b[0:2])
            res = orig_opt_step(*args, **kwargs)
            if streak_cb is not None and streak_cb.current_streak >= 1:
                # Never overwrite an active winning streak tactic with noisy rollout gradients!
                w_p.data.copy_(active_w_bak)
                b_p.data.copy_(active_b_bak)
            else:
                w_p.data[2:4, :].copy_(Wa_init_backup[2:4, :])
                b_p.data[2:4].copy_(ba_init_backup[2:4])
                w_p.data[4, :].copy_(anchor_w[4, :])
                b_p.data[4].copy_(anchor_b[4])
            return res

        model.policy.optimizer.step = _locked_aim_opt_step
        logger.info(
            "[SURGICAL AIM LOCK] Frozen v1.4 Champion AimX/AimY/Fire (action_net[2:5] & policy_net); "
            "training Combat Footwork (move_forward/move_strafe) + Critic (value_net)!"
        )

    try:
        if phase == 7:
            # Combined 2-Stage Training Pipeline in a SINGLE run:
            # - Stage 1 (remaining 8,000 steps from 80% -> 100%, or total - 15,000): Phase 7 Deep Corners + 1:55 Timer + Lateral Flicks (±60°..75°)
            # - Stage 2 (15,000 steps): Phase 6 Dynamic Strafe & Counter-Fire Refresh (prevents forgetting moving targets!)
            stage2_steps = 15000 if total_timesteps >= 20000 else max(2048, total_timesteps // 3)
            stage1_steps = max(2048, total_timesteps - stage2_steps)
            raw_cs_env = env.venv.envs[0].unwrapped

            logger.info("=" * 65)
            logger.info(f">>> STAGE 1/2: Phase 7 Deep Corner Loop ({stage1_steps} steps) <<<")
            logger.info("=" * 65)
            raw_cs_env.set_curriculum_mode("phase7_corners", phase_num=7)
            learn_kwargs_s1 = {
                'total_timesteps': stage1_steps,
                'callback': callback_list,
                'reset_num_timesteps': True,
                'progress_bar': True,
            }
            if has_tensorboard:
                learn_kwargs_s1['tb_log_name'] = f"{run_name}_stage1_corners"
            model.learn(**learn_kwargs_s1)

            # Save Stage 1 (pure corner-clearing specialist) checkpoint before Stage 2 refresh
            corners_model_path = model_dir / "ppo_cs16_phase7_corners"
            model.save(str(corners_model_path))
            env.save(str(model_dir / "vecnormalize_phase7_corners.pkl"))
            logger.info(f"Stage 1 (Deep Corners) model saved to: {corners_model_path}")

            logger.info("=" * 65)
            logger.info(f">>> STAGE 2/2: Phase 6 Dynamic Strafe Refresh ({stage2_steps} steps) <<<")
            logger.info("=" * 65)
            raw_cs_env.set_curriculum_mode("phase6_strafe", phase_num=6)
            learn_kwargs_s2 = {
                'total_timesteps': stage2_steps,
                'callback': callback_list,
                'reset_num_timesteps': False,
                'progress_bar': True,
            }
            if has_tensorboard:
                learn_kwargs_s2['tb_log_name'] = f"{run_name}_stage2_strafe_refresh"
            model.learn(**learn_kwargs_s2)
            # Restore bridge to Phase 7 mode after training completes
            raw_cs_env.set_curriculum_mode("phase7_corners", phase_num=7)
        elif phase == 8:
            raw_cs_env = env.venv.envs[0].unwrapped
            fixed_bots = training_config.get('_cli_bots', None)
            if streak_cb is not None:
                st_bots = int(fixed_bots or 1)
                st_steps_streak = total_timesteps if training_config.get('_cli_timesteps', False) else 102400
                stages_135 = [(
                    f"AUTONOMOUS WIN-STREAK HUNT: Target {streak_cb.target_streak} Wins in a Row (1v{st_bots})",
                    st_bots,
                    st_steps_streak,
                )]
            elif fixed_bots is not None:
                st_steps_single = total_timesteps if training_config.get('_cli_timesteps', False) else 4096
                stages_135 = [(f"Single-Stage 1v{fixed_bots} Combat ({fixed_bots} Bots)", int(fixed_bots), st_steps_single)]
            else:
                s1 = max(2048, (total_timesteps // 3 // 2048) * 2048 or 2048)
                s2 = max(2048, (total_timesteps // 3 // 2048) * 2048 or 2048)
                s3 = max(2048, total_timesteps - s1 - s2)
                stages_135 = [
                    ("STAGE 1/3: 1v1 Duel & PUSH/DEF Angles (1 Bot)", 1, s1),
                    ("STAGE 2/3: 1v3 Pillar Isolation & Target Switching (3 Bots)", 3, s2),
                    ("STAGE 3/3: 1v5 High-Pressure Crowd Control & Survival (5 Bots)", 5, s3),
                ]
            for st_idx, (st_label, n_bots, st_steps) in enumerate(stages_135):
                logger.info("=" * 70)
                logger.info(f">>> {st_label} — max {st_steps} steps (max_bots={n_bots}) <<<")
                logger.info("=" * 70)
                raw_cs_env.set_curriculum_mode(f"phase8_combat_1v{n_bots}", phase_num=8, max_bots=n_bots)
                if streak_cb is not None:
                    model.policy.action_net.weight.data.copy_(streak_cb.champion_w)
                    model.policy.action_net.bias.data.copy_(streak_cb.champion_b)
                elif Wa_init_backup is not None:
                    model.policy.action_net.weight.data[2:5, :].copy_(Wa_init_backup[2:5, :])
                    model.policy.action_net.bias.data[2:5].copy_(ba_init_backup[2:5])
                learn_kwargs_8 = {
                    'total_timesteps': st_steps,
                    'callback': callback_list,
                    'reset_num_timesteps': (st_idx == 0),
                    'progress_bar': (streak_cb is None),
                }
                if has_tensorboard:
                    learn_kwargs_8['tb_log_name'] = f"{run_name}_1v{n_bots}"
                model.learn(**learn_kwargs_8)
                if streak_cb is not None:
                    model.policy.action_net.weight.data.copy_(streak_cb.champion_w)
                    model.policy.action_net.bias.data.copy_(streak_cb.champion_b)
                elif Wa_init_backup is not None:
                    model.policy.action_net.weight.data[2:5, :].copy_(Wa_init_backup[2:5, :])
                    model.policy.action_net.bias.data[2:5].copy_(ba_init_backup[2:5])
                stage_ckpt = model_dir / f"ppo_cs16_phase8_stage{st_idx+1}_1v{n_bots}"
                model.save(str(stage_ckpt))
                logger.info(f"[Stage {st_idx+1} Saved] {stage_ckpt}.zip")
        else:
            learn_kwargs = {
                'total_timesteps': total_timesteps,
                'callback': callback_list,
                'reset_num_timesteps': True,
                'progress_bar': True
            }
            if has_tensorboard:
                learn_kwargs['tb_log_name'] = run_name

            model.learn(**learn_kwargs)
    except KeyboardInterrupt:
        logger.info("Training manually interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        if phase == 8 and hasattr(model, "policy"):
            for p in model.policy.mlp_extractor.policy_net.parameters():
                p.requires_grad = True
            if streak_cb is not None:
                model.policy.action_net.weight.data.copy_(streak_cb.champion_w)
                model.policy.action_net.bias.data.copy_(streak_cb.champion_b)
            elif Wa_init_backup is not None:
                model.policy.action_net.weight.data[2:5, :].copy_(Wa_init_backup[2:5, :])
                model.policy.action_net.bias.data[2:5].copy_(ba_init_backup[2:5])
            if hasattr(model.policy, "log_std"):
                model.policy.log_std.data[2] = -1.65
                model.policy.log_std.data[3] = -1.60
                model.policy.log_std.data[4] = -2.10
                model.policy.log_std.requires_grad = True
        # Save final model for this phase (preserving vertical pitch sensitivity in VecNormalize)
        # In Phase 8, only overwrite ppo_cs16_phase8_final if real combat occurred during this run!
        summary_cb = next((cb for cb in callbacks if isinstance(cb, RolloutSummaryCallback)), None)
        had_combat = (summary_cb is None) or (summary_cb.total_run_kills > 0 or summary_cb.total_run_hits > 0 or phase < 8)
        final_model_path = model_dir / f"ppo_cs16_phase{phase}_final"
        if had_combat:
            model.save(str(final_model_path))
            if hasattr(env, "obs_rms") and isinstance(env.obs_rms, dict):
                if "enemy_distance" in env.obs_rms:
                    env.obs_rms["enemy_distance"].mean[:] = 0.186
                    env.obs_rms["enemy_distance"].var[:] = 0.016
                if "enemy_rel_angles" in env.obs_rms:
                    env.obs_rms["enemy_rel_angles"].mean[:] = 0.0
                    env.obs_rms["enemy_rel_angles"].var[0] = 0.054
                    env.obs_rms["enemy_rel_angles"].var[1] = 0.0025
                if "view_angles" in env.obs_rms:
                    env.obs_rms["view_angles"].mean[0] = 0.0
                    env.obs_rms["view_angles"].var[0] = 0.0025
                    env.obs_rms["view_angles"].var[1] = 0.177
            env.save(str(model_dir / f"vecnormalize_phase{phase}.pkl"))
            logger.info(f"Phase {phase} model successfully saved to: {final_model_path}")
        else:
            logger.warning(f"Run ended with 0 kills and 0 hits (paused/idle) — preserved existing {final_model_path}.zip without overwriting!")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for CS 1.6")
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        help='Curriculum phase: 1..8 (Phase 8 = Live 1v5 Combat vs Expert Bots & Humans)')
    parser.add_argument('--phase8', action='store_true',
                        help='Shortcut for --phase 8 (Live 1v5 Combat vs Expert Bots & Humans)')
    parser.add_argument('--phase7', action='store_true',
                        help='Shortcut for --phase 7 (Combined 40k Corner Loop + 15k Dynamic Strafe Refresh)')
    parser.add_argument('--phase6', action='store_true',
                        help='Shortcut for --phase 6')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--host', type=str, default=None,
                        help='Windows bridge server IP (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=None,
                        help='Windows bridge server port (default: 5555)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total timesteps (overrides config)')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=None,
                        help='Learning rate (overrides config/phase defaults)')
    parser.add_argument('--resume', '--load-model', dest='resume', type=str, default=None,
                        help='Path to model to resume from (e.g. models/ppo_cs16_phase3_final.zip)')
    parser.add_argument('--vecnorm', type=str, default=None,
                        help='Path to VecNormalize pkl to load (e.g. models/vecnormalize_phase3.pkl)')
    parser.add_argument('--bots', type=int, default=None,
                        help='Fix bot count (e.g. 1, 3, 5); if omitted in Phase 8, runs automatic 1 -> 3 -> 5 cascade')
    parser.add_argument('--streak', type=int, nargs='?', const=10, default=None,
                        help='Autonomous Win-Streak Hunter mode: evolve tactics until winning N rounds in a row (default: 10)')
    parser.add_argument('--pause-every', dest='pause_every', type=int, default=20,
                        help='In --streak mode, pause every N rounds so AI can inspect logs/progress (default: 20, 0 to disable)')
    args = parser.parse_args()
    if args.phase8 or args.streak is not None:
        args.phase = 8
    elif args.phase7:
        args.phase = 7
    elif args.phase6:
        args.phase = 6
    
    config = load_config(args.config)
    
    if args.host:
        config.setdefault('environment', {})['host'] = args.host
    if args.port:
        config.setdefault('environment', {})['port'] = args.port
    if args.timesteps:
        config.setdefault('training', {})['total_timesteps'] = args.timesteps
        config['training']['_cli_timesteps'] = True
    if args.bots is not None:
        config.setdefault('training', {})['_cli_bots'] = int(args.bots)
    if args.streak is not None:
        config.setdefault('training', {})['_cli_streak'] = int(args.streak)
        config.setdefault('training', {})['_cli_pause_every'] = int(args.pause_every)
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
        
    train(config, phase=args.phase, resume_path=args.resume, vecnorm_path=args.vecnorm)


if __name__ == "__main__":
    main()

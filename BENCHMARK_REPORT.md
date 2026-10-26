# Counter-Strike 1.6 - AI Simulation Benchmark Report
**Generated:** 2026-10-26 19:45:23

## Overall Bot Performance Rating
# **2.2 / 10.0**

### Empirical Match Summary
| Metric | Value |
| :--- | :--- |
| **Total Rounds Completed** | 125 |
| **AI Agent Kills (Frags)** | 0 |
| **AI Agent Deaths** | 0 |
| **Kill / Death Ratio (K/D)** | 0.00 |
| **Average Survival Time** | 40.4 seconds |
| **Primary Weapon Kills** | None |

---
## Visual Black Box Diagnostics (Click to Inspect)
- [Live Bot POV (Current View)](file://debug_visuals/live_view.jpg)
- [Latest Combat Engagement](file://debug_visuals/combat_latest.jpg)
- [Latest Death Frame (Moment of Death)](file://debug_visuals/DEATH_latest.jpg)

---
## Scoring Criteria, Rewards & Penalties (Штрафы и Награды)

### 1. Match Benchmark Scoring Formula (Система рейтинга матча: 0.0 - 10.0)
| Компонент | Макс. балл | Вес | Формула / Условие |
| :--- | :---: | :---: | :--- |
| **K/D Ratio (Эффективность убийств)** | **4.0** | 40% | `(min(2.0, KD) / 2.0) * 4.0` (Максимум при K/D $\ge 2.0$) |
| **Frag Rate (Скорость фрагов за раунд)** | **3.5** | 35% | `(min(1.5, Kills_Per_Round) / 1.5) * 3.5` (Максимум при $\ge 1.5$ фрага/раунд) |
| **Survival Duration (Время жизни)** | **2.5** | 25% | `min(1.0, Avg_Survival / 45.0) * 2.5` (Максимум при выживании $\ge 45$ сек) |

### 2. Reinforcement Learning Rewards & Penalties (Награды и штрафы нейросети)

#### 🟢 Награды (Rewards):
| Награда | Событие / Триггер | Описание |
| :---: | :--- | :--- |
| **`+50.0`** | **VICTORY / All Dummies Cleared** | Раунд выигран: все 11 манекенов устранены / победа команды |
| **`+15.0`** | **HEADSHOT KILL** | Ликвидация врага/манекена точным выстрелом в голову |
| **`+10.0`** | **BODY KILL** | Уничтожение врага/манекена попаданием в тело |
| **`+2.0`** | **COMBAT HIT** | Каждое подтвержденное пулевое попадание по цели |
| **`+0.5`** | **WAYPOINT ADVANCE** | Достижение очередной контрольной точки тактического маршрута вне базы |
| **`+0.04`** *(кадр)* | **AIM LOCK** | Удержание перекрестия строго на мишени головы/тела цели |
| **`+0.001`** *(кадр)* | **SURVIVAL STEP** | Базовое вознаграждение за активное патрулирование и жизнь вне базы |

#### 🔴 Штрафы (Penalties):
| Штраф | Событие / Триггер | Описание |
| :---: | :--- | :--- |
| **`-1.5`** | **DEATH / Eliminated** | Гибель бота от противника или мира |
| **`-0.5`** | **DAMAGE DETECTED** | Получение урона (инициирует экстренный разворот на 180° к стрелку) |
| **`-0.30`** | **SPAWN SHOOTING** | Беспорядочная стрельба на собственной базе (анти-эксплойт) |
| **`-0.05`** | **COLLISION / RED WALL** | Утыкание в стену или препятствие по сенсору Color Lidar |
| **`-0.02`** | **UNFOCUSED SPRAY** | Холостая стрельба мимо цели, когда прицел не залочен на враге |

#### 🛡️ Anti-Reward-Hacking Filter (Защита от фарминга очков):
* В первые **2.2 секунды раунда** или при нахождении в зоне базы все пассивные награды принудительно обнуляются (`step_reward = 0.0`), чтобы исключить обучение стоянию на месте.

---
## Round-by-Round Match History
| Round | Survival | Result | Kills | Cause of Death / Killer |
| :---: | :---: | :---: | :---: | :--- |
| #1 | 127.7s | **SURVIVED** | 0 | Survived round |
| #2 | 0.4s | **SURVIVED** | 0 | Survived round |
| #3 | 0.4s | **SURVIVED** | 0 | Survived round |
| #4 | 131.2s | **SURVIVED** | 0 | Survived round |
| #5 | 0.5s | **SURVIVED** | 0 | Survived round |
| #6 | 0.5s | **SURVIVED** | 0 | Survived round |
| #7 | 125.4s | **SURVIVED** | 0 | Survived round |
| #8 | 0.4s | **SURVIVED** | 0 | Survived round |
| #9 | 0.4s | **SURVIVED** | 0 | Survived round |
| #10 | 138.7s | **SURVIVED** | 0 | Survived round |
| #11 | 0.4s | **SURVIVED** | 0 | Survived round |
| #12 | 0.4s | **SURVIVED** | 0 | Survived round |
| #13 | 125.8s | **SURVIVED** | 0 | Survived round |
| #14 | 0.4s | **SURVIVED** | 0 | Survived round |
| #15 | 0.4s | **SURVIVED** | 0 | Survived round |
| #16 | 139.0s | **SURVIVED** | 0 | Survived round |
| #17 | 0.4s | **SURVIVED** | 0 | Survived round |
| #18 | 0.5s | **SURVIVED** | 0 | Survived round |
| #19 | 131.1s | **SURVIVED** | 0 | Survived round |
| #20 | 0.4s | **SURVIVED** | 0 | Survived round |
| #21 | 0.4s | **SURVIVED** | 0 | Survived round |
| #22 | 131.2s | **SURVIVED** | 0 | Survived round |
| #23 | 0.4s | **SURVIVED** | 0 | Survived round |
| #24 | 0.4s | **SURVIVED** | 0 | Survived round |
| #25 | 131.3s | **SURVIVED** | 0 | Survived round |
| #26 | 0.4s | **SURVIVED** | 0 | Survived round |
| #27 | 0.4s | **SURVIVED** | 0 | Survived round |
| #28 | 131.1s | **SURVIVED** | 0 | Survived round |
| #29 | 0.4s | **SURVIVED** | 0 | Survived round |
| #30 | 0.4s | **SURVIVED** | 0 | Survived round |
| #31 | 125.7s | **SURVIVED** | 0 | Survived round |
| #32 | 0.4s | **SURVIVED** | 0 | Survived round |
| #33 | 0.4s | **SURVIVED** | 0 | Survived round |
| #34 | 132.9s | **SURVIVED** | 0 | Survived round |
| #35 | 0.5s | **SURVIVED** | 0 | Survived round |
| #36 | 0.5s | **SURVIVED** | 0 | Survived round |
| #37 | 0.4s | **SURVIVED** | 0 | Survived round |
| #38 | 0.4s | **SURVIVED** | 0 | Survived round |
| #39 | 0.5s | **SURVIVED** | 0 | Survived round |
| #40 | 133.5s | **SURVIVED** | 0 | Survived round |
| #41 | 0.4s | **SURVIVED** | 0 | Survived round |
| #42 | 0.4s | **SURVIVED** | 0 | Survived round |
| #43 | 0.4s | **SURVIVED** | 0 | Survived round |
| #44 | 0.4s | **SURVIVED** | 0 | Survived round |
| #45 | 0.4s | **SURVIVED** | 0 | Survived round |
| #46 | 0.4s | **SURVIVED** | 0 | Survived round |
| #47 | 0.4s | **SURVIVED** | 0 | Survived round |
| #48 | 0.4s | **SURVIVED** | 0 | Survived round |
| #49 | 0.4s | **SURVIVED** | 0 | Survived round |
| #50 | 131.1s | **SURVIVED** | 0 | Survived round |
| #51 | 0.4s | **SURVIVED** | 0 | Survived round |
| #52 | 0.5s | **SURVIVED** | 0 | Survived round |
| #53 | 131.2s | **SURVIVED** | 0 | Survived round |
| #54 | 0.5s | **SURVIVED** | 0 | Survived round |
| #55 | 0.5s | **SURVIVED** | 0 | Survived round |
| #56 | 130.9s | **SURVIVED** | 0 | Survived round |
| #57 | 0.5s | **SURVIVED** | 0 | Survived round |
| #58 | 0.4s | **SURVIVED** | 0 | Survived round |
| #59 | 131.0s | **SURVIVED** | 0 | Survived round |
| #60 | 0.4s | **SURVIVED** | 0 | Survived round |
| #61 | 0.4s | **SURVIVED** | 0 | Survived round |
| #62 | 131.3s | **SURVIVED** | 0 | Survived round |
| #63 | 0.4s | **SURVIVED** | 0 | Survived round |
| #64 | 0.4s | **SURVIVED** | 0 | Survived round |
| #65 | 131.1s | **SURVIVED** | 0 | Survived round |
| #66 | 0.4s | **SURVIVED** | 0 | Survived round |
| #67 | 0.4s | **SURVIVED** | 0 | Survived round |
| #68 | 131.5s | **SURVIVED** | 0 | Survived round |
| #69 | 0.4s | **SURVIVED** | 0 | Survived round |
| #70 | 0.4s | **SURVIVED** | 0 | Survived round |
| #71 | 131.5s | **SURVIVED** | 0 | Survived round |
| #72 | 0.4s | **SURVIVED** | 0 | Survived round |
| #73 | 0.5s | **SURVIVED** | 0 | Survived round |
| #74 | 131.1s | **SURVIVED** | 0 | Survived round |
| #75 | 0.4s | **SURVIVED** | 0 | Survived round |
| #76 | 0.4s | **SURVIVED** | 0 | Survived round |
| #77 | 131.4s | **SURVIVED** | 0 | Survived round |
| #78 | 0.4s | **SURVIVED** | 0 | Survived round |
| #79 | 0.5s | **SURVIVED** | 0 | Survived round |
| #80 | 131.2s | **SURVIVED** | 0 | Survived round |
| #81 | 0.4s | **SURVIVED** | 0 | Survived round |
| #82 | 0.4s | **SURVIVED** | 0 | Survived round |
| #83 | 131.3s | **SURVIVED** | 0 | Survived round |
| #84 | 0.5s | **SURVIVED** | 0 | Survived round |
| #85 | 0.6s | **SURVIVED** | 0 | Survived round |
| #86 | 131.2s | **SURVIVED** | 0 | Survived round |
| #87 | 0.4s | **SURVIVED** | 0 | Survived round |
| #88 | 0.4s | **SURVIVED** | 0 | Survived round |
| #89 | 131.3s | **SURVIVED** | 0 | Survived round |
| #90 | 0.4s | **SURVIVED** | 0 | Survived round |
| #91 | 0.5s | **SURVIVED** | 0 | Survived round |
| #92 | 131.3s | **SURVIVED** | 0 | Survived round |
| #93 | 0.4s | **SURVIVED** | 0 | Survived round |
| #94 | 0.4s | **SURVIVED** | 0 | Survived round |
| #95 | 131.3s | **SURVIVED** | 0 | Survived round |
| #96 | 0.4s | **SURVIVED** | 0 | Survived round |
| #97 | 0.5s | **SURVIVED** | 0 | Survived round |
| #98 | 131.4s | **SURVIVED** | 0 | Survived round |
| #99 | 0.5s | **SURVIVED** | 0 | Survived round |
| #100 | 0.5s | **SURVIVED** | 0 | Survived round |
| #101 | 131.1s | **SURVIVED** | 0 | Survived round |
| #102 | 0.4s | **SURVIVED** | 0 | Survived round |
| #103 | 0.4s | **SURVIVED** | 0 | Survived round |
| #104 | 131.0s | **SURVIVED** | 0 | Survived round |
| #105 | 0.4s | **SURVIVED** | 0 | Survived round |
| #106 | 0.5s | **SURVIVED** | 0 | Survived round |
| #107 | 131.1s | **SURVIVED** | 0 | Survived round |
| #108 | 0.4s | **SURVIVED** | 0 | Survived round |
| #109 | 0.4s | **SURVIVED** | 0 | Survived round |
| #110 | 64.3s | **SURVIVED** | 0 | Survived round |
| #111 | 1.5s | **SURVIVED** | 0 | Survived round |
| #112 | 114.6s | **SURVIVED** | 0 | Survived round |
| #113 | 0.6s | **SURVIVED** | 0 | Survived round |
| #114 | 184.8s | **SURVIVED** | 0 | Survived round |
| #115 | 0.7s | **SURVIVED** | 0 | Survived round |
| #116 | 0.6s | **SURVIVED** | 0 | Survived round |
| #117 | 128.7s | **SURVIVED** | 0 | Survived round |
| #118 | 0.6s | **SURVIVED** | 0 | Survived round |
| #119 | 79.5s | **SURVIVED** | 0 | Survived round |
| #120 | 0.5s | **SURVIVED** | 0 | Survived round |
| #121 | 0.2s | **SURVIVED** | 0 | Survived round |
| #122 | 0.5s | **SURVIVED** | 0 | Survived round |
| #123 | 112.5s | **SURVIVED** | 0 | Survived round |
| #124 | 0.2s | **SURVIVED** | 0 | Survived round |
| #125 | 0.2s | **SURVIVED** | 0 | Survived round |

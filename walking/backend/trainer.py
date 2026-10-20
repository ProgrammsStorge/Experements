"""
CCM Biped v7 — N агентов (1..20) делят 5 CCM-моделей.

Новое:
  * Мультиагентность: до 20 тел одновременно, общие мозги.
    Episode завершается, когда упали/дошли все агенты.
  * temperature можно ставить в 0 → чистый argmax (greedy).
  * n_agents хранится в config.json, меняется из UI.
"""
import copy
import json
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

from ccm import CCM


# --------------------------------------------------------------------------- #
#  Конфиг окружения (редактируется из UI, сохраняется в config.json)
# --------------------------------------------------------------------------- #
DEFAULT_ENV = {
    # гравитация / инерция
    "g_torso":          7.0,
    "g_leg":            5.0,
    "inertia_torso":    1.6,

    # приводы
    "ctrl_torso":      14.0,
    "ctrl_leg_hip":     7.0,
    "ctrl_leg_knee":   10.0,

    # ankle PD
    "ankle_kp":         9.0,
    "ankle_kd":         2.5,

    # геометрия
    "L_torso":          0.60,
    "L_thigh":          0.45,
    "L_shin":           0.45,
    "target_h":         0.86,

    # награды
    "w_torso":          8.0,
    "w_omega":          4.0,
    "w_stay_bonus":     6.0,
    "w_fall_warn":     30.0,
    "w_vx":             0.05,
    "w_drift":          1.5,
    "fall_penalty":   200.0,
    "survive_bonus":  150.0,

    # обучение
    "temperature":      0.30,
    "epsilon":          0.07,
    "max_steps":       300,
    "improved_tail":    0.35,

    # мультиагентность (1..20)
    "n_agents":         1,
}

MAX_AGENTS = 20


def load_env(path: Path) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out = dict(DEFAULT_ENV)
            out.update({k: v for k, v in data.items() if k in DEFAULT_ENV})
            return out
        except Exception:
            pass
    return dict(DEFAULT_ENV)


def save_env(path: Path, env: dict):
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Параметры моделей
# --------------------------------------------------------------------------- #
GRAM_COUNT = 7
N_BINS     = 5
DT         = 0.02

MAX_CTX    = 96
Q_DECAY    = 0.9
Q_LR       = 0.1
Q_WEIGHT   = 0.15

MODEL_SPECS = {
    "torso": dict(
        label="Torso", color="#ff7b72",
        actions=["lean_back_hard", "lean_back", "hold", "lean_fwd", "lean_fwd_hard"],
        values=[-2.0, -1.0, 0.0, 1.0, 2.0],
    ),
    "left_leg": dict(
        label="Left leg", color="#79c0ff",
        actions=["extend_hard", "extend", "hold", "bend", "bend_hard"],
        values=[2.0, 1.0, 0.0, -1.0, -2.0],
    ),
    "right_leg": dict(
        label="Right leg", color="#56d364",
        actions=["extend_hard", "extend", "hold", "bend", "bend_hard"],
        values=[2.0, 1.0, 0.0, -1.0, -2.0],
    ),
    "left_arm": dict(
        label="Left arm", color="#d2a8ff",
        actions=["back_hard", "back", "hold", "fwd", "fwd_hard"],
        values=[-2.0, -1.0, 0.0, 1.0, 2.0],
    ),
    "right_arm": dict(
        label="Right arm", color="#a371f7",
        actions=["back_hard", "back", "hold", "fwd", "fwd_hard"],
        values=[-2.0, -1.0, 0.0, 1.0, 2.0],
    ),
}


# --------------------------------------------------------------------------- #
#  Утилиты
# --------------------------------------------------------------------------- #
def ngrams(text, n=GRAM_COUNT):
    text = (text or "").strip()
    if not text:
        return frozenset()
    if len(text) < n:
        return frozenset({text})
    return frozenset(text[i:i + n] for i in range(len(text) - n + 1))


def q(v, lo, hi, bins=N_BINS):
    v = max(lo, min(hi, float(v)))
    idx = int(round((v - lo) / (hi - lo) * (bins - 1)))
    return idx - (bins // 2)


def encode_state(kind, b):
    if kind == "torso":
        return (f"T{q(b.torso,-0.9,0.9):+d} "
                f"W{q(b.omega,-2.5,2.5):+d} "
                f"H{q(b.hip_y,0.45,0.95):+d} "
                f"V{q(b.hip_vx,-1.0,1.0):+d}")
    if kind == "left_leg":
        return (f"T{q(b.torso,-0.9,0.9):+d} "
                f"W{q(b.omega,-2.5,2.5):+d} "
                f"P{q(b.hip_L,-0.7,0.7):+d} "
                f"K{q(b.knee_L,0.0,1.3):+d} "
                f"H{q(b.hip_y,0.45,0.95):+d} "
                f"V{q(b.hip_vx,-1.0,1.0):+d}")
    if kind == "right_leg":
        return (f"T{q(b.torso,-0.9,0.9):+d} "
                f"W{q(b.omega,-2.5,2.5):+d} "
                f"P{q(b.hip_R,-0.7,0.7):+d} "
                f"K{q(b.knee_R,0.0,1.3):+d} "
                f"H{q(b.hip_y,0.45,0.95):+d} "
                f"V{q(b.hip_vx,-1.0,1.0):+d}")
    if kind == "left_arm":
        return (f"T{q(b.torso,-0.9,0.9):+d} "
                f"W{q(b.omega,-2.5,2.5):+d} "
                f"A{q(b.arm_L,-1.5,1.5):+d} "
                f"H{q(b.hip_y,0.45,0.95):+d}")
    if kind == "right_arm":
        return (f"T{q(b.torso,-0.9,0.9):+d} "
                f"W{q(b.omega,-2.5,2.5):+d} "
                f"A{q(b.arm_R,-1.5,1.5):+d} "
                f"H{q(b.hip_y,0.45,0.95):+d}")
    return ""


# --------------------------------------------------------------------------- #
#  Физика
# --------------------------------------------------------------------------- #
class Body:
    HIP_W  = 0.06
    L_ARM  = 0.34
    HEAD_R = 0.11

    def __init__(self, env, seed=None):
        self.env = env
        self.rng = random.Random(seed)
        self.reset()

    @property
    def L_TORSO(self): return self.env["L_torso"]
    @property
    def L_THIGH(self): return self.env["L_thigh"]
    @property
    def L_SHIN(self):  return self.env["L_shin"]

    def reset(self):
        r = self.rng
        self.t = 0.0
        self.hip_x  = 0.0
        self.hip_vx = 0.0

        self.torso = r.uniform(-0.03, 0.03)
        self.omega = r.uniform(-0.05, 0.05)

        self.hip_L  = r.uniform(-0.03, 0.03); self.hip_L_v = 0.0
        self.knee_L = r.uniform(0.04, 0.14);  self.knee_L_v = 0.0
        self.hip_R  = r.uniform(-0.03, 0.03); self.hip_R_v = 0.0
        self.knee_R = r.uniform(0.04, 0.14);  self.knee_R_v = 0.0

        self.arm_L = r.uniform(-0.04, 0.04); self.arm_L_v = 0.0
        self.arm_R = r.uniform(-0.04, 0.04); self.arm_R_v = 0.0

        self.fallen = False

        self.torso_hist = deque(maxlen=30)
        self.torso_hist.append(self.torso)

        yL, a1L, a2L = self._leg_geom("L")
        yR, a1R, a2R = self._leg_geom("R")
        self.hip_y = 0.5 * (yL + yR)
        self.hip_vy = 0.0
        self.a1L, self.a2L = a1L, a2L
        self.a1R, self.a2R = a1R, a2R

        self.foot_L_grounded = True
        self.foot_R_grounded = True
        self.foot_L_planted_x = self.hip_x + self._foot_rel("L")[0]
        self.foot_R_planted_x = self.hip_x + self._foot_rel("R")[0]

    def _leg_geom(self, side):
        hip_a  = getattr(self, f"hip_{side}")
        knee_a = getattr(self, f"knee_{side}")
        a1 = self.torso + hip_a
        a2 = a1 - knee_a
        y = self.L_THIGH * math.cos(a1) + self.L_SHIN * math.cos(a2)
        return y, a1, a2

    def _foot_rel(self, side):
        a1 = getattr(self, f"a1{side}")
        a2 = getattr(self, f"a2{side}")
        dx = self.L_THIGH * math.sin(a1) + self.L_SHIN * math.sin(a2)
        dy = -self.L_THIGH * math.cos(a1) - self.L_SHIN * math.cos(a2)
        return dx, dy

    def step(self, u, dt=DT):
        e = self.env
        u_t  = u.get("torso", 0.0)
        u_ll = u.get("left_leg", 0.0)
        u_rl = u.get("right_leg", 0.0)
        u_la = u.get("left_arm", 0.0)
        u_ra = u.get("right_arm", 0.0)

        # ===== ТОРС =====
        grav  = e["g_torso"] * math.sin(self.torso)
        ankle = -e["ankle_kp"] * self.torso - e["ankle_kd"] * self.omega
        ctrl  = e["ctrl_torso"] * u_t
        legs  = 2.5 * (u_ll + u_rl)
        arms  = -2.2 * (u_la + u_ra) * math.sin(self.torso + 0.3)
        damp  = -4.0 * self.omega
        self.omega += (grav + ankle + ctrl + legs + arms + damp) / e["inertia_torso"] * dt
        self.torso += self.omega * dt

        # ===== НОГИ =====
        for side, u_leg in (("L", u_ll), ("R", u_rl)):
            hip   = getattr(self, f"hip_{side}")
            hipv  = getattr(self, f"hip_{side}_v")
            knee  = getattr(self, f"knee_{side}")
            kneev = getattr(self, f"knee_{side}_v")

            a1 = self.torso + hip
            t_hip = -e["g_leg"] * math.sin(a1) - e["ctrl_leg_hip"] * u_leg - 1.2 * hipv
            hipv += t_hip / 0.5 * dt
            hip  += hipv * dt
            if hip < -1.0: hip, hipv = -1.0, max(0.0, hipv)
            if hip >  1.0: hip, hipv =  1.0, min(0.0, hipv)

            t_knee = -8.0 * (knee - 0.04) + 0.3 - e["ctrl_leg_knee"] * u_leg - 1.5 * kneev
            kneev += t_knee / 0.3 * dt
            knee  += kneev * dt
            if knee < 0.0: knee, kneev = 0.0, max(0.0, kneev)
            if knee > 1.4: knee, kneev = 1.4, min(0.0, kneev)

            setattr(self, f"hip_{side}", hip);   setattr(self, f"hip_{side}_v", hipv)
            setattr(self, f"knee_{side}", knee); setattr(self, f"knee_{side}_v", kneev)

        # ===== РУКИ =====
        for side, u_arm in (("L", u_la), ("R", u_ra)):
            a  = getattr(self, f"arm_{side}")
            av = getattr(self, f"arm_{side}_v")
            t  = -5.5 * math.sin(a + self.torso) - 2.2 * av + 6.5 * u_arm
            av += t / 0.25 * dt
            a  += av * dt
            if a < -1.8: a, av = -1.8, max(0.0, av)
            if a >  1.8: a, av =  1.8, min(0.0, av)
            setattr(self, f"arm_{side}", a); setattr(self, f"arm_{side}_v", av)

        prev_y = self.hip_y
        yL, a1L, a2L = self._leg_geom("L")
        yR, a1R, a2R = self._leg_geom("R")
        self.hip_y = 0.5 * (yL + yR)
        self.hip_vy = (self.hip_y - prev_y) / dt
        self.a1L, self.a2L = a1L, a2L
        self.a1R, self.a2R = a1R, a2R

        self._apply_ground(dt)

        self.t += dt
        self.torso_hist.append(self.torso)

        if abs(self.torso) > 1.05 or self.hip_y < 0.40:
            self.fallen = True

    def _apply_ground(self, dt):
        force = 0.0
        grounded_count = 0
        for side in ("L", "R"):
            dx, dy = self._foot_rel(side)
            fy = self.hip_y + dy
            grounded = fy <= 0.02
            was = getattr(self, f"foot_{side}_grounded")

            if grounded:
                grounded_count += 1
                if not was:
                    setattr(self, f"foot_{side}_planted_x", self.hip_x + dx)
                px = getattr(self, f"foot_{side}_planted_x")
                err = (self.hip_x + dx - px)
                force += -20.0 * err
            setattr(self, f"foot_{side}_grounded", grounded)

        if grounded_count > 0:
            force += -5.0 * self.hip_vx * grounded_count
        self.hip_vx += force * dt
        self.hip_vx *= 0.998
        self.hip_x  += self.hip_vx * dt

        if self.hip_x >  6.0: self.hip_x, self.hip_vx =  6.0, min(0.0, self.hip_vx)
        if self.hip_x < -6.0: self.hip_x, self.hip_vx = -6.0, max(0.0, self.hip_vx)

    def step_reward(self):
        e = self.env
        r = 0.0

        r += 1.0
        r += e["w_stay_bonus"] * math.exp(-6.0 * self.torso * self.torso)

        r -= e["w_torso"] * (self.torso ** 2)
        r -= e["w_omega"] * (self.omega ** 2)

        r -= e["w_fall_warn"] * (abs(self.torso) / 1.05) ** 4

        if len(self.torso_hist) >= 5:
            drift = sum(self.torso_hist) / len(self.torso_hist)
            r -= e["w_drift"] * (drift ** 2) * 10.0

        h_err = e["target_h"] - self.hip_y
        r -= 12.0 * max(0.0, h_err) ** 2
        r += 2.0 * max(0.0, 1.0 - abs(h_err) * 5.0)
        r -= 0.6 * (self.hip_vy ** 2)

        if self.foot_L_grounded: r += 0.4
        if self.foot_R_grounded: r += 0.4

        r -= 0.20 * (self.knee_L + self.knee_R)
        r -= 0.10 * (abs(self.hip_L) + abs(self.hip_R))
        r -= 0.05 * (abs(self.arm_L) + abs(self.arm_R))

        r -= e["w_vx"] * (self.hip_vx ** 2)
        r += 0.4 * max(-1.0, min(1.0, self.omega * self.hip_vx))
        if self.foot_L_grounded != self.foot_R_grounded:
            r += 0.25

        return r

    def skeleton(self):
        hx = self.hip_x
        hip  = [hx, self.hip_y]
        neck = [hip[0] + self.L_TORSO * math.sin(self.torso),
                hip[1] + self.L_TORSO * math.cos(self.torso)]
        head = [neck[0] + 0.18 * math.sin(self.torso),
                neck[1] + 0.18 * math.cos(self.torso)]

        def leg(a1, a2, dx=0.0):
            kn = [hip[0] + self.L_THIGH * math.sin(a1) + dx,
                  hip[1] - self.L_THIGH * math.cos(a1)]
            an = [kn[0] + self.L_SHIN * math.sin(a2),
                  kn[1] - self.L_SHIN * math.cos(a2)]
            ft = [an[0] + 0.13, an[1]]
            return kn, an, ft

        knL, anL, ftL = leg(self.a1L, self.a2L, -self.HIP_W)
        knR, anR, ftR = leg(self.a1R, self.a2R, +self.HIP_W)

        sh = [neck[0], neck[1] - 0.04]
        def arm(a):
            ab = self.torso + a
            el = [sh[0] + self.L_ARM * math.sin(ab),
                  sh[1] - self.L_ARM * math.cos(ab)]
            hd = [el[0] + self.L_ARM * 0.9 * math.sin(ab + 0.25),
                  el[1] - self.L_ARM * 0.9 * math.cos(ab + 0.25)]
            return el, hd

        elL, hdL = arm(self.arm_L)
        elR, hdR = arm(self.arm_R)

        return dict(
            hip=hip, neck=neck, head=head, headR=self.HEAD_R,
            kneeL=knL, ankleL=anL, footL=ftL,
            kneeR=knR, ankleR=anR, footR=ftR,
            elbowL=elL, handL=hdL, elbowR=elR, handR=hdR,
            groundL=self.foot_L_grounded,
            groundR=self.foot_R_grounded,
        )


# --------------------------------------------------------------------------- #
#  Агент — тело + trace + счётчики
# --------------------------------------------------------------------------- #
class Agent:
    __slots__ = ("body", "trace", "reward", "step_i", "done",
                 "last_words", "last_actions")

    def __init__(self, env, seed):
        self.body = Body(env, seed=seed)
        self.trace = []
        self.reward = 0.0
        self.step_i = 0
        self.done = False
        self.last_words = {}
        self.last_actions = {}

    def reset(self, env, seed):
        self.body = Body(env, seed=seed)
        self.trace = []
        self.reward = 0.0
        self.step_i = 0
        self.done = False
        self.last_words = {}
        self.last_actions = {}


# --------------------------------------------------------------------------- #
#  Выбор действия
# --------------------------------------------------------------------------- #
def pick_action(model, state_str, rng, temperature, epsilon):
    sig = ngrams(state_str, model.gram_count)
    scores, best_ctx = {}, {}

    for w in model.actions:
        ctxs = model.wordset.get(w) or set()
        best = 0.0
        bc   = None
        for c in ctxs:
            inter = len(sig & c)
            if inter == 0:
                continue
            union = len(sig) + len(c) - inter
            s = inter / union
            qv = model.quality[w].get(c, 0.0)
            boost = 1.0 + Q_WEIGHT * max(-1.0, min(1.0, qv / 40.0))
            s *= boost
            if s > best:
                best, bc = s, c
        scores[w] = best
        best_ctx[w] = bc

    # epsilon-greedy random jump
    if epsilon > 0.0 and rng.random() < epsilon:
        w = rng.choice(model.actions)
        return w, best_ctx.get(w), sig

    # ★ temperature == 0 → чистый argmax
    if temperature <= 1e-9:
        best_w = model.actions[0]
        best_s = float("-inf")
        for w in model.actions:
            if scores[w] > best_s:
                best_s = scores[w]
                best_w = w
        return best_w, best_ctx.get(best_w), sig

    mx   = max(scores.values()) if scores else 0.0
    exps = {w: math.exp((scores[w] - mx) / temperature) for w in model.actions}
    tot  = sum(exps.values())
    r    = rng.random() * tot
    acc  = 0.0
    for w in model.actions:
        acc += exps[w]
        if r <= acc:
            return w, best_ctx.get(w), sig
    w = model.actions[-1]
    return w, best_ctx.get(w), sig


# --------------------------------------------------------------------------- #
#  Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        self.env = load_env(self.dir / "config.json")
        self.rng = random.Random(int(time.time()))

        n0 = max(1, min(MAX_AGENTS, int(self.env.get("n_agents", 1))))
        self.env["n_agents"] = n0
        self.num_agents = n0
        self.agents = [
            Agent(self.env, seed=self.rng.randint(0, 10**9))
            for _ in range(n0)
        ]

        self.running       = False
        self.speed         = 1.0
        self.episode       = 0
        self.ep_reward     = 0.0
        self.best_reward   = float("-inf")
        self.best_episode  = -1
        self.last_words    = {}
        self.last_actions  = {}
        self.log           = []
        self.watch_best    = False

        self.models      = {}
        self.best_models = None
        self._build_models()

    # ------------------------------------------------------------------ #
    def _build_models(self):
        for kind, spec in MODEL_SPECS.items():
            m = CCM(no_setup=True)
            m.gram_count = GRAM_COUNT
            m.format     = "{{ messages[-1] }}"
            m.actions    = list(spec["actions"])
            m.wordset    = {a: set() for a in m.actions}
            m.quality    = defaultdict(lambda: defaultdict(float))
            self._seed_random(m)
            self.models[kind] = m

    def _random_sig(self):
        toks = []
        for p in ("T", "W", "H", "V", "P", "K", "A"):
            toks.append(f"{p}{self.rng.randint(-2, 2):+d}")
        self.rng.shuffle(toks)
        return ngrams(" ".join(toks[:4]))

    def _seed_random(self, model):
        for w in model.actions:
            model.wordset[w] = set()
            for _ in range(6):
                model.wordset[w].add(self._random_sig())

    def _active_models(self):
        if self.watch_best and self.best_models is not None:
            return self.best_models
        return self.models

    # ------------------------------------------------------------------ #
    #  Управление числом агентов
    # ------------------------------------------------------------------ #
    def _set_num_agents(self, n: int) -> bool:
        """Возвращает True, если число агентов реально изменилось."""
        n = max(1, min(MAX_AGENTS, int(n)))
        cur = len(self.agents)
        if n == cur:
            self.num_agents = n
            return False
        if n < cur:
            self.agents = self.agents[:n]
        else:
            for _ in range(n - cur):
                self.agents.append(
                    Agent(self.env, seed=self.rng.randint(0, 10**9))
                )
        self.num_agents = n
        return True

    # ------------------------------------------------------------------ #
    def step(self):
        active = self._active_models()

        for ag in self.agents:
            if ag.done:
                continue

            b = ag.body
            acts, recs = {}, []

            for kind, model in active.items():
                st = encode_state(kind, b)
                word, ctx, sig = pick_action(
                    model, st, self.rng,
                    self.env["temperature"], self.env["epsilon"]
                )
                val = MODEL_SPECS[kind]["values"][model.actions.index(word)]
                acts[kind] = val
                ag.last_words[kind] = word
                recs.append((kind, word, ctx, sig))

            ag.last_actions = acts
            b.step(acts)
            ag.reward += b.step_reward()
            ag.trace.append(recs)
            ag.step_i += 1

            if b.fallen or ag.step_i >= self.env["max_steps"]:
                ag.done = True

        # верхнеуровневые — для UI модели-панели (первый агент)
        if self.agents:
            self.last_actions = self.agents[0].last_actions
            self.last_words   = self.agents[0].last_words

        if self.agents:
            self.ep_reward = sum(a.reward for a in self.agents) / len(self.agents)

        if all(a.done for a in self.agents):
            self.finish_episode()

    # ------------------------------------------------------------------ #
    def finish_episode(self):
        max_steps = self.env["max_steps"]

        # --- итоговые награды по каждому агенту + флаг выживания
        agent_outcomes = []
        agent_survived = []
        for ag in self.agents:
            survived = (not ag.body.fallen) and (ag.step_i >= max_steps)
            outcome = self.env["survive_bonus"] if survived else -self.env["fall_penalty"]
            agent_outcomes.append(outcome)
            agent_survived.append(survived)

        agent_totals = [ag.reward + oc for ag, oc in zip(self.agents, agent_outcomes)]
        total = sum(agent_totals) / len(agent_totals) if agent_totals else 0.0

        max_steps_seen = max((ag.step_i for ag in self.agents), default=0)
        any_fallen     = any(ag.body.fallen for ag in self.agents)
        mean_x         = sum(ag.body.hip_x for ag in self.agents) / max(1, len(self.agents))

        # ---------- watch-режим: без обучения ----------
        if self.watch_best:
            self.log.append({
                "episode":  self.episode,
                "reward":   round(total, 2),
                "best":     None if self.best_reward == float("-inf") else round(self.best_reward, 2),
                "steps":    max_steps_seen,
                "fallen":   bool(any_fallen),
                "survived": all(agent_survived),
                "improved": False,
                "added": 0, "replaced": 0,
                "hip_x":  round(mean_x, 3),
                "watch":  True,
                "agents": len(self.agents),
            })
            if len(self.log) > 400:
                self.log.pop(0)
            self.episode += 1
            self._reset_agents()
            return

        improved = total > self.best_reward
        if improved:
            self.best_reward  = total
            self.best_episode = self.episode
            self._snapshot_best()

        added = replaced = 0
        FALLOUT_STEPS = 8

        # ── 1) quality — по каждому агенту со своим outcome
        for ag, outcome, survived in zip(self.agents, agent_outcomes, agent_survived):
            n_trace = len(ag.trace)
            for i, recs in enumerate(ag.trace):
                is_late_fall = (not survived) and (i >= n_trace - FALLOUT_STEPS)
                weight = Q_LR * (2.5 if is_late_fall else 1.0)
                for kind, word, ctx, sig in recs:
                    m = self.models[kind]
                    if ctx is not None:
                        qw = m.quality[word]
                        qw[ctx] = Q_DECAY * qw.get(ctx, 0.0) + weight * outcome
                        if is_late_fall and outcome < 0:
                            qw[ctx] -= 5.0

        # ── 2) wordset — при улучшении только хвост у КАЖДОГО агента
        for ag in self.agents:
            n_trace = len(ag.trace)
            tail_start = int(n_trace * (1.0 - self.env["improved_tail"]))
            for i, recs in enumerate(ag.trace):
                for kind, word, ctx, sig in recs:
                    m = self.models[kind]
                    s = m.wordset.setdefault(word, set())
                    qw = m.quality[word]
                    if improved:
                        if i >= tail_start and sig not in s:
                            s.add(sig)
                            added += 1
                    else:
                        if (ctx is not None and ctx in s
                                and qw.get(ctx, 0.0) < 0.0
                                and sig not in s):
                            s.discard(ctx)
                            s.add(sig)
                            replaced += 1

        # ── 3) затухание качества
        for kind, m in self.models.items():
            for w in m.actions:
                for c in list(m.quality[w].keys()):
                    m.quality[w][c] *= 0.995

        # ── 4) лимит контекстов
        for kind, m in self.models.items():
            for w in m.actions:
                s = m.wordset.setdefault(w, set())
                if len(s) > MAX_CTX:
                    ranked = sorted(s, key=lambda c: m.quality[w].get(c, 0.0))
                    for c in ranked[:len(s) - MAX_CTX]:
                        s.discard(c)
                        m.quality[w].pop(c, None)

        # ── 5) не даём опустеть
        for kind, m in self.models.items():
            for w in m.actions:
                if not m.wordset.get(w):
                    for _ in range(3):
                        m.wordset[w].add(self._random_sig())

        self.log.append({
            "episode":  self.episode,
            "reward":   round(total, 2),
            "best":     round(self.best_reward, 2),
            "steps":    max_steps_seen,
            "fallen":   bool(any_fallen),
            "survived": all(agent_survived),
            "improved": bool(improved),
            "added":    added,
            "replaced": replaced,
            "hip_x":    round(mean_x, 3),
            "watch":    False,
            "agents":   len(self.agents),
        })
        if len(self.log) > 400:
            self.log.pop(0)

        self.episode += 1
        self._reset_agents()

    # ------------------------------------------------------------------ #
    def _reset_agents(self):
        for ag in self.agents:
            ag.reset(self.env, seed=self.rng.randint(0, 10**9))
        self.ep_reward = 0.0

    # ------------------------------------------------------------------ #
    def _snapshot_best(self):
        snap = {}
        for kind, m in self.models.items():
            nm = CCM(no_setup=True)
            nm.gram_count = m.gram_count
            nm.format     = m.format
            nm.actions    = list(m.actions)
            nm.wordset    = {w: set(ctxs) for w, ctxs in m.wordset.items()}
            nm.quality    = defaultdict(lambda: defaultdict(float))
            for w, d in m.quality.items():
                nm.quality[w] = defaultdict(float, dict(d))
            snap[kind] = nm
        self.best_models = snap

    # ------------------------------------------------------------------ #
    def snapshot(self):
        active = self._active_models()
        models = []
        for kind, m in active.items():
            spec = MODEL_SPECS[kind]
            ctx_count = sum(len(v) for v in m.wordset.values())
            q_all = [qv for d in m.quality.values() for qv in d.values()]
            avg_q = (sum(q_all) / len(q_all)) if q_all else 0.0
            models.append({
                "key": kind, "label": spec["label"], "color": spec["color"],
                "action": self.last_words.get(kind, "—"),
                "contexts": ctx_count,
                "quality": round(avg_q, 2),
            })

        agents_out = []
        for ag in self.agents:
            agents_out.append({
                "body":   ag.body.skeleton(),
                "fallen": bool(ag.body.fallen),
                "done":   bool(ag.done),
                "reward": round(ag.reward, 2),
                "step":   ag.step_i,
                "hip_x":  round(ag.body.hip_x, 3),
                "hip_vx": round(ag.body.hip_vx, 3),
                "torso":  round(ag.body.torso, 3),
                "actions": {k: round(v, 2) for k, v in ag.last_actions.items()},
            })

        max_step = max((a.step_i for a in self.agents), default=0)

        return {
            "type": "frame",
            "running": self.running,
            "speed":   self.speed,
            "episode": self.episode,
            "step":    max_step,
            "maxStep": self.env["max_steps"],
            "reward":  round(self.ep_reward, 2),
            "best":    None if self.best_reward == float("-inf") else round(self.best_reward, 2),
            "bestEpisode": self.best_episode,
            "watchBest": self.watch_best,
            "hasBest": self.best_models is not None,
            "nAgents": len(self.agents),
            "actions": {k: round(v, 2) for k, v in self.last_actions.items()},
            "agents":  agents_out,
            "models":  models,
            "log":     self.log[-80:],
            "env":     dict(self.env),
        }

    # ------------------------------------------------------------------ #
    def hard_reset(self):
        self._reset_agents()

    def reset_brains(self):
        self.best_reward  = float("-inf")
        self.best_episode = -1
        self.episode      = 0
        self.log          = []
        self.best_models  = None
        self.rng = random.Random(int(time.time()))
        for kind, spec in MODEL_SPECS.items():
            m = self.models[kind]
            m.wordset = {a: set() for a in m.actions}
            m.quality = defaultdict(lambda: defaultdict(float))
            self._seed_random(m)
        self.hard_reset()

    # ------------------------------------------------------------------ #
    #  Config update
    # ------------------------------------------------------------------ #
    def update_env(self, patch: dict):
        for k, v in patch.items():
            if k in DEFAULT_ENV:
                try:
                    if k == "n_agents":
                        self.env[k] = max(1, min(MAX_AGENTS, int(v)))
                    else:
                        self.env[k] = float(v) if not isinstance(v, bool) else v
                except Exception:
                    pass

        if "n_agents" in patch:
            if self._set_num_agents(int(self.env["n_agents"])):
                self.hard_reset()

        for ag in self.agents:
            ag.body.env = self.env

        save_env(self.dir / "config.json", self.env)

    def reset_env(self):
        self.env = dict(DEFAULT_ENV)
        self._set_num_agents(int(self.env["n_agents"]))
        for ag in self.agents:
            ag.body.env = self.env
        save_env(self.dir / "config.json", self.env)
        self.hard_reset()

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #
    def _model_payload(self, m):
        ws = {w: [sorted(list(c)) for c in ctxs] for w, ctxs in m.wordset.items()}
        return json.dumps({
            "format":     m.format,
            "gram_count": m.gram_count,
            "actions":    m.actions,
            "wordset":    ws,
        })

    def save(self):
        cctx = zstd.ZstdCompressor(level=22) if _HAS_ZSTD else None
        meta = {
            "episode":     self.episode,
            "best_reward": None if self.best_reward == float("-inf") else self.best_reward,
            "best_episode": self.best_episode,
            "saved_at":    time.time(),
            "n_agents":    self.num_agents,
            "models":      {},
        }
        for kind, m in self.models.items():
            payload = self._model_payload(m)
            if cctx:
                (self.dir / f"{kind}.cws").write_bytes(cctx.compress(payload.encode("utf-8")))
            else:
                (self.dir / f"{kind}.json").write_text(payload, encoding="utf-8")
            meta["models"][kind] = sum(len(v) for v in m.wordset.values())

        if self.best_models is not None:
            for kind, m in self.best_models.items():
                payload = self._model_payload(m)
                if cctx:
                    (self.dir / f"{kind}.best.cws").write_bytes(cctx.compress(payload.encode("utf-8")))
                else:
                    (self.dir / f"{kind}.best.json").write_text(payload, encoding="utf-8")
            meta["has_best_snapshot"] = True

        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        save_env(self.dir / "config.json", self.env)
        return str(self.dir)

    def load(self):
        dctx = zstd.ZstdDecompressor() if _HAS_ZSTD else None
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for kind, m in self.models.items():
                p_cws  = self.dir / f"{kind}.cws"
                p_json = self.dir / f"{kind}.json"
                if p_cws.exists() and dctx:
                    data = json.loads(dctx.decompress(p_cws.read_bytes()).decode("utf-8"))
                elif p_json.exists():
                    data = json.loads(p_json.read_text(encoding="utf-8"))
                else:
                    continue
                m.wordset = {w: set(frozenset(c) for c in ctxs)
                             for w, ctxs in data["wordset"].items()}
                if data.get("actions"):
                    m.actions = data["actions"]

            if meta.get("has_best_snapshot"):
                snap = {}
                for kind, m in self.models.items():
                    p_cws  = self.dir / f"{kind}.best.cws"
                    p_json = self.dir / f"{kind}.best.json"
                    if p_cws.exists() and dctx:
                        data = json.loads(dctx.decompress(p_cws.read_bytes()).decode("utf-8"))
                    elif p_json.exists():
                        data = json.loads(p_json.read_text(encoding="utf-8"))
                    else:
                        continue
                    nm = CCM(no_setup=True)
                    nm.gram_count = GRAM_COUNT
                    nm.format = "{{ messages[-1] }}"
                    nm.actions = data.get("actions", MODEL_SPECS[kind]["actions"])
                    nm.wordset = {w: set(frozenset(c) for c in ctxs)
                                  for w, ctxs in data["wordset"].items()}
                    nm.quality = defaultdict(lambda: defaultdict(float))
                    snap[kind] = nm
                if snap:
                    self.best_models = snap

            self.best_reward  = meta.get("best_reward") or float("-inf")
            self.best_episode = meta.get("best_episode", -1)
            self.episode      = meta.get("episode", 0)
            self.hard_reset()
            return True
        except Exception:
            return False
# -*- coding: utf-8 -*-
"""
Multi-Scene Boids + Perlin Wind (OpenCV)
- K 个场景；前 8 个为单一模式，其后为 2~3 种模式随机混合（权重 Dirichlet）
- 同一场景可含若干“子群”（物种/模式），子群内部相互作用、共享场景风场
- 每场景输出到独立文件夹：video.mp4 + tracks.txt（MOT 行：frame,id,x,y,1,1,-1,-1,-1）
- ★ MOT 的 x,y 为像素坐标，原点=画布左上，x 向右、y 向下（无裁剪）
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple

import numpy as np
import cv2
from collections import deque
import noise

# ============================= 画布 / 输出通用参数 =============================
SPACE_SIZE = 200.0                     # 世界坐标范围 [0, SPACE_SIZE] × [0, SPACE_SIZE]
CANVAS_W, CANVAS_H = 1000, 1000        # 视频分辨率（建议偶数）
BG_COLOR = (250, 250, 250)
FPS = 30
DRAW_BORDER = True                     # 便于检查播放器裁切（四周黑边）
SHOW_WIND = True
WIND_GRID = 14
COLORMAP = cv2.COLORMAP_TURBO

# 叠加标题/图例（中文需 PIL）
USE_PIL_TEXT = False                   # 如需中文标题设 True 并设置 FONT_PATH
FONT_PATH = ""                         # 例如 Windows: "C:/Windows/Fonts/simhei.ttf"
TITLE_COLOR = (10, 10, 10)

# 色条（风速大小）位置与尺寸
CB_W, CB_H = 24, 280
CB_X, CB_Y = CANVAS_W - 60, 70
CB_TICKS = 3
CB_LABEL = "Wind Speed"

# 参考箭头（quiver key）
QK_X0, QK_Y0 = CANVAS_W - 220, 40
QK_COLOR = (40, 40, 40)
QK_THICK = 3

# 轨迹可视化
BOID_COLOR  = (0, 0, 0)
BOID_RADIUS = 3
TRAIL_THICK = 2
TRAILS_PER_GROUP = 3                   # 每个子群最多绘制多少条轨迹
TRAIL_COLORS = [
    (255, 0, 0), (0, 140, 255), (0, 0, 255), (0, 180, 0),
    (180, 0, 180), (0, 165, 255), (128, 0, 255), (255, 128, 0)
]

# 随机源（会在 main 里按 seed 重置）
rng = np.random.default_rng(2025)

# ============================= 基础工具函数 =============================
def limit_vec(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > max_norm and n > 0:
        return v * (max_norm / n)
    return v

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0: return v
    return v / n

def even_hw(w: int, h: int) -> Tuple[int, int]:
    return (w if w % 2 == 0 else w - 1, h if h % 2 == 0 else h - 1)

# —— 世界坐标 -> 像素坐标（原点：左上；x→右；y→下）
# 使用 “满幅”缩放：y_px = (1 - y/SPACE_SIZE) * (CANVAS_H - 1)
SX = (CANVAS_W - 1) / SPACE_SIZE
SY = (CANVAS_H - 1) / SPACE_SIZE

def world_to_px_float(xy: np.ndarray) -> Tuple[float, float]:
    """浮点像素坐标，不裁剪：用于 MOT 输出"""
    x, y = float(xy[0]), float(xy[1])
    px = x * SX
    py = (SPACE_SIZE - y) * SY
    return px, py

def world_to_px_int(xy: np.ndarray) -> Tuple[int, int]:
    """整数像素坐标，裁剪到画布范围：用于绘制"""
    px, py = world_to_px_float(xy)
    px_i = int(np.clip(round(px), 0, CANVAS_W - 1))
    py_i = int(np.clip(round(py), 0, CANVAS_H - 1))
    return px_i, py_i

# —— Windows/Unix 兼容文件名 —— #
INVALID_CHARS = '<>:"/\\|?*'
RESERVED_NAMES = {
    'CON','PRN','AUX','NUL',
    'COM1','COM2','COM3','COM4','COM5','COM6','COM7','COM8','COM9',
    'LPT1','LPT2','LPT3','LPT4','LPT5','LPT6','LPT7','LPT8','LPT9'
}
def safe_name(s: str, maxlen: int = 120) -> str:
    t = ''.join((c if c not in INVALID_CHARS else '-') for c in s)
    t = t.replace(' ', '_').strip('. ')
    if t.upper() in RESERVED_NAMES: t = '_' + t
    if len(t) > maxlen: t = t[:maxlen].rstrip('. ')
    return t

# ============================= 文本绘制（标题） =============================
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

def put_text(img, text, org, color=(10,10,10), scale=0.8, thickness=2):
    """优先用 PIL（支持中文），否则 OpenCV 英文字体"""
    if USE_PIL_TEXT and PIL_AVAILABLE and FONT_PATH and os.path.exists(FONT_PATH):
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(FONT_PATH, int(44*scale))
        except Exception:
            font = ImageFont.load_default()
        draw.text(org, text, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
        return np.asarray(img_pil)
    else:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return img

# ============================= 配置与个体 =============================
@dataclass
class SpeciesCfg:
    # 基础动力学
    name: str
    mode: str = "baseline"
    DT: float = 1.0
    MAX_SPEED: float = 2.2
    MAX_FORCE: float = 0.06
    # 感知与权重
    SEPARATION_RADIUS: float = 3.0
    ALIGNMENT_RADIUS: float = 12.0
    COHESION_RADIUS: float = 24.0
    W_SEPARATION: float = 1.6
    W_ALIGNMENT: float = 1.0
    W_COHESION: float = 0.7
    COHESION_GAIN_MIN: float = 0.4
    COHESION_GAIN_MAX: float = 1.6
    # 初速度（截断正态）
    SPEED_MEAN: float = 0.8 * 2.2
    SPEED_STD: float  = 0.25 * 2.2
    SPEED_MIN: float  = 0.05 * 2.2
    SPEED_MAX: float  = 2.2
    # 模式相关参数
    MIGRATE_DIR: Tuple[float, float] = (1.0, 0.1)
    MIGRATE_WEIGHT: float = 0.35
    SWIRL_CENTER: Tuple[float, float] = (SPACE_SIZE/2, SPACE_SIZE/2)
    SWIRL_STRENGTH: float = 0.35
    SWIRL_CENTRIPETAL: float = 0.08
    SPLIT_A: Tuple[float, float] = (50.0, 150.0)
    SPLIT_B: Tuple[float, float] = (150.0, 50.0)
    SPLIT_PERIOD: int = 150
    SPLIT_GOAL_WEIGHT: float = 0.4
    JITTER_WEIGHT: float = 0.04
    LEADER_FRAC: float = 0.08
    LEADER_GOAL_WEIGHT: float = 0.6
    FOLLOW_TO_LEADERS: float = 0.35
    PATROL_GOAL_WEIGHT: float = 0.35
    # 每子群可调整风影响增益（相对场景风）
    WIND_GAIN: float = 1.0
    # 轨迹长度
    TRAIL_LEN: int = 250
    # 可选：函数句柄
    LEADER_PATH: Callable = None
    PATROL_PATH: Callable = None

@dataclass
class SceneCfg:
    frames: int = 400
    title: str = ""
    # 场景级风场
    NOISE_SCALE_XY: float = 0.06
    NOISE_TIME_SCALE: float = 0.03
    WIND_STRENGTH: float = 0.20
    ARROW_GAIN_WORLD: float = 24.0

@dataclass
class Boid:
    id: int
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    cohesion_gain: float
    trail: deque
    is_leader: bool
    species: int  # 所属子群索引

# ============================= 模式库（生成 SpeciesCfg） =============================
def leader_path_func(t, sp: SpeciesCfg):
    x = 30 + (0.5 * t) % (SPACE_SIZE - 60)
    y = 30 + (0.4 * t) % (SPACE_SIZE - 60)
    return x, y

def patrol_path_func(t, sp: SpeciesCfg):
    T = 160.0
    tt = (t % (4*T))
    m = SPACE_SIZE
    margin = 30.0
    if tt < T:
        frac = tt / T;              return margin + frac*(m-2*margin), m - margin
    elif tt < 2*T:
        frac = (tt - T)/T;          return m - margin, m - margin - frac*(m-2*margin)
    elif tt < 3*T:
        frac = (tt - 2*T)/T;        return m - margin - frac*(m-2*margin), margin
    else:
        frac = (tt - 3*T)/T;        return margin, margin + frac*(m-2*margin)

def sp_baseline():
    return SpeciesCfg(name="baseline")

def sp_tight_cohesion():
    cfg = SpeciesCfg(name="tight_cohesion")
    cfg.W_COHESION = 1.4; cfg.COHESION_RADIUS = 32.0; cfg.SEPARATION_RADIUS = 2.5
    return cfg

def sp_migrate_line():
    cfg = SpeciesCfg(name="migrate_line", mode="migrate_line")
    cfg.W_ALIGNMENT = 1.4; cfg.W_COHESION = 0.5
    cfg.MIGRATE_DIR = (1.0, 0.2); cfg.MIGRATE_WEIGHT = 0.5
    return cfg

def sp_vortex_swirl():
    cfg = SpeciesCfg(name="vortex_swirl", mode="vortex_swirl")
    cfg.SWIRL_STRENGTH = 0.45; cfg.SWIRL_CENTRIPETAL = 0.10; cfg.W_ALIGNMENT = 0.8
    return cfg

def sp_split_merge():
    cfg = SpeciesCfg(name="split_merge", mode="split_merge")
    cfg.SPLIT_A = (40.0, 160.0); cfg.SPLIT_B = (160.0, 40.0)
    cfg.SPLIT_PERIOD = 120; cfg.SPLIT_GOAL_WEIGHT = 0.45
    return cfg

def sp_turbulent_wander():
    cfg = SpeciesCfg(name="turbulent_wander", mode="turbulent_wander")
    cfg.JITTER_WEIGHT = 0.06; cfg.W_ALIGNMENT = 0.7; cfg.W_COHESION = 0.5
    return cfg

def sp_leader_follow():
    cfg = SpeciesCfg(name="leader_follow", mode="leader_follow")
    cfg.LEADER_FRAC = 0.10; cfg.LEADER_GOAL_WEIGHT = 0.8; cfg.FOLLOW_TO_LEADERS = 0.45
    cfg.W_ALIGNMENT = 1.2; cfg.LEADER_PATH = leader_path_func
    return cfg

def sp_patrol_loop():
    cfg = SpeciesCfg(name="patrol_loop", mode="patrol_loop")
    cfg.PATROL_GOAL_WEIGHT = 0.45; cfg.W_ALIGNMENT = 1.1; cfg.PATROL_PATH = patrol_path_func
    return cfg

SPECIES_LIBRARY = [
    sp_baseline, sp_tight_cohesion, sp_migrate_line, sp_vortex_swirl,
    sp_split_merge, sp_turbulent_wander, sp_leader_follow, sp_patrol_loop
]

# ============================= 风场（场景级） =============================
def scene_wind_vec(x, y, t, scfg: SceneCfg) -> np.ndarray:
    sx = x * scfg.NOISE_SCALE_XY
    sy = y * scfg.NOISE_SCALE_XY
    st = t * scfg.NOISE_TIME_SCALE
    wx = noise.pnoise2(sx, sy + 37.123 + st, octaves=3, persistence=0.55, lacunarity=2.0,
                       repeatx=1024, repeaty=1024, base=0)
    wy = noise.pnoise2(sx + 101.7 + st, sy,  octaves=3, persistence=0.55, lacunarity=2.0,
                       repeatx=1024, repeaty=1024, base=0)
    return np.array([wx, wy], dtype=np.float64) * scfg.WIND_STRENGTH

# ============================= 行为力（模式专属） =============================
def steer_to_point(b, target: np.ndarray, sp: SpeciesCfg) -> np.ndarray:
    d = target - b.position
    if np.linalg.norm(d) == 0: return np.zeros(2)
    desired = unit(d) * sp.MAX_SPEED
    return limit_vec(desired - b.velocity, sp.MAX_FORCE)

def steer_to_dir(b, dir_unit: np.ndarray, sp: SpeciesCfg) -> np.ndarray:
    desired = dir_unit * sp.MAX_SPEED
    return limit_vec(desired - b.velocity, sp.MAX_FORCE)

def extra_forces(b, t: float, leaders_pos: np.ndarray, sp: SpeciesCfg) -> np.ndarray:
    f = np.zeros(2)
    m = sp.mode

    if m == 'migrate_line':
        d = np.array(sp.MIGRATE_DIR, dtype=np.float64)
        du = d if np.linalg.norm(d)==0 else unit(d)
        f += sp.MIGRATE_WEIGHT * steer_to_dir(b, du, sp)

    if m == 'vortex_swirl':
        center = np.array(sp.SWIRL_CENTER)
        radial = center - b.position
        if np.linalg.norm(radial) > 0:
            tang = unit(np.array([-radial[1], radial[0]]))
            f += sp.SWIRL_STRENGTH * limit_vec(tang * sp.MAX_SPEED - b.velocity, sp.MAX_FORCE)
            if sp.SWIRL_CENTRIPETAL > 0:
                f += sp.SWIRL_CENTRIPETAL * steer_to_point(b, center, sp)

    if m == 'split_merge':
        TA = np.array(sp.SPLIT_A, dtype=np.float64)
        TB = np.array(sp.SPLIT_B, dtype=np.float64)
        period = sp.SPLIT_PERIOD
        target = TA if int(t // period) % 2 == 0 else TB
        f += sp.SPLIT_GOAL_WEIGHT * steer_to_point(b, target, sp)

    if m == 'turbulent_wander':
        jitter = unit(np.random.normal(size=2)) * sp.MAX_SPEED
        f += sp.JITTER_WEIGHT * limit_vec(jitter - b.velocity, sp.MAX_FORCE)

    if m == 'leader_follow':
        if b.is_leader and sp.LEADER_PATH is not None:
            gx, gy = sp.LEADER_PATH(t, sp)
            f += sp.LEADER_GOAL_WEIGHT * steer_to_point(b, np.array([gx, gy]), sp)
        elif (not b.is_leader) and leaders_pos.size > 0:
            lc = leaders_pos.mean(axis=0)
            f += sp.FOLLOW_TO_LEADERS * steer_to_point(b, lc, sp)

    if m == 'patrol_loop' and sp.PATROL_PATH is not None:
        gx, gy = sp.PATROL_PATH(t, sp)
        f += sp.PATROL_GOAL_WEIGHT * steer_to_point(b, np.array([gx, gy]), sp)

    return f

# ============================= 子群相互作用（仅群内交互） =============================
def flock_group(boids: List, t: float, sp: SpeciesCfg, scfg: SceneCfg):
    if not boids: return
    leaders_pos = np.array([b.position for b in boids if b.is_leader])
    for b in boids:
        sep_sum = np.zeros(2); ali_sum = np.zeros(2); coh_pos_sum = np.zeros(2)
        n_sep = n_ali = n_coh = 0

        for o in boids:
            if o is b: continue
            dvec = o.position - b.position
            dist = np.linalg.norm(dvec)
            if dist == 0: continue
            if dist < sp.SEPARATION_RADIUS:
                sep_sum -= dvec / (dist * dist + 1e-6); n_sep += 1
            if dist < sp.ALIGNMENT_RADIUS:
                ali_sum += o.velocity; n_ali += 1
            if dist < sp.COHESION_RADIUS:
                coh_pos_sum += o.position; n_coh += 1

        steer_sep = limit_vec(unit(sep_sum)*sp.MAX_SPEED - b.velocity, sp.MAX_FORCE) if n_sep else np.zeros(2)
        steer_ali = limit_vec(unit(ali_sum/n_ali)*sp.MAX_SPEED - b.velocity, sp.MAX_FORCE) if n_ali else np.zeros(2)
        steer_coh = np.zeros(2)
        if n_coh:
            center = coh_pos_sum / n_coh
            ddir = center - b.position
            if np.linalg.norm(ddir) > 0:
                steer_coh = limit_vec(unit(ddir)*sp.MAX_SPEED - b.velocity, sp.MAX_FORCE)
            steer_coh *= b.cohesion_gain

        # 风场（场景级）+ 子群风增益
        f_wind = scene_wind_vec(b.position[0], b.position[1], t, scfg) * sp.WIND_GAIN
        # 模式外力
        f_extra = extra_forces(b, t, leaders_pos, sp)

        # 合力
        b.acceleration += limit_vec(sp.W_SEPARATION * steer_sep, sp.MAX_FORCE)
        b.acceleration += limit_vec(sp.W_ALIGNMENT  * steer_ali, sp.MAX_FORCE)
        b.acceleration += limit_vec(sp.W_COHESION   * steer_coh, sp.MAX_FORCE)
        b.acceleration += limit_vec(f_wind, sp.MAX_FORCE)
        b.acceleration += limit_vec(f_extra, sp.MAX_FORCE)

    # 状态积分
    for b in boids:
        b.velocity += b.acceleration
        b.velocity = limit_vec(b.velocity, sp.MAX_SPEED)
        b.position += b.velocity * sp.DT
        b.acceleration[:] = 0.0
        b.trail.append(b.position.copy())

# ============================= 可视化：风场/叠加 =============================
def precompute_colorbar():
    grad = np.linspace(255, 0, CB_H, dtype=np.uint8).reshape(CB_H, 1)
    grad = np.repeat(grad, CB_W, axis=1)
    return cv2.applyColorMap(grad, COLORMAP)

CB_IMG = precompute_colorbar()

def draw_wind(img, t, scfg: SceneCfg):
    xs = np.linspace(0, SPACE_SIZE, WIND_GRID)
    ys = np.linspace(0, SPACE_SIZE, WIND_GRID)
    gain_px = scfg.ARROW_GAIN_WORLD * 0.5 * ((CANVAS_W - 1)/SPACE_SIZE + (CANVAS_H - 1)/SPACE_SIZE)
    MAG_VMAX = math.sqrt(2.0) * scfg.WIND_STRENGTH
    for y in ys:
        for x in xs:
            w = scene_wind_vec(x, y, t, scfg)
            mag = float(np.linalg.norm(w))
            p0 = world_to_px_int((x, y))
            vec_px = (w[0] * gain_px, -w[1] * gain_px)  # 注意像素 y 向下
            p1 = (int(round(p0[0] + vec_px[0])), int(round(p0[1] + vec_px[1])))
            val = int(np.clip(mag / MAG_VMAX, 0.0, 1.0) * 255)
            color = cv2.applyColorMap(np.uint8([[val]]), COLORMAP)[0, 0].tolist()
            cv2.arrowedLine(img, p0, p1, color, thickness=2, tipLength=0.25)

def draw_colorbar(img, scfg: SceneCfg):
    MAG_VMAX = math.sqrt(2.0) * scfg.WIND_STRENGTH
    x0, y0 = CB_X, CB_Y
    h, w = CB_IMG.shape[:2]
    img[y0:y0+h, x0:x0+w] = CB_IMG
    cv2.rectangle(img, (x0-1, y0-1), (x0+w, y0+h), (0,0,0), 1)
    for i in range(CB_TICKS):
        frac = i/(CB_TICKS-1) if CB_TICKS>1 else 0
        y = int(y0 + (1-frac)*h)
        cv2.line(img, (x0+w+2, y), (x0+w+8, y), (0,0,0), 1)
        val = frac * MAG_VMAX
        label = f"{val:.2f}"
        cv2.putText(img, label, (x0+w+10, max(12, y+4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, CB_LABEL, (x0-8, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

def draw_quiver_key(img, scfg: SceneCfg):
    gain_px = scfg.ARROW_GAIN_WORLD * 0.5 * ((CANVAS_W - 1)/SPACE_SIZE + (CANVAS_H - 1)/SPACE_SIZE)
    L = int(round(scfg.WIND_STRENGTH * gain_px))
    p0 = (QK_X0, QK_Y0); p1 = (QK_X0 + L, QK_Y0)
    cv2.arrowedLine(img, p0, p1, QK_COLOR, thickness=QK_THICK, tipLength=0.25)
    cv2.putText(img, f"Ref |wind|={scfg.WIND_STRENGTH:.2f}", (p1[0]+8, p1[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

def draw_title(img, title_text: str):
    return put_text(img, title_text, (16, 28), TITLE_COLOR, scale=0.8, thickness=2)

def draw_trails(img, groups: List[List]):
    color_idx = 0
    for g in groups:
        cnt = min(TRAILS_PER_GROUP, len(g))
        for k in range(cnt):
            tr = np.stack(g[k].trail, axis=0)
            pts = [world_to_px_int(p) for p in tr]
            if len(pts) >= 2:
                cv2.polylines(img, [np.int32(pts)], False,
                              TRAIL_COLORS[color_idx % len(TRAIL_COLORS)], TRAIL_THICK)
                color_idx += 1

def draw_boids(img, groups: List[List]):
    for g in groups:
        for b in g:
            px = world_to_px_int(b.position)
            if b.is_leader:
                cv2.circle(img, px, BOID_RADIUS+2, (0,0,180), -1)
            else:
                cv2.circle(img, px, BOID_RADIUS, BOID_COLOR, -1)

def draw_trail_legend(img, groups: List[List]):
    x = 16; y0 = 56; dy = 18
    cv2.putText(img, "Trails:", (x, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    color_idx = 0; line = 0
    for g in groups:
        cnt = min(TRAILS_PER_GROUP, len(g))
        for k in range(cnt):
            color = TRAIL_COLORS[color_idx % len(TRAIL_COLORS)]
            cv2.line(img, (x, y0 + line*dy), (x+18, y0 + line*dy), color, 3)
            cv2.putText(img, f"ID={g[k].id}", (x+24, y0 + 4 + line*dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            color_idx += 1; line += 1

# ============================= 场景配方（单/混合） =============================
def make_scene_recipes(K: int):
    """
    返回 K 个“配方”列表；每个配方为 [(species_cfg, weight), ...]
    - 前 8 个：8 种模式一一对应，权重=1
    - 后续：随机混合 2~3 种模式，权重 ~ Dirichlet
    """
    recipes = []
    # 单一模式
    for i in range(min(K, len(SPECIES_LIBRARY))):
        sp = SPECIES_LIBRARY[i]()
        recipes.append([(sp, 1.0)])
    # 混合
    for _ in range(len(recipes), K):
        m = 2 if rng.uniform() < 0.8 else 3
        idxs = rng.choice(len(SPECIES_LIBRARY), size=m, replace=False)
        ws = rng.dirichlet(alpha=np.ones(m))
        combo = []
        for j, idx in enumerate(idxs):
            sp = SPECIES_LIBRARY[idx]()
            # 轻微扰动以增加多样性
            jitter = 1.0 + rng.uniform(-0.08, 0.08)
            sp.SPEED_MEAN = float(np.clip(sp.SPEED_MEAN * jitter, 0.2, sp.SPEED_MAX))
            sp.WIND_GAIN  = float(np.clip(jitter, 0.8, 1.2))
            combo.append((sp, float(ws[j])))
        recipes.append(combo)
    return recipes

def recipe_name(recipe) -> str:
    if len(recipe) == 1:
        return recipe[0][0].name
    parts = [f"{sp.name}-{int(round(w*100))}p" for sp, w in recipe]  # 百分比简名，避免非法字符
    return "mix_" + "_".join(parts)

# ============================= 单场景运行并导出 =============================
def sample_speed_norm(sp: SpeciesCfg) -> float:
    for _ in range(16):
        v = rng.normal(loc=sp.SPEED_MEAN, scale=sp.SPEED_STD)
        if sp.SPEED_MIN <= v <= sp.SPEED_MAX:
            return float(v)
    v = rng.normal(loc=sp.SPEED_MEAN, scale=sp.SPEED_STD)
    return float(np.clip(v, sp.SPEED_MIN, sp.SPEED_MAX))

def create_group(n: int, sp: SpeciesCfg, next_id: int, species_idx: int):
    n_leaders = int(round(n * (sp.LEADER_FRAC if sp.mode == "leader_follow" else 0.0)))
    leader_flags = np.zeros(n, dtype=bool)
    if n_leaders > 0:
        leader_flags[rng.choice(n, n_leaders, replace=False)] = True

    group = []
    for i in range(n):
        pos = np.array([rng.uniform(0, SPACE_SIZE), rng.uniform(0, SPACE_SIZE)], dtype=np.float64)
        direction = rng.normal(size=2); 
        if np.allclose(direction, 0): direction = np.array([1.0, 0.0])
        direction = unit(direction)
        speed0 = sample_speed_norm(sp)
        vel = direction * speed0
        acc = np.zeros(2, dtype=np.float64)
        cohesion_gain = rng.uniform(sp.COHESION_GAIN_MIN, sp.COHESION_GAIN_MAX)
        trail = deque(maxlen=sp.TRAIL_LEN); trail.append(pos.copy())
        b = Boid(id=next_id+i, position=pos, velocity=vel, acceleration=acc,
                 cohesion_gain=cohesion_gain, trail=trail,
                 is_leader=bool(leader_flags[i]), species=species_idx)
        group.append(b)
    return group

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def run_scene(scene_idx: int, recipe, scfg: SceneCfg, out_root: str, frames: int):
    # —— 场景目录名（Windows 安全） —— #
    name_raw = recipe_name(recipe)
    dir_name = safe_name(f"scene_{scene_idx+1:02d}_{name_raw}")
    scene_dir = os.path.join(out_root, dir_name)
    ensure_dir(scene_dir)
    video_path = os.path.join(scene_dir, "video.mp4")
    mot_path   = os.path.join(scene_dir, "tracks.txt")

    # —— VideoWriter —— #
    tmp = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    if DRAW_BORDER:
        cv2.rectangle(tmp, (0,0), (tmp.shape[1]-1, tmp.shape[0]-1), (0,0,0), 2)
    w_out, h_out = even_hw(tmp.shape[1], tmp.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 如需 H.264 且系统支持可改为 'avc1'
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w_out, h_out))
    assert writer.isOpened(), "VideoWriter 打开失败"

    # —— 人数分配（各子群数量按权重切分，总数随机 60~120） —— #
    N_total = int(rng.integers(60, 121))
    weights = np.array([w for _, w in recipe], dtype=np.float64)
    weights = weights / weights.sum()
    ns = np.floor(weights * N_total).astype(int)
    while ns.sum() < N_total:
        # 将剩余名额分配给当前剩余最大的权重项
        idx = int(np.argmax(weights - ns / max(1, ns.sum() or 1)))
        ns[idx] += 1

    # —— 创建子群 —— #
    groups: List[List[Boid]] = []
    next_id = 1
    for gi, ((sp, _w), n) in enumerate(zip(recipe, ns)):
        if n <= 0: continue
        groups.append(create_group(n, sp, next_id, gi))
        next_id += n

    mot_rows = []

    # —— 帧循环 —— #
    for f in range(frames):
        frame_idx = f + 1
        # 使用第一个子群的 DT（若需支持不同 DT，可细化到每个子群）
        t = f * recipe[0][0].DT

        # 动力学：各子群独立交互；共享风场
        for gi, (sp, _w) in enumerate(recipe):
            if gi >= len(groups): continue
            flock_group(groups[gi], t, sp, scfg)

        # 画布
        img = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

        # 风场（场景级）
        if SHOW_WIND: draw_wind(img, t, scfg)

        # 轨迹/个体
        draw_trails(img, groups)
        draw_boids(img, groups)

        # 叠加
        title_text = f"Scene {scene_idx+1}: {name_raw}"
        img = draw_title(img, title_text)
        draw_colorbar(img, scfg)
        draw_quiver_key(img, scfg)
        draw_trail_legend(img, groups)

        # —— MOT 行（像素坐标，原点=左上；x→右，y→下；不裁剪）—— #
        for g in groups:
            for b in g:
                px, py = world_to_px_float(b.position)  # ★ 关键：用于 txt 的像素坐标
                mot_rows.append(f"{frame_idx},{b.id},{px:.3f},{py:.3f},1,1,-1,-1,-1")

        # 边框与尺寸
        frame_out = img
        if DRAW_BORDER:
            cv2.rectangle(frame_out, (0,0), (frame_out.shape[1]-1, frame_out.shape[0]-1), (0,0,0), 2)
        if (frame_out.shape[1] != w_out) or (frame_out.shape[0] != h_out):
            frame_out = cv2.resize(frame_out, (w_out, h_out), interpolation=cv2.INTER_AREA)

        writer.write(frame_out)

    writer.release()

    # 写 MOT 文本
    with open(mot_path, "w", encoding="utf-8") as ftxt:
        ftxt.write("\n".join(mot_rows))

    print(f"[OK] Scene {scene_idx+1:02d} -> {video_path} | {mot_path}  "
          f"(groups={len(groups)}, N={sum(len(g) for g in groups)}, frames={frames})")

# ============================= 主程序 =============================
def make_scene_cfg() -> SceneCfg:
    # 为增加多样性，对风场做小扰动
    return SceneCfg(
        frames=400,
        NOISE_SCALE_XY = 0.06 * (1.0 + rng.uniform(-0.05, 0.05)),
        NOISE_TIME_SCALE = 0.03 * (1.0 + rng.uniform(-0.10, 0.10)),
        WIND_STRENGTH = float(np.clip(0.20 * (1.0 + rng.uniform(-0.20, 0.20)), 0.08, 0.8)),
        ARROW_GAIN_WORLD = 24.0
    )

def main():
    parser = argparse.ArgumentParser(description="Boids multi-scene MOT generator (mode mixtures)")
    parser.add_argument("--K", type=int, default=50, help="要生成的场景数量（默认50）")
    parser.add_argument("--outdir", type=str, default="dataset", help="输出根目录（每场景一个子目录）")
    parser.add_argument("--frames", type=int, default=400, help="每场景帧数（默认400）")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    args = parser.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)

    # 场景配方
    recipes = make_scene_recipes(args.K)

    # 输出根目录
    os.makedirs(args.outdir, exist_ok=True)

    # 逐场景生成
    for i, recipe in enumerate(recipes):
        scfg = make_scene_cfg()
        scfg.frames = args.frames  # 统一帧数（可按需改为 per-scene）
        run_scene(i, recipe, scfg, args.outdir, args.frames)

if __name__ == "__main__":
    main()

# 生成 50 个场景，输出到 dataset/scene_xx_* 目录
#python .\Biods+wind.py --K 20 --outdir biodsdataset --frames 400 --seed 2025

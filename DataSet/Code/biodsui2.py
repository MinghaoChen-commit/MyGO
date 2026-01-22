import sys
import os
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QPushButton, QFileDialog, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QLineEdit, QScrollArea, QMessageBox, QFrame
)     # noqa
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from noise import pnoise3


# --------------------- Boid 类 ---------------------
class Boid:
    def __init__(self, position, velocity, bid=None, is_leader=False, personal_max_speed=None):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.id = int(bid) if bid is not None else -1
        self.trail = deque(maxlen=200)
        self.is_leader = bool(is_leader)
        self.personal_max_speed = float(personal_max_speed) if personal_max_speed is not None else None
        # 每个 Boid 随机一个固定颜色（RGB 0~1）
        self.color = np.random.uniform(0.2, 0.9, 3).astype(np.float32)

    def update(self, boids, params, wind_func, dt, time_now):
        if getattr(self, 'is_leader', False):
            # 领航者：外部更新，只记录轨迹
            self.trail.append(self.position.copy())
            return True

        align_weight = params['align_weight']
        cohesion_weight = params['cohesion_weight']
        separation_weight = params['separation_weight']
        max_speed = params['max_speed']
        max_force = params['max_force']
        perception = params['perception']

        alignment = np.zeros(3, dtype=np.float32)
        cohesion = np.zeros(3, dtype=np.float32)
        separation = np.zeros(3, dtype=np.float32)
        total = 0

        for other in boids:
            if other is self:
                continue
            distance = np.linalg.norm(other.position - self.position)
            if 0 < distance < perception:
                alignment += other.velocity
                cohesion += other.position
                diff = self.position - other.position
                separation += diff / (distance * distance)
                total += 1

        if total > 0:
            alignment /= total
            alignment = self._set_magnitude(alignment, max_speed)
            alignment -= self.velocity
            alignment = self._limit_force(alignment, max_force)

            cohesion /= total
            cohesion = cohesion - self.position
            cohesion = self._set_magnitude(cohesion, max_speed)
            cohesion -= self.velocity
            cohesion = self._limit_force(cohesion, max_force)

            separation /= total
            separation = self._set_magnitude(separation, max_speed)
            separation -= self.velocity
            separation = self._limit_force(separation, max_force)

        # 风场 + 随机扰动
        wind = wind_func(self.position, time_now)
        acceleration = (alignment * align_weight +
                        cohesion * cohesion_weight +
                        separation * separation_weight +
                        wind)
        jitter = np.random.normal(0.0, 1.0, 3).astype(np.float32)
        acceleration += jitter * float(params.get('random_accel', 0.0))

        self.velocity += acceleration * dt
        speed = np.linalg.norm(self.velocity)

        cap_max = max_speed
        pmax = self.personal_max_speed if self.personal_max_speed is not None else cap_max
        cap_max = min(cap_max, pmax, float(params.get('max_speed_max', cap_max)))
        cap_min = float(params.get('min_speed', 0.0))
        pmin = getattr(self, 'personal_min_speed', None)
        if pmin is not None:
            cap_min = max(0.0, min(cap_min, cap_max - 1e-6))
            cap_min = max(cap_min, min(pmin, cap_max - 1e-6))

        if speed > cap_max:
            self.velocity = (self.velocity / (speed + 1e-6)) * cap_max
        elif speed < cap_min:
            if speed < 1e-6:
                dirv = np.random.normal(0.0, 1.0, 3).astype(np.float32)
                dirv /= (np.linalg.norm(dirv) + 1e-6)
            else:
                dirv = self.velocity / (speed + 1e-6)
            self.velocity = dirv * cap_min

        self.position += self.velocity * dt

        in_bounds = bool(np.all(self.position >= 0) and np.all(self.position <= 100))
        if in_bounds:
            self.trail.append(self.position.copy())
        return in_bounds

    @staticmethod
    def _set_magnitude(vector, magnitude):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return (vector / norm) * magnitude

    @staticmethod
    def _limit_force(force, max_force):
        norm = np.linalg.norm(force)
        if norm > max_force:
            return (force / norm) * max_force
        return force


# --------------------- 3D 主视图 ---------------------
class BoidsGLWidget(QGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.boids = []
        self.params = {
            'align_weight': 1.0,
            'cohesion_weight': 1.0,
            'separation_weight': 1.5,
            'max_speed': 10.0,
            'min_speed': 1.0,
            'max_force': 0.5,
            'perception': 15.0,
            'wind_strength': 1.0,
            'wind_scale': 0.1,
            'wind_speed': 1.0,
            'sim_speed': 1.0,
            'num_boids': 100,
            'trail_len': 200,
            'export_frames': 300,
            'random_accel': 0.2,
            'max_speed_max': 20.0,
            'speed_randomness': 0.5,
            'leader_weight': 1.5,
            'leader_speed': 12.0,
            'has_leader': True,
            'trail_show': 60,
            'trail_gap_thresh': 15.0,
            'leader_count': 1,
        }
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_scene)
        self.timer.start(16)

        self.sim_time = 0.0
        self.view_angle = [45, 45]
        self.distance = 150
        self.setMinimumSize(700, 700)
        self.next_id = 1
        self.leaders = []
        self.init_boids()
        self.mouse_last_pos = None

    # ---------- 初始化 boids ----------
    def init_boids(self):
        self.boids = []
        tlen = int(self.params.get('trail_len', 200))
        n = int(self.params.get('num_boids', 100))
        rng = float(self.params.get('speed_randomness', 0.5))
        base = float(self.params.get('max_speed', 10.0))
        min_base = float(self.params.get('min_speed', 1.0))
        capmax = float(self.params.get('max_speed_max', max(base, 20.0)))

        self.leaders = []
        want_leaders = int(self.params.get('leader_count', 1)) if self.params.get('has_leader', True) else 0
        L = max(0, min(n, want_leaders))
        for _ in range(L):
            pos = np.random.rand(3) * 100
            vel = (np.random.rand(3) - 0.5) * 2.0
            leader = Boid(pos, vel, bid=self.next_id, is_leader=True, personal_max_speed=capmax)
            leader.personal_min_speed = float(self.params.get('min_speed', 1.0))
            self.next_id += 1
            leader.trail = deque(maxlen=tlen)
            leader.trail.append(leader.position.copy())
            self.leaders.append(leader)
            self.boids.append(leader)

        count_rest = max(0, n - L)
        lo_max = max(min_base + 1e-3, base * max(0.0, 1.0 - rng))
        hi_max = capmax
        for _ in range(count_rest):
            pos = np.random.rand(3) * 100
            vel = (np.random.rand(3) - 0.5) * 10
            factor = np.random.uniform(1.0 - rng, 1.0 + rng)
            personal_max = np.clip(base * factor, lo_max, hi_max)
            lo_min = max(0.0, min_base * max(0.0, 1.0 - rng))
            hi_min = min(personal_max - 1e-3, min_base * (1.0 + rng))
            if hi_min < lo_min:
                hi_min = max(lo_min, personal_max * 0.2)
            personal_min = np.random.uniform(lo_min, hi_min) if hi_min > lo_min else lo_min
            b = Boid(pos, vel, bid=self.next_id, is_leader=False, personal_max_speed=personal_max)
            b.personal_min_speed = float(personal_min)
            self.next_id += 1
            b.trail = deque(maxlen=tlen)
            b.trail.append(b.position.copy())
            self.boids.append(b)

    def set_leader_count(self, count):
        self.params['leader_count'] = int(max(0, count))
        self.init_boids()

    def set_num_boids(self, n):
        self.params['num_boids'] = int(n)
        self.init_boids()

    # ---------- OpenGL 基础 ----------
    def initializeGL(self):
        # 如果上一帧有 GL 错误，清一次
        while glGetError() != GL_NO_ERROR:
            pass

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h, 1.0, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        azimuth = np.radians(self.view_angle[0])
        elevation = np.radians(self.view_angle[1])
        eye_x = self.distance * np.cos(elevation) * np.sin(azimuth)
        eye_y = self.distance * np.sin(elevation)
        eye_z = self.distance * np.cos(elevation) * np.cos(azimuth)
        gluLookAt(eye_x, eye_y, eye_z, 50, 50, 50, 0, 1, 0)

        self.draw_bounding_box()
        self.draw_grid_3d()          # 三维网格
        self.draw_cfd_trails_3d()    # 拖尾

        for boid in self.boids:
            self.draw_boid(boid)

    # ---------- 3D 网格 ----------
    def draw_grid_3d(self):
        glDisable(GL_LIGHTING)
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.0)
        step = 10

        glBegin(GL_LINES)
        # XY 平面 (z = 0)
        for i in range(0, 101, step):
            glVertex3f(0, i, 0)
            glVertex3f(100, i, 0)
            glVertex3f(i, 0, 0)
            glVertex3f(i, 100, 0)
        # XZ 平面 (y = 0)
        for i in range(0, 101, step):
            glVertex3f(0, 0, i)
            glVertex3f(100, 0, i)
            glVertex3f(i, 0, 0)
            glVertex3f(i, 0, 100)
        # YZ 平面 (x = 0)
        for i in range(0, 101, step):
            glVertex3f(0, 0, i)
            glVertex3f(0, 100, i)
            glVertex3f(0, i, 0)
            glVertex3f(0, i, 100)
        glEnd()

        glEnable(GL_LIGHTING)

    # ---------- 3D 拖尾：粗细渐变 + 深色 + 与 Boid 同色 ----------
    def draw_cfd_trails_3d(self):
        show_n = int(self.params.get('trail_show', 60))
        gap = float(self.params.get('trail_gap_thresh', 15.0))

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        min_w = 1.0   # 尾部线宽
        max_w = 5.0   # 头部线宽

        for boid in self.boids:
            if len(boid.trail) < 2:
                continue
            pts = list(boid.trail)
            if show_n > 0:
                pts = pts[-show_n:]

            n_seg = len(pts) - 1
            if n_seg <= 0:
                continue

            prev = pts[0]
            for idx in range(1, len(pts)):
                p = pts[idx]
                dist = float(np.linalg.norm(p - prev))
                if dist > gap:
                    prev = p
                    continue

                age = idx / max(1, len(pts) - 1)
                alpha = 0.5 + 0.5 * age   # 越接近目标越深
                width = min_w + (max_w - min_w) * age
                glLineWidth(width)

                r, g, b = boid.color
                glBegin(GL_LINES)
                glColor4f(r, g, b, alpha)
                glVertex3f(prev[0], prev[1], prev[2])
                glVertex3f(p[0], p[1], p[2])
                glEnd()

                prev = p

        glEnable(GL_LIGHTING)

    # ---------- 其它绘制 ----------
    @staticmethod
    def draw_bounding_box():
        glColor3f(0.7, 0.7, 0.7)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for x in [0, 100]:
            for y in [0, 100]:
                glVertex3f(x, y, 0)
                glVertex3f(x, y, 100)
        for x in [0, 100]:
            for z in [0, 100]:
                glVertex3f(x, 0, z)
                glVertex3f(x, 100, z)
        for y in [0, 100]:
            for z in [0, 100]:
                glVertex3f(0, y, z)
                glVertex3f(100, y, z)
        glEnd()

    @staticmethod
    def draw_boid(boid):
        glPushMatrix()
        glTranslatef(*boid.position)
        # 使用 Boid 自己的颜色
        r, g, b = boid.color
        glColor3f(r, g, b)
        # 小球尺寸减小
        if getattr(boid, 'is_leader', False):
            radius = 0.6
        else:
            radius = 0.4
        quad = gluNewQuadric()
        gluSphere(quad, radius, 12, 12)
        gluDeleteQuadric(quad)
        glPopMatrix()

    # ---------- 仿真步进 ----------
    def update_scene(self):
        dt = 0.016 * float(self.params.get('sim_speed', 1.0))
        self.step_sim(dt)
        self.update()

    def update_leader(self, dt):
        if not self.leaders:
            return
        Lspd = float(self.params.get('leader_speed', 12.0))
        for idx, leader in enumerate(self.leaders):
            t = self.sim_time + idx * 0.7
            dirv = np.array([
                np.sin(0.7 * t) + 0.3 * np.sin(1.7 * t + 1.0),
                np.sin(0.5 * t + 0.8),
                np.cos(0.6 * t + 0.3)
            ], dtype=np.float32)
            dirv /= (np.linalg.norm(dirv) + 1e-6)

            wind = self.perlin_wind(leader.position, t)
            jitter = np.random.normal(0.0, 1.0, 3).astype(np.float32) * float(self.params.get('random_accel', 0.0))
            vel = dirv * Lspd + wind + jitter

            cap = min(leader.personal_max_speed or Lspd, float(self.params.get('max_speed_max', 1e9)))
            sp = np.linalg.norm(vel)
            if sp > cap:
                vel = vel / (sp + 1e-6) * cap

            pmin = getattr(leader, 'personal_min_speed', float(self.params.get('min_speed', 1.0)))
            sp = np.linalg.norm(vel)
            if sp < pmin:
                if sp < 1e-6:
                    dirr = np.random.normal(0.0, 1.0, 3).astype(np.float32)
                    dirr /= (np.linalg.norm(dirr) + 1e-6)
                else:
                    dirr = vel / (sp + 1e-6)
                vel = dirr * pmin

            leader.velocity = vel
            leader.position += leader.velocity * dt
            leader.position = np.mod(leader.position, 100.0)
            leader.trail.append(leader.position.copy())

    def update_follower(self, boid, dt):
        if self.leaders:
            dmin = 1e9
            target = None
            for L in self.leaders:
                d = np.linalg.norm(L.position - boid.position)
                if d < dmin:
                    dmin = d
                    target = L
            params = self.params
            to_leader = target.position - boid.position
            dist = np.linalg.norm(to_leader)
            if dist > 1e-6:
                desired = to_leader / dist * params['max_speed']
                steer = desired - boid.velocity
                mf = params['max_force']
                s = np.linalg.norm(steer)
                if s > mf:
                    steer = steer / (s + 1e-6) * mf
            else:
                steer = np.zeros(3, dtype=np.float32)

            def wind_with_leader(pos, t):
                return self.perlin_wind(pos, t) + steer * float(self.params.get('leader_weight', 1.0))

            return boid.update(self.boids, dict(self.params), wind_with_leader, dt, self.sim_time)
        else:
            return boid.update(self.boids, self.params, self.perlin_wind, dt, self.sim_time)

    def step_sim(self, dt):
        self.sim_time += dt
        tlen = int(self.params.get('trail_len', 200))
        self.update_leader(dt)

        out_idx = []
        for i, boid in enumerate(self.boids):
            if getattr(boid, 'is_leader', False):
                continue
            in_bounds = self.update_follower(boid, dt)
            if not in_bounds:
                out_idx.append(i)

        for i in out_idx:
            base = float(self.params.get('max_speed', 10.0))
            min_base = float(self.params.get('min_speed', 1.0))
            capmax = float(self.params.get('max_speed_max', 20.0))
            rng = float(self.params.get('speed_randomness', 0.5))
            factor = np.random.uniform(1.0 - rng, 1.0 + rng)
            lo_max = max(min_base + 1e-3, base * max(0.0, 1.0 - rng))
            hi_max = capmax
            personal_max = float(np.clip(base * factor, lo_max, hi_max))
            lo_min = max(0.0, min_base * max(0.0, 1.0 - rng))
            hi_min = min(personal_max - 1e-3, min_base * (1.0 + rng))
            if hi_min < lo_min:
                hi_min = max(lo_min, personal_max * 0.2)
            personal_min = float(np.random.uniform(lo_min, hi_min)) if hi_min > lo_min else float(lo_min)
            pos = np.random.rand(3) * 100
            vel = (np.random.rand(3) - 0.5) * 10
            nb = Boid(pos, vel, bid=self.next_id, is_leader=False, personal_max_speed=personal_max)
            nb.personal_min_speed = personal_min
            self.next_id += 1
            nb.trail = deque(maxlen=tlen)
            nb.trail.append(nb.position.copy())
            self.boids[i] = nb

    # ---------- 状态快照 ----------
    def get_state(self):
        snap = []
        for b in self.boids:
            snap.append((
                b.id,
                b.position.copy(),
                b.velocity.copy(),
                deque(b.trail, maxlen=b.trail.maxlen),
                bool(getattr(b, 'is_leader', False)),
                float(getattr(b, 'personal_max_speed', self.params.get('max_speed', 10.0))
                      if hasattr(b, 'personal_max_speed') else self.params.get('max_speed', 10.0)),
                float(getattr(b, 'personal_min_speed', self.params.get('min_speed', 1.0))),
                b.color.copy(),
            ))
        return snap, int(self.next_id), float(self.sim_time)

    def set_state(self, snapshot):
        snap, next_id, sim_time = snapshot
        self.boids = []
        self.leaders = []
        for bid, pos, vel, tr, is_leader, pmax, pmin, color in snap:
            b = Boid(pos.copy(), vel.copy(), bid=bid, is_leader=is_leader, personal_max_speed=pmax)
            b.personal_min_speed = float(pmin)
            b.trail = deque(tr, maxlen=tr.maxlen)
            b.color = color.copy()
            self.boids.append(b)
            if is_leader:
                self.leaders.append(b)
        self.next_id = int(next_id)
        self.sim_time = float(sim_time)

    # ---------- Perlin 风场 ----------
    def perlin_wind(self, position, t):
        scale = self.params['wind_scale']
        strength = self.params['wind_strength']
        speed = self.params['wind_speed']
        x, y, z = position * scale
        t_scaled = t * speed
        wind_x = pnoise3(x + t_scaled, y, z)
        wind_y = pnoise3(x, y + t_scaled, z)
        wind_z = pnoise3(x, y, z + t_scaled)
        wind = np.array([wind_x, wind_y, wind_z], dtype=np.float32)
        return wind * strength

    # ---------- 交互 ----------
    def mousePressEvent(self, event):
        self.mouse_last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if not self.mouse_last_pos:
            return
        dx = event.x() - self.mouse_last_pos.x()
        dy = event.y() - self.mouse_last_pos.y()
        self.view_angle[0] += dx * 0.5
        self.view_angle[1] += dy * 0.5
        self.view_angle[1] = np.clip(self.view_angle[1], -89, 89)
        self.mouse_last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120
        self.distance -= delta * 5
        self.distance = np.clip(self.distance, 20, 300)
        self.update()


# --------------------- 2D 投影视图 ---------------------
class ProjectionView(QGLWidget):
    def __init__(self, parent=None, view='XY', source=None):
        super().__init__(parent)
        self.boids = []
        self.view = view
        self.source = source
        self.setMinimumSize(350, 350)

        self.plane_normal = None
        self.plane_up = None
        self.plane_name = ""
        self.plane_x = None
        self.plane_y = None

    def set_source(self, src):
        self.source = src
        self.update()

    def set_boids(self, boids):
        self.boids = boids
        self.update()

    # ---------- OpenGL ----------
    def initializeGL(self):
        while glGetError() != GL_NO_ERROR:
            pass
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)

    def resizeGL(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 100, 0, 100, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    # ---------- 自定义平面 ----------
    def set_custom_plane(self, normal, up_vector, name="Custom"):
        self.view = 'CUSTOM'
        self.plane_normal = np.array(normal, dtype=np.float32)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)

        up_vector = np.array(up_vector, dtype=np.float32)
        up_vector = up_vector - np.dot(up_vector, self.plane_normal) * self.plane_normal
        up_vector = up_vector / np.linalg.norm(up_vector)
        self.plane_up = np.array(up_vector, dtype=np.float32)
        self.plane_name = name

        self.plane_x = np.cross(self.plane_up, self.plane_normal)
        self.plane_x = self.plane_x / (np.linalg.norm(self.plane_x) + 1e-6)
        self.plane_y = np.cross(self.plane_normal, self.plane_x)
        self.plane_y = self.plane_y / (np.linalg.norm(self.plane_y) + 1e-6)

        self.update()

    def project_to_custom_plane(self, point_3d):
        if self.plane_normal is None:
            return 0, 0
        point_vec = np.array(point_3d, dtype=np.float32)
        x = np.dot(point_vec, self.plane_x)
        y = np.dot(point_vec, self.plane_y)
        x = (x + 100) % 100
        y = (y + 100) % 100
        return x, y

    # ---------- 2D 坐标系网格 ----------
    def draw_axes_2d(self):
        # 网格线
        glLineWidth(1.0)
        glColor3f(0.7, 0.7, 0.7)
        step = 10
        glBegin(GL_LINES)
        for v in range(0, 101, step):
            # 水平线
            glVertex2f(0, v)
            glVertex2f(100, v)
            # 垂直线
            glVertex2f(v, 0)
            glVertex2f(v, 100)
        glEnd()
        # 边框
        glLineWidth(2.0)
        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(0, 0)
        glVertex2f(100, 0)
        glVertex2f(100, 100)
        glVertex2f(0, 100)
        glEnd()

    # ---------- 2D 拖尾（与小球同色，粗细渐变） ----------
    def draw_cfd_trails_2d(self, current_boids):
        if not current_boids:
            return

        show_n = int(getattr(self.source, 'params', {}).get('trail_show', 60)) if self.source is not None else 60
        gap3d = float(getattr(self.source, 'params', {}).get('trail_gap_thresh', 15.0)) if self.source is not None else 15.0

        min_w = 1.0
        max_w = 4.0

        for boid in current_boids:
            if len(getattr(boid, 'trail', [])) < 2:
                continue
            pts = list(boid.trail)
            if show_n > 0:
                pts = pts[-show_n:]

            n_seg = len(pts) - 1
            if n_seg <= 0:
                continue

            prev3d = pts[0]
            prev2d = self._project(prev3d)
            for idx in range(1, len(pts)):
                p3d = pts[idx]
                p2d = self._project(p3d)
                dist3d = float(np.linalg.norm(p3d - prev3d))
                if dist3d > gap3d:
                    prev3d = p3d
                    prev2d = p2d
                    continue

                age = idx / max(1, len(pts) - 1)
                alpha = 0.5 + 0.5 * age
                width = min_w + (max_w - min_w) * age
                glLineWidth(width)

                r, g, b = getattr(boid, 'color', (0.2, 0.2, 0.2))
                glBegin(GL_LINES)
                glColor4f(r, g, b, alpha)
                glVertex2f(prev2d[0], prev2d[1])
                glVertex2f(p2d[0], p2d[1])
                glEnd()

                prev3d = p3d
                prev2d = p2d

    def _project(self, p):
        if self.view == 'XY':
            return np.array([p[0], p[1]], dtype=np.float32)
        elif self.view == 'XZ':
            return np.array([p[0], p[2]], dtype=np.float32)
        elif self.view == 'YZ':
            return np.array([p[1], p[2]], dtype=np.float32)
        elif self.view == 'CUSTOM' and self.plane_normal is not None:
            x, y = self.project_to_custom_plane(p)
            return np.array([x, y], dtype=np.float32)
        else:
            return np.array([0.0, 0.0], dtype=np.float32)

    # ---------- 绘制 ----------
    def paintGL(self):
        if self.source is not None and hasattr(self.source, 'boids'):
            current_boids = self.source.boids
        else:
            current_boids = self.boids

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 先画坐标网格
        self.draw_axes_2d()
        # 再画拖尾
        self.draw_cfd_trails_2d(current_boids)

        # 最后画目标点
        for boid in current_boids:
            p = boid.position
            if self.view == 'XY':
                x, y = p[0], p[1]
            elif self.view == 'XZ':
                x, y = p[0], p[2]
            elif self.view == 'YZ':
                x, y = p[1], p[2]
            elif self.view == 'CUSTOM' and self.plane_normal is not None:
                x, y = self.project_to_custom_plane(p)
            else:
                x, y = 0, 0

            r, g, b = getattr(boid, 'color', (0.0, 0.0, 0.0))
            glColor3f(r, g, b)
            # 2D 点也稍微小一点
            if getattr(boid, 'is_leader', False):
                glPointSize(5)
            else:
                glPointSize(3)
            glBegin(GL_POINTS)
            glVertex2f(x, y)
            glEnd()

    # 保留接口，不再使用
    def draw_wind_field_2d(self, boids):
        return


# --------------------- 控制面板 ---------------------
class ControlPanel(QWidget):
    def __init__(self, boids_gl_widget):
        super().__init__()
        self.boids_gl_widget = boids_gl_widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.num_boids_spin = QSpinBox()
        self.num_boids_spin.setRange(1, 10000)
        self.num_boids_spin.setValue(self.boids_gl_widget.params.get('num_boids', 100))
        self.num_boids_spin.valueChanged.connect(lambda v: self.boids_gl_widget.set_num_boids(v))

        self.align_spin = QDoubleSpinBox()
        self.align_spin.setRange(0, 5)
        self.align_spin.setSingleStep(0.1)
        self.align_spin.setValue(self.boids_gl_widget.params['align_weight'])
        self.align_spin.valueChanged.connect(self.update_params)

        self.cohesion_spin = QDoubleSpinBox()
        self.cohesion_spin.setRange(0, 5)
        self.cohesion_spin.setSingleStep(0.1)
        self.cohesion_spin.setValue(self.boids_gl_widget.params['cohesion_weight'])
        self.cohesion_spin.valueChanged.connect(self.update_params)

        self.separation_spin = QDoubleSpinBox()
        self.separation_spin.setRange(0, 5)
        self.separation_spin.setSingleStep(0.1)
        self.separation_spin.setValue(self.boids_gl_widget.params['separation_weight'])
        self.separation_spin.valueChanged.connect(self.update_params)

        self.max_speed_spin = QDoubleSpinBox()
        self.max_speed_spin.setRange(1, 50)
        self.max_speed_spin.setSingleStep(1)
        self.max_speed_spin.setValue(self.boids_gl_widget.params['max_speed'])
        self.max_speed_spin.valueChanged.connect(self.update_params)

        self.min_speed_spin = QDoubleSpinBox()
        self.min_speed_spin.setRange(0.0, 50)
        self.min_speed_spin.setSingleStep(0.5)
        self.min_speed_spin.setValue(self.boids_gl_widget.params.get('min_speed', 1.0))
        self.min_speed_spin.valueChanged.connect(self.update_params)

        self.max_speed_max_spin = QDoubleSpinBox()
        self.max_speed_max_spin.setRange(1, 100)
        self.max_speed_max_spin.setSingleStep(1)
        self.max_speed_max_spin.setValue(self.boids_gl_widget.params.get('max_speed_max', 20.0))
        self.max_speed_max_spin.valueChanged.connect(self.update_params)

        self.speed_rand_spin = QDoubleSpinBox()
        self.speed_rand_spin.setRange(0.0, 1.0)
        self.speed_rand_spin.setSingleStep(0.05)
        self.speed_rand_spin.setValue(self.boids_gl_widget.params.get('speed_randomness', 0.5))
        self.speed_rand_spin.valueChanged.connect(self.update_params)

        self.leader_weight_spin = QDoubleSpinBox()
        self.leader_weight_spin.setRange(0.0, 5.0)
        self.leader_weight_spin.setSingleStep(0.1)
        self.leader_weight_spin.setValue(self.boids_gl_widget.params.get('leader_weight', 1.5))
        self.leader_weight_spin.valueChanged.connect(self.update_params)

        self.leader_speed_spin = QDoubleSpinBox()
        self.leader_speed_spin.setRange(0.1, 50.0)
        self.leader_speed_spin.setSingleStep(0.5)
        self.leader_speed_spin.setValue(self.boids_gl_widget.params.get('leader_speed', 12.0))
        self.leader_speed_spin.valueChanged.connect(self.update_params)

        self.max_force_spin = QDoubleSpinBox()
        self.max_force_spin.setRange(0.1, 5)
        self.max_force_spin.setSingleStep(0.1)
        self.max_force_spin.setValue(self.boids_gl_widget.params['max_force'])
        self.max_force_spin.valueChanged.connect(self.update_params)

        self.perception_spin = QDoubleSpinBox()
        self.perception_spin.setRange(1, 50)
        self.perception_spin.setSingleStep(1)
        self.perception_spin.setValue(self.boids_gl_widget.params['perception'])
        self.perception_spin.valueChanged.connect(self.update_params)

        self.wind_strength_spin = QDoubleSpinBox()
        self.wind_strength_spin.setRange(0, 10)
        self.wind_strength_spin.setSingleStep(0.1)
        self.wind_strength_spin.setValue(self.boids_gl_widget.params['wind_strength'])
        self.wind_strength_spin.valueChanged.connect(self.update_params)

        self.wind_scale_spin = QDoubleSpinBox()
        self.wind_scale_spin.setRange(0.01, 1)
        self.wind_scale_spin.setSingleStep(0.01)
        self.wind_scale_spin.setValue(self.boids_gl_widget.params['wind_scale'])
        self.wind_scale_spin.valueChanged.connect(self.update_params)

        self.wind_speed_spin = QDoubleSpinBox()
        self.wind_speed_spin.setRange(0, 10)
        self.wind_speed_spin.setSingleStep(0.1)
        self.wind_speed_spin.setValue(self.boids_gl_widget.params['wind_speed'])
        self.wind_speed_spin.valueChanged.connect(self.update_params)

        self.sim_speed_spin = QDoubleSpinBox()
        self.sim_speed_spin.setRange(0.1, 10.0)
        self.sim_speed_spin.setSingleStep(0.1)
        self.sim_speed_spin.setValue(self.boids_gl_widget.params['sim_speed'])
        self.sim_speed_spin.valueChanged.connect(self.update_params)

        self.export_frames_spin = QSpinBox()
        self.export_frames_spin.setRange(1, 20000)
        self.export_frames_spin.setValue(self.boids_gl_widget.params.get('export_frames', 300))
        self.export_frames_spin.valueChanged.connect(self.update_params)

        self.jitter_spin = QDoubleSpinBox()
        self.jitter_spin.setRange(0.0, 5.0)
        self.jitter_spin.setSingleStep(0.05)
        self.jitter_spin.setValue(self.boids_gl_widget.params.get('random_accel', 0.2))
        self.jitter_spin.valueChanged.connect(self.update_params)

        self.trail_show_spin = QSpinBox()
        self.trail_show_spin.setRange(0, 10000)
        self.trail_show_spin.setValue(self.boids_gl_widget.params.get('trail_show', 60))
        self.trail_show_spin.valueChanged.connect(self.update_params)

        self.leader_count_spin = QSpinBox()
        self.leader_count_spin.setRange(0, 1000)
        self.leader_count_spin.setValue(self.boids_gl_widget.params.get('leader_count', 1))
        self.leader_count_spin.valueChanged.connect(lambda v: self.boids_gl_widget.set_leader_count(v))

        form = QFormLayout()
        form.addRow("目标数量", self.num_boids_spin)
        form.addRow("对齐权重", self.align_spin)
        form.addRow("聚合权重", self.cohesion_spin)
        form.addRow("分离权重", self.separation_spin)
        form.addRow("最大速度", self.max_speed_spin)
        form.addRow("最小速度", self.min_speed_spin)
        form.addRow("最大速度上限", self.max_speed_max_spin)
        form.addRow("速度随机度", self.speed_rand_spin)
        form.addRow("领航权重", self.leader_weight_spin)
        form.addRow("领航速度", self.leader_speed_spin)
        form.addRow("领航者数量", self.leader_count_spin)
        form.addRow("最大转向力", self.max_force_spin)
        form.addRow("感知半径", self.perception_spin)
        form.addRow("风强度", self.wind_strength_spin)
        form.addRow("风尺度", self.wind_scale_spin)
        form.addRow("风速度", self.wind_speed_spin)
        form.addRow("仿真倍速", self.sim_speed_spin)
        form.addRow("导出帧数", self.export_frames_spin)
        form.addRow("随机扰动", self.jitter_spin)
        form.addRow("轨迹显示长度", self.trail_show_spin)

        group_box = QGroupBox("仿真参数")
        group_box.setLayout(form)
        layout.addWidget(group_box)

        self.export_button = QPushButton("导出目标状态")
        self.export_button.clicked.connect(self.export_boids)
        layout.addWidget(self.export_button)

        self.save_views_button = QPushButton("保存当前画面")
        self.save_views_button.clicked.connect(self.save_views)
        layout.addWidget(self.save_views_button)

        layout.addStretch()
        self.setLayout(layout)

    def update_params(self):
        p = self.boids_gl_widget.params
        p['align_weight'] = self.align_spin.value()
        p['cohesion_weight'] = self.cohesion_spin.value()
        p['separation_weight'] = self.separation_spin.value()
        p['max_speed'] = self.max_speed_spin.value()
        p['min_speed'] = self.min_speed_spin.value()
        p['max_force'] = self.max_force_spin.value()
        p['perception'] = self.perception_spin.value()
        p['wind_strength'] = self.wind_strength_spin.value()
        p['wind_scale'] = self.wind_scale_spin.value()
        p['wind_speed'] = self.wind_speed_spin.value()
        p['sim_speed'] = self.sim_speed_spin.value()
        p['export_frames'] = self.export_frames_spin.value()
        p['random_accel'] = self.jitter_spin.value()
        p['max_speed_max'] = self.max_speed_max_spin.value()
        p['speed_randomness'] = self.speed_rand_spin.value()
        p['leader_weight'] = self.leader_weight_spin.value()
        p['leader_speed'] = self.leader_speed_spin.value()
        p['trail_show'] = self.trail_show_spin.value()

    def save_views(self):
        main_window = None
        widget = self.boids_gl_widget
        while widget is not None:
            parent = widget.parent()
            if parent is None:
                if hasattr(widget, 'save_all_views'):
                    main_window = widget
                break
            elif hasattr(parent, 'save_all_views'):
                main_window = parent
                break
            widget = parent

        if main_window is None:
            QMessageBox.warning(self, "保存失败", "未找到主窗口，无法保存画面。")
            return

        main_window.save_all_views()

    def export_boids(self):
        main_window = None
        widget = self.boids_gl_widget
        while widget is not None:
            parent = widget.parent()
            if parent is None:
                if hasattr(widget, 'custom_views'):
                    main_window = widget
                    break
            elif hasattr(parent, 'custom_views'):
                main_window = parent
                break
            widget = parent

        outdir = QFileDialog.getExistingDirectory(self, "选择导出目录", os.getcwd())
        if not outdir:
            return

        frames = int(self.export_frames_spin.value())
        snapshot = self.boids_gl_widget.get_state()
        self.boids_gl_widget.timer.stop()

        f_xy = open(os.path.join(outdir, 'tracks_xy.txt'), 'w', encoding='utf-8')
        f_xz = open(os.path.join(outdir, 'tracks_xz.txt'), 'w', encoding='utf-8')
        f_yz = open(os.path.join(outdir, 'tracks_yz.txt'), 'w', encoding='utf-8')
        f_xyz = open(os.path.join(outdir, 'tracks_xyz.txt'), 'w', encoding='utf-8')

        custom_files = {}
        if main_window is not None:
            for i, (container, view) in enumerate(main_window.custom_views):
                if hasattr(view, 'plane_name') and view.plane_name:
                    filename = f"tracks_custom_{view.plane_name.replace(' ', '_')}.txt"
                    filepath = os.path.join(outdir, filename)
                    custom_files[i] = open(filepath, 'w', encoding='utf-8')

        try:
            dt = 0.016 * float(self.boids_gl_widget.params.get('sim_speed', 1.0))
            for frame in range(1, frames + 1):
                self.boids_gl_widget.step_sim(dt)
                for b in self.boids_gl_widget.boids:
                    f_xyz.write(
                        f"{frame},{b.id},"
                        f"{b.position[0]:.3f},{b.position[1]:.3f},{b.position[2]:.3f},"
                        "1,1,-1,-1,-1\n"
                    )
                    f_xy.write(f"{frame},{b.id},{b.position[0]:.3f},{b.position[1]:.3f},1,1,-1,-1,-1\n")
                    f_xz.write(f"{frame},{b.id},{b.position[0]:.3f},{b.position[2]:.3f},1,1,-1,-1,-1\n")
                    f_yz.write(f"{frame},{b.id},{b.position[1]:.3f},{b.position[2]:.3f},1,1,-1,-1,-1\n")

                for i, (container, view) in enumerate(getattr(main_window, 'custom_views', [])):
                    if i in custom_files:
                        for b in self.boids_gl_widget.boids:
                            x, y = view.project_to_custom_plane(b.position)
                            custom_files[i].write(f"{frame},{b.id},{x:.3f},{y:.3f},1,1,-1,-1,-1\n")
        finally:
            f_xy.close()
            f_xz.close()
            f_yz.close()
            f_xyz.close()
            for f in custom_files.values():
                f.close()
            self.boids_gl_widget.set_state(snapshot)
            self.boids_gl_widget.timer.start(16)


# --------------------- 自定义平面对话框 ---------------------
class CustomPlaneDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        form = QFormLayout()

        self.name_edit = QLineEdit("Custom Plane")
        form.addRow("平面名称", self.name_edit)

        self.normal_x = QDoubleSpinBox()
        self.normal_x.setRange(-1.0, 1.0)
        self.normal_x.setSingleStep(0.1)
        self.normal_x.setValue(1.0)

        self.normal_y = QDoubleSpinBox()
        self.normal_y.setRange(-1.0, 1.0)
        self.normal_y.setSingleStep(0.1)
        self.normal_y.setValue(0.0)

        self.normal_z = QDoubleSpinBox()
        self.normal_z.setRange(-1.0, 1.0)
        self.normal_z.setSingleStep(0.1)
        self.normal_z.setValue(0.0)

        normal_layout = QHBoxLayout()
        normal_layout.addWidget(QLabel("X:"))
        normal_layout.addWidget(self.normal_x)
        normal_layout.addWidget(QLabel("Y:"))
        normal_layout.addWidget(self.normal_y)
        normal_layout.addWidget(QLabel("Z:"))
        normal_layout.addWidget(self.normal_z)
        form.addRow("法向量", normal_layout)

        self.up_x = QDoubleSpinBox()
        self.up_x.setRange(-1.0, 1.0)
        self.up_x.setSingleStep(0.1)
        self.up_x.setValue(0.0)

        self.up_y = QDoubleSpinBox()
        self.up_y.setRange(-1.0, 1.0)
        self.up_y.setSingleStep(0.1)
        self.up_y.setValue(1.0)

        self.up_z = QDoubleSpinBox()
        self.up_z.setRange(-1.0, 1.0)
        self.up_z.setSingleStep(0.1)
        self.up_z.setValue(0.0)

        up_layout = QHBoxLayout()
        up_layout.addWidget(QLabel("X:"))
        up_layout.addWidget(self.up_x)
        up_layout.addWidget(QLabel("Y:"))
        up_layout.addWidget(self.up_y)
        up_layout.addWidget(QLabel("Z:"))
        up_layout.addWidget(self.up_z)
        form.addRow("上方向", up_layout)

        preset_layout = QHBoxLayout()
        xy_btn = QPushButton("XY平面")
        xz_btn = QPushButton("XZ平面")
        yz_btn = QPushButton("YZ平面")
        iso_btn = QPushButton("等轴视图")

        xy_btn.clicked.connect(lambda: self.set_preset(0, 0, 1, 0, 1, 0))
        xz_btn.clicked.connect(lambda: self.set_preset(0, 1, 0, 0, 0, 1))
        yz_btn.clicked.connect(lambda: self.set_preset(1, 0, 0, 0, 1, 0))
        iso_btn.clicked.connect(lambda: self.set_preset(1, 1, 1, 0, 1, 0))

        preset_layout.addWidget(xy_btn)
        preset_layout.addWidget(xz_btn)
        preset_layout.addWidget(yz_btn)
        preset_layout.addWidget(iso_btn)
        form.addRow("预设", preset_layout)

        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("添加平面")
        cancel_btn = QPushButton("取消")
        add_btn.clicked.connect(self.add_plane)
        cancel_btn.clicked.connect(self.close)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setWindowTitle("添加自定义投影平面")
        self.setFixedSize(600, 300)

    def showEvent(self, event):
        super().showEvent(event)
        if self.parent:
            parent_geometry = self.parent.geometry()
            self.move(parent_geometry.center() - self.rect().center())

    def set_preset(self, nx, ny, nz, ux, uy, uz):
        self.normal_x.setValue(nx)
        self.normal_y.setValue(ny)
        self.normal_z.setValue(nz)
        self.up_x.setValue(ux)
        self.up_y.setValue(uy)
        self.up_z.setValue(uz)

    def add_plane(self):
        name = self.name_edit.text()
        normal = [self.normal_x.value(), self.normal_y.value(), self.normal_z.value()]
        up = [self.up_x.value(), self.up_y.value(), self.up_z.value()]

        normal_norm = np.array(normal) / (np.linalg.norm(normal) + 1e-6)
        up_norm = np.array(up) / (np.linalg.norm(up) + 1e-6)

        if self.parent:
            self.parent.add_custom_projection(name, normal_norm, up_norm)

        self.close()


# --------------------- 主窗口 ---------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维 Boids + Perlin 风场 仿真 (CFD 风格轨迹)")
        self.resize(1500, 900)

        self.custom_views = []

        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.boids_gl_widget = BoidsGLWidget()
        self.control_panel = ControlPanel(self.boids_gl_widget)
        main_layout.addWidget(self.control_panel, 1)
        main_layout.addWidget(self.boids_gl_widget, 4)

        self.right_scroll = QScrollArea()
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_widget.setLayout(self.right_layout)
        self.right_scroll.setWidget(self.right_widget)
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setMinimumWidth(350)

        self.view_xy = ProjectionView(view='XY', source=self.boids_gl_widget)
        self.view_xz = ProjectionView(view='XZ', source=self.boids_gl_widget)
        self.view_yz = ProjectionView(view='YZ', source=self.boids_gl_widget)

        self.right_layout.addWidget(QLabel("XY 投影"))
        self.right_layout.addWidget(self.view_xy)
        self.right_layout.addWidget(QLabel("XZ 投影"))
        self.right_layout.addWidget(self.view_xz)
        self.right_layout.addWidget(QLabel("YZ 投影"))
        self.right_layout.addWidget(self.view_yz)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.right_layout.addWidget(line)

        self.add_plane_btn = QPushButton("+ 添加自定义投影平面")
        self.add_plane_btn.clicked.connect(self.show_custom_plane_dialog)
        self.right_layout.addWidget(self.add_plane_btn)

        self.right_layout.addStretch()
        main_layout.addWidget(self.right_scroll, 1)

        self.proj_timer = QTimer()
        self.proj_timer.timeout.connect(self.update_projections)
        self.proj_timer.start(30)

    def show_custom_plane_dialog(self):
        self.plane_dialog = CustomPlaneDialog(self)
        self.plane_dialog.setWindowModality(Qt.ApplicationModal)
        self.plane_dialog.setWindowFlags(
            Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
        )
        self.plane_dialog.show()
        self.plane_dialog.raise_()
        self.plane_dialog.activateWindow()

    def add_custom_projection(self, name, normal, up):
        custom_view = ProjectionView(source=self.boids_gl_widget)
        custom_view.set_custom_plane(normal, up, name)

        view_container = QWidget()
        view_container.setMinimumSize(320, 320)
        view_layout = QVBoxLayout()
        view_container.setLayout(view_layout)

        title_layout = QHBoxLayout()
        title_label = QLabel(f"自定义: {name}")
        title_label.setStyleSheet("font-weight: bold; color: blue;")
        close_btn = QPushButton("×")
        close_btn.setFixedSize(25, 25)
        close_btn.setStyleSheet("QPushButton { background-color: red; color: white; font-weight: bold; }")
        close_btn.clicked.connect(lambda: self.remove_custom_view(view_container))

        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(close_btn)

        view_layout.addLayout(title_layout)
        view_layout.addWidget(custom_view)

        index = self.right_layout.indexOf(self.add_plane_btn)
        self.right_layout.insertWidget(index, view_container)

        self.custom_views.append((view_container, custom_view))

        view_container.show()
        custom_view.show()
        self.right_widget.update()
        self.right_scroll.ensureWidgetVisible(view_container)

    def remove_custom_view(self, view_container):
        for i, (container, view) in enumerate(self.custom_views):
            if container == view_container:
                self.right_layout.removeWidget(container)
                container.deleteLater()
                self.custom_views.pop(i)
                break

    def update_projections(self):
        self.view_xy.set_boids(self.boids_gl_widget.boids)
        self.view_xz.set_boids(self.boids_gl_widget.boids)
        self.view_yz.set_boids(self.boids_gl_widget.boids)
        self.view_xy.update()
        self.view_xz.update()
        self.view_yz.update()

        for container, view in self.custom_views:
            view.set_boids(self.boids_gl_widget.boids)
            view.update()

    # ---------- 高分辨率保存 ----------
    def save_all_views(self):
        outdir = QFileDialog.getExistingDirectory(self, "选择保存画面目录", os.getcwd())
        if not outdir:
            return

        def grab_gl_widget(widget, scale_factor=2.0):
            img = widget.grabFrameBuffer()
            if scale_factor != 1.0:
                w = int(img.width() * scale_factor)
                h = int(img.height() * scale_factor)
                img = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return img

        img_3d = grab_gl_widget(self.boids_gl_widget, scale_factor=2.5)
        img_3d.save(os.path.join(outdir, "view_3d.png"))

        views = [
            ("XY", self.view_xy, "view_xy.png"),
            ("XZ", self.view_xz, "view_xz.png"),
            ("YZ", self.view_yz, "view_yz.png"),
        ]
        for name, widget, fname in views:
            img = grab_gl_widget(widget, scale_factor=2.5)
            img.save(os.path.join(outdir, fname))

        for idx, (container, view) in enumerate(self.custom_views):
            plane_name = getattr(view, "plane_name", f"custom_{idx}")
            safe_name = plane_name.replace(" ", "_")
            img = grab_gl_widget(view, scale_factor=2.5)
            img.save(os.path.join(outdir, f"view_custom_{safe_name}.png"))

        QMessageBox.information(self, "保存完成", f"当前画面已保存到：\n{outdir}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

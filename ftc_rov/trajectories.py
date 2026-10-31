"""
24 reference motions for the BlueROV2 Heavy.

Each builder returns a Motion with a time grid and reference p, v, a (NED), heading
psi and yaw rate r, a dense 'planned path' for cross-track evaluation, the
environment (current / disturbance) and optionally a reactive hook that modifies
the reference online from (noisy) sensor data, per vehicle.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

DT = 0.02
V_CRUISE = 0.30
ACC = 0.05


class Motion:
    def __init__(self, name, t, p, psi, desc, heading_free=False, current=(0.05, np.pi / 4, 0.0),
                 wave=0.0, reactive=None, path=None, extra=None, eval_mode="path"):
        self.name, self.desc = name, desc
        self.t = t
        self.p = p
        self.v = np.gradient(p, DT, axis=0)
        self.a = gaussian_filter1d(np.gradient(self.v, DT, axis=0), 5, axis=0)
        self.psi = psi
        self.r = gaussian_filter1d(np.gradient(psi, DT), 3)
        self.heading_free = heading_free
        self.current = current          # (speed, direction, vertical)
        self.wave = wave                # amplitude of wave-induced surge/sway force [N]
        self.reactive = reactive
        self.path = p[::5] if path is None else path
        self.extra = extra or {}
        self.eval_mode = eval_mode
        self.T = len(t)


# ------------------------------------------------------------------ helpers
def chaikin(P, iters=4):
    P = np.asarray(P, float)
    for _ in range(iters):
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R = 0.25 * P[:-1] + 0.75 * P[1:]
        mid = np.empty((2 * len(Q), P.shape[1]))
        mid[0::2], mid[1::2] = Q, R
        P = np.vstack([P[:1], mid, P[-1:]])
    return P


def resample(P, ds=0.01):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    P, s = P[keep], s[keep]
    ss = np.arange(0, s[-1], ds)
    return np.stack([np.interp(ss, s, P[:, i]) for i in range(3)], 1), ss


def speed_profile(L, vmax=V_CRUISE, acc=ACC, hold_end=8.0, hold_start=3.0):
    acc = acc if acc else ACC
    """Smooth trapezoid s(t) over arc length L."""
    t_acc = vmax / acc
    if L < vmax * t_acc:
        vmax = np.sqrt(L * acc); t_acc = vmax / acc
    t_cruise = (L - vmax * t_acc) / vmax
    Ttot = hold_start + 2 * t_acc + t_cruise + hold_end
    t = np.arange(0, Ttot, DT)
    tt = t - hold_start
    v = np.where(tt < 0, 0, np.where(tt < t_acc, acc * tt, np.where(tt < t_acc + t_cruise, vmax,
                 np.clip(vmax - acc * (tt - t_acc - t_cruise), 0, None))))
    v = gaussian_filter1d(v, 25)
    s = np.cumsum(v) * DT
    s *= L / max(s[-1], 1e-9)
    return t, np.clip(s, 0, L)


def path_motion(name, waypoints, desc, vmax=V_CRUISE, smooth=4, heading="tangent", psi_fixed=0.0,
                hold_end=8.0, acc=None, **kw):
    P = chaikin(np.asarray(waypoints, float), smooth) if smooth else np.asarray(waypoints, float)
    Pd, s = resample(P)
    t, st = speed_profile(s[-1], vmax, acc=acc, hold_end=hold_end)
    p = np.stack([np.interp(st, s, Pd[:, i]) for i in range(3)], 1)
    psi = heading_from(p, heading, psi_fixed)
    return Motion(name, t, p, psi, desc, path=Pd[::10], **kw)


def heading_from(p, mode, psi_fixed=0.0):
    if mode == "fixed":
        return np.full(len(p), psi_fixed)
    d = np.gradient(gaussian_filter1d(p, 10, axis=0), axis=0)
    sp = np.hypot(d[:, 0], d[:, 1]) / DT
    ang = np.arctan2(d[:, 1], d[:, 0])
    psi = np.empty(len(p)); last = psi_fixed
    valid = np.nonzero(sp > 0.05)[0]
    first = ang[valid[0]] if len(valid) else psi_fixed
    last = first
    for i in range(len(p)):
        if sp[i] > 0.05:
            last = ang[i]
        psi[i] = last
    psi = np.unwrap(psi)
    return gaussian_filter1d(psi, 40)


def timed(name, t, p, psi, desc, **kw):
    return Motion(name, t, p, psi, desc, **kw)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x * x)   # min-jerk


# ------------------------------------------------------------------ motions
def m_dive():
    t = np.arange(0, 90, DT)
    z = 1 + 19 * smoothstep((t - 5) / 70)
    p = np.stack([np.zeros_like(t), np.zeros_like(t), z], 1)
    return timed("Dive-In", t, p, np.zeros_like(t), "Descend 1 m -> 20 m", heading_free=False)


def m_ascend():
    t = np.arange(0, 90, DT)
    z = 20 - 19 * smoothstep((t - 5) / 70)
    p = np.stack([np.zeros_like(t), np.zeros_like(t), z], 1)
    return timed("Surface/Ascend", t, p, np.zeros_like(t), "Ascend 20 m -> 1 m")


def m_lawnmower():
    L, sp, n = 16.0, 4.0, 4
    wps = []
    for i in range(n):
        y = i * sp
        xs = [0, L] if i % 2 == 0 else [L, 0]
        wps += [[xs[0], y, 10], [xs[1], y, 10]]
    return path_motion("Lawnmower", wps, "4 lanes x 16 m, 4 m spacing, z=10 m", smooth=5)


def m_zigzag():
    wps = [[3 * i, (2.5 if i % 2 else -2.5) if 0 < i < 8 else 0, 10] for i in range(9)]
    return path_motion("Zigzag", wps, "8 diagonal legs, +-2.5 m, z=10 m", smooth=4)


def m_spiral():
    th = np.linspace(0, 3 * 2 * np.pi, 4000)
    r = 1.0 + 7.0 * th / th[-1]
    P = np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, 10)], 1)
    return path_motion("Spiral", P, "Expanding Archimedean spiral r 1->8 m, 3 turns", smooth=0)


def m_helical():
    th = np.linspace(0, 3 * 2 * np.pi, 4000)
    R = 4.0
    P = np.stack([R * np.cos(th), R * np.sin(th), 5 + 2.5 * th / (2 * np.pi)], 1)
    return path_motion("Helical", P, "Helix R=4 m, 2.5 m/turn, z 5 -> 12.5 m", smooth=0)


def m_orbit():
    th = np.linspace(0, 2 * np.pi * 1.25, 3000)
    R = 5.0
    P = np.stack([R * np.cos(th), R * np.sin(th), np.full_like(th, 10)], 1)
    m = path_motion("Circular Orbit", P, "Orbit R=5 m around target, nose on target", smooth=0)
    m.psi = gaussian_filter1d(np.unwrap(np.arctan2(-m.p[:, 1], -m.p[:, 0])), 20)
    m.r = gaussian_filter1d(np.gradient(m.psi, DT), 3)
    return m


def m_figure8():
    th = np.linspace(0, 2 * np.pi, 4000)
    a = 7.0
    P = np.stack([a * np.sin(th), a * np.sin(th) * np.cos(th), np.full_like(th, 10)], 1)
    return path_motion("Figure-Eight", P, "Lemniscate of Gerono, 14 m x 7 m", smooth=0)


def m_square():
    s = 10.0
    wps = [[0, 0, 10], [s, 0, 10], [s, s, 10], [0, s, 10], [0, 0, 10]]
    return path_motion("Square", wps, "10 m square, rounded corners", smooth=3)


def m_waypoints():
    wps = [[0, 0, 5], [8, 2, 8], [12, 10, 12], [4, 14, 10], [-2, 8, 6], [0, 0, 5]]
    return path_motion("Waypoint Nav", wps, "6 3-D waypoints", smooth=4)


def m_straight():
    return path_motion("Straight Line", [[0, 0, 8], [40, 0, 8]], "40 m line at 8 m depth", smooth=0)


def m_staircase():
    t = np.arange(0, 145, DT)
    x = 0.15 * np.clip(t - 5, 0, None)
    z = 5 + sum(3 * smoothstep((t - (10 + 25 * k)) / 15) for k in range(5))
    p = np.stack([x, np.zeros_like(t), z], 1)
    return timed("Depth Staircase", t, p, np.zeros_like(t), "5 depth steps of 3 m (5->20 m) at 0.15 m/s")


def m_sawtooth():
    x = np.linspace(0, 36, 8000)
    tri = 2 * np.abs((x / 12) % 1 - 0.5)
    z = 6 + 8 * (1 - tri)
    P = chaikin(np.stack([x, np.zeros_like(x), z], 1)[::400], 4)
    return path_motion("Sawtooth/Yo-Yo", P, "3 yo-yo cycles 6-14 m over 36 m", smooth=0)


def m_scurve():
    x = np.linspace(0, 45, 5000)
    P = np.stack([x, 3 * np.sin(2 * np.pi * x / 15), np.full_like(x, 8)], 1)
    return path_motion("S-Curve", P, "Sinusoid A=3 m, lambda=15 m", smooth=0)


def m_station():
    t = np.arange(0, 120, DT)
    p = np.tile([0, 0, 10.0], (len(t), 1))
    return timed("Station Keeping", t, p, np.zeros_like(t), "Hover at 10 m in 0.15 m/s current + waves",
                 current=(0.15, 0.6, 0.0), wave=6.0, eval_mode="time")


def m_transect():
    wps = [[0, 0, 12], [30, 0, 12]]
    return path_motion("Transect", wps, "30 m survey line, fixed heading, 0.15 m/s cross-current",
                       smooth=0, heading="fixed", psi_fixed=0.0, current=(0.15, np.pi / 2, 0.0))


# ---- reactive motions -------------------------------------------------------
def wall_y(x):
    return 6 + 1.0 * np.sin(2 * np.pi * x / 20) + 0.04 * x


def m_wall():
    x = np.linspace(0, 36, 4000)
    d0 = 2.0
    P = np.stack([x, wall_y(x) - d0, np.full_like(x, 8)], 1)
    m = path_motion("Wall Following", P, "Keep 2 m stand-off from curved wall (sonar)", smooth=0)

    def hook(k, meas, ref, mem, rng):
        x_m = meas["p"][:, 0]
        rng_meas = wall_y(meas["p_true"][:, 0]) - meas["p_true"][:, 1] + rng.normal(0, 0.05, len(x_m))
        mem.setdefault("filt", rng_meas.copy())
        mem["filt"] += 0.1 * (rng_meas - mem["filt"])
        y_wall_est = meas["p"][:, 1] + mem["filt"]
        ref["p"][:, 1] = y_wall_est - d0
        slope = (wall_y(x_m + 0.01) - wall_y(x_m - 0.01)) / 0.02
        ref["v"][:, 1] = slope * ref["v"][:, 0]
    m.reactive = hook
    m.extra["d0"] = d0
    return m


def seabed_z(x):
    return 18 - 2.0 * np.sin(2 * np.pi * x / 25) - 1.0 * np.sin(2 * np.pi * x / 9 + 1) - 0.05 * x


def m_terrain():
    x = np.linspace(0, 36, 4000)
    h0 = 3.0
    P = np.stack([x, np.zeros_like(x), seabed_z(x) - h0], 1)
    m = path_motion("Terrain Following", P, "Hold 3 m altitude over undulating seabed (altimeter)", smooth=0)

    def hook(k, meas, ref, mem, rng):
        xt = meas["p_true"][:, 0]
        alt = seabed_z(xt) - meas["p_true"][:, 2] + rng.normal(0, 0.03, len(xt))
        mem.setdefault("filt", alt.copy())
        mem["filt"] += 0.15 * (alt - mem["filt"])
        zbed = meas["p"][:, 2] + mem["filt"]
        ref["p"][:, 2] = zbed - h0
        xm = meas["p"][:, 0]
        slope = (seabed_z(xm + 0.05) - seabed_z(xm - 0.05)) / 0.1
        ref["v"][:, 2] = slope * ref["v"][:, 0]
    m.reactive = hook
    return m


def m_docking():
    # dock at origin, axis +x (vehicle must arrive heading +x into the dock at x=0)
    t = np.arange(0, 130, DT)
    A = np.array([-12.0, 5.0, 9.0])      # start
    Bp = np.array([-4.0, 0.0, 6.0])      # pre-dock (aligned)
    C = np.array([0.0, 0.0, 6.0])        # dock
    s1 = smoothstep((t - 3) / 55)
    s2 = smoothstep((t - 70) / 45)
    p = A + (Bp - A) * s1[:, None] + (C - Bp) * s2[:, None]
    psi = np.deg2rad(-60) + np.deg2rad(60) * smoothstep((t - 45) / 20)
    return timed("Docking Approach", t, p, psi, "Transit, align with dock axis, 0.1 m/s final approach",
                 eval_mode="time", extra={"dock": C})


OBST = np.array([[10.0, 0.6, 2.0, 1.0], [22.0, -0.5, 2.5, 1.0], [32.0, 0.0, 1.5, 1.0]])  # x, y, radius, side


def bump(x, xc, half=6.0):
    u = np.clip((x - (xc - half)) / (2 * half), 0, 1)
    return np.sin(np.pi * u) ** 2


def m_obstacle():
    x = np.linspace(0, 42, 5000)
    y_path = np.zeros_like(x)
    for o in OBST:
        y_path += (o[1] + np.sign(o[3]) * (o[2] + 1.5) - 0) * bump(x, o[0])
    P = np.stack([x, y_path, np.full_like(x, 8)], 1)
    m = path_motion("Obstacle Avoidance", P, "3 pillars detected at 8 m by sonar, local detours (1.5 m clearance)",
                    smooth=0)
    m.extra["obstacles"] = OBST
    # straight nominal reference + online detours triggered by sonar detection

    def hook(k, meas, ref, mem, rng):
        N = meas["p"].shape[0]
        act = mem.setdefault("active", np.zeros((N, len(OBST)), bool))
        d = np.linalg.norm(meas["p"][:, None, :2] - OBST[None, :, :2], axis=2)
        act |= d < 8.0
        xr = ref["p"][:, 0]
        y = np.zeros(N); vy = np.zeros(N)
        for j, o in enumerate(OBST):
            amp = o[1] + np.sign(o[3]) * (o[2] + 1.5)
            b = bump(xr, o[0]); db = (bump(xr + 0.01, o[0]) - bump(xr - 0.01, o[0])) / 0.02
            y += act[:, j] * amp * b; vy += act[:, j] * amp * db * ref["v"][:, 0]
        ref["p"][:, 1] = y; ref["v"][:, 1] = vy
    m.reactive = hook
    return m


def m_sar():
    legs = [3, 3, 6, 6, 9, 9, 12, 12]
    p = np.array([0.0, 0.0]); wps = [[0, 0, 10]]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for i, L in enumerate(legs):
        p = p + np.array(dirs[i % 4]) * L
        wps.append([p[0], p[1], 10])
    return path_motion("SAR Sweep", wps, "Expanding-square search, 3 m track spacing", smooth=3)


def m_random_walk():
    rng = np.random.default_rng(7)
    p = np.array([0.0, 0.0, 10.0]); wps = [p.copy()]; hd = 0.0
    for i in range(9):
        hd += rng.uniform(-np.pi / 2, np.pi / 2)
        step = np.array([5 * np.cos(hd), 5 * np.sin(hd), rng.uniform(-1.5, 1.5)])
        p = p + step; p[2] = np.clip(p[2], 6, 14); wps.append(p.copy())
    return path_motion("Random Walk", wps, "9 random-heading 5 m segments (seed 7)", smooth=4)


SRC = np.array([14.0, 9.0, 13.0])


def field(p):
    d = p - SRC
    return np.exp(-(d[:, 0] ** 2 + d[:, 1] ** 2) / (2 * 8.0 ** 2) - d[:, 2] ** 2 / (2 * 5.0 ** 2))


def m_adaptive():
    t = np.arange(0, 150, DT)
    p0 = np.array([0.0, 0.0, 8.0])
    p = np.tile(p0, (len(t), 1))
    m = timed("Adaptive Exploration", t, p, np.zeros_like(t), "Gradient-seeking plume search, replans every 10 s",
              eval_mode="time", heading_free=False)
    seg = 10.0
    step = 2.5

    def hook(k, meas, ref, mem, rng):
        N = meas["p"].shape[0]
        tk = ref["tv"]                                  # per-vehicle virtual time
        if "wp0" not in mem:
            mem["wp0"] = np.tile(p0, (N, 1)); mem["wp1"] = np.tile(p0, (N, 1)); mem["t0"] = np.full(N, -1e9)
            mem["psi"] = np.zeros(N); mem["psi_prev"] = np.zeros(N)
        rep = tk - mem["t0"] >= seg - 1e-9
        if rep.any():
            pm = meas["p_true"][rep]
            eps = 0.5
            grad = np.zeros((pm.shape[0], 3))
            for i in range(3):
                e = np.zeros(3); e[i] = eps
                grad[:, i] = (field(pm + e) - field(pm - e)) / (2 * eps)
            grad += rng.normal(0, 0.003, grad.shape)
            g = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-9)
            stp = np.where(field(pm) > 0.9, 0.0, step)[:, None]
            mem["wp0"][rep] = mem["wp1"][rep]
            mem["wp1"][rep] = mem["wp1"][rep] + stp * g * np.array([1, 1, 0.5])
            mem["t0"][rep] = np.where(mem["t0"][rep] < 0, 0.0, mem["t0"][rep] + seg)
            dxy = mem["wp1"][rep, :2] - mem["wp0"][rep, :2]
            mv = np.hypot(dxy[:, 0], dxy[:, 1]) > 0.1
            old = mem["psi"][rep]
            mem["psi_prev"][rep] = old
            newpsi = old + np.angle(np.exp(1j * (np.arctan2(dxy[:, 1], dxy[:, 0]) - old)))
            mem["psi"][rep] = np.where(mv, newpsi, old)
        tau = np.clip((tk - mem["t0"]) / seg, 0, 1)
        kap = ref["kappa"]
        s = smoothstep(tau)
        ds = 30 * tau**2 * (1 - tau)**2 / seg * kap
        dd = 60 * tau * (1 - tau) * (1 - 2 * tau) / seg**2 * kap**2
        D = mem["wp1"] - mem["wp0"]
        ref["p"][:] = mem["wp0"] + D * s[:, None]
        ref["v"][:] = D * ds[:, None]
        ref["a"][:] = D * dd[:, None]
        sh = smoothstep(tau / 0.5)
        ref["psi"][:] = mem["psi_prev"] + (mem["psi"] - mem["psi_prev"]) * sh
        ref["r"][:] = 0.0
    m.reactive = hook
    m.extra["source"] = SRC
    return m


def m_emergency():
    m = path_motion("Emergency Ascent", [[0, 0, 25.0], [0, 0, 1.0]],
                    "Safety ascent 25 m -> 1 m at <=0.5 m/s, level attitude", vmax=0.5, acc=0.1,
                    smooth=0, heading="fixed", psi_fixed=0.0, eval_mode="time", hold_end=5.0)
    return m


BUILDERS = [m_dive, m_ascend, m_lawnmower, m_zigzag, m_spiral, m_helical, m_orbit, m_figure8, m_square,
            m_waypoints, m_straight, m_staircase, m_sawtooth, m_scurve, m_station, m_transect, m_wall,
            m_terrain, m_docking, m_obstacle, m_sar, m_random_walk, m_adaptive, m_emergency]


def all_motions():
    return [b() for b in BUILDERS]

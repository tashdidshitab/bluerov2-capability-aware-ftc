"""
BlueROV2 Heavy 6-DOF model (Fossen formulation), batched over N vehicles.

Parameters: von Benzon et al., "An Open-Source Benchmark Simulator: Control of a
BlueROV2 Underwater Robot", JMSE 10(12):1898, 2022 (heavy configuration).
Thrusters: Blue Robotics T200 @16 V (51.5 N fwd / 40.2 N rev, ~3500 RPM max).

Frames: NED world, body x-fwd / y-stbd / z-down. Attitude integrated as a unit
quaternion (no Euler singularity), Euler angles only exposed to the controller.
"""
import numpy as np

G = 9.82
RHO = 1000.0

# ----------------------------------------------------------------- geometry
S2 = 1.0 / np.sqrt(2.0)
R_TH = np.array([
    [0.156, 0.111, 0.085],   # T1 horizontal
    [0.156, -0.111, 0.085],  # T2
    [-0.156, 0.111, 0.085],  # T3
    [-0.156, -0.111, 0.085], # T4
    [0.120, 0.218, 0.0],     # T5 vertical
    [0.120, -0.218, 0.0],    # T6
    [-0.120, 0.218, 0.0],    # T7
    [-0.120, -0.218, 0.0],   # T8
])
E_TH = np.array([
    [S2, -S2, 0], [S2, S2, 0], [-S2, -S2, 0], [-S2, S2, 0],
    [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
], dtype=float)
B_ALLOC = np.vstack([E_TH.T, np.cross(R_TH, E_TH).T])   # 6x8 thrust configuration matrix
N_TH = 8
THRUSTER_NAMES = [f"T{i+1}" for i in range(8)]
HORIZ = np.array([0, 1, 2, 3])
VERT = np.array([4, 5, 6, 7])

# --------------------------------------------------------------- T200 model
RPM_MAX = 3500.0
T_FWD_MAX = 51.5      # N   (5.25 kgf @16 V)
T_REV_MAX = 40.2      # N   (4.10 kgf @16 V)
K_FWD = T_FWD_MAX / RPM_MAX**2
K_REV = T_REV_MAX / RPM_MAX**2
TAU_MOTOR = 0.08      # s  first-order RPM response
P_FULL = 390.0        # W  full-throttle electrical power @16 V


def thrust_to_rpm(T):
    return np.where(T >= 0, np.sqrt(np.abs(T) / K_FWD), -np.sqrt(np.abs(T) / K_REV))


def rpm_to_thrust(n):
    return np.where(n >= 0, K_FWD * n * n, -K_REV * n * n)


def thruster_power(T):
    frac = np.where(T >= 0, T / T_FWD_MAX, -T / T_REV_MAX)
    return P_FULL * np.clip(frac, 0, 1) ** 1.5


# ---------------------------------------------------------------- params
def nominal_params():
    m = 13.5
    W = m * G
    return dict(
        m=m, W=W,
        B=W + 1.3,                                   # trimmed slightly positive (fail-safe)
        I=np.array([0.26, 0.23, 0.37]),
        A1=np.array([6.36, 7.12, 18.68]),
        A2=np.array([0.189, 0.135, 0.222]),
        Dl=np.array([13.7, 0.0, 33.0, 0.0, 0.8, 0.0]),
        Dq=np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5]),
        rg=np.array([0.0, 0.0, 0.0]),
        rb=np.array([0.0, 0.0, -0.01]),
        kth=np.ones(8),                              # thruster effectiveness multipliers
    )


def batch_params(N, rng=None, unc=0.0, base=None):
    """Return per-vehicle parameter arrays; unc = relative uniform uncertainty."""
    p = base or nominal_params()
    out = {}
    def pert(x, scale=unc):
        x = np.asarray(x, float)
        shape = (N,) + x.shape
        if rng is None or scale == 0:
            return np.broadcast_to(x, shape).copy()
        return x * (1 + rng.uniform(-scale, scale, shape))
    out["m"] = pert(p["m"], unc * 0.25)
    out["I"] = pert(p["I"])
    out["A1"] = pert(p["A1"])
    out["A2"] = pert(p["A2"])
    out["Dl"] = pert(p["Dl"])
    out["Dq"] = pert(p["Dq"])
    out["W"] = out["m"] * G
    if rng is None or unc == 0:
        out["B"] = np.full(N, p["B"])
        out["rb"] = np.broadcast_to(p["rb"], (N, 3)).copy()
        out["kth"] = np.ones((N, 8))
    else:
        out["B"] = out["W"] + rng.uniform(0.0, 3.0, N)          # 0..3 N positive buoyancy
        out["rb"] = np.zeros((N, 3)); out["rb"][:, 2] = -rng.uniform(0.005, 0.02, N)
        out["kth"] = 1 + rng.uniform(-0.1, 0.1, (N, 8))          # +-10 % thruster gain
    out["rg"] = np.zeros((N, 3))
    out["M1"] = out["m"][:, None] + out["A1"]
    out["M2"] = out["I"] + out["A2"]
    return out


# ------------------------------------------------------------ quaternions
def quat_to_R(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_to_euler(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    phi = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    th = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    psi = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([phi, th, psi], 1)


def euler_to_quat(phi, th, psi):
    cr, sr = np.cos(phi / 2), np.sin(phi / 2)
    cp, sp = np.cos(th / 2), np.sin(th / 2)
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], -1)


def quat_dot(q, w):
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    p, r_, s = w[:, 0], w[:, 1], w[:, 2]
    return 0.5 * np.stack([-qx * p - qy * r_ - qz * s,
                           qw * p + qy * s - qz * r_,
                           qw * r_ - qx * s + qz * p,
                           qw * s + qx * r_ - qy * p], 1)


# ---------------------------------------------------------------- dynamics
def restoring(R, P):
    """Gravity + buoyancy wrench in body frame (RHS sign)."""
    zW = R[:, 2, :]                        # world z-axis expressed in body (row 3 of R = R^T e3)
    fg = P["W"][:, None] * zW
    fb = -P["B"][:, None] * zW
    m = np.cross(P["rg"], fg) + np.cross(P["rb"], fb)
    return np.concatenate([fg + fb, m], 1)


def accel(q, nu, tau, vc_world, P):
    """nu_dot for batch. tau: body wrench (N,6); vc_world: current (N,3)."""
    R = quat_to_R(q)
    vc_b = np.einsum("nji,nj->ni", R, vc_world)
    nur = nu.copy(); nur[:, :3] -= vc_b
    v1, v2 = nu[:, :3], nu[:, 3:]
    vr1 = nur[:, :3]
    # Coriolis (rigid body on nu, added mass on relative nu)
    m = P["m"][:, None]
    f_c = np.cross(v2, m * v1) + np.cross(v2, P["A1"] * vr1)
    m_c = np.cross(v2, (P["I"] + P["A2"]) * v2) + np.cross(vr1, P["A1"] * vr1)
    Dnu = P["Dl"] * nur + P["Dq"] * np.abs(nur) * nur
    g = restoring(R, P)
    rhs = tau + g - Dnu
    rhs[:, :3] -= f_c
    rhs[:, 3:] -= m_c
    Mdiag = np.concatenate([P["M1"], P["M2"]], 1)
    return rhs / Mdiag, R


def wrench_from_thrust(T, kth=None):
    if kth is not None:
        T = T * kth
    return T @ B_ALLOC.T

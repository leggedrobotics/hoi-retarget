# Derived from GMR (General Motion Retargeting).
# Source: https://github.com/YanjieZe/GMR @ 5bac4bd (2025-10-16)
# Copyright 2025 Yanjie Ze
#
# SPDX-License-Identifier: MIT
#
# Modified by ETH Zurich, 2026, under the same MIT terms:
#   - added CasADi-compatible variants of the quaternion helpers, so the same
#     rotation maths serves both the NumPy IK stage and the symbolic
#     trajectory-optimization stage;
#   - added quat_mul_wxyz / quat_conjugate_wxyz / rot_from_quat_wxyz /
#     axis_angle_from_quat_wxyz.

import numpy as np
import torch
import casadi as ca
from scipy.spatial.transform import Rotation as R


def quatToEuler(quat):
    """ 将四元数转换为欧拉角(roll, pitch, yaw)。 """
    eulerVec = np.zeros(3)
    qw, qx, qy, qz = quat
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)
    else:
        eulerVec[1] = np.arcsin(sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return eulerVec



def quat_mul_np(x, y, scalar_first=True):
    """
    Performs quaternion multiplication on arrays of quaternions
    :param x: tensor of quaternions of shape (..., 4)
    :param y: tensor of quaternions of shape (..., 4)
    :param scalar_first: True if quaternions are in [w, x, y, z] format
    :return: quaternion multiplication result in same format
    """
    if scalar_first:
        pass
    else: # convert to scalar-first
        x = x[..., [3, 0, 1, 2]]
        y = y[..., [3, 0, 1, 2]]

    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
        x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
        x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
        x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0
    ], axis=-1)

    if scalar_first:
        pass
    else:
        res = res[..., [1, 2, 3, 0]]  # back to [w, x, y, z]

    return res

def quat_rotate_inverse(q, v):
    """
    将向量 v 以四元数 q 的逆旋转进行变换。  
    为保持一致，以下代码与原脚本中的实现相同。
    """
    q = np.asarray(q)
    v = np.asarray(v)

    q_w = q[:, -1]      # w
    q_vec = q[:, :3]    # x, y, z

    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * (2.0 * q_w)[:, np.newaxis]
    dot = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * (2.0 * dot)

    return a - b + c

def quat_rotate_inverse_torch(q, v, scalar_first=True):
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quat_rotate_inverse_np(q, v, scalar_first=True):
    q = np.asarray(q)
    v = np.asarray(v)
    if scalar_first:
        q = q[..., [1, 2, 3, 0]]
    else:
        q = q[..., [0, 1, 2, 3]]
    q_w = q[..., -1]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a - b + c

def euler_from_quaternion_torch(quat_angle, scalar_first=True):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    if scalar_first:
        quat_angle = quat_angle[..., [1, 2, 3, 0]]
    else:
        quat_angle = quat_angle[..., [0, 1, 2, 3]]
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def euler_from_quaternion_np(quat, scalar_first=True):
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]
    else:
        quat = quat[..., [0, 1, 2, 3]]
    
    x = quat[:,0]; y = quat[:,1]; z = quat[:,2]; w = quat[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def quat_diff_np(q1, q2, scalar_first=True):
    # Ensure quaternions are numpy arrays
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Convert to scipy Rotation object (scalar-first)
    r1 = R.from_quat(q1, scalar_first=scalar_first)
    r2 = R.from_quat(q2, scalar_first=scalar_first)

    # Relative rotation
    r_rel = r2 * r1.inv()

    # Rotation vector (axis * angle)
    rotvec = r_rel.as_rotvec()  # returns angle * axis vector

    return rotvec


def rot_from_quat_wxyz(q):
    """
    Quaternion [w, x, y, z] -> 3x3 rotation matrix (NumPy).
    """
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    return np.array([
        [ww + xx - yy - zz, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), ww - xx + yy - zz, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), ww - xx - yy + zz],
    ])

def normalize_quat_xyzw_np(
    q: np.ndarray,
    eps: float = 1e-8,
    identity_on_zero: bool = True,
) -> np.ndarray:
    q = np.asarray(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    out = q / np.maximum(n, eps)
    if identity_on_zero and q.shape[-1] == 4:
        valid = n > eps
        identity = np.zeros_like(out)
        identity[..., -1] = 1.0
        out = np.where(valid, out, identity)
    return out

def quat_conjugate_wxyz(q):
    """
    Conjugate of quaternion [w, x, y, z] for NumPy or CasADi types.
    """
    if isinstance(q, (ca.MX, ca.SX, ca.DM)):
        return ca.vertcat(q[0], -q[1], -q[2], -q[3])
    q = np.asarray(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul_wxyz(q1, q2):
    """
    Hamilton product q = q1 * q2 for [w, x, y, z] (NumPy or CasADi).
    """
    if isinstance(q1, (ca.MX, ca.SX, ca.DM)) or isinstance(q2, (ca.MX, ca.SX, ca.DM)):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return ca.vertcat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    return quat_mul_np(np.asarray(q1), np.asarray(q2), scalar_first=True)


def axis_angle_from_quat_wxyz(q, eps: float = 1e-6):
    """
    Axis-angle vector (axis * angle) from quaternion [w, x, y, z].
    Works with CasADi or NumPy.
    """
    if isinstance(q, (ca.MX, ca.SX, ca.DM)):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        sign = ca.if_else(qw < 0, -1.0, 1.0)
        qw = sign * qw
        qx = sign * qx
        qy = sign * qy
        qz = sign * qz

        mag = ca.sqrt(qx * qx + qy * qy + qz * qz)
        half_angle = ca.atan2(mag, qw)
        angle = 2.0 * half_angle
        sin_half = ca.sin(half_angle)
        sin_over_angle = ca.if_else(
            ca.fabs(angle) > eps,
            sin_half / angle,
            0.5 - (angle * angle) / 48.0,
        )
        return ca.vertcat(qx / sin_over_angle, qy / sin_over_angle, qz / sin_over_angle)

    q = np.asarray(q, dtype=float)
    if q[0] < 0:
        q = -q
    qw, qx, qy, qz = q
    mag = np.linalg.norm([qx, qy, qz])
    if mag < eps:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(mag, qw)
    axis = np.array([qx, qy, qz]) / mag
    return axis * angle


def finite_difference_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """
    Simple finite-difference velocities for a (T, 3) position array.
    Central differences for interior points, forward/backward for endpoints.
    """
    T, _ = positions.shape
    vel = np.zeros_like(positions)
    if T < 2:
        return vel
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    seq = np.asarray(seq)
    if seq.shape[0] == target_len:
        return seq
    if seq.shape[0] > target_len:
        return seq[:target_len]
    pad = np.repeat(seq[-1:], target_len - seq.shape[0], axis=0)
    return np.concatenate([seq, pad], axis=0)


def pad_or_truncate_1d(seq: np.ndarray, target_len: int) -> np.ndarray:
    seq = np.asarray(seq).reshape(-1)
    if seq.shape[0] == target_len:
        return seq
    if seq.shape[0] > target_len:
        return seq[:target_len]
    pad = np.repeat(seq[-1], target_len - seq.shape[0], axis=0)
    return np.concatenate([seq, pad], axis=0)

def rodrigues_casadi(aa):
    """(3,1) axis-angle MX -> (3,3) rotation matrix MX via Rodrigues formula."""
    theta_sq = ca.dot(aa, aa)
    theta = ca.sqrt(theta_sq + 1e-10)
    K = ca.vertcat(
        ca.horzcat(0, -aa[2], aa[1]),
        ca.horzcat(aa[2], 0, -aa[0]),
        ca.horzcat(-aa[1], aa[0], 0),
    )
    I3 = ca.DM.eye(3)
    sinc = ca.sin(theta) / theta
    versinc = (1 - ca.cos(theta)) / (theta ** 2)
    R = I3 + sinc * K + versinc * (K @ K)
    return R

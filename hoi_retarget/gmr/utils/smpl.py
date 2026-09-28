# Derived from GMR (General Motion Retargeting).
# Source: https://github.com/YanjieZe/GMR @ 5bac4bd (2025-10-16)
# Copyright 2025 Yanjie Ze
#
# SPDX-License-Identifier: MIT
#
# Modified by ETH Zurich, 2026, under the same MIT terms:
#   - reads the InterMimic/OMOMO sources through hoi_retarget.datasets;
#   - the one call into the LAFAN1 helpers was replaced by rot_utils.quat_mul_wxyz
#     and its loader has since been removed, so nothing here reaches that tree;
#   - dropped five loaders no entry point used (SMPL/SMPL-X file readers and the
#     GVHMR path);
#   - the SMPL-X file extension comes from hoi_retarget.paths.smplx_ext;
#   - InterMimic clips drive the retarget with their own recorded skeleton, and
#     source_body_mesh draws the human on it;
#   - tgt_fps: non-integer downsampling ratios resample, and a rate above the
#     source's is refused with a warning instead of being written as the label.

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from smplx.joint_names import JOINT_NAMES
from hoi_retarget.paths import smplx_ext
from hoi_retarget.datasets.intermimic import (
    build_intermimic_pose_body,
    intermimic_joints_in_smplx_order,
    load_intermimic_hoi_2d,
    parse_intermimic_hoi,
)


GYM_TO_SMPL_IDX = torch.tensor([
    0, 1, 5, 10, 2, 6, 11, 3, 7, 12, 4, 8, 13, 15, 34, 14,
    16, 35, 17, 36, 18, 37, 9, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51, 52
])

def load_smplx_data(smplx_data, smplx_body_model_path):
    body_model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        ext=smplx_ext(),
        use_pca=False,
        num_betas=int(np.asarray(smplx_data["betas"]).reshape(-1).shape[0]),
    )

    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, body_model.num_expression_coeffs).float(),
        return_full_pose=True,
    )

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    obj_data = None
    if "obj_state" in smplx_data:
        obj_data = {
            "pos": smplx_data["obj_state"][:, :3],
            "rot": smplx_data["obj_state"][:, 3:6],
            "contact": smplx_data["obj_contact"][:, :],
        }

    return body_model, smplx_output, human_height, obj_data


def convert_intermimic_to_smplx(input_path, gender="neutral"):
    """InterMimic clip -> SMPL-X parameters plus ``source_joints``, the clip's own skeleton.

    The skeleton was built from the subject's own body model and shape (SMPL-H for
    NeuralDome/IMHD², SMPL-X otherwise); the zero-beta SMPL-X body only supplies
    joint orientations, which depend on the rotations alone.
    """
    hoi_2d = load_intermimic_hoi_2d(input_path)
    parsed = parse_intermimic_hoi(hoi_2d)

    root_orient = (
        R.from_quat(parsed["root_rot_xyzw"])
        * R.from_matrix(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
    ).as_rotvec().astype(np.float32) # Reorder from InterMimic y-up to SMPL-X z-up convention
    pose_body = build_intermimic_pose_body(parsed)
    obj_rot = (
        R.from_quat(parsed["obj_rot_xyzw"]) 
    ).as_rotvec().astype(np.float32)
    obj_state = np.concatenate([parsed["obj_pos"], obj_rot], axis=1).astype(np.float32)

    return {
        "gender": np.array(gender),
        "betas": np.zeros(16, dtype=np.float32),
        "mocap_frame_rate": np.array(30.0, dtype=np.float32), # Fixed frame rate for InterMimic data since it's not explicitly provided
        "trans": parsed["root_pos"].astype(np.float32),
        "root_orient": root_orient,
        "pose_body": pose_body,
        "obj_state": obj_state,
        "obj_contact": parsed["contact_obj"].astype(np.int32),
        "contact_human": parsed["contact_human"].astype(np.int32),
        "source_joints": intermimic_joints_in_smplx_order(parsed["body_pos"]),
        "hand_pose": build_intermimic_pose_body(parsed, JOINT_NAMES[25:55]),  # flat-hand relative
    }


def source_body_mesh(smplx_data, smplx_body_model_path):
    """``(vertices, joints, body_model)`` of the SMPL-X body a display should draw.

    With ``source_joints`` (InterMimic clips) the zero-beta mesh is fitted to the recorded
    skeleton, so hands and feet sit where the subject's did while girth stays average: the clip's
    finger pose; each body vertex moved by its skinning-weighted joint offsets, blended along each
    limb bone so a length change spreads over the bone; each hand scaled about its wrist to the
    recorded wrist-to-middle3 length. Without it (CARI4D) this is ``load_smplx_data``'s mesh.
    """
    hands = smplx_data.get("hand_pose")
    if hands is not None and not np.any(hands):
        hands = None  # no finger motion (OMOMO): the recording carries SMPL-X's relaxed mean hand
    betas = np.asarray(smplx_data["betas"], dtype=np.float32).reshape(1, -1)
    n = smplx_data["pose_body"].shape[0]
    model = smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=str(smplx_data["gender"]),
        ext=smplx_ext(),
        use_pca=False,
        num_betas=betas.shape[1],
        flat_hand_mean=hands is not None,
    )
    hp = torch.zeros(n, 90) if hands is None else torch.as_tensor(hands, dtype=torch.float32)
    with torch.no_grad():
        out = model(
            betas=torch.as_tensor(betas),
            global_orient=torch.as_tensor(smplx_data["root_orient"]).float(),
            body_pose=torch.as_tensor(smplx_data["pose_body"]).float(),
            transl=torch.as_tensor(smplx_data["trans"]).float(),
            left_hand_pose=hp[:, :45],
            right_hand_pose=hp[:, 45:],
            jaw_pose=torch.zeros(n, 3),
            leye_pose=torch.zeros(n, 3),
            reye_pose=torch.zeros(n, 3),
            expression=torch.zeros(n, model.num_expression_coeffs),
        )
    verts, joints = out.vertices.numpy(), out.joints.numpy()
    src = smplx_data.get("source_joints")
    if src is None:
        return verts, joints, model

    parents, w = model.parents.numpy(), model.lbs_weights.numpy()[:, :55]
    d = np.zeros((n, 55, 3), dtype=np.float32)  # recorded minus model joint, pelvis pinned
    d[:, :22] = (src[:, :22] - src[:, :1]) - (joints[:, :22] - joints[:, :1])
    for k in range(22, 55):  # hands, jaw, eyes ride their parent
        d[:, k] = d[:, parents[k]]
    child = np.arange(55)  # a limb joint's only child; itself elsewhere
    for k in range(1, 22):
        kids = [c for c in range(1, 22) if parents[c] == k]
        child[k] = kids[0] if len(kids) == 1 else k
    rest = model.v_template.numpy()
    jr = model.J_regressor.numpy()[:55] @ rest
    bone = jr[child] - jr
    along = np.einsum("vkc,kc->vk", rest[:, None] - jr, bone) / np.maximum((bone**2).sum(1), 1e-9)
    verts = verts + np.einsum("vk,tkc->tvc", w, d) + np.einsum("vk,tkc->tvc", w * along.clip(0, 1), d[:, child] - d)
    for wrist, hand in ((20, range(25, 40)), (21, range(40, 55))):
        chain = [wrist, hand[3], hand[4], hand[5]]  # wrist, middle1-3
        size = [np.median(np.linalg.norm(np.diff(x[:, chain], axis=1), axis=-1).sum(1)) for x in (src, joints)]
        pivot = (joints[:, wrist] + d[:, wrist])[:, None]
        verts = verts + (size[0] / size[1] - 1) * w[:, [wrist, *hand]].sum(1)[:, None] * (verts - pivot)
    return verts, joints, model


def slerp(rot1, rot2, t):
    """Spherical linear interpolation between two rotations."""
    # Convert to quaternions
    q1 = rot1.as_quat()
    q2 = rot2.as_quat()
    
    # Normalize quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Compute dot product
    dot = np.sum(q1 * q2)
    
    # If the dot product is negative, slerp won't take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # If the inputs are too close, linearly interpolate
    if dot > 0.9995:
        return R.from_quat(q1 + t * (q2 - q1))
    
    # Perform SLERP
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q1 + s1 * q2
    
    return R.from_quat(q)

def get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30, object_data=None):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    src_fps = smplx_data["mocap_frame_rate"].item()
    frame_skip = int(src_fps / tgt_fps)
    num_frames = smplx_data["pose_body"].shape[0]
    global_orient = smplx_output.global_orient.squeeze()
    full_body_pose = smplx_output.full_pose.reshape(num_frames, -1, 3)
    joints = smplx_output.joints.detach().numpy().squeeze()
    if "source_joints" in smplx_data:
        # The subject's own skeleton, pinned to the model's pelvis so the world frame
        # (and the object offset derived from it) is unchanged.
        src = smplx_data["source_joints"]
        src = src + (joints[:, :1] - src[:, :1])
        joints = joints.copy()
        joints[:, : src.shape[1]] = np.where(np.isnan(src), joints[:, : src.shape[1]], src)
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    parents = body_model.parents

    has_object = object_data is not None
    if has_object:
        obj_pos = np.asarray(object_data["pos"])            # (T,3)
        obj_rot = np.asarray(object_data["rot"])            # (T,3) rotvec or (T,4) quat (wxyz)
        obj_contact = np.asarray(object_data["contact"]).reshape(-1)  # (T,) binary
        assert obj_pos.shape[0] == num_frames and obj_rot.shape[0] == num_frames and obj_contact.shape[0] == num_frames

    if tgt_fps < src_fps:
        # perform fps alignment with proper interpolation
        ratio = src_fps / tgt_fps
        if abs(ratio - round(ratio)) < 1e-6:
            new_num_frames = num_frames // frame_skip
        else:
            new_num_frames = max(2, int(round(num_frames / ratio)))
        
        # Create time points for interpolation
        original_time = np.arange(num_frames)
        target_time = np.linspace(0, num_frames-1, new_num_frames)
        
        # Interpolate global orientation using SLERP
        global_orient_interp = []
        for i in range(len(target_time)):
            t = target_time[i]
            idx1 = int(np.floor(t))
            idx2 = min(idx1 + 1, num_frames - 1)
            alpha = t - idx1
            
            rot1 = R.from_rotvec(global_orient[idx1])
            rot2 = R.from_rotvec(global_orient[idx2])
            interp_rot = slerp(rot1, rot2, alpha)
            global_orient_interp.append(interp_rot.as_rotvec())
        global_orient = np.stack(global_orient_interp, axis=0)
        
        # Interpolate full body pose using SLERP
        full_body_pose_interp = []
        for i in range(full_body_pose.shape[1]):  # For each joint
            joint_rots = []
            for j in range(len(target_time)):
                t = target_time[j]
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                
                rot1 = R.from_rotvec(full_body_pose[idx1, i])
                rot2 = R.from_rotvec(full_body_pose[idx2, i])
                interp_rot = slerp(rot1, rot2, alpha)
                joint_rots.append(interp_rot.as_rotvec())
            full_body_pose_interp.append(np.stack(joint_rots, axis=0))
        full_body_pose = np.stack(full_body_pose_interp, axis=1)
        
        # Interpolate joint positions using linear interpolation
        joints_interp = []
        for i in range(joints.shape[1]):  # For each joint
            for j in range(3):  # For each coordinate
                interp_func = interp1d(original_time, joints[:, i, j], kind='linear')
                joints_interp.append(interp_func(target_time))
        joints = np.stack(joints_interp, axis=1).reshape(new_num_frames, -1, 3)

        # --- Interpolate object (if provided) ---
        if has_object:
            # positions: linear
            obj_pos_interp = []
            for j in range(3):
                interp_func = interp1d(original_time, obj_pos[:, j], kind='linear')
                obj_pos_interp.append(interp_func(target_time))
            obj_pos = np.stack(obj_pos_interp, axis=1)  # (new_T,3)

            # rotations: SLERP between neighbors
            obj_rot_interp_quat = []
            is_quat = (obj_rot.shape[-1] == 4)
            if is_quat:
                def to_R(k): return R.from_quat(obj_rot[k], scalar_first=True)
            else:
                def to_R(k): return R.from_rotvec(obj_rot[k])
            for t in target_time:
                idx1 = int(np.floor(t))
                idx2 = min(idx1 + 1, num_frames - 1)
                alpha = t - idx1
                rot1 = to_R(idx1)
                rot2 = to_R(idx2)
                interp_rot = slerp(rot1, rot2, alpha)
                obj_rot_interp_quat.append(interp_rot.as_quat(scalar_first=True))
            obj_rot = np.stack(obj_rot_interp_quat, axis=0)  # (new_T,4)

            # contact: nearest-neighbor (keeps binary)
            contact_interp_func = interp1d(original_time, obj_contact.astype(float), kind='nearest')
            obj_contact = contact_interp_func(target_time).astype(np.int32).reshape(-1, 1)  # (new_T,1)

        aligned_fps = len(global_orient) / num_frames * src_fps
    else:
        if tgt_fps > src_fps:
            print(f"[WARN] tgt_fps={tgt_fps} exceeds the source's {src_fps:g} fps; keeping {src_fps:g} fps.")
        aligned_fps = src_fps
        # Keep original object data if any (no resampling)
        if has_object:
            if obj_rot.shape[-1] == 3:
                # keep orientation as quaternion for output consistency
                obj_rot = R.from_rotvec(obj_rot).as_quat(scalar_first=True)
            obj_contact = obj_contact.astype(np.int32).reshape(-1, 1)  # ensure int

    smplx_data_frames = []
    for curr_frame in range(len(global_orient)):
        result = {}
        single_global_orient = global_orient[curr_frame]
        single_full_body_pose = full_body_pose[curr_frame]
        single_joints = joints[curr_frame]
        joint_orientations = []
        for i, joint_name in enumerate(joint_names):
            if i == 0:
                rot = R.from_rotvec(single_global_orient)
            else:
                rot = joint_orientations[parents[i]] * R.from_rotvec(
                    single_full_body_pose[i].squeeze()
                )
            joint_orientations.append(rot)
            result[joint_name] = (single_joints[i], rot.as_quat(scalar_first=True))
        # Synthetic palm landmark for palm-contact retargeting: midpoint(wrist, middle3),
        # inheriting the wrist orientation. Injected inside the (already FPS-resampled) loop.
        for _sd in ("left", "right"):
            _wp, _wq = result[f"{_sd}_wrist"]
            _mp = result[f"{_sd}_middle3"][0]
            result[f"{_sd}_palm"] = (0.5 * (_wp + _mp), _wq)
        smplx_data_frames.append(result)

    # --- Build object frames aligned with SMPL frames (optional) ---
    object_frames = None
    if has_object:
        # Ensure lengths match (in case of rounding)
        T = len(smplx_data_frames)
        if obj_pos.shape[0] != T:
            obj_pos = obj_pos[:T]
            obj_rot = obj_rot[:T]
            obj_contact = obj_contact[:T]  # NEW
        object_frames = [(obj_pos[k], obj_rot[k], obj_contact[k]) for k in range(T)]  # NEW

    return smplx_data_frames, aligned_fps, object_frames




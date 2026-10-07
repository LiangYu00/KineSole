# file: test.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from smplx import SMPLX
from collections import defaultdict
import numpy as np
import math
import os
import kornia.geometry.conversions as K

# Import model and dataset classes from local files
from model import KineSole
from dataset import ActionPoseEstimationDataset


# --- Evaluation Metrics and Utility Functions ---
def get_coco_joints_from_smplx(smplx_output):
    smplx_joints = smplx_output.joints
    coco_joint_indices = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 16, 17]
    return smplx_joints[:, coco_joint_indices, :]


def procrustes_alignment(pred, target):
    pred_centroid = pred.mean(dim=1, keepdim=True)
    target_centroid = target.mean(dim=1, keepdim=True)
    pred_centered = pred - pred_centroid
    target_centered = target - target_centroid
    H = torch.bmm(target_centered.transpose(1, 2), pred_centered)
    U, S, V = torch.svd(H)
    R = torch.bmm(U, V.transpose(1, 2))
    det = torch.det(R)
    sign = torch.sign(det)
    V_reflect = V.clone()
    V_reflect[:, :, -1] *= sign.unsqueeze(-1)
    R = torch.bmm(U, V_reflect.transpose(1, 2))
    aligned_pred = torch.bmm(pred_centered, R.transpose(1, 2)) + target_centroid
    return aligned_pred


def to_rotation_matrix(pose):
    num_frames = pose.shape[0]
    pose_reshaped = pose.reshape(-1, 3)
    rot_mats = K.axis_angle_to_rotation_matrix(pose_reshaped)[:, :3, :3]
    return rot_mats.reshape(num_frames, -1)


def angle_between(R1, R2):
    num_frames, num_joints = R1.shape[0], R1.shape[1] // 9
    R1 = R1.view(num_frames, num_joints, 3, 3)
    R2 = R2.view(num_frames, num_joints, 3, 3)
    R_diff = torch.matmul(R1, R2.transpose(-1, -2))
    trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-7, 1.0 - 1e-7))
    return angle


def calculate_mpjre(pose_p, pose_t):
    pose_local_p = to_rotation_matrix(pose_p)
    pose_local_t = to_rotation_matrix(pose_t)
    local_angle_error = angle_between(pose_local_p, pose_local_t)
    mpjre = (local_angle_error * 180.0 / math.pi).mean()
    return mpjre


# --- Main Program ---
if __name__ == "__main__":
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use anonymized usernames S01, S02...
    all_usernames = [f"S{i:02d}" for i in range(1, 21)]
    # Assuming data folders are renamed accordingly
    all_user_roots = [f"../outputs/demo/{username}" for username in all_usernames]

    seq_len = 100
    batch_size = 1024
    fps = 30.0

    # --- Data Loading ---
    test_dataset = ActionPoseEstimationDataset(
        all_user_roots,
        seq_len=seq_len,
        step_size=1,  # Use denser step size during testing
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Initialization and Weight Loading ---
    model = KineSole(
        input_size_positional=6,
        input_size_pressure=10,  # 修改点1：从 8 改为 10
        hidden_size=128,
        output_size_body_poses=51,
        output_size_global_orient=3,
        num_encoder_layers=2,
        gcn_hidden_size=128,  # 兼容传参
        num_heads=8,
        max_seq_len=seq_len,
        causal=False
    ).to(device)

    model_path = 'weight.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please place the pre-trained model file here.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize SMPLX model
    smplx_model = SMPLX(
        model_path='../inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz',
        batch_size=batch_size,
        use_pca=False,
    ).to(device)

    # --- Test Process ---
    print("\n--- Running Final Test on Held-out Test Set ---")

    grouped_data = defaultdict(lambda: {
        'pred_joints': [], 'gt_joints': [],
        'raw_body_poses': [], 'gt_body_poses': []
    })

    with torch.no_grad():
        for positional_input, pressure_input, targets in test_loader:
            positional_input = positional_input.to(device)
            pressure_input = pressure_input.to(device)
            current_batch_size = positional_input.shape[0]

            if smplx_model.batch_size != current_batch_size:
                smplx_model = SMPLX(model_path='../inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz',
                                    batch_size=current_batch_size, use_pca=False).to(device)

            body_pose_pred, global_orient_pred = model(positional_input, pressure_input)

            full_body_pose_pred = F.pad(body_pose_pred, (0, 63 - 51), "constant", 0)
            pred_output = smplx_model(body_pose=full_body_pose_pred, global_orient=global_orient_pred)
            pred_joints = get_coco_joints_from_smplx(pred_output)

            gt_body_pose = F.pad(targets['body_poses'][:, :51].to(device), (0, 63 - 51), "constant", 0)
            gt_output = smplx_model(body_pose=gt_body_pose, global_orient=targets['global_orient'].to(device))
            gt_joints = get_coco_joints_from_smplx(gt_output)

            user_ids = targets['user_id'].numpy()
            action_ids = targets['action_id'].numpy()
            for i in range(current_batch_size):
                key = (user_ids[i], action_ids[i])
                grouped_data[key]['pred_joints'].append(pred_joints[i].cpu())
                grouped_data[key]['gt_joints'].append(gt_joints[i].cpu())
                grouped_data[key]['raw_body_poses'].append(body_pose_pred[i].cpu())
                grouped_data[key]['gt_body_poses'].append(targets['body_poses'][i, :51].cpu())

    # --- Calculate and Summarize Metrics ---
    total_mpjpe_error_sum = 0.0
    total_mpjre_error_sum = 0.0
    total_jerk_error_sum = 0.0
    total_frame_count = 0
    total_jerk_frame_count = 0
    per_action_results = defaultdict(dict)

    for (uid, aid), data in grouped_data.items():
        pred_j = torch.stack(data['pred_joints'])
        gt_j = torch.stack(data['gt_joints'])
        raw_pred_p = torch.stack(data['raw_body_poses'])
        raw_gt_p = torch.stack(data['gt_body_poses'])

        num_frames = pred_j.shape[0]
        total_frame_count += num_frames

        aligned_pred_j = procrustes_alignment(pred_j, gt_j)

        per_frame_mpjpe = torch.norm(aligned_pred_j - gt_j, p=2, dim=-1).mean(dim=1) * 1000
        total_mpjpe_error_sum += per_frame_mpjpe.sum().item()

        sequence_mpjre = calculate_mpjre(raw_pred_p, raw_gt_p).item()
        total_mpjre_error_sum += sequence_mpjre * num_frames

        sequence_jitter_avg = 0.0
        if num_frames >= 4:
            jerk_magnitudes = ((pred_j[3:] - 3 * pred_j[2:-1] + 3 * pred_j[1:-2] - pred_j[:-3]) * (fps ** 3)).norm(
                dim=2)
            total_jerk_error_sum += jerk_magnitudes.sum().item()
            total_jerk_frame_count += (num_frames - 3) * pred_j.shape[1]
            sequence_jitter_avg = jerk_magnitudes.mean().item()

        per_action_results[uid][aid] = {
            'MPJPE': per_frame_mpjpe.mean().item(),
            'MPJRE': sequence_mpjre,
            'Jitter': sequence_jitter_avg,
        }

    # --- Print Results ---
    print("\n" + "=" * 60)
    print(" " * 15 + "Detailed Test Results per User/Action")
    print("=" * 60)

    for uid in sorted(per_action_results.keys()):
        username = all_usernames[uid]
        print(f"\n--- User: {username} (ID: {uid}) ---")
        print(f"{'Action':<10}{'MPJPE (mm)':<15}{'MPJRE (°)':<15}{'Jitter (m/s³)':<15}")
        print("-" * 60)
        for aid in sorted(per_action_results[uid].keys()):
            metrics = per_action_results[uid][aid]
            print(f"{aid:<10}{metrics['MPJPE']:<15.2f}{metrics['MPJRE']:<15.2f}{metrics['Jitter']:<15.2f}")

    final_mpjpe = total_mpjpe_error_sum / total_frame_count if total_frame_count > 0 else 0
    final_mpjre = total_mpjre_error_sum / total_frame_count if total_frame_count > 0 else 0
    final_jitter = total_jerk_error_sum / total_jerk_frame_count if total_jerk_frame_count > 0 else 0

    print("\n" + "=" * 65)
    print(" " * 10 + "FINAL Overall Average Test Results (Per-Frame Weighted)")
    print("=" * 65)
    print(f"Average Test MPJPE: {final_mpjpe:.2f} mm")
    print(f"Average Test MPJRE: {final_mpjre:.2f} °")
    print(f"Average Test JITTER: {final_jitter:.2f} m/s³")
    print("=" * 65)
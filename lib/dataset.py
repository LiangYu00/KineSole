# file: dataset.py (Optimized for Test-Only Scenario)

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActionPoseEstimationDataset(Dataset):
    """
    Dataset class designed specifically for loading test data.
    This version simplifies data split logic, defining and loading only test set actions.
    """

    def __init__(self, data_root_paths, seq_len=100, step_size=1):

        self.seq_len = seq_len
        self.step_size = step_size
        self.mode = 'test'  # Hardcoded to 'test' mode

        # Simplified split definition: contains only test action IDs for each user
        self.TEST_ACTION_SPLITS = {
            0: {43, 44, 47, 52},
            1: {1, 12, 34, 47},
            2: {20, 46, 49},
            3: {30, 31, 48, 56},
            4: {23, 44, 45, 46},
            5: {14, 32, 54},
            6: {16, 22, 43, 53, 56},
            7: {11, 40, 59},
            8: {11, 15, 40},
            9: {30, 32, 57},
            10: {14, 41},
            11: {19, 51},
            12: {17, 31, 34, 41, 49, 54},
            13: {13, 17, 42, 53},
            14: {22, 45, 48},
            15: {55, 57},
            16: {13, 18, 52, 55, 59},
            17: {15, 20, 21, 33, 51, 58},
            18: {18, 19, 23, 33, 42},
            19: {1, 12, 16, 21, 58},
        }

        # Set of valid action IDs
        self.valid_action_ids = {1} | set(range(11, 24)) | set(range(30, 35)) | set(range(40, 61))

        self.user_data_list = []
        self.indices = []

        # Preload data and filter based on rules
        for user_idx, user_path in enumerate(data_root_paths):
            f_path = os.path.join(user_path, "cache_foot.npy")
            t_path = os.path.join(user_path, "cache_time.npy")
            g_path = os.path.join(user_path, "cache_gt.npy")
            fid_path = os.path.join(user_path, "fid.npy")
            l_path = os.path.join(user_path, "labels.csv")

            if not all(os.path.exists(p) for p in [f_path, t_path, g_path, l_path]):
                self.user_data_list.append(None)
                print(f"Warning: Data not found for user S{user_idx + 1:02d}, skipping.")
                continue

            foot_data, time_data, gt_data, fid_data = np.load(f_path), np.load(t_path), np.load(g_path), np.load(
                fid_path)
            labels_df = pd.read_csv(l_path)
            self.user_data_list.append({'foot': foot_data, 'time': time_data, 'gt': gt_data, 'fid': fid_data})

            # Get test action set for current user
            test_actions_for_user = self.TEST_ACTION_SPLITS.get(user_idx, set())

            for _, row in labels_df.iterrows():
                act_id = int(row['id'])
                if act_id not in self.valid_action_ids:
                    continue

                # --- Core logic simplification: Directly check if current action is in test set ---
                if act_id in test_actions_for_user:
                    start_t, end_t = row['start'], row['end']
                    indices_in_window = np.where((time_data >= start_t) & (time_data <= end_t))[0]

                    if len(indices_in_window) > self.seq_len + 5:
                        s_idx = indices_in_window[0]
                        e_idx = indices_in_window[-1]
                        for start_pos in range(s_idx + 1, e_idx - self.seq_len + 1, self.step_size):
                            # Ensure sequence is continuous
                            if fid_data[start_pos + self.seq_len - 1] - fid_data[start_pos] == self.seq_len - 1:
                                self.indices.append((user_idx, start_pos, act_id))

        print(f"[Dataset] Mode='{self.mode}': Generated {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        user_idx, start, act_id = self.indices[idx]
        data = self.user_data_list[user_idx]

        foot_curr = data['foot'][start: start + self.seq_len]
        gt_all = data['gt'][start + self.seq_len - 1]

        # Define indices for position and pressure data in sensor input
        # Define indices for position and pressure data in sensor input
        pos_idx = [7, 8, 9, 17, 18, 19]
        pres_idx = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]

        return (
            torch.tensor(foot_curr[:, pos_idx], dtype=torch.float32),
            torch.tensor(foot_curr[:, pres_idx], dtype=torch.float32),
            # Return targets dictionary
            {
                'body_poses': torch.tensor(gt_all[2:19].reshape(-1), dtype=torch.float32),
                'global_orient': torch.tensor(gt_all[1], dtype=torch.float32),
                'user_id': torch.tensor(user_idx, dtype=torch.long),
                'action_id': torch.tensor(act_id, dtype=torch.long)
            }
        )
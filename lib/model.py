# file: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia.geometry.conversions as K
from pytorch3d import transforms

def axis_angle_to_rot6d(aa: torch.Tensor) -> torch.Tensor:
    rotmat = K.axis_angle_to_rotation_matrix(aa.reshape(-1, 3))[:, :3, :3]
    rot6d = rotmat[..., :2].reshape(aa.shape[:-1] + (6,))
    return rot6d

def rot6d_to_rotmat(rot6d: torch.Tensor) -> torch.Tensor:
    x_raw = rot6d[..., 0:3]
    y_raw = rot6d[..., 3:6]
    x = F.normalize(x_raw, dim=-1, eps=1e-6)
    dot = torch.sum(x * y_raw, dim=-1, keepdim=True)
    y = y_raw - dot * x
    y = F.normalize(y, dim=-1, eps=1e-6)
    z = torch.cross(x, y, dim=-1)
    rotmat = torch.stack((x, y, z), dim=-1)
    return rotmat

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class ConstrainedKinematicAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_joints=18, dropout=0.1, core_joint_ids=None, delta_scale=0.25):
        super().__init__()
        self.num_joints = num_joints
        self.delta_scale = delta_scale
        self.core_joint_ids = core_joint_ids or[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        fixed_bias = torch.zeros(num_joints, num_joints)
        anatomical_connections =[
            (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
            (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
            (9, 12), (12, 13), (12, 14), (12, 15), (13, 16), (14, 17),
        ]
        for i, j in anatomical_connections:
            fixed_bias[i, j] = 1.0
            fixed_bias[j, i] = 1.0
        for i in range(num_joints):
            fixed_bias[i, i] = 2.0
        for i in self.core_joint_ids:
            for j in self.core_joint_ids:
                if i != j:
                    fixed_bias[i, j] += 0.15

        self.register_buffer("fixed_bias", fixed_bias)
        self.delta_bias = nn.Parameter(torch.zeros(num_joints, num_joints))

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size), nn.Dropout(dropout),
        )

    def forward(self, x, core_gate=None):
        identity = x
        bias = self.fixed_bias + self.delta_scale * torch.tanh(self.delta_bias)
        bias = 0.5 * (bias + bias.transpose(0, 1))

        if core_gate is not None:
            x = x.clone()
            gain = 1.0 + 0.25 * core_gate.view(-1, 1, 1)
            x[:, self.core_joint_ids, :] = x[:, self.core_joint_ids, :] * gain

        attn_out, _ = self.self_attn(query=x, key=x, value=x, attn_mask=bias, need_weights=False)
        x = self.norm1(identity + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class ContactPhaseEncoder(nn.Module):
    def __init__(self, pressure_dim, accel_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.pressure_dim = pressure_dim
        feat_dim = pressure_dim * 3 + accel_dim * 2 + hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(),
        )
        self.core_gate_head = nn.Linear(hidden_size, 1)

    def forward(self, positional_input, pressure_input, memory):
        half_p = pressure_input.shape[-1] // 2
        left_p = pressure_input[..., :half_p]
        right_p = pressure_input[..., half_p:]

        left_mean, right_mean = left_p.mean(dim=1), right_p.mean(dim=1)
        left_max, right_max = left_p.max(dim=1).values, right_p.max(dim=1).values
        balance = left_mean - right_mean
        total_load = left_mean + right_mean

        acc_mean, acc_std = positional_input.mean(dim=1), positional_input.std(dim=1)
        mem_pool = memory.mean(dim=1)

        feat = torch.cat([left_mean, right_mean, left_max, right_max, balance, total_load, acc_mean, acc_std, mem_pool], dim=-1)
        z_contact = self.mlp(feat)
        core_gate = torch.sigmoid(self.core_gate_head(z_contact))
        return z_contact, core_gate

class JointQueryConditioner(nn.Module):
    def __init__(self, hidden_size, num_joints, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.base_joint_tokens = nn.Parameter(torch.randn(1, num_joints, hidden_size))
        self.joint_role_embed = nn.Parameter(torch.randn(1, num_joints, hidden_size))
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )
        self.out_norm = nn.LayerNorm(hidden_size)

    def forward(self, z_contact):
        B, H = z_contact.shape
        base = self.base_joint_tokens.expand(B, -1, -1) + self.joint_role_embed.expand(B, -1, -1)
        cond = z_contact.unsqueeze(1).expand(-1, self.num_joints, -1)
        scale, shift = self.cond_mlp(torch.cat([base, cond], dim=-1)).chunk(2, dim=-1)
        queries = base * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        return self.out_norm(queries)

class KineSole(nn.Module):
    def __init__(self, input_size_positional, input_size_pressure, hidden_size,
                 output_size_body_poses=51, output_size_global_orient=3,
                 num_encoder_layers=2, gcn_hidden_size=256, num_heads=8,
                 dropout=0.1, max_seq_len=100, num_joints=18, causal=False, core_joint_ids=None):
        super().__init__()
        self.num_joints = num_joints
        self.causal = causal
        self.core_joint_ids = core_joint_ids or[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        total_input_size = input_size_positional + input_size_pressure

        self.input_encoder = nn.Sequential(
            nn.Linear(total_input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.pos_encoding = PositionalEncoding(hidden_size, max_seq_len)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_encoder_layers)

        self.contact_encoder = ContactPhaseEncoder(pressure_dim=input_size_pressure, accel_dim=input_size_positional, hidden_size=hidden_size, dropout=dropout)
        self.query_conditioner = JointQueryConditioner(hidden_size=hidden_size, num_joints=num_joints, dropout=dropout)

        self.core_decoder = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, activation='gelu', batch_first=True)
        self.core_fuse = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.full_decoder = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout, activation='gelu', batch_first=True)

        self.spatial_refiner = nn.ModuleList([
            ConstrainedKinematicAttention(hidden_size, num_heads, num_joints, dropout, core_joint_ids=self.core_joint_ids),
            ConstrainedKinematicAttention(hidden_size, num_heads, num_joints, dropout, core_joint_ids=self.core_joint_ids),
        ])

        self.pose_head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 6))

    def _build_causal_mask(self, T, device):
        mask = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, positional_input, pressure_input):
        batch_size, T, _ = positional_input.shape
        combined_features = torch.cat([positional_input, pressure_input], dim=-1)
        embedded_input = self.input_encoder(combined_features)
        embedded_input = self.pos_encoding(embedded_input)

        if self.causal:
            causal_mask = self._build_causal_mask(T, embedded_input.device)
            memory = self.temporal_encoder(embedded_input, mask=causal_mask)
        else:
            memory = self.temporal_encoder(embedded_input)

        z_contact, core_gate = self.contact_encoder(positional_input, pressure_input, memory)
        dynamic_joint_queries = self.query_conditioner(z_contact)

        core_queries = dynamic_joint_queries[:, self.core_joint_ids, :]
        core_feats = self.core_decoder(tgt=core_queries, memory=memory)

        full_queries = dynamic_joint_queries.clone()
        full_queries[:, self.core_joint_ids, :] = full_queries[:, self.core_joint_ids, :] + self.core_fuse(core_feats)
        joint_feats = self.full_decoder(tgt=full_queries, memory=memory)

        for block in self.spatial_refiner:
            joint_feats = block(joint_feats, core_gate=core_gate)

        pred_6d = self.pose_head(joint_feats)
        pred_rotmats = rot6d_to_rotmat(pred_6d)
        pred_axis_angle = transforms.matrix_to_axis_angle(pred_rotmats)

        global_orient = pred_axis_angle[:, 0]
        body_pose = pred_axis_angle[:, 1:].reshape(batch_size, -1)

        return body_pose, global_orient
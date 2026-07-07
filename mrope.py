"""
This module implements the Multimodal RoPE (MROPE) positional encoding for vision transformers, 
extending the traditional RoPE to handle 3D positional information (time, height, width) following
the approach from Qwen-VL series.
"""
import torch
import torch.nn as nn

class MultimodalRoPE(nn.Module):

    def __init__(self, head_dim, mrope_section, base = 10000.0):
        super().__init__()

        half_dim = head_dim // 2   # RoPE works on pairs of channels
        if not mrope_section:
            base_dim = half_dim // 3
            remaining = half_dim - base_dim
            height_dim = remaining // 2
            width_dim = half_dim - base_dim - height_dim
            mrope_section = [base_dim, height_dim, width_dim]

        if sum(mrope_section) != half_dim:
            raise ValueError(f"mrope_section must sum to half the head dimension ({half_dim}), got {sum(mrope_section)}")

        self.head_dim = head_dim
        self.mrope_section = mrope_section
        self._split_sizes = mrope_section * 2

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k, positions_3d):
        if positions_3d.dim() != 3 or positions_3d.size(-1) != 3:
            raise ValueError("positions_3d must have shape (batch, seq_len, 3)")

        cos, sin = self._compute_cos_sin(positions_3d)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    def _compute_cos_sin(self, positions_3d):
        batch, _, _ = positions_3d.shape
        position_ids = positions_3d.permute(2, 0, 1)

        inv_freq = self.inv_freq.to(positions_3d.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(3, batch, -1, 1)
        position_ids_expanded = position_ids[:, :, None, :]

        freqs = torch.matmul(inv_freq_expanded.float(), position_ids_expanded.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        cos = self._interleave_axes(cos)
        sin = self._interleave_axes(sin)
        return cos, sin

    def _interleave_axes(self, tensor):
        chunks = tensor.split(self._split_sizes, dim=-1)
        reordered = [chunk[i % 3] for i, chunk in enumerate(chunks)]
        return torch.cat(reordered, dim=-1)

    def _rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat([-x2, x1], dim=-1)

def create_3d_positions_for_images(
    batch_size,
    num_patches_h,
    num_patches_w,
    num_frames = 1,
    device = 'cuda'
):
    positions = []
    
    for t in range(num_frames):
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                positions.append([t, h, w])
    
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    positions = positions.unsqueeze(0).expand(batch_size, -1, -1)

    return positions

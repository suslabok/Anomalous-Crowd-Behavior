import torch
import torch.nn as nn
import torch.nn.functional as F

from vad.models.decoder import TokenDecoder
from vad.models.temporal import TemporalBranch
from vad.models.vit import CrossAttention, SpatialBranch

_TIN_CH = {"flow": 2, "diff": 3, "both": 5}


class CrossFusionBlock(nn.Module):
    """Bidirectional cross-attention: appearance tokens query motion tokens and vice versa (both use the
    pre-update values, so the two directions are symmetric), each followed by its own MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.ln_s, self.ln_m, self.ln_s2, self.ln_m2 = (nn.LayerNorm(dim) for _ in range(4))
        self.ca_s, self.ca_m = CrossAttention(dim, heads), CrossAttention(dim, heads)
        hid = int(dim * mlp_ratio)
        self.mlp_s = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))
        self.mlp_m = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, s, m):
        ns, nm = self.ln_s(s), self.ln_m(m)
        s = s + self.ca_s(ns, nm)               # appearance asks "what is moving here?"
        m = m + self.ca_m(nm, ns)               # motion asks "what is this, and where?"
        s = s + self.mlp_s(self.ln_s2(s))
        m = m + self.mlp_m(self.ln_m2(m))
        return s, m


class DualBranchPredictor(nn.Module):
    """Phase 4: spatial branch (appearance) + temporal branch (motion) + cross-attention fusion -> next frame.

    temporal_input: 'flow' (optical flow, needs data.flow_mode precomputed|online), 'diff' (frame differences,
    computed here, no flow needed) or 'both'. This is the with/without-optical-flow ablation switch.
    The memory bank of normal patterns is inserted between fusion and decoder in Phase 5."""

    def __init__(self, img_size=224, patch=16, clip_len=8, embed_dim=256, depth_spatial=6, depth_temporal=2,
                 depth_fusion=2, heads=8, mlp_ratio=4.0, spatial_frames=1, temporal_input="flow",
                 flow_scale=12.0, residual=True):
        super().__init__()
        if temporal_input not in _TIN_CH:
            raise ValueError(f"temporal_input must be one of {list(_TIN_CH)}")
        self.k, self.tin, self.flow_scale, self.residual = min(spatial_frames, clip_len), temporal_input, flow_scale, residual
        self.spatial = SpatialBranch(img_size, patch, self.k, embed_dim, depth_spatial, heads, mlp_ratio)
        self.temporal = TemporalBranch(img_size, patch, _TIN_CH[temporal_input], clip_len - 1, embed_dim,
                                       depth_temporal, heads, mlp_ratio)
        self.fusion = nn.ModuleList([CrossFusionBlock(embed_dim, heads, mlp_ratio) for _ in range(depth_fusion)])
        self.merge = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.head = TokenDecoder(embed_dim, patch, zero_init=residual)

    def _motion_input(self, clip, flow):
        parts = []
        if self.tin in ("flow", "both"):
            if flow is None:
                raise ValueError("temporal_input='flow' needs optical flow: set data.flow_mode=precomputed "
                                 "(after `python -m vad.data.flow <dataset>`) or data.flow_mode=online")
            b, t1, _, fh, fw = flow.shape
            f = F.interpolate((flow / self.flow_scale).reshape(b * t1, 2, fh, fw), size=clip.shape[-2:],
                              mode="bilinear", align_corners=False)
            parts.append(f.reshape(b, t1, 2, *clip.shape[-2:]))
        if self.tin in ("diff", "both"):
            parts.append(clip[:, 1:] - clip[:, :-1])
        return torch.cat(parts, dim=2)

    def forward(self, clip, flow=None):                 # clip: B,T,C,H,W ; flow: B,T-1,2,F,F
        b, t, c, h, w = clip.shape
        # Appearance branch: what is in the scene (latest frame). The motion branch on the next line: how things move.
        s = self.spatial(clip[:, -self.k:].reshape(b, self.k * c, h, w))        # [B,N,D] appearance
        m = self.temporal(self._motion_input(clip, flow))                       # [B,N,D] motion
        # Let appearance and motion exchange information (cross-attention).
        for blk in self.fusion:
            s, m = blk(s, m)
        # Merge both, decode to pixels, and add the predicted change to the last frame.
        delta = self.head(self.merge(torch.cat([s, m], dim=-1)), self.spatial.grid)
        return clip[:, -1] + delta if self.residual else torch.sigmoid(delta)

    def set_keep_attn(self, flag: bool) -> None:
        """Store attention maps of every attention module (used for localization in Phase 7)."""
        for mod in self.modules():
            if hasattr(mod, "keep_attn"):
                mod.keep_attn = flag

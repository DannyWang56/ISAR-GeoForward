"""Point-cloud heads for the VGGT ISAR reconstruction pipeline.

The spatial patch head keeps a direct path from VGGT patch tokens to output
points.  Unlike a free learned-query decoder, every output point is attached to
an image patch proposal slot.  This makes it harder for surplus points to all
collapse onto the easiest central structure while retaining VGGT's global
multi-view context in each token.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudHead(nn.Module):
    """Learned-query fallback head used by the VGGT reconstruction model."""

    def __init__(self, dim=1024, n_points=2048, n_layers=4, n_heads=8,
                 point_scale=3.0):
        super().__init__()
        self.n_points = n_points
        self.point_scale = point_scale
        self.query = nn.Parameter(torch.randn(1, n_points, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.0, activation='gelu', batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, memory):
        batch_size = memory.shape[0]
        queries = self.query.repeat(batch_size, 1, 1)
        decoded = self.decoder(queries, memory)
        raw = self.out(decoded)
        xyz = torch.tanh(raw[..., :3]) * self.point_scale
        conf_logits = raw[..., 3]
        return xyz, conf_logits


class SpatialPatchPointHead(nn.Module):
    """Expand fused per-view patch tokens into a fixed-size point cloud.

    The design is inspired by spatial token-to-point reconstruction heads, but
    is adapted to four RLOS-conditioned VGGT views.  It avoids a dense
    ``image_pixels -> n_points`` layer and therefore supports arbitrary square
    patch grids with a modest parameter count.
    """

    def __init__(self, dim=512, n_points=5000, n_views=4,
                 patch_start_idx=5, n_layers=2, n_heads=8,
                 point_scale=3.0, dropout=0.0):
        super().__init__()
        if n_points <= 0:
            raise ValueError("n_points must be positive")
        if n_views <= 0:
            raise ValueError("n_views must be positive")
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")

        self.dim = dim
        self.n_points = n_points
        self.n_views = n_views
        self.patch_start_idx = patch_start_idx
        self.point_scale = point_scale

        self.input_norm = nn.LayerNorm(dim)
        self.spatial_embed = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.refiner = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.refiner = nn.Identity()

        # The exact number of slots needed depends on the runtime patch grid.
        # Sixteen covers the current 4 x 9 x 9 grid for a 5000-point output.
        self.max_slots = 32
        self.slot_embed = nn.Parameter(torch.randn(1, 1, self.max_slots, dim) * 0.02)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    @staticmethod
    def _spatial_grid(n_patch, device, dtype):
        side = int(round(math.sqrt(n_patch)))
        if side * side != n_patch:
            raise ValueError(
                f"SpatialPatchPointHead requires a square patch grid; got {n_patch} tokens")
        axis = torch.linspace(-1.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(1, n_patch, 2)

    def forward(self, memory):
        if memory.ndim != 3:
            raise ValueError(f"memory must be (B,L,D), got {tuple(memory.shape)}")
        B, length, dim = memory.shape
        if dim != self.dim:
            raise ValueError(f"memory dim {dim} does not match head dim {self.dim}")
        if length % self.n_views != 0:
            raise ValueError(
                f"token length {length} is not divisible by n_views={self.n_views}")

        tokens_per_view = length // self.n_views
        n_patch = tokens_per_view - self.patch_start_idx
        if n_patch <= 0:
            raise ValueError(
                f"patch_start_idx={self.patch_start_idx} leaves no patch tokens")

        tokens = memory.reshape(B, self.n_views, tokens_per_view, dim)
        patch = tokens[:, :, self.patch_start_idx:]
        grid = self._spatial_grid(n_patch, memory.device, memory.dtype)
        grid_embed = self.spatial_embed(grid).unsqueeze(1)

        patch = self.input_norm(patch) + grid_embed
        patch = patch.reshape(B, self.n_views * n_patch, dim)
        patch = self.refiner(patch)

        n_anchor = patch.shape[1]
        slots_per_anchor = math.ceil(self.n_points / n_anchor)
        if slots_per_anchor > self.max_slots:
            raise ValueError(
                f"Need {slots_per_anchor} slots per patch for {self.n_points} points; "
                f"increase max_slots={self.max_slots}")

        global_context = self.global_proj(patch.mean(dim=1, keepdim=True)).unsqueeze(2)
        proposals = patch.unsqueeze(2)
        proposals = proposals + self.slot_embed[:, :, :slots_per_anchor] + global_context
        raw = self.out(proposals).reshape(B, -1, 4)

        # Evenly retain proposals so all views/patches contribute when the
        # proposal count is slightly larger than the requested fixed count.
        if raw.shape[1] != self.n_points:
            select = torch.linspace(
                0, raw.shape[1] - 1, self.n_points,
                device=raw.device, dtype=torch.float32,
            ).round().long()
            raw = raw.index_select(1, select)

        xyz = torch.tanh(raw[..., :3]) * self.point_scale
        conf_logits = raw[..., 3]
        return xyz, conf_logits


class CoarseToFinePointHead(nn.Module):
    """Decode a structural skeleton, then unfold local surface children.

    Independent global queries can place many surplus points on the same easy
    central structure. Here only the coarse queries attend globally. Each
    coarse point owns a small, bounded group of fine offsets, which gives the
    output a local topology while retaining a fixed point count.
    """

    def __init__(self, dim=512, n_points=5000, n_coarse=1000,
                 upsample_ratio=5, n_layers=3, n_heads=8,
                 point_scale=3.0, fine_offset_scale=0.03):
        super().__init__()
        if n_points <= 0 or n_coarse <= 0 or upsample_ratio <= 0:
            raise ValueError("point counts and upsample_ratio must be positive")
        if n_coarse * upsample_ratio < n_points:
            raise ValueError(
                f"n_coarse*upsample_ratio={n_coarse * upsample_ratio} "
                f"cannot produce n_points={n_points}")
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")

        self.n_points = n_points
        self.n_coarse = n_coarse
        self.upsample_ratio = upsample_ratio
        self.point_scale = point_scale
        self.fine_offset_scale = fine_offset_scale
        self.query = nn.Parameter(torch.randn(1, n_coarse, dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.coarse_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.fine_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, upsample_ratio * 4),
        )

        # Distinct local directions break child symmetry at initialization.
        i = torch.arange(upsample_ratio, dtype=torch.float32)
        z = 1.0 - 2.0 * (i + 0.5) / upsample_ratio
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i
        radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        template = torch.stack(
            [radius * torch.cos(theta), radius * torch.sin(theta), z], dim=-1)
        self.register_buffer("child_template", template.view(1, 1, upsample_ratio, 3))

        # Start fine residuals small so the initial cloud remains near its
        # structural parents while children still have distinct templates.
        nn.init.normal_(self.fine_out[-1].weight, std=1e-3)
        nn.init.zeros_(self.fine_out[-1].bias)
        self.last_coarse_points = None

    def forward(self, memory):
        if memory.ndim != 3:
            raise ValueError(f"memory must be (B,L,D), got {tuple(memory.shape)}")
        batch = memory.shape[0]
        features = self.decoder(self.query.expand(batch, -1, -1), memory)

        coarse_raw = self.coarse_out(features)
        coarse_xyz = torch.tanh(coarse_raw[..., :3]) * self.point_scale
        self.last_coarse_points = coarse_xyz

        fine_raw = self.fine_out(features).reshape(
            batch, self.n_coarse, self.upsample_ratio, 4)
        offset = torch.tanh(
            fine_raw[..., :3] + 0.5 * self.child_template.to(fine_raw.dtype)
        ) * self.fine_offset_scale
        xyz = coarse_xyz.unsqueeze(2) + offset
        conf_logits = fine_raw[..., 3] + coarse_raw[..., 3].unsqueeze(2)

        xyz = xyz.reshape(batch, -1, 3)[:, :self.n_points]
        conf_logits = conf_logits.reshape(batch, -1)[:, :self.n_points]
        return xyz, conf_logits


class RlosPointMapHead(nn.Module):
    """Lift image-conditioned pixels into canonical 3D along known RLOS rays.

    Free learned queries can minimize multi-object Chamfer losses with an
    object-agnostic cloud near the common target center. This head removes that
    shortcut: every output point is attached to a high-response ISAR pixel and
    can only move along its orthographic LOS plus a small in-plane residual.
    """

    requires_geometry_inputs = True
    requires_resolution_scaled_ortho = True

    def __init__(self, dim=512, n_points=5000, n_views=4,
                 patch_start_idx=5, feature_dim=128,
                 depth_center=2.0, depth_range=0.75,
                 offset_scale=0.025, activity_floor=0.08,
                 intensity_gamma=1.5):
        super().__init__()
        if n_points <= 0 or n_views <= 0:
            raise ValueError("n_points and n_views must be positive")
        self.dim = dim
        self.n_points = n_points
        self.n_views = n_views
        self.patch_start_idx = patch_start_idx
        self.depth_center = depth_center
        self.depth_range = depth_range
        self.offset_scale = offset_scale
        if not 0.0 <= activity_floor < 1.0:
            raise ValueError("activity_floor must be in [0, 1)")
        if intensity_gamma <= 0:
            raise ValueError("intensity_gamma must be positive")
        self.activity_floor = activity_floor
        self.intensity_gamma = intensity_gamma

        self.patch_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, feature_dim),
            nn.GELU(),
        )
        self.dense_refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GELU(),
        )
        self.geom_embed = nn.Sequential(
            nn.Linear(7, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, feature_dim),
            nn.GELU(),
        )
        # A sparse ISAR foreground can contain fewer active pixels than the
        # requested point count. Reuse only foreground pixels and give each
        # repeated proposal a learned depth slot instead of pulling white
        # background pixels into the cloud.
        self.max_depth_slots = 32
        self.depth_slot_embed = nn.Parameter(
            torch.randn(self.max_depth_slots, feature_dim) * 0.02)
        self.out = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 4),
        )
        # Start from the physical prior: target center is roughly two camera
        # units along the LOS and has no in-plane residual.
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

        self.last_pixel_indices = None
        self.last_view_ids = None
        self.last_activity = None
        self.last_slot_ids = None
        # Geometry/feature metadata consumed by the optional cross-view
        # consensus head.  Keeping it on the anchor head also makes it easy to
        # diagnose which branch produced inaccurate points.
        self.last_anchor_features = None
        self.last_ray_origins = None
        self.last_ray_directions = None

    def _select_active_pixels(self, activity, k):
        """Select only non-background pixels, repeating them via depth slots."""
        B, S, _ = activity.shape
        all_indices, all_activity, all_slots = [], [], []
        for b in range(B):
            view_indices, view_activity, view_slots = [], [], []
            for s in range(S):
                scores = activity[b, s]
                active = torch.nonzero(scores > 1e-6, as_tuple=False).squeeze(-1)
                if active.numel() == 0:
                    _, idx = torch.topk(scores, k=min(k, scores.numel()), sorted=True)
                    if idx.numel() < k:
                        idx = idx.repeat(math.ceil(k / max(idx.numel(), 1)))[:k]
                    slots = torch.zeros(k, dtype=torch.long, device=scores.device)
                else:
                    order = torch.argsort(scores[active], descending=True)
                    ranked = active[order]
                    if ranked.numel() >= k:
                        idx = ranked[:k]
                        slots = torch.zeros(k, dtype=torch.long, device=scores.device)
                    else:
                        repeats = math.ceil(k / ranked.numel())
                        idx = ranked.repeat(repeats)[:k]
                        slots = torch.arange(k, device=scores.device) // ranked.numel()
                        slots = slots.clamp_max(self.max_depth_slots - 1)
                view_indices.append(idx)
                view_activity.append(scores[idx])
                view_slots.append(slots)
            all_indices.append(torch.stack(view_indices))
            all_activity.append(torch.stack(view_activity))
            all_slots.append(torch.stack(view_slots))
        return (torch.stack(all_indices), torch.stack(all_activity),
                torch.stack(all_slots))

    @staticmethod
    def _canonical_ray_geometry(rays):
        """Convert world ray position/direction fields to view-0 coordinates."""
        if rays.ndim != 5 or rays.shape[2] != 6:
            raise ValueError(f"expected rays (B,S,6,H,W), got {tuple(rays.shape)}")
        los_world = rays[:, :, :3].permute(0, 1, 3, 4, 2).float()
        pos_world = rays[:, :, 3:].permute(0, 1, 3, 4, 2).float()
        if pos_world.shape[2] < 2 or pos_world.shape[3] < 2:
            raise ValueError("ray grid must be at least 2x2")

        t_ref = pos_world[:, 0].mean(dim=(1, 2))
        ex_ref = (pos_world[:, 0, :, 1:] - pos_world[:, 0, :, :-1]).mean(dim=(1, 2))
        ey_ref = (pos_world[:, 0, 1:] - pos_world[:, 0, :-1]).mean(dim=(1, 2))
        ex_ref = F.normalize(ex_ref, dim=-1)
        ey_ref = ey_ref - (ey_ref * ex_ref).sum(dim=-1, keepdim=True) * ex_ref
        ey_ref = F.normalize(ey_ref, dim=-1)
        ez_ref = F.normalize(los_world[:, 0].mean(dim=(1, 2)), dim=-1)
        basis_ref = torch.stack([ex_ref, ey_ref, ez_ref], dim=-1)
        world_to_ref = basis_ref.transpose(1, 2)

        pos_centered = pos_world - t_ref[:, None, None, None]
        pos_canon = torch.einsum("bij,bshwj->bshwi", world_to_ref, pos_centered)
        los_canon = torch.einsum("bij,bshwj->bshwi", world_to_ref, los_world)
        los_canon = F.normalize(los_canon, dim=-1)

        ex_world = (pos_world[:, :, :, 1:] - pos_world[:, :, :, :-1]).mean(dim=(2, 3))
        ey_world = (pos_world[:, :, 1:] - pos_world[:, :, :-1]).mean(dim=(2, 3))
        ex_canon = torch.einsum("bij,bsj->bsi", world_to_ref, ex_world)
        ey_canon = torch.einsum("bij,bsj->bsi", world_to_ref, ey_world)
        ex_canon = F.normalize(ex_canon, dim=-1)
        ey_canon = F.normalize(ey_canon, dim=-1)
        return pos_canon, los_canon, ex_canon, ey_canon

    def forward(self, memory, images=None, rays=None, alphas=None):
        if images is None or rays is None:
            raise ValueError("RlosPointMapHead requires images and ISAR rays")
        B, S, _, H, W = images.shape
        if S != self.n_views:
            raise ValueError(f"expected {self.n_views} views, got {S}")
        if memory.shape[1] % S != 0:
            raise ValueError("memory token count is not divisible by n_views")

        tokens_per_view = memory.shape[1] // S
        tokens = memory.reshape(B, S, tokens_per_view, self.dim)
        patch = tokens[:, :, self.patch_start_idx:]
        side = int(round(math.sqrt(patch.shape[2])))
        if side * side != patch.shape[2]:
            raise ValueError(f"point-map head requires square patch grid, got {patch.shape[2]}")

        dense = self.patch_proj(patch)
        dense = dense.reshape(B * S, side, side, -1).permute(0, 3, 1, 2)
        dense = F.interpolate(dense, size=(H, W), mode="bilinear", align_corners=False)
        dense = self.dense_refine(dense)
        dense = dense.reshape(B, S, dense.shape[1], H * W).transpose(2, 3)

        rgb = images.float().clamp(0.0, 1.0)
        # The simulated PNGs use MATLAB's hot colormap: black/red is weak and
        # yellow/white is strong. Background has already been separated by
        # alpha, so ranking with 1-RGB would invert the radar intensity and
        # preferentially select dark sidelobes.
        signal = (0.299 * rgb[:, :, 0]
                  + 0.587 * rgb[:, :, 1]
                  + 0.114 * rgb[:, :, 2])
        if alphas is not None:
            alpha = alphas.float().squeeze(2).clamp(0.0, 1.0)
            masked_signal = signal * alpha
            peak = masked_signal.flatten(2).amax(dim=-1).clamp_min(1e-6)
            relative = masked_signal / peak[:, :, None, None]
            activity = alpha * relative.pow(self.intensity_gamma)
            activity = torch.where(
                relative >= self.activity_floor, activity,
                torch.zeros_like(activity))

            # Retain a safe fallback for an empty/very weak frame.
            empty = activity.flatten(2).amax(dim=-1) <= 0
            if empty.any():
                fallback = alpha * relative.clamp_min(1e-4)
                activity = torch.where(
                    empty[:, :, None, None], fallback, activity)
        else:
            peak = signal.flatten(2).amax(dim=-1).clamp_min(1e-6)
            relative = signal / peak[:, :, None, None]
            activity = torch.where(
                relative >= self.activity_floor,
                relative.pow(self.intensity_gamma),
                torch.zeros_like(relative))
        activity = activity.reshape(B, S, H * W)

        points_per_view = math.ceil(self.n_points / S)
        points_per_view = min(points_per_view, H * W)
        selected_idx, selected_activity, selected_slots = \
            self._select_active_pixels(activity, points_per_view)
        gather_feat = selected_idx.unsqueeze(-1).expand(-1, -1, -1, dense.shape[-1])
        selected_feat = torch.gather(dense, 2, gather_feat)

        pos_canon, los_canon, ex_canon, ey_canon = self._canonical_ray_geometry(rays)
        pos_flat = pos_canon.reshape(B, S, H * W, 3)
        los_flat = los_canon.reshape(B, S, H * W, 3)
        gather_xyz = selected_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        anchor = torch.gather(pos_flat, 2, gather_xyz)
        los = torch.gather(los_flat, 2, gather_xyz)

        geom = torch.cat([anchor, los, selected_activity.unsqueeze(-1)], dim=-1)
        global_context = self.global_proj(tokens.mean(dim=(1, 2)))[:, None, None]
        slot_embed = self.depth_slot_embed[selected_slots]
        fused = selected_feat + self.geom_embed(geom) + global_context + slot_embed
        raw = self.out(fused)

        du = torch.tanh(raw[..., 0]) * self.offset_scale
        dv = torch.tanh(raw[..., 1]) * self.offset_scale
        depth = self.depth_center + torch.tanh(raw[..., 2]) * self.depth_range
        ex = ex_canon[:, :, None]
        ey = ey_canon[:, :, None]
        xyz = (anchor + du.unsqueeze(-1) * ex + dv.unsqueeze(-1) * ey
               + depth.unsqueeze(-1) * los)
        conf_logits = raw[..., 3] + 2.0 * (selected_activity - 0.5)

        view_ids = torch.arange(S, device=memory.device).view(1, S, 1)
        view_ids = view_ids.expand(B, S, points_per_view)
        xyz = xyz.reshape(B, -1, 3)[:, :self.n_points]
        conf_logits = conf_logits.reshape(B, -1)[:, :self.n_points]
        self.last_pixel_indices = selected_idx.reshape(B, -1)[:, :self.n_points]
        self.last_view_ids = view_ids.reshape(B, -1)[:, :self.n_points]
        self.last_activity = selected_activity.reshape(B, -1)[:, :self.n_points]
        self.last_slot_ids = selected_slots.reshape(B, -1)[:, :self.n_points]
        self.last_anchor_features = fused.reshape(B, -1, fused.shape[-1])[
            :, :self.n_points]
        self.last_ray_origins = anchor.reshape(B, -1, 3)[:, :self.n_points]
        self.last_ray_directions = los.reshape(B, -1, 3)[:, :self.n_points]
        return xyz.to(memory.dtype), conf_logits.to(memory.dtype)

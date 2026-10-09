"""VGGT/RLOS backbone, LoRA configuration, and safe weight loading.

This is a selected model component, not the complete reconstruction pipeline.
The point decoder, training loop, losses, and trained weights are not included.
VGGT and ``minlora`` are external dependencies and are not redistributed here.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


log = logging.getLogger(__name__)


class VggtRlosBackbone(nn.Module):
    """Fuse six-channel ISAR/RLOS rays with the final VGGT aggregator tokens.

    ``images`` has shape ``[B, S, 3, H, W]`` and ``rays`` has shape
    ``[B, S, 6, H, W]``. The first three ray channels are direction and the
    last three encode the corresponding RLOS-plane position.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        embed_dim: int = 1024,
        patch_size: int = 14,
        init_scale: float = 0.02,
        n_views: int = 4,
        fusion: str = "film",
        view_embed_mode: str = "refsrc",
        use_rd_pose_token: bool = False,
        rd_pose_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if fusion not in {"film", "add"}:
            raise ValueError("fusion must be 'film' or 'add'")
        if view_embed_mode not in {"perview", "shared", "refsrc"}:
            raise ValueError("view_embed_mode must be perview, shared, or refsrc")
        if n_views < 1:
            raise ValueError("n_views must be positive")

        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.inner_dim = 2 * embed_dim
        self.patch_size = patch_size
        self.n_views = n_views
        self.fusion = fusion
        self.view_embed_mode = view_embed_mode
        self.use_rd_pose_token = use_rd_pose_token

        self.ray_embed = nn.Conv2d(
            6, self.inner_dim, kernel_size=patch_size, stride=patch_size,
            padding=0, bias=False,
        )
        nn.init.normal_(self.ray_embed.weight, std=init_scale)
        self.ray_norm = nn.LayerNorm(self.inner_dim)

        if fusion == "film":
            self.film_gamma = nn.Linear(self.inner_dim, self.inner_dim)
            self.film_beta = nn.Linear(self.inner_dim, self.inner_dim)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        if view_embed_mode == "perview":
            self.view_embed = nn.Parameter(
                torch.randn(1, n_views, self.inner_dim) * 0.02
            )
        elif view_embed_mode == "shared":
            self.src_embed = nn.Parameter(
                torch.randn(1, 1, self.inner_dim) * 0.02
            )
        else:
            self.ref_embed = nn.Parameter(
                torch.randn(1, 1, self.inner_dim) * 0.02
            )
            self.src_embed = nn.Parameter(
                torch.randn(1, 1, self.inner_dim) * 0.02
            )

        if use_rd_pose_token:
            self.rd_pose_proj = nn.Sequential(
                nn.LayerNorm(11 + n_views),
                nn.Linear(11 + n_views, rd_pose_hidden_dim),
                nn.GELU(),
                nn.Linear(rd_pose_hidden_dim, self.inner_dim),
            )
            nn.init.normal_(self.rd_pose_proj[-1].weight, std=0.01)
            nn.init.zeros_(self.rd_pose_proj[-1].bias)

    def build_view_embed(self, n: int) -> torch.Tensor:
        if self.view_embed_mode == "perview":
            if n <= self.n_views:
                return self.view_embed[:, :n]
            extra = self.view_embed[:, -1:].repeat(1, n - self.n_views, 1)
            return torch.cat([self.view_embed, extra], dim=1)
        if self.view_embed_mode == "shared":
            return self.src_embed.repeat(1, n, 1)
        src = self.src_embed.repeat(1, max(n - 1, 0), 1)
        return torch.cat([self.ref_embed, src], dim=1)

    def build_rd_pose_token(self, rays: torch.Tensor) -> torch.Tensor:
        """Express each RLOS frame relative to the first input view."""
        if not self.use_rd_pose_token:
            raise RuntimeError("RD pose token was disabled at construction")
        _, n_views, _, _, _ = rays.shape
        if n_views > self.n_views:
            raise ValueError("RD pose token received more views than configured")

        direction = F.normalize(rays[:, :, :3].mean(dim=(-1, -2)), dim=-1)
        position = rays[:, :, 3:].permute(0, 1, 3, 4, 2).float()
        du_raw = (position[:, :, :, 1:] - position[:, :, :, :-1]).mean(dim=(2, 3))
        dv_raw = (position[:, :, 1:] - position[:, :, :-1]).mean(dim=(2, 3))
        du_scale = du_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dv_scale = dv_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis_u = F.normalize(du_raw, dim=-1)
        axis_v = F.normalize(dv_raw, dim=-1)

        ref_basis = torch.stack(
            [axis_u[:, 0], axis_v[:, 0], direction[:, 0]], dim=-1
        )
        world_to_ref = ref_basis.transpose(1, 2)
        axis_u = torch.einsum("bij,bsj->bsi", world_to_ref, axis_u)
        axis_v = torch.einsum("bij,bsj->bsi", world_to_ref, axis_v)
        axis_n = torch.einsum("bij,bsj->bsi", world_to_ref, direction)
        pairwise = torch.einsum("bsi,bti->bst", direction, direction)
        if n_views < self.n_views:
            pairwise = F.pad(pairwise, (0, self.n_views - n_views))
        scale = torch.cat(
            [torch.log1p(100.0 * du_scale), torch.log1p(100.0 * dv_scale)],
            dim=-1,
        )
        pose = torch.cat([axis_u, axis_v, axis_n, scale, pairwise], dim=-1)
        return self.rd_pose_proj(pose.to(self.rd_pose_proj[1].weight.dtype))

    def forward(self, images: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or rays.ndim != 5:
            raise ValueError("images and rays must have shape [B, S, C, H, W]")
        batch, n_views, image_channels, height, width = images.shape
        if image_channels != 3 or rays.shape != (batch, n_views, 6, height, width):
            raise ValueError("expected RGB images and matching six-channel rays")

        token_layers, patch_start_idx = self.aggregator(images)
        tokens = token_layers[-1]
        _, _, token_count, dim = tokens.shape
        if dim != self.inner_dim:
            raise ValueError("VGGT token dimension does not match embed_dim")

        rays_flat = rays.reshape(batch * n_views, 6, height, width)
        ray_tokens = self.ray_embed(rays_flat).flatten(2).transpose(1, 2)
        ray_tokens = self.ray_norm(ray_tokens)
        ray_tokens = ray_tokens.reshape(batch, n_views, ray_tokens.shape[1], dim)

        n_patch = min(token_count - patch_start_idx, ray_tokens.shape[2])
        ray_full = tokens.new_zeros(batch, n_views, token_count, dim)
        ray_full[:, :, patch_start_idx:patch_start_idx + n_patch] = (
            ray_tokens[:, :, :n_patch]
        )

        if self.fusion == "film":
            tokens = tokens * (1.0 + self.film_gamma(ray_full)) + self.film_beta(ray_full)
        else:
            tokens = tokens + ray_full
        tokens = tokens + self.build_view_embed(n_views).unsqueeze(2)
        if self.use_rd_pose_token:
            tokens = tokens + self.build_rd_pose_token(rays).unsqueeze(2)
        return tokens.reshape(batch, n_views * token_count, dim)


def enable_lora_finetuning(
    backbone: VggtRlosBackbone, rank: int = 16, target: str = "aggregator"
) -> VggtRlosBackbone:
    """Freeze base VGGT and enable LoRA plus the RLOS fusion parameters.

    This mirrors the archived model-assembly protocol; it does not provide a
    training loop or optimizer. ``target`` is aggregator, patch, or all.
    """
    if rank < 0 or target not in {"aggregator", "patch", "all"}:
        raise ValueError("rank must be non-negative; target must be aggregator, patch, or all")
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    if rank:
        try:
            from minlora import LoRAParametrization, add_lora
        except ImportError as exc:
            raise ImportError("LoRA setup requires the external 'minlora' package") from exc
        config = {
            nn.Linear: {"weight": partial(LoRAParametrization.from_linear, rank=rank)},
            nn.Conv2d: {"weight": partial(LoRAParametrization.from_conv2d, rank=rank)},
        }
        if target in {"aggregator", "all"}:
            add_lora(backbone.aggregator.frame_blocks, lora_config=config)
            add_lora(backbone.aggregator.global_blocks, lora_config=config)
        if target in {"patch", "all"}:
            add_lora(backbone.aggregator.patch_embed, lora_config=config)

    for module in (backbone.ray_embed, backbone.ray_norm):
        for parameter in module.parameters():
            parameter.requires_grad = True
    if backbone.fusion == "film":
        for module in (backbone.film_gamma, backbone.film_beta):
            for parameter in module.parameters():
                parameter.requires_grad = True
    for name in ("view_embed", "ref_embed", "src_embed"):
        if hasattr(backbone, name):
            getattr(backbone, name).requires_grad = True
    return backbone


def _read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: object = load_file(str(path), device="cpu")
    elif path.suffix in {".pt", ".pth", ".ckpt"}:
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("weights must be .safetensors, .pt, .pth, or .ckpt")

    for key in ("model", "state_dict"):
        if isinstance(state, Mapping) and key in state and isinstance(state[key], Mapping):
            state = state[key]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
           for k, v in state.items()):
        raise ValueError("checkpoint state_dict must contain tensor values")
    return dict(state)


def _normalized_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name in normalized:
            raise ValueError(f"duplicate checkpoint key after prefix removal: {name}")
        normalized[name] = tensor
    return normalized


def load_model_weights(
    model: nn.Module, path: str | Path, *, strict: bool = True
) -> dict[str, object]:
    """Load matching weights and report omissions, including shape mismatches.

    With ``strict=True`` (the default), no tensor is copied unless the full
    checkpoint matches the supplied module. ``strict=False`` permits a partial
    load, but callers must inspect the returned report before using the model.
    """
    state = _normalized_keys(_read_state_dict(path))
    own = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in own and own[k].shape == v.shape}
    shape_mismatch = sorted(k for k, v in state.items()
                            if k in own and own[k].shape != v.shape)
    unexpected = sorted(k for k in state if k not in own)
    missing = sorted(k for k in own if k not in matched)
    report: dict[str, object] = {
        "loaded": len(matched),
        "checkpoint_tensors": len(state),
        "loaded_fraction": len(matched) / len(state),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }
    if not matched:
        raise ValueError("checkpoint has no tensors matching this model")
    if strict and (missing or unexpected or shape_mismatch):
        raise ValueError(
            "checkpoint does not fully match model: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatch={len(shape_mismatch)}"
        )
    model.load_state_dict(matched, strict=False)
    log.info("Loaded %s/%s checkpoint tensors from %s", len(matched), len(state), path)
    return report


def load_vggt_weights(vggt: nn.Module, path: str | Path) -> dict[str, object]:
    """Load the public VGGT checkpoint, requiring every aggregator tensor."""
    state = _normalized_keys(_read_state_dict(path))
    own = vggt.state_dict()
    matched = {k: v for k, v in state.items()
               if k in own and own[k].shape == v.shape}
    aggregator_missing = sorted(k for k in own
                                if k.startswith("aggregator.") and k not in matched)
    if not matched or aggregator_missing:
        raise ValueError(
            f"VGGT aggregator checkpoint is incomplete: "
            f"{len(aggregator_missing)} aggregator tensors missing"
        )
    vggt.load_state_dict(matched, strict=False)
    report: dict[str, object] = {
        "loaded": len(matched),
        "checkpoint_tensors": len(state),
        "aggregator_missing": aggregator_missing,
        "unexpected": sorted(k for k in state if k not in own),
    }
    log.info("Loaded %s VGGT tensors from %s", len(matched), path)
    return report


def build_vggt_rlos_backbone(
    vggt_repo: str | Path,
    vggt_checkpoint: str | Path | None = None,
    *,
    image_size: int = 518,
    patch_size: int = 14,
    embed_dim: int = 1024,
    n_views: int = 4,
    fusion: str = "film",
    view_embed_mode: str = "refsrc",
    use_rd_pose_token: bool = False,
    rd_pose_hidden_dim: int = 256,
    ray_init_scale: float = 0.02,
    lora_rank: int = 16,
    lora_target: str = "aggregator",
) -> VggtRlosBackbone:
    """Construct the VGGT feature extractor and attach RLOS/LoRA modules.

    The VGGT source and its pretrained weights are external; this repository
    does not bundle either one. A missing checkpoint leaves VGGT random and is
    therefore not suitable for reconstruction experiments.
    """
    repo = Path(vggt_repo).resolve()
    if not (repo / "vggt" / "models" / "vggt.py").is_file():
        raise FileNotFoundError(f"VGGT source not found in {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from vggt.models.vggt import VGGT

    vggt = VGGT(
        img_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
        enable_camera=False, enable_point=False, enable_depth=False,
        enable_track=False,
    )
    if vggt_checkpoint is None:
        log.warning("VGGT checkpoint not supplied; backbone is randomly initialized")
    else:
        load_vggt_weights(vggt, vggt_checkpoint)

    backbone = VggtRlosBackbone(
        vggt.aggregator, embed_dim=embed_dim, patch_size=patch_size,
        init_scale=ray_init_scale, n_views=n_views, fusion=fusion,
        view_embed_mode=view_embed_mode,
        use_rd_pose_token=use_rd_pose_token,
        rd_pose_hidden_dim=rd_pose_hidden_dim,
    )
    return enable_lora_finetuning(backbone, rank=lora_rank, target=lora_target)

"""
renderer.py
=================
ISAR 可微分渲染器 v2（minimum viable，基于 ISAR-GS 论文 Eq.7, 14, 15）。

与 v1 改进:
  - ★ 各向异性 2D 协方差（用 scale + rotation 计算 Σ_3D → Σ_2D）
  - ★ 单通道散射强度（物理正确，从 SH 取均值）
  - ★ 内置 hot colormap 可视化（与 GT PNG 数据生成方式匹配）

物理模型（ISAR-GS Eq.15 简化版，省略 transmittance）:
  I(a, b) = Σ_i σ_i · α_i · exp(-0.5 · δ_i)

  其中:
    σ_i ∈ R+:  散射强度（标量，从 SH 取均值）
    α_i ∈ [0,1]: 不透明度（sigmoid）
    δ_i = (x_pix - μ_proj)^T Σ_2D^{-1} (x_pix - μ_proj):  2D 马氏距离

  各向异性协方差投影（ISAR-GS Eq.14）:
    Σ_3D_world = R · S·S^T · R^T   （来自 scale + rotation 四元数）
    Σ_3D_cam   = R_w2c · Σ_3D_world · R_w2c^T   （转到相机系）
    Σ_2D       = ortho_scale² · Σ_3D_cam[:2,:2]  （正交投影到图像）

简化点:
  - 省略 transmittance T(x)（前景遮挡的能量衰减）
  - 用标量强度替代完整 SH

用法:
  renderer = ISARRenderer(image_size=512)
  intensity = renderer.render(gaussians, c2w, ortho_scale)
  rgb_viz = renderer.colorize(intensity)   # 用 hot colormap 转 RGB
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint


# ─── 四元数 → 旋转矩阵 ────────────────────────────────────────────────────────

def quat_to_rotmat(q):
    """
    q: (..., 4) 四元数 (w, x, y, z) 格式
    返回: (..., 3, 3) 旋转矩阵
    """
    # 归一化
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),   1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


# ─── ISAR 渲染器 ─────────────────────────────────────────────────────────────

class ISARRenderer(nn.Module):
    """
    各向异性 ISAR 正交可微分渲染器。

    输入: FreeSplatter latent (N, C) [xyz(3), sh(sh_dim), opacity(1), scale(2), rot(4)]
          FreeSplatter-2dgs 默认: C=22, sh_dim=12 (SH degree 1, 4 系数 × 3 通道)
    输出: 单通道散射强度图 (H, W)
    """

    def __init__(self, image_size=512, sh_dim=None, scale_dim=2,
                  bbox_sigma=3.0):
        super().__init__()
        self.H = image_size
        self.W = image_size
        self.sh_dim = sh_dim   # None = 自动从 latent 总维度推断
        self.scale_dim = scale_dim
        self.bbox_sigma = bbox_sigma   # 渲染范围 = N 倍标准差

    def split_latent(self, latent, use_sh_intensity=False):
        """解析 FreeSplatter latent（自动推断 sh_dim）

        use_sh_intensity:
          False (默认) → intensity = opacity（旧行为，opacity 双重担任 σ 和 α）
          True         → intensity = sigmoid(SH[0])（σ/α 解耦，SH 第 0 维专职散射强度）
        """
        # 自动推断 sh_dim: total - 3(xyz) - 1(opacity) - scale_dim - 4(rot)
        if self.sh_dim is None:
            total_dim = latent.shape[-1]
            self.sh_dim = total_dim - 3 - 1 - self.scale_dim - 4
            assert self.sh_dim > 0, f'latent 维度 {total_dim} 太小'

        xyz, _sh, opacity, scale, rotation = latent.split(
            [3, self.sh_dim, 1, self.scale_dim, 4], dim=-1)

        opacity = opacity.sigmoid().squeeze(-1)  # (N,) 几何存在性
        # ★ σ/α 解耦：use_sh_intensity=True 时让 SH 第 0 维当散射强度，
        #   opacity 专职"几何存在性"。这样 12 维 SH capacity 不再闲置。
        if use_sh_intensity and self.sh_dim >= 1:
            intensity = _sh[..., 0].sigmoid()    # (N,) ISAR 散射强度
        else:
            intensity = opacity.clone()          # (N,) 旧行为：opacity 兼任 σ
        scale_3d = scale.exp()                  # (N, scale_dim)
        # 2DGS: 只有 2 个 scale，第三个补一个小值（薄盘）
        if self.scale_dim == 2:
            scale_z = torch.ones_like(scale_3d[:, :1]) * 0.001
            scale_3d = torch.cat([scale_3d, scale_z], dim=-1)   # (N, 3)
        return xyz, intensity, opacity, scale_3d, rotation

    def build_3d_covariance(self, scale, rotation):
        """
        Σ_3D = R · diag(s^2) · R^T  （从 quat 构建）

        scale:    (N, 3) 已 exp，正值
        rotation: (N, 4) 四元数 (w, x, y, z)
        返回: (N, 3, 3) 3D 协方差
        """
        R = quat_to_rotmat(rotation)
        return self.build_3d_covariance_from_R(scale, R)

    def build_3d_covariance_from_R(self, scale, R):
        """从已有旋转矩阵直接构建协方差（避免重复 quat→matrix 转换）"""
        S2 = (scale ** 2).unsqueeze(-1)     # (N, 3, 1) 对角项
        Sigma = R @ (S2 * R.transpose(-1, -2))   # (N, 3, 3)
        return Sigma

    def _prepare_gaussians(self, latent, use_sh_intensity=False):
        """Decode view-independent attributes for one render call.

        The established v2 renderer recomputes these tensors every call. A
        derived inference renderer can cache them without changing the render
        equation; gradient-enabled training keeps this differentiable path.
        """
        xyz, intensity, opacity, scale_3d, rotation = self.split_latent(
            latent, use_sh_intensity=use_sh_intensity)
        R_gauss = quat_to_rotmat(rotation)
        Sigma_3D = self.build_3d_covariance_from_R(scale_3d, R_gauss)
        return xyz, intensity, opacity, R_gauss, Sigma_3D

    def _pixel_grid(self, device, y0=0, y1=None, x0=0, x1=None):
        """Build a float32 pixel-centre grid (uncached v2 behaviour)."""
        y1 = self.H if y1 is None else int(y1)
        x1 = self.W if x1 is None else int(x1)
        px = torch.arange(int(x0), x1, device=device).float() + 0.5
        py = torch.arange(int(y0), y1, device=device).float() + 0.5
        return torch.meshgrid(py, px, indexing='ij')

    def _build_tile_bins(self, u_pix, v_pix, radius, tile_size):
        """Optional per-frame sparse Gaussian-to-tile index.

        The established v2/v3 path returns ``None`` and performs its original
        per-tile boolean cull. A derived renderer may return one LongTensor of
        Gaussian indices per tile, in row-major tile order.
        """
        return None

    def render(self, latent, c2w, ortho_scale, chunk_size=2000,
                opacity_thresh=0.01, return_alpha=False,
                use_shading=True, shading_weight=1.0,
                use_sh_intensity=False,
                use_directional_sh_intensity=False,
                use_quadratic_directional_sh_intensity=False,
                use_energy_attenuation=False,
                attenuation_strength=0.1,
                use_psf_conv=False, psf_sigma_px=1.5, psf_kernel_size=11,
                use_grad_checkpoint=True,
                tile_size=0):
        """
        渲染单视角 ISAR 强度图。

        latent:        (N, 14) Gaussian 参数（canonical 系下）
        c2w:           (4, 4) 该视角的相机外参（canonical 系）
        ortho_scale:   float 像素/世界单位
        chunk_size:    分块大小（控显存）
        opacity_thresh: 滤掉小 opacity 的 Gaussian
        use_shading:   ★ 启用 G2-SDF 风格的 facing-radar shading 因子
                       (max(-LOS · normal, 0), 模拟 bright vs dark area)
        shading_weight: shading 权重（0=不用，1=完全用）
        use_sh_intensity: ★ σ/α 解耦开关
                       False → 旧行为：intensity = opacity（双重计数）
                       True  → 新行为：intensity = sigmoid(SH[0])，opacity 专职几何
        use_energy_attenuation: ★ 启用 ISAR-GS 风格 transmittance（前到后合成）
                       按深度排序 Gaussian，每个 chunk 累积 T_pixel
                       近端 Gaussian 衰减远端的可见性，模拟自遮挡
        attenuation_strength: 能量衰减强度 [0,1]（默认 0.1）
                       0=完全加性（等价于禁用），1=完整 alpha compositing（过激）
                       0.1 是推荐值：FreeSplatter 像素对齐 Gaussian 叠层严重，
                       全量 compositing 会让后方 Gaussian 梯度归零
        use_psf_conv:  ★ 渲染后用 sinc-Gaussian PSF 做 2D 卷积（RNSplat 风格）
        psf_sigma_px:  PSF 主瓣宽度（像素，默认 1.5）
        psf_kernel_size: PSF kernel 尺寸（默认 11×11，应为奇数）

        返回:
          intensity: (H, W) 单通道强度图 [0, +∞)（未截断）
          (可选) alpha_map: (H, W) 累积不透明度
        """
        device = latent.device

        # === 1. 解析 Gaussian 参数 ===
        xyz, intensity, opacity, R_gauss, Sigma_3D = \
            self._prepare_gaussians(
                latent, use_sh_intensity=use_sh_intensity)

        # === 2. 转到相机系 ===
        w2c = torch.linalg.inv(c2w)                                 # (4, 4)
        R_w2c = w2c[:3, :3]                                          # (3, 3)
        t_w2c = w2c[:3, 3]                                           # (3,)
        xyz_cam = xyz @ R_w2c.T + t_w2c                              # (N, 3)

        if use_directional_sh_intensity and self.sh_dim >= 4:
            # First-order scalar SH-like response in the canonical frame.
            # The camera forward direction is shared by all points under the
            # orthographic ISAR model, so this adds only three coefficients per
            # Gaussian and remains inexpensive in the sparse renderer.
            sh = latent[:, 3:3 + self.sh_dim]
            view_dir = c2w[:3, 2]
            view_dir = view_dir / view_dir.norm().clamp(min=1e-8)
            directional_logit = sh[:, 0] + (
                sh[:, 1:4] * view_dir[None]).sum(dim=-1)
            if (use_quadratic_directional_sh_intensity and
                    self.sh_dim >= 9):
                x, y, z = view_dir.unbind()
                quadratic_basis = torch.stack((
                    x * y, y * z, z * x, x.square() - y.square(),
                    3.0 * z.square() - 1.0))
                directional_logit = directional_logit + (
                    sh[:, 4:9] * quadratic_basis[None]).sum(dim=-1)
            intensity = directional_logit.sigmoid()

        # === 2.5. ★ G2-SDF 风格 shading 因子 ===
        # 2DGS Gaussian 是平面盘，法向 = 旋转矩阵第 3 列（disk 的 z 轴）
        # 相机 LOS 方向（雷达 → 目标）= 相机系 +z 方向
        # shading = max(-LOS · normal_cam, 0) = max(-normal_cam[2], 0)
        # 物理含义: 圆盘"面对雷达"的程度（背对的接收不到入射波）
        if use_shading:
            normal_world = R_gauss[:, :, 2]                          # (N, 3) 2DGS 圆盘法向
            normal_cam = normal_world @ R_w2c.T                       # (N, 3) 相机系
            # LOS = +z in camera frame; 入射波方向 -LOS = -z
            # bright area: -K_EM · K_n > 0 → -(- z) · normal = z · normal > 0
            # 即 normal_cam[2] > 0 (法向指向相机)
            shading_factor = normal_cam[:, 2].clamp(min=0)            # (N,)
            # 混合: 1 - w + w · shading（w=0 退化为均匀，w=1 完全 shading）
            shading_factor = (1.0 - shading_weight) + shading_weight * shading_factor
        else:
            shading_factor = torch.ones_like(opacity)
        # 协方差转相机系: Σ_cam = R_w2c · Σ_world · R_w2c^T
        Sigma_3D_cam = R_w2c.unsqueeze(0) @ Sigma_3D @ R_w2c.T.unsqueeze(0)
        # 取 2D 子矩阵（前两行/列） + 正交缩放
        Sigma_2D = (ortho_scale ** 2) * Sigma_3D_cam[:, :2, :2]      # (N, 2, 2)

        # ★ 关键修复: 加正则避免奇异（2DGS 薄盘投影易出现 det≈0）
        # 至少 0.5 像素的标准差（最小 Gaussian 宽度）
        eye_2 = torch.eye(2, device=device).unsqueeze(0)
        Sigma_2D = Sigma_2D + eye_2 * 0.25   # 0.25 = (0.5 px)^2

        # === 3. 投影中心到像素 ===
        u_pix = xyz_cam[:, 0] * ortho_scale + self.W / 2.0           # (N,)
        v_pix = xyz_cam[:, 1] * ortho_scale + self.H / 2.0

        # === 4. 滤可见 ===
        # 计算 bounding box 半径（最大特征值开方 × bbox_sigma）
        # det = a*d - b*c; trace = a + d; eigvals = (tr ± √(tr²-4det))/2
        a = Sigma_2D[:, 0, 0]
        b = Sigma_2D[:, 0, 1]
        d = Sigma_2D[:, 1, 1]
        trace = a + d
        det = a * d - b * b
        disc = (trace ** 2 - 4 * det).clamp(min=0)
        lam_max = 0.5 * (trace + disc.sqrt())                        # (N,) 最大特征值
        radius = self.bbox_sigma * lam_max.clamp(min=1e-4).sqrt()    # (N,) 像素半径
        radius = radius.clamp(max=20.0)                              # 防止极端值

        valid = (opacity > opacity_thresh) & \
                 (u_pix > -radius) & (u_pix < self.W + radius) & \
                 (v_pix > -radius) & (v_pix < self.H + radius) & \
                 (intensity > 1e-3)

        if not valid.any():
            # Connect to parameters so loss.backward() has a grad_fn
            diff_zero = opacity.sum() * 0.0 + intensity.sum() * 0.0
            zeros = torch.zeros(self.H, self.W, device=device) + diff_zero
            return (zeros, zeros) if return_alpha else zeros

        u_pix = u_pix[valid]
        v_pix = v_pix[valid]
        intensity = intensity[valid]
        opacity = opacity[valid]
        shading_factor = shading_factor[valid]   # ★ G2-SDF shading
        Sigma_2D = Sigma_2D[valid]
        radius = radius[valid]
        z_cam = xyz_cam[:, 2][valid]              # ★ 深度（沿 LOS）

        # ★ Module A: 标准 3DGS 风格 per-pixel per-Gaussian alpha compositing
        # 物理上严格正确：仅在 Gaussian g 投影覆盖的像素上对后方 Gaussian 衰减
        # 实现：chunk 内用 cumprod 并行；chunk 间 T_global 累乘
        if use_energy_attenuation:
            sort_idx = z_cam.argsort()       # 近端在前
            u_pix = u_pix[sort_idx]
            v_pix = v_pix[sort_idx]
            intensity = intensity[sort_idx]
            opacity = opacity[sort_idx]
            shading_factor = shading_factor[sort_idx]
            Sigma_2D = Sigma_2D[sort_idx]
            radius = radius[sort_idx]
            z_cam = z_cam[sort_idx]

        # === 5. 求 Σ_2D 的逆（用于马氏距离）===
        # Σ^{-1} = 1/det * [[d, -b], [-b, a]]
        a = Sigma_2D[:, 0, 0]
        b = Sigma_2D[:, 0, 1]
        d = Sigma_2D[:, 1, 1]
        det = a * d - b * b
        det_inv = 1.0 / det.clamp(min=1e-8)
        # (N, 2, 2)
        Sigma_inv = torch.stack([
            torch.stack([d * det_inv, -b * det_inv], dim=-1),
            torch.stack([-b * det_inv, a * det_inv], dim=-1),
        ], dim=-2)

        # === 6. 像素网格 ===
        py_grid, px_grid = self._pixel_grid(device)                  # (H, W)

        # === 7. 累积渲染（分块 + 梯度检查点）===
        # ★ 把 image 初始化连接到参数：即使后续 chunk 全被过滤、checkpoint 没有把
        #   grad 传回 leaf，loss.backward() 也能找到至少一条到参数的反向路径。
        param_anchor = (opacity.sum() + intensity.sum() +
                         Sigma_inv.sum()) * 0.0
        image = torch.zeros(self.H, self.W, device=device) + param_anchor
        alpha_acc = torch.zeros(self.H, self.W, device=device)
        bbox_sigma_sq = self.bbox_sigma ** 2

        # ★ 加性渲染版本（use_energy_attenuation=False 时使用）
        # 返回 chunk 内 (radiance_sum, alpha_sum)
        def _render_chunk_additive(u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                                     px_grid_, py_grid_):
            du = px_grid_.unsqueeze(0) - u_c.view(-1, 1, 1)
            dv = py_grid_.unsqueeze(0) - v_c.view(-1, 1, 1)
            s00 = Sinv_c[:, 0, 0].view(-1, 1, 1)
            s01 = Sinv_c[:, 0, 1].view(-1, 1, 1)
            s11 = Sinv_c[:, 1, 1].view(-1, 1, 1)
            delta = du * du * s00 + 2 * du * dv * s01 + dv * dv * s11
            delta = delta.clamp(min=0.0, max=40.0)
            gauss_weight = torch.exp(-0.5 * delta)
            gauss_weight = gauss_weight * (delta < bbox_sigma_sq).float()
            alpha_per_g = op_c.view(-1, 1, 1) * gauss_weight
            radiance_per_g = (int_c.view(-1, 1, 1) *
                              sh_c.view(-1, 1, 1) *
                              alpha_per_g)
            return radiance_per_g.sum(dim=0), alpha_per_g.sum(dim=0)

        # ★ Alpha compositing 版本（use_energy_attenuation=True 时使用）
        # 给定输入 T_global，返回 (chunk_image, T_global_updated)
        # chunk 内用 cumprod 做严格 per-pixel per-Gaussian front-to-back
        def _render_chunk_compositing(u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                                        T_global_in, px_grid_, py_grid_):
            du = px_grid_.unsqueeze(0) - u_c.view(-1, 1, 1)
            dv = py_grid_.unsqueeze(0) - v_c.view(-1, 1, 1)
            s00 = Sinv_c[:, 0, 0].view(-1, 1, 1)
            s01 = Sinv_c[:, 0, 1].view(-1, 1, 1)
            s11 = Sinv_c[:, 1, 1].view(-1, 1, 1)
            delta = du * du * s00 + 2 * du * dv * s01 + dv * dv * s11
            delta = delta.clamp(min=0.0, max=40.0)
            gauss_weight = torch.exp(-0.5 * delta)
            gauss_weight = gauss_weight * (delta < bbox_sigma_sq).float()
            alpha_per_g = (op_c.view(-1, 1, 1) * gauss_weight).clamp(min=0.0, max=0.999)
            # (chunk, H, W) per-Gaussian per-pixel alpha
            radiance_per_g = (int_c.view(-1, 1, 1) *
                              sh_c.view(-1, 1, 1) *
                              alpha_per_g)
            # front-to-back: T_within[i] = prod_{j<i in chunk}(1 - alpha_j)^strength
            # strength=1: 完整 alpha compositing; strength<1: 更柔和的衰减
            # 用 exp(strength * log(cumprod)) 避免对负数开幂
            one_minus_alpha = (1.0 - alpha_per_g).clamp(min=1e-8)
            log_cumprod = torch.cumsum(torch.log(one_minus_alpha), dim=0)  # (chunk,H,W)
            T_within_cumprod = torch.exp(log_cumprod)  # = prod(1-α)
            T_within_soft = torch.exp(attenuation_strength * log_cumprod)  # = prod(1-α)^s
            # 右移一位：T_within[0]=1, T_within[i]=prod_{j<i}
            T_within = torch.cat(
                [torch.ones_like(T_within_soft[:1]),
                 T_within_soft[:-1]], dim=0)
            # chunk 总贡献 = Σ_i radiance_i · T_within_i · T_global_before_chunk
            chunk_image = (radiance_per_g * T_within).sum(dim=0) * T_global_in
            # chunk 净衰减（用 soft 版 T）
            T_global_out = T_global_in * T_within_soft[-1]
            chunk_alpha_sum = alpha_per_g.sum(dim=0)
            return chunk_image, T_global_out, chunk_alpha_sum

        # 是否启用梯度检查点（训练时启用省显存；关掉可大幅提速但耗显存）
        use_checkpoint = (use_grad_checkpoint and
                           torch.is_grad_enabled() and
                           any(t.requires_grad for t in
                               [opacity, intensity, shading_factor]))

        tile_bins = None
        if tile_size and tile_size > 0 and tile_size < max(self.H, self.W):
            tile_size = int(tile_size)
            tile_bins = self._build_tile_bins(
                u_pix, v_pix, radius, tile_size)

        def _render_tile(y0, y1, x0, x1, tile_indices=None):
            h = y1 - y0
            w = x1 - x0
            tile_image = torch.zeros(h, w, device=device) + param_anchor
            tile_alpha = torch.zeros(h, w, device=device)

            if tile_indices is None:
                tile_valid = ((u_pix + radius) > x0) & ((u_pix - radius) < x1) & \
                             ((v_pix + radius) > y0) & ((v_pix - radius) < y1)
                if not tile_valid.any():
                    return tile_image, tile_alpha
                selector = tile_valid
            else:
                if tile_indices.numel() == 0:
                    return tile_image, tile_alpha
                selector = tile_indices

            tu = u_pix[selector]
            tv = v_pix[selector]
            ti = intensity[selector]
            to = opacity[selector]
            ts = shading_factor[selector]
            tinv = Sigma_inv[selector]
            tz = z_cam[selector]

            if use_energy_attenuation:
                order = tz.argsort()
                tu = tu[order]
                tv = tv[order]
                ti = ti[order]
                to = to[order]
                ts = ts[order]
                tinv = tinv[order]

            tpy_grid, tpx_grid = self._pixel_grid(
                device, y0=y0, y1=y1, x0=x0, x1=x1)

            tile_T = (torch.ones(h, w, device=device)
                      if use_energy_attenuation else None)
            mt = tu.shape[0]
            for start_t in range(0, mt, chunk_size):
                end_t = min(start_t + chunk_size, mt)
                u_c = tu[start_t:end_t]
                v_c = tv[start_t:end_t]
                int_c = ti[start_t:end_t]
                op_c = to[start_t:end_t]
                sh_c = ts[start_t:end_t]
                Sinv_c = tinv[start_t:end_t]

                if use_energy_attenuation:
                    if use_checkpoint:
                        chunk_image, tile_T, chunk_alpha = torch.utils.checkpoint.checkpoint(
                            _render_chunk_compositing,
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            tile_T, tpx_grid, tpy_grid,
                            use_reentrant=False)
                    else:
                        chunk_image, tile_T, chunk_alpha = _render_chunk_compositing(
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            tile_T, tpx_grid, tpy_grid)
                    tile_image = tile_image + chunk_image
                else:
                    if use_checkpoint:
                        chunk_radiance, chunk_alpha = torch.utils.checkpoint.checkpoint(
                            _render_chunk_additive,
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            tpx_grid, tpy_grid,
                            use_reentrant=False)
                    else:
                        chunk_radiance, chunk_alpha = _render_chunk_additive(
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            tpx_grid, tpy_grid)
                    tile_image = tile_image + chunk_radiance

                if return_alpha:
                    with torch.no_grad():
                        tile_alpha = tile_alpha + chunk_alpha
            return tile_image, tile_alpha

        # T_global 维护：仅 use_energy_attenuation=True 时需要
        T_global = (torch.ones(self.H, self.W, device=device)
                    if use_energy_attenuation else None)

        if tile_size and tile_size > 0 and tile_size < max(self.H, self.W):
            tile_size = int(tile_size)
            rows = []
            alpha_rows = [] if return_alpha else None
            tile_id = 0
            for y0 in range(0, self.H, tile_size):
                y1 = min(y0 + tile_size, self.H)
                row_tiles = []
                alpha_tiles = [] if return_alpha else None
                for x0 in range(0, self.W, tile_size):
                    x1 = min(x0 + tile_size, self.W)
                    tile_indices = (None if tile_bins is None
                                    else tile_bins[tile_id])
                    tile_img, tile_alpha = _render_tile(
                        y0, y1, x0, x1, tile_indices=tile_indices)
                    tile_id += 1
                    row_tiles.append(tile_img)
                    if return_alpha:
                        alpha_tiles.append(tile_alpha)
                rows.append(torch.cat(row_tiles, dim=1))
                if return_alpha:
                    alpha_rows.append(torch.cat(alpha_tiles, dim=1))
            image = torch.cat(rows, dim=0)
            if return_alpha:
                alpha_acc = torch.cat(alpha_rows, dim=0)
        else:
            M = u_pix.shape[0]
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                u_c = u_pix[start:end]
                v_c = v_pix[start:end]
                int_c = intensity[start:end]
                op_c = opacity[start:end]
                sh_c = shading_factor[start:end]
                Sinv_c = Sigma_inv[start:end]

                if use_energy_attenuation:
                    if use_checkpoint:
                        chunk_image, T_global, chunk_alpha = torch.utils.checkpoint.checkpoint(
                            _render_chunk_compositing,
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            T_global, px_grid, py_grid,
                            use_reentrant=False)
                    else:
                        chunk_image, T_global, chunk_alpha = _render_chunk_compositing(
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            T_global, px_grid, py_grid)
                    image = image + chunk_image
                else:
                    if use_checkpoint:
                        chunk_radiance, chunk_alpha = torch.utils.checkpoint.checkpoint(
                            _render_chunk_additive,
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            px_grid, py_grid,
                            use_reentrant=False)
                    else:
                        chunk_radiance, chunk_alpha = _render_chunk_additive(
                            u_c, v_c, int_c, op_c, sh_c, Sinv_c,
                            px_grid, py_grid)
                    image = image + chunk_radiance

                if return_alpha:
                    with torch.no_grad():
                        alpha_acc = alpha_acc + chunk_alpha

        # ★ Module B: 渲染后 PSF 卷积（RNSplat 风格 sinc-Gaussian kernel）
        if use_psf_conv:
            psf = self._build_psf_kernel(
                size=psf_kernel_size, sigma_px=psf_sigma_px, device=device)
            # conv2d 需要 (1, 1, H, W) 输入与 (1, 1, k, k) kernel
            image_4d = image.unsqueeze(0).unsqueeze(0)
            psf_4d = psf.unsqueeze(0).unsqueeze(0)
            image = torch.nn.functional.conv2d(
                image_4d, psf_4d, padding=psf_kernel_size // 2)[0, 0]

        if return_alpha:
            return image, alpha_acc.clamp(0, 1)
        return image

    @staticmethod
    def _build_psf_kernel(size=11, sigma_px=1.5, device='cpu'):
        """
        构造 ISAR sinc-Gaussian PSF kernel（取绝对值避免负值副作用）。
        来源：RNSplat Eq.6 ϕ(b) = sinc(πb) · exp(-b²/2)
        sinc(π b) 描述 LFM 矩形窗的频域响应，Gaussian 描述天线方向图加权。
        """
        coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        r = torch.sqrt(x * x + y * y)
        # torch.sinc(x) = sin(πx)/(πx); 这里 x = r/sigma_px
        sinc_val = torch.sinc(r / sigma_px)
        gauss = torch.exp(-(r ** 2) / (2.0 * sigma_px * sigma_px))
        psf = (sinc_val * gauss).abs()    # 取 |·| 避免负 sidelobe 引入数值问题
        psf = psf / psf.sum().clamp(min=1e-8)
        return psf

    @staticmethod
    def colorize(intensity, cmap_name='hot', percentile_clip=99.5):
        """
        把单通道强度图转 RGB（用 matplotlib colormap）。

        intensity:   (H, W) float tensor 或 ndarray
        cmap_name:   colormap 名（'hot' 是 ISAR 标准）
        percentile_clip: 截断上限百分位（避免少数高亮像素压扁其他）

        返回: (H, W, 3) RGB tensor / ndarray, [0, 1]
        """
        import matplotlib.cm as cm
        if isinstance(intensity, torch.Tensor):
            intensity_np = intensity.detach().cpu().numpy()
        else:
            intensity_np = np.asarray(intensity)

        # 归一化到 [0, 1]（按上限百分位截断）
        v_max = np.percentile(intensity_np[intensity_np > 0], percentile_clip) \
                if (intensity_np > 0).any() else 1.0
        if v_max < 1e-6:
            v_max = 1.0
        normed = (intensity_np / v_max).clip(0, 1)

        cmap = cm.get_cmap(cmap_name)
        rgb = cmap(normed)[..., :3]   # (H, W, 3), drop alpha

        if isinstance(intensity, torch.Tensor):
            return torch.from_numpy(rgb).float()
        return rgb


class ISARRendererV3(ISARRenderer):
    """Numerically equivalent inference path with conservative caching.

    V3 intentionally keeps the v2 projection and accumulation equations. It
    only caches view-independent Gaussian decoding and immutable pixel grids
    while gradients are disabled. Gradient-enabled training automatically
    bypasses the static cache.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_cache = None
        self._grid_cache = {}
        self._static_cache_hits = 0
        self._grid_cache_hits = 0

    def clear_inference_cache(self):
        self._static_cache = None
        self._grid_cache.clear()
        self._static_cache_hits = 0
        self._grid_cache_hits = 0

    def cache_info(self):
        return {
            'static_cache_hits': int(self._static_cache_hits),
            'grid_cache_hits': int(self._grid_cache_hits),
            'grid_entries': int(len(self._grid_cache)),
        }

    def _prepare_gaussians(self, latent, use_sh_intensity=False):
        if torch.is_grad_enabled():
            return super()._prepare_gaussians(
                latent, use_sh_intensity=use_sh_intensity)

        # Tensors created inside torch.inference_mode() intentionally have no
        # version counter.  The FPS path keeps one immutable latent object, so
        # object identity is sufficient there; ordinary no-grad tensors retain
        # the version check for in-place invalidation.
        try:
            version = int(latent._version)
        except RuntimeError:
            version = None
        cache = self._static_cache
        if (cache is not None
                and cache['source'] is latent
                and cache['version'] == version
                and cache['use_sh_intensity'] == bool(use_sh_intensity)):
            self._static_cache_hits += 1
            return cache['prepared']

        prepared = super()._prepare_gaussians(
            latent, use_sh_intensity=use_sh_intensity)
        # Retain the source reference so allocator/id reuse cannot produce a
        # false cache hit after a temporary latent tensor is released.
        self._static_cache = {
            'source': latent,
            'version': version,
            'use_sh_intensity': bool(use_sh_intensity),
            'prepared': prepared,
        }
        return prepared

    def _pixel_grid(self, device, y0=0, y1=None, x0=0, x1=None):
        y1 = self.H if y1 is None else int(y1)
        x1 = self.W if x1 is None else int(x1)
        # Grids created in inference_mode are inference tensors. Reusing one
        # in a later gradient-enabled render makes activation checkpointing
        # fail when it tries to save the grid for backward. Training is not
        # the hot path for this cache, so create a regular grid in that mode.
        if torch.is_grad_enabled():
            return super()._pixel_grid(
                device, y0=y0, y1=y1, x0=x0, x1=x1)
        key = (str(device), int(y0), y1, int(x0), x1)
        cached = self._grid_cache.get(key)
        if cached is not None:
            self._grid_cache_hits += 1
            return cached
        grid = super()._pixel_grid(
            device, y0=y0, y1=y1, x0=x0, x1=x1)
        self._grid_cache[key] = grid
        return grid


class ISARRendererV4SparseTiles(ISARRendererV3):
    """Cached renderer with a per-frame sparse Gaussian-to-tile index.

    V2/v3 test every visible Gaussian against every tile. V4 computes each
    Gaussian's covered tile rectangle once, expands only the actual overlap
    references, and stores them in a row-major CSR-like list. The raster and
    accumulation equations remain inherited and unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_sparse_bin_info = None

    def sparse_bin_info(self):
        return (None if self._last_sparse_bin_info is None
                else dict(self._last_sparse_bin_info))

    def _build_tile_bins(self, u_pix, v_pix, radius, tile_size):
        n_cols = (self.W + tile_size - 1) // tile_size
        n_rows = (self.H + tile_size - 1) // tile_size
        n_tiles = n_rows * n_cols
        device = u_pix.device

        # Tile membership is a discrete visibility decision in both v3 and
        # v4. Keep bin construction outside autograd while the selected
        # Gaussian attributes remain fully differentiable during rasterizing.
        with torch.no_grad():
            tx_min = torch.floor((u_pix - radius) / tile_size).long()
            tx_max = torch.ceil((u_pix + radius) / tile_size).long() - 1
            ty_min = torch.floor((v_pix - radius) / tile_size).long()
            ty_max = torch.ceil((v_pix + radius) / tile_size).long() - 1

            tx_min = tx_min.clamp(0, n_cols - 1)
            tx_max = tx_max.clamp(0, n_cols - 1)
            ty_min = ty_min.clamp(0, n_rows - 1)
            ty_max = ty_max.clamp(0, n_rows - 1)

            has_tile = (tx_max >= tx_min) & (ty_max >= ty_min)
            source_ids = torch.arange(
                u_pix.shape[0], device=device, dtype=torch.long)[has_tile]
            tx_min = tx_min[has_tile]
            tx_max = tx_max[has_tile]
            ty_min = ty_min[has_tile]
            ty_max = ty_max[has_tile]

            nx = tx_max - tx_min + 1
            ny = ty_max - ty_min + 1
            references_per_gaussian = nx * ny

            if source_ids.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                self._last_sparse_bin_info = {
                    'tiles': int(n_tiles),
                    'nonempty_tiles': 0,
                    'gaussians': int(u_pix.shape[0]),
                    'references': 0,
                }
                return tuple(empty for _ in range(n_tiles))

            gaussian_ids = torch.repeat_interleave(
                source_ids, references_per_gaussian)
            starts = (torch.cumsum(references_per_gaussian, dim=0) -
                      references_per_gaussian)
            local = (torch.arange(gaussian_ids.shape[0], device=device) -
                     torch.repeat_interleave(
                         starts, references_per_gaussian))
            nx_repeated = torch.repeat_interleave(
                nx, references_per_gaussian)
            tile_x = (torch.repeat_interleave(
                tx_min, references_per_gaussian) + local % nx_repeated)
            tile_y = (torch.repeat_interleave(
                ty_min, references_per_gaussian) +
                torch.div(local, nx_repeated, rounding_mode='floor'))
            tile_ids = tile_y * n_cols + tile_x

            # Stable ordering preserves the original Gaussian order within
            # each tile, making additive rendering bitwise-equivalent to v3.
            order = torch.argsort(tile_ids, stable=True)
            tile_ids = tile_ids[order]
            gaussian_ids = gaussian_ids[order]
            counts = torch.bincount(tile_ids, minlength=n_tiles)
            counts_cpu = counts.cpu().tolist()  # one synchronization/frame
            bins = tuple(torch.split(gaussian_ids, counts_cpu))

        self._last_sparse_bin_info = {
            'tiles': int(n_tiles),
            'nonempty_tiles': int(sum(c > 0 for c in counts_cpu)),
            'gaussians': int(u_pix.shape[0]),
            'references': int(gaussian_ids.numel()),
        }
        return bins


class ISARRendererV5SparseScatter(ISARRendererV3):
    """Inference-only sparse pixel scatter renderer.

    V4 removes unnecessary Gaussian/tile tests, but still synchronizes tile
    counts back to the CPU and launches Python work for every tile.  V5 skips
    tiles entirely: it expands only each visible Gaussian's clipped pixel
    rectangle, evaluates its Mahalanobis support, and accumulates radiance with
    one flat ``index_add_`` per Gaussian chunk.

    The fast path implements the established additive equation exactly (up to
    floating-point accumulation order).  Gradient-enabled calls and the less
    common depth-attenuation mode fall back to V3, preserving training and the
    existing physical compositing implementation.
    """

    def __init__(self, *args, splat_chunk_size=2048, prefer_triton=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.splat_chunk_size = max(1, int(splat_chunk_size))
        self.prefer_triton = bool(prefer_triton)
        self._last_v5_info = None

    def v5_info(self):
        return None if self._last_v5_info is None else dict(self._last_v5_info)

    def render(self, latent, c2w, ortho_scale, chunk_size=2000,
               opacity_thresh=0.01, return_alpha=False,
               use_shading=True, shading_weight=1.0,
               use_sh_intensity=False,
               use_directional_sh_intensity=False,
               use_quadratic_directional_sh_intensity=False,
               use_energy_attenuation=False,
               attenuation_strength=0.1,
               use_psf_conv=False, psf_sigma_px=1.5, psf_kernel_size=11,
               use_grad_checkpoint=True,
               tile_size=0):
        # V5 is intentionally an inference renderer.  Keep the differentiable
        # and attenuation paths numerically identical to the established V3.
        if torch.is_grad_enabled() or use_energy_attenuation:
            reason = ('gradient_enabled' if torch.is_grad_enabled()
                      else 'energy_attenuation')
            self._last_v5_info = {
                'fast_path': False,
                'fallback': reason,
                'tile_size_ignored': False,
            }
            return super().render(
                latent, c2w, ortho_scale,
                chunk_size=chunk_size,
                opacity_thresh=opacity_thresh,
                return_alpha=return_alpha,
                use_shading=use_shading,
                shading_weight=shading_weight,
                use_sh_intensity=use_sh_intensity,
                use_directional_sh_intensity=use_directional_sh_intensity,
                use_quadratic_directional_sh_intensity=(
                    use_quadratic_directional_sh_intensity),
                use_energy_attenuation=use_energy_attenuation,
                attenuation_strength=attenuation_strength,
                use_psf_conv=use_psf_conv,
                psf_sigma_px=psf_sigma_px,
                psf_kernel_size=psf_kernel_size,
                use_grad_checkpoint=use_grad_checkpoint,
                tile_size=tile_size)

        device = latent.device
        xyz, intensity, opacity, R_gauss, Sigma_3D = \
            self._prepare_gaussians(
                latent, use_sh_intensity=use_sh_intensity)

        w2c = torch.linalg.inv(c2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        xyz_cam = xyz @ R_w2c.T + t_w2c

        if use_directional_sh_intensity and self.sh_dim >= 4:
            sh = latent[:, 3:3 + self.sh_dim]
            view_dir = c2w[:3, 2]
            view_dir = view_dir / view_dir.norm().clamp(min=1e-8)
            directional_logit = sh[:, 0] + (
                sh[:, 1:4] * view_dir[None]).sum(dim=-1)
            if (use_quadratic_directional_sh_intensity and
                    self.sh_dim >= 9):
                x, y, z = view_dir.unbind()
                quadratic_basis = torch.stack((
                    x * y, y * z, z * x, x.square() - y.square(),
                    3.0 * z.square() - 1.0))
                directional_logit = directional_logit + (
                    sh[:, 4:9] * quadratic_basis[None]).sum(dim=-1)
            intensity = directional_logit.sigmoid()

        if use_shading:
            normal_world = R_gauss[:, :, 2]
            normal_cam = normal_world @ R_w2c.T
            shading_factor = normal_cam[:, 2].clamp(min=0)
            shading_factor = (
                (1.0 - shading_weight) + shading_weight * shading_factor)
        else:
            shading_factor = torch.ones_like(opacity)

        Sigma_3D_cam = (
            R_w2c.unsqueeze(0) @ Sigma_3D @ R_w2c.T.unsqueeze(0))
        Sigma_2D = (ortho_scale ** 2) * Sigma_3D_cam[:, :2, :2]
        Sigma_2D = Sigma_2D + torch.eye(
            2, device=device, dtype=Sigma_2D.dtype).unsqueeze(0) * 0.25

        u_pix = xyz_cam[:, 0] * ortho_scale + self.W / 2.0
        v_pix = xyz_cam[:, 1] * ortho_scale + self.H / 2.0
        a = Sigma_2D[:, 0, 0]
        b = Sigma_2D[:, 0, 1]
        d = Sigma_2D[:, 1, 1]
        trace = a + d
        det = a * d - b * b
        disc = (trace.square() - 4 * det).clamp(min=0)
        lam_max = 0.5 * (trace + disc.sqrt())
        radius = self.bbox_sigma * lam_max.clamp(min=1e-4).sqrt()
        radius = radius.clamp(max=20.0)

        valid = ((opacity > opacity_thresh) &
                 (u_pix > -radius) & (u_pix < self.W + radius) &
                 (v_pix > -radius) & (v_pix < self.H + radius) &
                 (intensity > 1e-3))
        u_pix = u_pix[valid]
        v_pix = v_pix[valid]
        intensity = intensity[valid]
        opacity = opacity[valid]
        shading_factor = shading_factor[valid]
        Sigma_2D = Sigma_2D[valid]
        radius = radius[valid]

        a = Sigma_2D[:, 0, 0]
        b = Sigma_2D[:, 0, 1]
        d = Sigma_2D[:, 1, 1]
        det_inv = 1.0 / (a * d - b * b).clamp(min=1e-8)
        s00 = d * det_inv
        s01 = -b * det_inv
        s11 = a * det_inv

        n_visible = u_pix.shape[0]
        if n_visible == 0:
            image = torch.zeros(
                self.H, self.W, device=device, dtype=latent.dtype)
            self._last_v5_info = {
                'fast_path': True,
                'backend': 'empty',
                'fallback': None,
                'visible_gaussians': 0,
                'tile_size_ignored': bool(tile_size),
            }
            return (image, image) if return_alpha else image

        if self.prefer_triton and device.type == 'cuda':
            try:
                from isar_renderer_v5_triton import (
                    TRITON_AVAILABLE, triton_sparse_scatter)
                if TRITON_AVAILABLE:
                    image_flat, alpha_flat = triton_sparse_scatter(
                        u_pix, v_pix, intensity, opacity, shading_factor,
                        s00, s01, s11, radius,
                        height=self.H, width=self.W,
                        bbox_sigma_sq=self.bbox_sigma ** 2,
                        return_alpha=return_alpha)
                    image = image_flat.view(self.H, self.W).to(latent.dtype)
                    if use_psf_conv:
                        psf = self._build_psf_kernel(
                            size=psf_kernel_size, sigma_px=psf_sigma_px,
                            device=device)
                        image = torch.nn.functional.conv2d(
                            image[None, None], psf[None, None],
                            padding=psf_kernel_size // 2)[0, 0]
                    self._last_v5_info = {
                        'fast_path': True,
                        'backend': 'triton_fused',
                        'fallback': None,
                        'visible_gaussians': int(n_visible),
                        'tile_size_ignored': bool(tile_size),
                    }
                    if return_alpha:
                        alpha_map = alpha_flat.view(
                            self.H, self.W).clamp(0, 1).to(latent.dtype)
                        return image, alpha_map
                    return image
            except Exception as exc:
                # Keep deployment robust on unsupported Triton/GPU versions;
                # the exact sparse PyTorch implementation remains available.
                triton_error = f'{type(exc).__name__}: {exc}'
            else:
                triton_error = 'Triton unavailable'
        else:
            triton_error = ('disabled' if not self.prefer_triton
                            else 'non-CUDA device')

        image_flat = torch.zeros(
            self.H * self.W, device=device, dtype=latent.dtype)
        alpha_flat = (torch.zeros_like(image_flat) if return_alpha else None)
        bbox_sigma_sq = self.bbox_sigma ** 2
        # The old chunk size is tuned for dense [chunk,H,W] tensors.  Sparse
        # expansion is much smaller, so use at least the V5 launch-efficient
        # chunk while retaining a larger explicit user value.
        splat_chunk = max(int(chunk_size), self.splat_chunk_size)
        n_rect_refs = 0
        n_support_refs = 0

        for start in range(0, n_visible, splat_chunk):
            end = min(start + splat_chunk, n_visible)
            uc = u_pix[start:end]
            vc = v_pix[start:end]
            rc = radius[start:end]

            # Pixel centres are x+0.5/y+0.5.  The final Mahalanobis mask is
            # strict, exactly matching the reference renderer.
            x0 = torch.ceil(uc - rc - 0.5).long().clamp(0, self.W - 1)
            x1 = torch.floor(uc + rc - 0.5).long().clamp(0, self.W - 1)
            y0 = torch.ceil(vc - rc - 0.5).long().clamp(0, self.H - 1)
            y1 = torch.floor(vc + rc - 0.5).long().clamp(0, self.H - 1)
            widths = (x1 - x0 + 1).clamp(min=0)
            heights = (y1 - y0 + 1).clamp(min=0)
            counts = widths * heights
            source = torch.arange(
                end - start, device=device, dtype=torch.long)
            gid = torch.repeat_interleave(source, counts)
            starts = torch.cumsum(counts, dim=0) - counts
            local = (torch.arange(gid.shape[0], device=device) -
                     torch.repeat_interleave(starts, counts))
            repeated_width = torch.repeat_interleave(widths, counts)
            px_i = (torch.repeat_interleave(x0, counts) +
                    local.remainder(repeated_width))
            py_i = (torch.repeat_interleave(y0, counts) +
                    torch.div(local, repeated_width, rounding_mode='floor'))

            du = px_i.to(uc.dtype) + 0.5 - u_pix[start:end][gid]
            dv = py_i.to(vc.dtype) + 0.5 - v_pix[start:end][gid]
            delta = (du.square() * s00[start:end][gid] +
                     2.0 * du * dv * s01[start:end][gid] +
                     dv.square() * s11[start:end][gid])
            support = delta < bbox_sigma_sq
            pixel_ids = (py_i * self.W + px_i)[support]
            gid = gid[support]
            weight = torch.exp(-0.5 * delta[support].clamp(0.0, 40.0))
            alpha = opacity[start:end][gid] * weight
            radiance = (intensity[start:end][gid] *
                        shading_factor[start:end][gid] * alpha)
            image_flat.index_add_(0, pixel_ids, radiance)
            if return_alpha:
                alpha_flat.index_add_(0, pixel_ids, alpha)
            n_rect_refs += int(local.numel())
            n_support_refs += int(pixel_ids.numel())

        image = image_flat.view(self.H, self.W)
        if use_psf_conv:
            psf = self._build_psf_kernel(
                size=psf_kernel_size, sigma_px=psf_sigma_px, device=device)
            image = torch.nn.functional.conv2d(
                image[None, None], psf[None, None],
                padding=psf_kernel_size // 2)[0, 0]

        self._last_v5_info = {
            'fast_path': True,
            'backend': 'torch_sparse_scatter',
            'fallback': None,
            'triton_fallback_reason': triton_error,
            'visible_gaussians': int(n_visible),
            'rectangle_references': int(n_rect_refs),
            'support_references': int(n_support_refs),
            'splat_chunk_size': int(splat_chunk),
            'tile_size_ignored': bool(tile_size),
        }
        if return_alpha:
            return image, alpha_flat.view(self.H, self.W).clamp(0, 1)
        return image


# ─── 快速测试 ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """加载随机 Gaussian → 渲染 → 显示 hot colormap"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N = 5000
    # FreeSplatter 2DGS: xyz(3) + sh(3) + opacity(1) + scale(2) + rot(4) = 13
    latent = torch.randn(N, 13, device=device)
    latent[:, :3] = (torch.rand(N, 3, device=device) - 0.5) * 0.5    # xyz
    latent[:, 3:6] = 2.0     # SH 强度较高
    latent[:, 6] = 1.0       # opacity logit -> sigmoid 0.73
    latent[:, 7:9] = -2.5    # scale log -> exp 0.082
    latent[:, 9] = 1.0       # quat w = 1（单位四元数）
    latent[:, 10:13] = 0.0   # quat xyz = 0

    c2w = torch.eye(4, device=device)
    c2w[:3, 3] = torch.tensor([0, 0, -2.0], device=device)

    renderer = ISARRenderer(image_size=512).to(device)
    import time
    t0 = time.time()
    intensity = renderer.render(latent, c2w, ortho_scale=482.61)
    print(f'渲染时间: {time.time()-t0:.2f}s')
    print(f'强度范围: [{intensity.min():.4f}, {intensity.max():.4f}]')
    print(f'非零像素: {(intensity > 0.01).sum()} / {intensity.numel()}')

    # Hot colormap 可视化
    rgb = renderer.colorize(intensity)
    print(f'Colormap 输出: {rgb.shape}, range [{rgb.min():.3f}, {rgb.max():.3f}]')

    # 保存测试图
    from PIL import Image
    img = (rgb.numpy() * 255).clip(0, 255).astype('uint8')
    Image.fromarray(img).save('test_isar_render.png')
    print('[OK] 测试图保存: test_isar_render.png')

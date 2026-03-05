
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F

def rot(psi: float) -> np.ndarray:
    c = np.cos(psi)
    s = np.sin(psi)

    T = np.array([
        [ c, -s,  0,  0],
        [ s,  c,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ], dtype=float)

    return T

def calculation_extrinsic(pan, tilt, T_tc, T_pt, T_lp) :
    T_final = T_tc@rot(tilt)@T_pt@rot(pan)@T_lp

    return T_final 

def compute_scale_and_shift(prediction, target, device="cuda"):
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction).to(device, dtype=torch.float32, non_blocking=True)
    else:
        prediction = prediction.to(device, dtype=torch.float32, non_blocking=True)
    
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).to(device, dtype=torch.float32, non_blocking=True)
    else:
        target = target.to(device, dtype=torch.float32, non_blocking=True)

    pred_flat = prediction.ravel()
    tgt_flat  = target.ravel()

    mask = (pred_flat > 0) & (tgt_flat > 0)
    pred_m = pred_flat[mask]  
    tgt_m  = tgt_flat[mask]   

    ones  = torch.ones_like(pred_m)
    A_mat = torch.stack([ones, pred_m], dim=1)   
    rhs   = tgt_m.reciprocal().unsqueeze(1)      

    solution = torch.linalg.lstsq(A_mat, rhs, driver='gels').solution
    A, B = solution[0, 0], solution[1, 0]

    output = (A + B * prediction).reciprocal_()  
    return output

@torch.no_grad()
def depth_map_to_pcd(depth_map, K, device="cuda"):
    if isinstance(depth_map, np.ndarray):
        depth = torch.from_numpy(depth_map).pin_memory().to(device, non_blocking=True).float()
    else:
        depth = depth_map.to(device, non_blocking=True).float()

    H, W = depth.shape

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    mask = depth > 0
    z = depth[mask]
    u = u[mask]
    v = v[mask]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    x = (u - cx) * z * inv_fx
    y = (v - cy) * z * inv_fy

    points = torch.stack((x, y, z), dim=1).contiguous()

    return points

@torch.no_grad()
def project_and_mask(
    pts_lidar: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    w: int, h: int,
    z_min: float = 0.0,
    z_max: float = 200.0,
    margin_px: int = 0,
    device: str = "cuda"
) -> torch.Tensor:

    pts = torch.from_numpy(pts_lidar).float().pin_memory().to(device, non_blocking=True)  # (N,3)
    R_t = torch.from_numpy(R).float().to(device)  # (3,3)
    t_t = torch.from_numpy(t).float().to(device)  # (3,)

    pts_cam = pts @ R_t.T + t_t    # (N,3) broadcasting

    xc = pts_cam[:, 0]
    yc = pts_cam[:, 1]
    zc = pts_cam[:, 2]

    z_mask = (zc > z_min) & (zc < z_max)

    inv_zc = 1.0 / zc.clamp(min=1e-6)   # avoid division by zero
    u = fx * xc * inv_zc + cx
    v = fy * yc * inv_zc + cy

    fov_mask = (
        (u >= -margin_px) & (u < w + margin_px) &
        (v >= -margin_px) & (v < h + margin_px)
    )

    mask = z_mask & fov_mask

    return mask  # (N,) bool tensor on GPU

@torch.no_grad()
def pcd_to_fov_npy(
    pcd_file: str,
    T_cam_lidar: np.ndarray,
    K: np.ndarray,
    img_wh: tuple,
    z_min: float = 0.0,
    z_max: float = 200.0,
    margin_px: int = 0,
    device: str = "cuda"
) -> np.ndarray:

    pcd = o3d.io.read_point_cloud(pcd_file)
    pts_lidar = np.asarray(pcd.points, dtype=np.float32)  # (N,3)

    R = T_cam_lidar[:3, :3].astype(np.float32)
    t = T_cam_lidar[:3,  3].astype(np.float32)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h   = img_wh

    mask_gpu = project_and_mask(
        pts_lidar,
        R, t,
        fx, fy, cx, cy,
        w, h,
        z_min, z_max,
        margin_px,
        device=device
    )

    pts_gpu     = torch.from_numpy(pts_lidar).to(device)
    pts_fov_gpu = pts_gpu[mask_gpu]         

    return pts_fov_gpu

@torch.no_grad()
def points_npy_to_sparse_depth_map(
    pts_lidar: torch.Tensor,
    img_wh: tuple,
    K: np.ndarray,
    T_cam_lidar: np.ndarray,
    device: str = "cuda",
    radius_px: int = 10, # inspection radius
    eps: float = 1e-4       
):
    if pts_lidar.ndim != 2 or pts_lidar.shape[1] < 3:
        raise ValueError(f"Expected (N,3) points, got {pts_lidar.shape}")

    W, H = img_wh

    if isinstance(pts_lidar, np.ndarray):
        pts = torch.from_numpy(pts_lidar[:, :3]).float().pin_memory().to(device, non_blocking=True)
    else:
        pts = pts_lidar[:, :3].to(device=device, dtype=torch.float32)

    T   = torch.from_numpy(T_cam_lidar).to(device=device, dtype=torch.float32)  # (4,4)
    K_t = torch.from_numpy(K).to(device=device, dtype=torch.float32)            # (3,3)

    R = T[:3, :3]
    t = T[:3, 3]
    pts_cam = pts @ R.T + t  # (N,3)

    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    valid = Z > 1e-6
    X, Y, Z = X[valid], Y[valid], Z[valid]

    fx, fy = K_t[0, 0], K_t[1, 1]
    cx, cy = K_t[0, 2], K_t[1, 2]

    u = torch.round(fx * (X / Z) + cx).to(torch.int64)
    v = torch.round(fy * (Y / Z) + cy).to(torch.int64)

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Z = u[in_img], v[in_img], Z[in_img]

    if u.numel() == 0:
        return torch.zeros((H, W), device=device, dtype=torch.float32)

    pixel_idx = (v * W + u).to(torch.int64)

    zbuf_flat = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
    zbuf_flat.scatter_reduce_(0, pixel_idx, Z.to(torch.float32), reduce="amin", include_self=True)
    zbuf = zbuf_flat.view(H, W)

    # 2) Leave only the local minimum depth within radius_px
    if radius_px > 0:
        big = 1e9
        z_for_pool = torch.where(torch.isinf(zbuf), torch.full_like(zbuf, big), zbuf)

        k = 2 * radius_px + 1
        # min-pooling = -max_pool(-x)
        local_min = -F.max_pool2d((-z_for_pool)[None, None, ...],
                                  kernel_size=k, stride=1, padding=radius_px)[0, 0]

        local_min_flat = local_min.view(-1)

        z_loc = local_min_flat[pixel_idx]
        keep = Z <= (z_loc + eps)

        pixel_idx_k = pixel_idx[keep]
        Z_k = Z[keep].to(torch.float32)

        zbuf_flat2 = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
        zbuf_flat2.scatter_reduce_(0, pixel_idx_k, Z_k, reduce="amin", include_self=True)
        zbuf = zbuf_flat2.view(H, W)
        
    zbuf = torch.where(torch.isinf(zbuf), torch.zeros_like(zbuf), zbuf)
    return zbuf


@torch.no_grad()
def project_and_sample_color_full_gpu(points_lidar_np, R_np, t_np, K_np, image_bgr_np,
                                      device="cuda", z_min=0.0):
    """
    points_lidar_np: (N,3) float
    image_bgr_np: (H,W,3) uint8 (undistorted image)
    returns:
      mask: (N,) bool (torch, GPU)
      colors_full: (N,3) float32 RGB [0,1] (torch, GPU)  mask False is 0
    """
    # to GPU
    pts = torch.as_tensor(points_lidar_np, device=device, dtype=torch.float32)  # (N,3)
    R = torch.as_tensor(R_np, device=device, dtype=torch.float32)              # (3,3)
    t = torch.as_tensor(t_np, device=device, dtype=torch.float32).view(3, 1)   # (3,1)
    K = torch.as_tensor(K_np, device=device, dtype=torch.float32)              # (3,3)

    img = torch.as_tensor(image_bgr_np, device=device, dtype=torch.uint8)      # (H,W,3)
    H, W = img.shape[0], img.shape[1]

    # LiDAR -> Camera: Pc = R*P + t
    Pc = (R @ pts.t()) + t   # (3,N)
    X, Y, Z = Pc[0], Pc[1], Pc[2]

    valid = Z > z_min

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = valid & inside

    # full-length color buffer (N,3)
    colors_full = torch.zeros((pts.shape[0], 3), device=device, dtype=torch.float32)

    if mask.any():
        u_i = u[mask].to(torch.int64)
        v_i = v[mask].to(torch.int64)

        # img[v,u] is BGR uint8 -> RGB float
        bgr = img[v_i, u_i, :]  # (M,3) uint8
        rgb = bgr[:, [2,1,0]].to(torch.float32) / 255.0
        colors_full[mask] = rgb

    return mask, colors_full


@torch.no_grad()
def render_points_zbuffer_gpu(points_lidar_np, colors_rgb01_torch, R_np, t_np, K_np,
                              image_shape_hw, device="cuda", eps=1e-4):
    """
    points_lidar_np: (N,3) numpy
    colors_rgb01_torch: (N,3) torch float32 on GPU (RGB [0,1]), points와 1:1 대응
    image_shape_hw: (H,W) 또는 (H,W,3) shape tuple
    returns:
      overlay_bgr: (H,W,3) uint8 numpy (GPU->CPU 복사)
    """
    H, W = image_shape_hw[:2]

    pts = torch.as_tensor(points_lidar_np, device=device, dtype=torch.float32)  # (N,3)
    R = torch.as_tensor(R_np, device=device, dtype=torch.float32)
    t = torch.as_tensor(t_np, device=device, dtype=torch.float32).view(3, 1)
    K = torch.as_tensor(K_np, device=device, dtype=torch.float32)

    Pc = (R @ pts.t()) + t
    X, Y, Z = Pc[0], Pc[1], Pc[2]

    valid = Z > 0
    if not valid.any():
        return np.zeros((H, W, 3), dtype=np.uint8)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = valid & inside

    if not mask.any():
        return np.zeros((H, W, 3), dtype=np.uint8)

    u_i = u[mask].to(torch.int64)
    v_i = v[mask].to(torch.int64)
    z_i = Z[mask].to(torch.float32)
    col = colors_rgb01_torch[mask]  # (M,3) RGB float

    # pixel index
    pix = v_i * W + u_i  # (M,)

    # z-buffer: per-pixel minimum z using scatter_reduce (torch 2.0+)
    zbuf = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
    zbuf.scatter_reduce_(0, pix, z_i, reduce="amin", include_self=True)

    # keep points close to per-pixel min z
    zmin = zbuf[pix]
    keep = z_i <= (zmin + eps)
    if not keep.any():
        return np.zeros((H, W, 3), dtype=np.uint8)

    pix_k = pix[keep]
    col_k = col[keep]

    # write overlay (flat)
    overlay_flat = torch.zeros((H * W, 3), device=device, dtype=torch.uint8)

    # RGB -> BGR uint8
    bgr_k = (col_k[:, [2,1,0]].clamp(0, 1) * 255.0).to(torch.uint8)

    # index_put: last write wins (동일 픽셀에 keep가 여러개면 가장 마지막이 남음)
    overlay_flat.index_put_((pix_k,), bgr_k, accumulate=False)

    overlay = overlay_flat.view(H, W, 3).cpu().numpy()
    return overlay


def fuse_and_render_viewpoint_gpu(points_np,
                                  img_rgb_undist_bgr_np, K_rgb, R_LtoRGB, t_LtoRGB,
                                  img_th_undist_bgr_np, K_th, R_LtoTH, t_LtoTH,
                                  device="cuda"):
    
    mask_rgb, rgb_full = project_and_sample_color_full_gpu(
        points_np, R_LtoRGB, t_LtoRGB, K_rgb, img_rgb_undist_bgr_np, device=device
    )

    mask_both = mask_rgb 
    if not mask_both.any():
        H, W = img_th_undist_bgr_np.shape[:2]
        return np.zeros((H, W, 3), dtype=np.uint8)

    fused_full =  rgb_full 
    overlay = render_points_zbuffer_gpu(
        points_lidar_np=points_np,
        colors_rgb01_torch=fused_full,
        R_np=R_LtoTH, t_np=t_LtoTH, K_np=K_th,
        image_shape_hw=img_th_undist_bgr_np.shape,
        device=device
    )
    return overlay
import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch
import time
import util
from depth_anything_v2.dpt import DepthAnythingV2

T_lp = [
    [0.6510493062082845, -0.7590226062595021, 0.004414076658157591, 0.08314794370421265],
    [0.759010580110259, 0.6510640407127174, 0.004307455341268353, 0.0911459477578342],
    [-0.00614330256455169, 0.0005459650735034168, 0.9999809806970026, -0.04876299402464784],
    [0.0, 0.0, 0.0, 1.0]
  ]

T_pt = [
    [-0.06949323601098757, -0.054157502185163076, -0.9961112664284975, 0.01743560337132007],
    [0.6644233126157826, 0.7423090580417944, -0.08671172932062778, -0.011124962203000738],
    [0.7441185065575181, -0.6678654260449242, -0.015601951558729647, -0.022324137469471126],
    [0.0, 0.0, 0.0, 1.0]
  ]

T_tc = [
    [-0.02002046791088496, 0.008696392401634863, 0.9997617484300076, 0.043856019711015645],
    [0.9979891944164593, 0.06032301654016546, 0.019460254455717564, 0.009121746390669117],
    [-0.06013941047778534, 0.9981410253239229, -0.009886600664167274, -0.05751367924835556],
    [0.0, 0.0, 0.0, 1.0]
  ]

rgb_intrin = np.array([
    [1.55427854e+03, 0, 9.12672296e+02],
    [0, 1.55126077e+03, 5.15098474e+02],
    [0, 0, 1]
])
rgb_dist = np.array([-0.15659469, 0.12623196, -0.00123247, -0.00324473, -0.10057865])

th_intrin = np.array([
    [617.13379153, 0, 310.21086098],
    [0, 617.81378068, 266.67229058],
    [0, 0, 1]
])
th_dist = np.array([-0.38931623, 0.35394636, 0.00115199, 0.00073078, -0.47373078])

tr_rotation = np.array([[ 9.99860301e-01, 1.50497496e-02, 7.27204907e-03],
                    [-1.49052066e-02, 9.99698008e-01, -1.95378606e-02],
                    [-7.56389288e-03, 1.94267398e-02, 9.99782671e-01]])
tr_translation_vector = np.array([1.0265196,-52.50769629,19.63499906])/1000

def undistorting(img_src, domain='rgb') :
    global rgb_intrin, rgb_dist, th_intrin, th_dist
    img = cv2.imread(img_src)
    h, w = img.shape[:2]
    if domain == 'rgb' :
        newK, roi = cv2.getOptimalNewCameraMatrix(rgb_intrin, rgb_dist, (w, h), 0.0, (w, h))
        undist = cv2.undistort(img, rgb_intrin, rgb_dist, None, newK)
        rgb_intrin = newK
    else :
        newK, roi = cv2.getOptimalNewCameraMatrix(th_intrin, th_dist, (w, h), 0.0, (w, h))
        undist = cv2.undistort(img, th_intrin, th_dist, None, newK)
        th_intrin = newK
        
    return undist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--pcd-path', type=str)
    parser.add_argument('--thermal-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--pan', type=float, default=0.0)
    parser.add_argument('--tilt', type=float, default=0.0)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--verbose', dest="verbose", action='store_true', help="Choose whether to save intermediate results")
    parser.add_argument('--th-fused-level', type=float, default=0.5, help="Choose the intensity of thermal(0~1)")
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    start = time.perf_counter()

    print(f'Progress {args.img_path}')
    rgb_undist = undistorting(args.img_path, 'rgb') # cv2.undistort(cv2.imread(args.img_path), rgb_intrin, rgb_dist)
    
    relative_depth = depth_anything.infer_image(rgb_undist, args.input_size)
    
    if args.verbose : 
        depth = (relative_depth - relative_depth.min()) / (relative_depth.max() - relative_depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        cv2.imwrite('relative_depth.png', depth)

    T_cam_lidar = util.calculation_extrinsic(args.pan, args.tilt, T_tc, T_pt, T_lp)

    # Align pcd to the pan-tilt camera's field of view (FOV)
    lidar_fov_torch = util.pcd_to_fov_npy(args.pcd_path, T_cam_lidar, rgb_intrin, rgb_undist.shape[:2][::-1])

    # Generate sparse depth map by projecting LiDAR points
    # into the RGB camera coordinate system (intrinsic + extrinsic)
    # (Project only points with minimum distance within a certain radius_default 10px)
    depth_map_torch = util.points_npy_to_sparse_depth_map(lidar_fov_torch, rgb_undist.shape[:2][::-1], rgb_intrin, T_cam_lidar, radius_px=0)
    
    if args.verbose : 
        import matplotlib.pyplot as plt
        img_rgb = cv2.cvtColor(rgb_undist, cv2.COLOR_BGR2RGB)

        depth = depth_map_torch.detach().cpu().numpy()
        ys, xs = np.nonzero(depth > 0)
        vals = depth[ys, xs]

        plt.figure(figsize=(10,6))
        plt.imshow(img_rgb)
        plt.scatter(xs, ys, c=vals, s=5, cmap="turbo", alpha=0.9)
        plt.colorbar(label="Depth")
        plt.axis("off")
        plt.show()


    # Estimate scale and bias parameters to align relative depth with metric (absolute) depth
    # and convert relative depth to absolute depth
    scale_and_shift = util.compute_scale_and_shift (relative_depth, depth_map_torch)

    # Converting from depth map to pcd
    absolute_depth = util.depth_map_to_pcd(scale_and_shift, rgb_intrin)

    points_cpu = absolute_depth.detach().cpu().numpy()   

    if args.verbose : 
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cpu)
        o3d.io.write_point_cloud("absolut_depth.pcd", pcd)
        print("saved")

    th_undist = undistorting(args.thermal_path, 'thermal')

    I = np.eye(4)
    rt_rotation = tr_rotation.T
    rt_translation_vector = -tr_rotation.T@tr_translation_vector

    # Projecting colored pcd to fit thermal internal matrix
    aligned_rgb = util.fuse_and_render_viewpoint_gpu(absolute_depth,
                                rgb_undist, rgb_intrin, I[:3, :3], I[:3,3],
                                th_undist, th_intrin, rt_rotation, rt_translation_vector,
                                device="cuda")

    cv2.imwrite("aligned_rgb.jpg", aligned_rgb)
    cv2.imwrite("th_undist.jpg", th_undist)

    fused_img = aligned_rgb*(1-args.th_fused_level) + th_undist*args.th_fused_level
    cv2.imwrite("fusing.jpg", fused_img)

    print(f'processing tile : {time.perf_counter()-start}')

        
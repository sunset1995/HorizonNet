import json
import open3d
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from scipy.ndimage import map_coordinates

from misc.post_proc import np_coor2xy, np_coory2v
from misc.panostretch import pano_connect_points


def xyz_2_coorxy(xs, ys, zs, H, W):
    us = np.arctan2(xs, -ys)
    vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
    coorx = (us / (2 * np.pi) + 0.5) * W
    coory = (vs / np.pi + 0.5) * H
    return coorx, coory


def create_ceiling_floor_mask(cor_id, H, W):
    # Prepare 1d ceiling-wall/floor-wall boundary
    c_pts = []
    f_pts = []
    n_cor = len(cor_id)
    for i in range(n_cor // 2):
        # Ceiling boundary points
        xys = pano_connect_points(cor_id[i*2],
                                  cor_id[(i*2+2) % n_cor],
                                  z=-50, w=W, h=H)
        c_pts.extend(xys)

        # Floor boundary points
        xys = pano_connect_points(cor_id[i*2+1],
                                  cor_id[(i*2+3) % n_cor],
                                  z=50, w=W, h=H)
        f_pts.extend(xys)

    # Sort for interpolate
    c_pts = np.array(c_pts)
    c_pts = c_pts[np.argsort(c_pts[:, 0] * H - c_pts[:, 1])]
    f_pts = np.array(f_pts)
    f_pts = f_pts[np.argsort(f_pts[:, 0] * H + f_pts[:, 1])]

    # # Removed duplicated point
    c_pts = np.concatenate([c_pts[:1], c_pts[1:][np.diff(c_pts[:, 0]) > 0]], 0)
    f_pts = np.concatenate([f_pts[:1], f_pts[1:][np.diff(f_pts[:, 0]) > 0]], 0)

    # Generate boundary for each image column
    c_bon = np.interp(np.arange(W), c_pts[:, 0], c_pts[:, 1])
    f_bon = np.interp(np.arange(W), f_pts[:, 0], f_pts[:, 1])

    # Generate mask
    mask = np.zeros((H, W), np.bool)
    for i in range(W):
        u = max(0, int(round(c_bon[i])) + 1)
        b = min(W, int(round(f_bon[i])))
        mask[:u, i] = 1
        mask[b:, i] = 1

    return mask


def warp_walls(equirect_texture, xy, floor_z, ceil_z, ppm, alpha):
    H, W = equirect_texture.shape[:2]
    all_rgba = []
    all_xyz = []
    for i in trange(len(xy), desc='Processing walls'):
        next_i = (i + 1) % len(xy)
        xy_a = xy[i]
        xy_b = xy[next_i]
        xy_w = np.sqrt(((xy_a - xy_b)**2).sum())
        t_h = int(round((ceil_z - floor_z) * ppm))
        t_w = int(round(xy_w * ppm))
        xs = np.linspace(xy_a[0], xy_b[0], t_w)[None].repeat(t_h, 0)
        ys = np.linspace(xy_a[1], xy_b[1], t_w)[None].repeat(t_h, 0)
        zs = np.linspace(floor_z, ceil_z, t_h)[:, None].repeat(t_w, 1)
        coorx, coory = xyz_2_coorxy(xs, ys, zs, H, W)

        plane_texture = np.stack([
            map_coordinates(equirect_texture[..., 0], [coory, coorx], order=1, mode='wrap'),
            map_coordinates(equirect_texture[..., 1], [coory, coorx], order=1, mode='wrap'),
            map_coordinates(equirect_texture[..., 2], [coory, coorx], order=1, mode='wrap'),
            np.zeros([t_h, t_w]) + alpha,
        ], -1)
        plane_xyz = np.stack([xs, ys, zs], axis=-1)

        all_rgba.extend(plane_texture.reshape(-1, 4))
        all_xyz.extend(plane_xyz.reshape(-1, 3))

    return all_rgba, all_xyz


def warp_floor_ceiling(equirect_texture, mask, xy, z_floor, z_ceiling, ppm, alpha, n_thread):
    assert equirect_texture.shape[:2] == mask.shape[:2]
    H, W = equirect_texture.shape[:2]
    min_x = xy[:, 0].min()
    max_x = xy[:, 0].max()
    min_y = xy[:, 1].min()
    max_y = xy[:, 1].max()
    t_h = int(round((max_y - min_y) * ppm))
    t_w = int(round((max_x - min_x) * ppm))
    xs = np.linspace(min_x, max_x, t_w)[None].repeat(t_h, 0)
    ys = np.linspace(min_y, max_y, t_h)[:, None].repeat(t_w, 1)
    zs_floor = np.zeros_like(xs) + z_floor
    zs_ceil = np.zeros_like(xs) + z_ceiling
    coorx_floor, coory_floor = xyz_2_coorxy(xs, ys, zs_floor, H, W)
    coorx_ceil, coory_ceil = xyz_2_coorxy(xs, ys, zs_ceil, H, W)

    # Project view
    floor_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_floor, coorx_floor], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1)
    floor_mask = map_coordinates(mask, [coory_floor, coorx_floor], order=0)
    floor_xyz = np.stack([xs, ys, zs_floor], axis=-1)

    ceil_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1)
    ceil_mask = map_coordinates(mask, [coory_ceil, coorx_ceil], order=0)
    ceil_xyz = np.stack([xs, ys, zs_ceil], axis=-1)

    floor_texture = floor_texture[floor_mask]
    floor_xyz = floor_xyz[floor_mask]
    ceil_texture = ceil_texture[ceil_mask]
    ceil_xyz = ceil_xyz[ceil_mask]

    return floor_texture, floor_xyz, ceil_texture, ceil_xyz


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', required=True,
                        help='Image texture in equirectangular format')
    parser.add_argument('--layout', required=True,
                        help='Txt file containing layout corners (cor_id)')
    parser.add_argument('--camera_height', default=1.6, type=float,
                        help='Camera height in meter (not the viewer camera)')
    parser.add_argument('--ppm', default=80, type=int,
                        help='Points per meter')
    parser.add_argument('--point_size', default=0.0025, type=int,
                        help='Point size')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Opacity of the texture')
    parser.add_argument('--threads', default=10, type=int,
                        help='Number of threads to use')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    args = parser.parse_args()

    # Reading source (texture img, cor_id txt)
    equirect_texture = np.array(Image.open(args.img)) / 255.0
    H, W = equirect_texture.shape[:2]
    with open(args.layout) as f:
        inferenced_result = json.load(f)
    cor_id = np.array(inferenced_result['uv'], np.float32)
    cor_id[:, 0] *= W
    cor_id[:, 1] *= H

    mask = create_ceiling_floor_mask(cor_id, H, W)

    # Convert cor_id to 3d xyz
    N = len(cor_id) // 2
    floor_z = -args.camera_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H, floorW=1, floorH=1)
    c = np.sqrt((floor_xy**2).sum(1))
    v = np_coory2v(cor_id[0::2, 1], H)
    ceil_z = (c * np.tan(v)).mean()

    # Warp each wall
    all_rgba, all_xyz = warp_walls(equirect_texture, floor_xy, floor_z, ceil_z, args.ppm, args.alpha)

    # Warp floor and ceiling
    if not args.ignore_floor or not args.ignore_ceiling:
        fi, fp, ci, cp = warp_floor_ceiling(equirect_texture, mask,
                                            floor_xy, floor_z, ceil_z,
                                            ppm=args.ppm,
                                            alpha=args.alpha,
                                            n_thread=args.threads)

        if not args.ignore_floor:
            all_rgba.extend(fi)
            all_xyz.extend(fp)

        if not args.ignore_ceiling:
            all_rgba.extend(ci)
            all_xyz.extend(cp)

    # Launch point cloud viewer
    print('Showing %d of points...' % len(all_rgba))
    all_xyz = np.array(all_xyz)
    all_rgb = np.array(all_rgba)[:, :3]
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(all_xyz)
    pcd.colors = open3d.Vector3dVector(all_rgb)
    open3d.draw_geometries([pcd])

import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial import HalfspaceIntersection
from scipy.spatial import ConvexHull

from misc import post_proc, panostretch


def tri2halfspace(pa, pb, p):
    ''' Helper function for evaluating 3DIoU '''
    v1 = pa - p
    v2 = pb - p
    vn = np.cross(v1, v2)
    if -vn @ p > 0:
        vn = -vn
    return [*vn, -vn @ p]


def xyzlst2halfspaces(xyz_floor, xyz_ceil):
    '''
    Helper function for evaluating 3DIoU
    return halfspace enclose (0, 0, 0)
    '''
    N = xyz_floor.shape[0]
    halfspaces = []
    for i in range(N):
        last_i = (i - 1 + N) % N
        next_i = (i + 1) % N

        p_floor_a = xyz_floor[last_i]
        p_floor_b = xyz_floor[next_i]
        p_floor = xyz_floor[i]
        p_ceil_a = xyz_ceil[last_i]
        p_ceil_b = xyz_ceil[next_i]
        p_ceil = xyz_ceil[i]
        halfspaces.append(tri2halfspace(p_floor_a, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_floor_a, p_ceil, p_floor))
        halfspaces.append(tri2halfspace(p_ceil, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_ceil_a, p_ceil_b, p_ceil))
        halfspaces.append(tri2halfspace(p_ceil_a, p_floor, p_ceil))
        halfspaces.append(tri2halfspace(p_floor, p_ceil_b, p_ceil))
    return np.array(halfspaces)


def eval_3diou(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor, ch=-1.6,
               coorW=1024, coorH=512, floorW=1024, floorH=512):
    ''' Evaluate 3D IoU using halfspace intersection '''
    dt_floor_coor = np.array(dt_floor_coor)
    dt_ceil_coor = np.array(dt_ceil_coor)
    gt_floor_coor = np.array(gt_floor_coor)
    gt_ceil_coor = np.array(gt_ceil_coor)
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0
    N = len(dt_floor_coor)
    dt_floor_xyz = np.hstack([
        post_proc.np_coor2xy(dt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
        np.zeros((N, 1)) + ch,
    ])
    gt_floor_xyz = np.hstack([
        post_proc.np_coor2xy(gt_floor_coor, ch, coorW, coorH, floorW=1, floorH=1),
        np.zeros((N, 1)) + ch,
    ])
    dt_c = np.sqrt((dt_floor_xyz[:, :2] ** 2).sum(1))
    gt_c = np.sqrt((gt_floor_xyz[:, :2] ** 2).sum(1))
    dt_v2 = post_proc.np_coory2v(dt_ceil_coor[:, 1], coorH)
    gt_v2 = post_proc.np_coory2v(gt_ceil_coor[:, 1], coorH)
    dt_ceil_z = dt_c * np.tan(dt_v2)
    gt_ceil_z = gt_c * np.tan(gt_v2)

    dt_ceil_xyz = dt_floor_xyz.copy()
    dt_ceil_xyz[:, 2] = dt_ceil_z
    gt_ceil_xyz = gt_floor_xyz.copy()
    gt_ceil_xyz[:, 2] = gt_ceil_z

    dt_halfspaces = xyzlst2halfspaces(dt_floor_xyz, dt_ceil_xyz)
    gt_halfspaces = xyzlst2halfspaces(gt_floor_xyz, gt_ceil_xyz)

    in_halfspaces = HalfspaceIntersection(np.concatenate([dt_halfspaces, gt_halfspaces]),
                                          np.zeros(3))
    dt_halfspaces = HalfspaceIntersection(dt_halfspaces, np.zeros(3))
    gt_halfspaces = HalfspaceIntersection(gt_halfspaces, np.zeros(3))

    in_volume = ConvexHull(in_halfspaces.intersections).volume
    dt_volume = ConvexHull(dt_halfspaces.intersections).volume
    gt_volume = ConvexHull(gt_halfspaces.intersections).volume
    un_volume = dt_volume + gt_volume - in_volume

    return 100 * in_volume / un_volume


def gen_reg_from_xy(xy, w):
    xy = xy[np.argsort(xy[:, 0])]
    return np.interp(np.arange(w), xy[:, 0], xy[:, 1], period=w)


def test(dt_cor_id, z0, z1, gt_cor_id, w, h, losses):
    # Eval corner error
    mse = np.sqrt(((gt_cor_id - dt_cor_id)**2).sum(1)).mean()
    ce_loss = 100 * mse / np.sqrt(w**2 + h**2)

    # Pixel surface error (3 labels: ceiling, wall, floor)
    y0_dt = []
    y0_gt = []
    y1_gt = []
    for j in range(4):
        coorxy = panostretch.pano_connect_points(dt_cor_id[j * 2],
                                                 dt_cor_id[(j * 2 + 2) % 8],
                                                 -z0)
        y0_dt.append(coorxy)

        coorxy = panostretch.pano_connect_points(gt_cor_id[j * 2],
                                                 gt_cor_id[(j * 2 + 2) % 8],
                                                 -z0)
        y0_gt.append(coorxy)

        coorxy = panostretch.pano_connect_points(gt_cor_id[j * 2 + 1],
                                                 gt_cor_id[(j * 2 + 3) % 8],
                                                 z0)
        y1_gt.append(coorxy)
    y0_dt = gen_reg_from_xy(np.concatenate(y0_dt, 0), w)
    y1_dt = post_proc.infer_coory(y0_dt, z1 - z0, z0)
    y0_gt = gen_reg_from_xy(np.concatenate(y0_gt, 0), w)
    y1_gt = gen_reg_from_xy(np.concatenate(y1_gt, 0), w)

    surface = np.zeros((h, w), dtype=np.int32)
    surface[np.round(y0_dt).astype(int), np.arange(w)] = 1
    surface[np.round(y1_dt).astype(int), np.arange(w)] = 1
    surface = np.cumsum(surface, axis=0)
    surface_gt = np.zeros((h, w), dtype=np.int32)
    surface_gt[np.round(y0_gt).astype(int), np.arange(w)] = 1
    surface_gt[np.round(y1_gt).astype(int), np.arange(w)] = 1
    surface_gt = np.cumsum(surface_gt, axis=0)

    pe_loss = 100 * (surface != surface_gt).sum() / (h * w)

    # Eval 3d IoU
    iou3d = eval_3diou(dt_cor_id[1::2], dt_cor_id[0::2], gt_cor_id[1::2], gt_cor_id[0::2])

    losses['CE'].append(ce_loss)
    losses['PE'].append(pe_loss)
    losses['3DIoU'].append(iou3d)


def prepare_gtdt_pairs(gt_glob, dt_glob):
    gt_paths = sorted(glob.glob(gt_glob))
    dt_paths = dict([(os.path.split(v)[-1].split('.')[0], v)
                     for v in glob.glob(dt_glob) if v.endswith('json')])

    gtdt_pairs = []
    for gt_path in gt_paths:
        k = os.path.split(gt_path)[-1].split('.')[0]
        if k in dt_paths:
            gtdt_pairs.append((gt_path, dt_paths[k]))

    return gtdt_pairs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt_glob', required=True,
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be json from inference.py')
    parser.add_argument('--gt_glob', default='data/test/label_cor/*txt',
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be txt')
    parser.add_argument('--w', default=1024, type=int,
                        help='GT images width')
    parser.add_argument('--h', default=512, type=int,
                        help='GT images height')
    args = parser.parse_args()

    # Prepare (gt, dt) pairs
    gtdt_pairs = prepare_gtdt_pairs(args.gt_glob, args.dt_glob)

    # Testing
    losses = {
        'CE': [],
        'PE': [],
        '3DIoU': [],
    }
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        with open(gt_path) as f:
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        dt_cor_id[:, 0] *= args.w
        dt_cor_id[:, 1] *= args.h

        test(dt_cor_id, dt['z0'], dt['z1'], gt_cor_id, args.w, args.h, losses)

    print(' Testing Result '.center(50, '='))
    print('Corner Error (%):', np.mean(losses['CE']))
    print('Pixel  Error (%):', np.mean(losses['PE']))
    print('3DIoU        (%):', np.mean(losses['3DIoU']))
    print('=' * 50)

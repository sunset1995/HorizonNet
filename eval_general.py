import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

from eval_cuboid import prepare_gtdt_pairs
from misc import post_proc


def test_general(dt_cor_id, gt_cor_id, w, h, losses):
    dt_floor_coor = dt_cor_id[1::2]
    dt_ceil_coor = dt_cor_id[0::2]
    gt_floor_coor = gt_cor_id[1::2]
    gt_ceil_coor = gt_cor_id[0::2]
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 3d IoU and height error(in meter)
    N = len(dt_floor_coor)
    ch = -1.6
    dt_floor_xy = post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    gt_floor_xy = post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)

    area_dt = dt_poly.area
    area_gt = gt_poly.area
    area_inter = dt_poly.intersection(gt_poly).area
    iou2d = area_inter / (area_gt + area_dt - area_inter)
    cch_dt = post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
    cch_gt = post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)

    h_dt = abs(cch_dt.mean() - ch)
    h_gt = abs(cch_gt.mean() - ch)
    iouH = min(h_dt, h_gt) / max(h_dt, h_gt)
    iou3d = iou2d * iouH

    # Add a result
    n_corners = len(gt_floor_coor)
    n_corners = str(n_corners) if n_corners < 10 else '10+'
    losses[n_corners]['2DIoU'].append(iou2d)
    losses[n_corners]['3DIoU'].append(iou3d)
    losses['overall']['2DIoU'].append(iou2d)
    losses['overall']['3DIoU'].append(iou3d)


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
    losses = dict([
        (n_corner, {'2DIoU': [], '3DIoU': []})
        for n_corner in ['4', '6', '8', '10+', 'overall']
    ])
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        # Parse ground truth
        with open(gt_path) as f:
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        # Parse inferenced result
        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        dt_cor_id[:, 0] *= args.w
        dt_cor_id[:, 1] *= args.h

        test_general(dt_cor_id, gt_cor_id, args.w, args.h, losses)

    for k, result in losses.items():
        iou2d = np.array(result['2DIoU'])
        iou3d = np.array(result['3DIoU'])
        if len(iou2d) == 0:
            continue
        print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
        print('    2DIoU: %.2f' % (
            iou2d.mean() * 100,
        ))
        print('    3DIoU: %.2f' % (
            iou3d.mean() * 100,
        ))


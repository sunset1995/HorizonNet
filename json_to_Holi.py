import os
import json
import glob
import numpy as np
from shapely.geometry import LineString

from misc import panostretch

def find_occlusion(coor):
    u = panostretch.coorx2u(coor[:, 0])
    v = panostretch.coory2v(coor[:, 1])
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i+1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ori_glob', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'v0'), exist_ok=True)

    paths = glob.glob(args.ori_glob)
    for path in paths:
        with open(path) as f:
            dt = json.load(f)
        cor = np.array(dt['uv'], np.float32)
        cor[:, 0] *= 1024
        cor[:, 1] *= 512
        cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)
        occlusion = find_occlusion(cor[::2].copy()).repeat(2)

        cor_v0 = cor[~occlusion]
        cor_v0 = np.roll(cor_v0[:, :2], -2 * np.argmin(cor_v0[::2, 0]), 0)
        if (np.diff(cor_v0[:, 0]) < 0).sum():
            cor_v0[2::2] = np.flip(cor_v0[2::2], 0)
            cor_v0[3::2] = np.flip(cor_v0[3::2], 0)
        with open(os.path.join(args.output_dir, 'v0', f'{os.path.split(path)[1][:-5]}.txt'), 'w') as f:
            for u, v in cor_v0:
                f.write(f'{u:.0f} {v:.0f}\n')
        # if occlusion.sum() == 2:
        #     print(occlusion.sum())


import os
import json
import glob
import numpy as np
from shapely.geometry import LineString

from misc import panostretch

def cor_2_1d(cor, H=512, W=1024):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = panostretch.pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    #bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys

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

    paths = glob.glob(args.ori_glob)
    for path in paths:
        if path.endswith('json'):
            with open(path) as f:
                dt = json.load(f)
            cor = np.array(dt['uv'], np.float32)
            cor[:, 0] *= 1024
            cor[:, 1] *= 512
        else:
            with open(path) as f:
                cor = np.array([l.strip().split() for l in f]).astype(np.float32)
        cor = cor.reshape(-1, 4)
        duplicated = [False] * len(cor)
        for i in range(len(duplicated)):
            for j in range(i+1, len(duplicated)):
                if (cor[j] ==  cor[i]).sum() == 4:
                    duplicated[j] = True
        cor = cor[~np.array(duplicated)].reshape(-1, 2)
        cor = np.roll(cor[:, :2], -2 * np.argmin(cor[::2, 0]), 0)
        occlusion = find_occlusion(cor[::2].copy()).repeat(2)

        '''
        cor_v0 = cor[~occlusion]
        cor_v0 = np.roll(cor_v0[:, :2], -2 * np.argmin(cor_v0[::2, 0]), 0)
        if (np.diff(cor_v0[:, 0]) < 0).sum():
            cor_v0[2::2] = np.flip(cor_v0[2::2], 0)
            cor_v0[3::2] = np.flip(cor_v0[3::2], 0)
        with open(os.path.join(args.output_dir, f'{os.path.split(path)[1][:-4]}.txt'), 'w') as f:
            for u, v in cor_v0:
                f.write(f'{u:.0f} {v:.0f}\n')
        '''

        bon = cor_2_1d(cor)

        cor_v1 = []
        for i in range(0, len(cor), 2):
            if occlusion[i] & ~occlusion[(i+2) % len(cor)]:
                cur_x = cor[i, 0]
                next_x = cor[(i+2) % len(cor), 0]
                prev_x, j = None, i-2
                while prev_x is None:
                    if j < 0:
                        j += len(cor)
                    if ~occlusion[j]:
                        prev_x = cor[j, 0]
                        break
                    j -= 2
                dist2next = min(abs(next_x-cur_x), abs(next_x+1024-cur_x), abs(next_x-1024-cur_x))
                dist2prev = min(abs(prev_x-cur_x), abs(prev_x+1024-cur_x), abs(prev_x-1024-cur_x))
                # print(cor[i], prev_x, next_x, dist2next, dist2prev)
                if dist2prev < dist2next:
                    cor_v1.append([prev_x, bon[0, (int(prev_x)+1) % 1024]])
                    cor_v1.append([prev_x, bon[1, (int(prev_x)+1) % 1024]])
                else:
                    cor_v1.append([next_x, bon[0, (int(next_x)-1) % 1024]])
                    cor_v1.append([next_x, bon[1, (int(next_x)-1) % 1024]])
            elif ~occlusion[i]:
                cor_v1.extend(cor[i:i+2])

        cor_v1 = np.stack(cor_v1, 0)
        for _ in range(len(cor_v1)):
            if np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
                break
            cor_v1 = np.roll(cor_v1, 2, axis=0)
        if not np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
            cor_v1[2::2] = np.flip(cor_v1[2::2], 0)
            cor_v1[3::2] = np.flip(cor_v1[3::2], 0)
        for _ in range(len(cor_v1)):
            if np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
                break
            cor_v1 = np.roll(cor_v1, 2, axis=0)
        if not np.alltrue(cor_v1[::2, 0][1:] - cor_v1[::2, 0][:-1] >= 0):
            import pdb; pdb.set_trace()
        #cor_v1 = np.roll(cor_v1[:, :2], -2 * np.argmin(cor_v1[::2, 0]), 0)
        '''
        if (np.diff(cor_v1[:, 0]) < 0).sum():
            cor_v1[2::2] = np.flip(cor_v1[2::2], 0)
            cor_v1[3::2] = np.flip(cor_v1[3::2], 0)
        '''
        with open(os.path.join(args.output_dir, f'{os.path.split(path)[1].replace("json", "txt")}'), 'w') as f:
            for u, v in cor_v1:
                f.write(f'{u:.0f} {v:.0f}\n')


import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from sklearn.linear_model import HuberRegressor
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HorizonNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils


def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate, rotate_flip):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
        if rotate_flip:
            aug_type.append('rotate_flip %d' % shift)
            x_imgs_augmented.append(np.flip(x_imgs_augmented[-1], axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            if 'flip' in aug:
                x_img = np.flip(x_img, axis=-1)
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def inference(net, x, device, flip=False, rotate=[], rotate_flip=False, visualize=False,
              force_manhattan=False, min_v=0.05, r=0.05,
              td_flip=False, refine=False):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    x, aug_type = augment(x, flip, rotate, rotate_flip)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_cor_ = y_cor_[0, 0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    if td_flip:
        y_bon_ = H - 1 - y_bon_[::-1]
        z1 = z0 * z0 / z1

    # Detech wall-wall peaks
    r = int(round(W * r / 2))
    N = None
    coor_xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    us = panostretch.coorx2u(np.arange(W))
    vs = panostretch.coory2v(y_bon_[0])
    cs = -1 / np.tan(vs)
    xs = cs * np.cos(us)
    ys = cs * np.sin(us)
    coor_ys_ = []
    fit_lines_ = []
    for i in range(len(coor_xs_)):
        # Get points in topdown view
        s, t = coor_xs_[[i-1, i]]
        s, t = min(s, t), max(s, t)
        if t - s >= W / 2:
            cur_xs = np.concatenate([xs[t:], xs[:s]])
            cur_ys = np.concatenate([ys[t:], ys[:s]])
            cur_vs = np.concatenate([vs[t:], vs[:s]])
        else:
            cur_xs = xs[s:t]
            cur_ys = ys[s:t]
            cur_vs = vs[s:t]

        # Fit line
        if force_manhattan:
            fit_x = np.median(cur_xs)
            fit_y = np.median(cur_ys)
            if np.abs(cur_xs - fit_x).mean() < np.abs(cur_ys - fit_y).mean():
                fit_line = np.cross([fit_x, 0, 1], [fit_x, 1, 1])
            else:
                fit_line = np.cross([0, fit_y, 1], [1, fit_y, 1])
        else:
            if cur_xs.max()-cur_xs.min() > cur_ys.max()-cur_ys.min():
                huber = HuberRegressor(alpha=0).fit(cur_xs[:,None], cur_ys)
                a0, a1 = huber.predict(cur_xs[[0, -1]][:,None])
                fit_line = np.cross([cur_xs[0], a0, 1], [cur_xs[-1], a1, 1])
            else:
                huber = HuberRegressor(alpha=0).fit(cur_ys[:,None], cur_xs)
                a0, a1 = huber.predict(cur_ys[[0, -1]][:,None])
                fit_line = np.cross([a0, cur_ys[0], 1], [a1, cur_ys[-1], 1])

        # Project to wall-wall bonary
        prev_u, next_u = panostretch.coorx2u(coor_xs_[[i-1, i]])
        prev_line = np.cross([np.cos(prev_u), np.sin(prev_u), 1], [0, 0, 1])
        next_line = np.cross([np.cos(next_u), np.sin(next_u), 1], [0, 0, 1])
        prev_pt = np.cross(fit_line, prev_line)
        prev_pt /= prev_pt[-1]
        next_pt = np.cross(fit_line, next_line)
        next_pt /= next_pt[-1]
        prev_v = np.arctan2(-1, np.sqrt(prev_pt[0]**2 + prev_pt[1]**2))
        next_v = np.arctan2(-1, np.sqrt(next_pt[0]**2 + next_pt[1]**2))
        prev_coory = panostretch.v2coory(prev_v)
        next_coory = panostretch.v2coory(next_v)
        coor_ys_.append((prev_coory, next_coory))
        fit_lines_.append(fit_line)

    cor = []
    for i in range(len(coor_xs_)):
        cur_coorx = coor_xs_[i]
        cur_coory_a = coor_ys_[i][1]
        cur_coory_b = coor_ys_[(i+1)%len(coor_ys_)][0]
        line_a, line_b = fit_lines_[i], fit_lines_[(i+1)%len(fit_lines_)]
        refine_xy1 = np.cross(line_a, line_b)
        if refine and np.abs(refine_xy1[2]) > 1e-3:
            refine_x, refine_y = refine_xy1[:2] / refine_xy1[2]
            refine_u = np.arctan2(refine_y, refine_x)
            refine_c = np.sqrt(refine_x**2 + refine_y**2)
            refine_v = np.arctan2(-1, refine_c)
            refine_coorx = panostretch.u2coorx(refine_u)
            refine_coory = panostretch.v2coory(refine_v)
            drift_x = abs(refine_coorx - cur_coorx)
            drift_y = max(abs(refine_coory - cur_coory_a), abs(refine_coory - cur_coory_b))
            cur_u = panostretch.coorx2u(coor_xs_[i])
            pre_u = panostretch.coorx2u(coor_xs_[i-1])
            nex_u = panostretch.coorx2u(coor_xs_[(i+1)%len(coor_xs_)])
            L2i = np.cross([np.cos(pre_u), np.sin(pre_u), 0], [np.cos(cur_u), np.sin(cur_u), 0])[-1]
            i2R = np.cross([np.cos(cur_u), np.sin(cur_u), 0], [np.cos(nex_u), np.sin(nex_u), 0])[-1]
            if drift_x < 10 and drift_y < 10 and L2i > 1e-4 and i2R > 1e-4:
                cur_coorx = refine_coorx
                cur_coory_a, cur_coory_b = refine_coory, refine_coory
        if abs(cur_coory_a - cur_coory_b) < 10:
            cor.append((cur_coorx, (cur_coory_a+cur_coory_b)/2))
        else:
            cor.append((cur_coorx, cur_coory_a))
            cor.append((cur_coorx, cur_coory_b))

    cor = np.array(cor)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    if td_flip:
        cor_id[:, 1] = H - 1 - cor_id[:, 1]
        floor_cor = cor_id[0::2]
        ceil_cor = cor_id[1::2]
        cor_id = np.concatenate([ceil_cor, floor_cor], -1).reshape(-1, 2)

    return cor_id, vis_out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True,
                        help='path to load saved checkpoint.')
    parser.add_argument('--img_glob', required=True,
                        help='NOTE: Remeber to quote your glob path. '
                             'All the given images are assumed to be aligned'
                             'or you should use preporcess.py to do so.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--visualize', action='store_true')
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    parser.add_argument('--rotate_flip', action='store_true')
    # Post-processing realted
    parser.add_argument('--r', default=0.05, type=float)
    parser.add_argument('--min_v', default=0.05, type=float)
    parser.add_argument('--force_manhattan', action='store_true')
    parser.add_argument('--td_flip', action='store_true')
    parser.add_argument('--refine', action='store_true')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()

    # Prepare image to processed
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('no images found')
    for path in paths:
        assert os.path.isfile(path), '%s not found' % path

    # Check target directory
    if not os.path.isdir(args.output_dir):
        print('Output directory %s not existed. Create one.' % args.output_dir)
        os.makedirs(args.output_dir)
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Loaded trained model
    net = utils.load_trained_model(HorizonNet, args.pth).to(device)
    net.eval()

    # Inferencing
    with torch.no_grad():
        for i_path in tqdm(paths, desc='Inferencing'):
            k = os.path.split(i_path)[-1][:-4]

            # Load image
            img_pil = Image.open(i_path)
            if img_pil.size != (1024, 512):
                img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
            img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
            x = torch.FloatTensor([img_ori / 255])

            # Inferenceing corners
            cor_id, vis_out = inference(net, x, device,
                                        flip=args.flip, rotate=args.rotate, rotate_flip=args.rotate_flip,
                                        visualize=args.visualize,
                                        force_manhattan=args.force_manhattan,
                                        min_v=args.min_v, r=args.r,
                                        td_flip=args.td_flip,
                                        refine=args.refine)

            # Output result
            with open(os.path.join(args.output_dir, k + '.txt'), 'w') as f:
                for x, y in cor_id:
                    f.write('%d %d\n' % (x, y))

            if vis_out is not None:
                vis_path = os.path.join(args.output_dir, k + '.raw.png')
                vh, vw = vis_out.shape[:2]
                Image.fromarray(vis_out)\
                     .resize((vw//2, vh//2), Image.LANCZOS)\
                     .save(vis_path)


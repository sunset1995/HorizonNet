import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import HorizonNet
from utils import load_trained_model
from dataset import PanoCorBonDataset
import manhattan
import panostretch


def find_4_peaks(signal):
    W = signal.shape[0]
    assert W % 4 == 0
    signal_part = np.stack(np.split(signal, 4))
    pk_loc = signal_part.argmax(1)
    pk_loc += np.arange(4) * (W // 4)
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate):
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
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def test(corid, y_bon_, y_cor_, img_hw, fpaths, losses):
    corid = corid.cpu().numpy()
    y_cor_ = y_cor_[:, 0]
    for i in range(len(corid)):
        xs_ = find_4_peaks(y_cor_[i])[0]

        # Init floor/ceil plane
        z0 = 50
        _, z1 = manhattan.np_refine_by_fix_z(*y_bon_[i, :], z0)

        # Init axis aligned manhattan cuboid
        cor, _ = manhattan.init_cuboid(xs_, y_bon_[i, 0], z0, tol=abs(0.16 * z1 / 1.6))

        # Expand with btn coory
        cor = np.hstack([cor, manhattan.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

        # Collect corner position in equirectangular
        dt_floor_coor = []
        dt_ceil_coor = []
        gt_floor_coor = corid[i, 1::2]
        gt_ceil_coor = corid[i, 0::2]
        for j in range(4):
            dt_floor_coor.append([cor[j, 0], cor[j, 2]])
            dt_ceil_coor.append([cor[j, 0], cor[j, 1]])

        # Eval corner error
        diff = np.concatenate([
            np.array(dt_floor_coor) - gt_floor_coor,
            np.array(dt_ceil_coor) - gt_ceil_coor,
        ])
        ce_loss = 100 * np.sqrt((diff ** 2).sum(1)).mean() / np.sqrt(np.sum(img_hw**2))

        # Pixel surface error (3 labels: ceiling, wall, floor)
        y0 = np.zeros(img_hw[1])
        y0_gt = np.zeros(img_hw[1])
        y1_gt = np.zeros(img_hw[1])
        for j in range(4):
            coorxy = panostretch.pano_connect_points(dt_ceil_coor[j], dt_ceil_coor[(j+1)%4], -z0)
            y0[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

            coorxy = panostretch.pano_connect_points(gt_ceil_coor[j], gt_ceil_coor[(j+1)%4], -z0)
            y0_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

            coorxy = panostretch.pano_connect_points(gt_floor_coor[j], gt_floor_coor[(j+1)%4], z0)
            y1_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]
        assert (y0 == 0).sum() == 0
        y1 = manhattan.infer_coory(y0, z1 - z0, z0)

        surface = np.zeros(img_hw, dtype=np.int32)
        surface[np.round(y0).astype(int), np.arange(img_hw[1])] = 1
        surface[np.round(y1).astype(int), np.arange(img_hw[1])] = 1
        surface = np.cumsum(surface, axis=0)
        surface_gt = np.zeros(img_hw, dtype=np.int32)
        surface_gt[np.round(y0_gt).astype(int), np.arange(img_hw[1])] = 1
        surface_gt[np.round(y1_gt).astype(int), np.arange(img_hw[1])] = 1
        surface_gt = np.cumsum(surface_gt, axis=0)

        pe_loss = 100 * (surface != surface_gt).sum() / np.prod(img_hw)

        # Eval 3d IoU
        iou3d = manhattan.eval_3d(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor)

        if fpaths[i].split('/')[-1].startswith('pano'):
            losses['pano']['CE'].append(ce_loss)
            losses['pano']['PE'].append(pe_loss)
            losses['pano']['3DIoU'].append(iou3d)
        else:
            assert fpaths[i].split('/')[-1].startswith('camera')
            losses['2d3d']['CE'].append(ce_loss)
            losses['2d3d']['PE'].append(pe_loss)
            losses['2d3d']['3DIoU'].append(iou3d)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True,
                        help='path to saved checkpoint.')
    # Dataset related arguments
    parser.add_argument('--test_root_dir', default='data/test',
                        help='root directory to testing dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--num_workers', default=6, type=int,
                        help='numbers of workers for dataloaders')
    parser.add_argument('--batch_size_test', default=4, type=int,
                        help='training mini-batch size')
    # Augmentation related
    parser.add_argument('--flip', action='store_true',
                        help='whether to perfome left-right flip. '
                             '# of input x2.')
    parser.add_argument('--rotate', nargs='*', default=[], type=float,
                        help='whether to perfome horizontal rotate. '
                             'each elements indicate fraction of image width. '
                             '# of input xlen(rotate).')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    args = parser.parse_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    # Create dataloader
    dataset_test = PanoCorBonDataset(
        root_dir=args.test_root_dir,
        flip=False, rotate=False, gamma=False,
        return_cor=True, return_path=True)
    loader_test = DataLoader(dataset_test, args.batch_size_test,
                             shuffle=False, drop_last=False,
                             num_workers=args.num_workers,
                             pin_memory=not args.no_cuda)

    # Create model
    net = load_trained_model(HorizonNet, args.pth).to(device)
    net.eval()

    # Testing
    losses = {
        'pano': {
            'CE': [],
            'PE': [],
            '3DIoU': [],
        },
        '2d3d': {
            'CE': [],
            'PE': [],
            '3DIoU': [],
        },
    }
    with torch.no_grad():
        for x, y_bon, y_cor, corid, fpaths in tqdm(loader_test):
            img_hw = np.array(x.shape[-2:])
            x, aug_type = augment(x, args.flip, args.rotate)
            x = x.to(device)
            y_bon_, y_cor_ = net(x)
            y_bon_ = augment_undo(y_bon_, aug_type).mean(0)
            y_bon_ = (y_bon_ / np.pi + 0.5) * 512 - 0.5
            y_cor_ = augment_undo(torch.sigmoid(y_cor_), aug_type).mean(0)

            test(corid, y_bon_, y_cor_, img_hw, fpaths, losses)

    print(' Testing Result '.center(50, '='))
    print('PanoContext  Dataset (%d instances)' % len(losses['pano']['CE']))
    print('\t Corner Error (%):', np.mean(losses['pano']['CE']))
    print('\t Pixel  Error (%):', np.mean(losses['pano']['PE']))
    print('\t 3DIoU        (%):', np.mean(losses['pano']['3DIoU']))
    print('')
    print('Stanford2D3D Dataset (%d instances)' % len(losses['2d3d']['CE']))
    print('\t Corner Error (%):', np.mean(losses['2d3d']['CE']))
    print('\t Pixel  Error (%):', np.mean(losses['2d3d']['PE']))
    print('\t 3DIoU        (%):', np.mean(losses['2d3d']['3DIoU']))
    print('')
    print('Overall              (%d instances)' % len(losses['2d3d']['CE'] + losses['pano']['CE']))
    print('\t Corner Error (%):', np.mean(losses['pano']['CE'] + losses['2d3d']['CE']))
    print('\t Pixel  Error (%):', np.mean(losses['pano']['PE'] + losses['2d3d']['PE']))
    print('\t 3DIoU        (%):', np.mean(losses['pano']['3DIoU'] + losses['2d3d']['3DIoU']))
    print('=' * 50)

import os
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
from PIL import Image
import functools


def label_iterator(label):
    for floor_id, floor_data in label['merger'].items():
        for complete_room_id, complete_room_data in floor_data.items():
            for partial_room_id, partial_room_data in complete_room_data.items():
                for pano_id, pano_data in partial_room_data.items():
                    yield floor_id, partial_room_id, pano_id, pano_data


def show_statistic(root, scene_id_lst):
    sta = []
    for scene_id in scene_id_lst:
        label = json.load(open(os.path.join(root, scene_id, 'zind_data.json')))
        for floor_id, partial_room_id, pano_id, pano_data in label_iterator(label):
            sta.append([pano_data['is_primary'], pano_data['is_inside'], pano_data.get('is_ceiling_flat', False)])
    sta = np.array(sta)
    is_primary = sta[:,0].astype(bool)
    is_inside = sta[:,1].astype(bool)
    is_ceiling_flat = sta[:,2].astype(bool)
    print(f'is_primary                                : {is_primary.sum()} ({is_primary.mean()*100:.1f}%)')
    print(f'is_inside                                 : {is_inside.sum()} ({is_inside.mean()*100:.1f}%)')
    print(f'is_ceiling_flat                           : {is_ceiling_flat.sum()} ({is_ceiling_flat.mean()*100:.1f}%)')
    print(f'is_inside       given that is_primary     : {is_inside[is_primary].sum()} ({is_inside[is_primary].mean()*100:.1f}%)')
    print(f'is_ceiling_flat given that is_primary     : {is_ceiling_flat[is_primary].sum()} ({is_ceiling_flat[is_primary].mean()*100:.1f}%)')


def run(scene_id, split, args):
    label = json.load(open(os.path.join(args.indir, scene_id, 'zind_data.json')))
    for floor_id, partial_room_id, pano_id, pano_data in label_iterator(label):
        if args.geometry not in pano_data:
            continue
        if args.is_primary != -1:
            if (args.is_primary == 0 and pano_data['is_primary']) or\
                    (args.is_primary == 1 and not pano_data['is_primary']):
                continue
        if args.is_inside != -1:
            if (args.is_inside == 0 and pano_data['is_inside']) or\
                    (args.is_inside == 1 and not pano_data['is_inside']):
                continue
        if args.is_ceiling_flat != -1:
            if (args.is_ceiling_flat == 0 and pano_data.get('is_ceiling_flat', False)) or\
                    (args.is_ceiling_flat == 1 and not pano_data.get('is_ceiling_flat', False)):
                continue
        key = f'{floor_id}_{partial_room_id}_{pano_id}'
        img_path = os.path.join(args.indir, scene_id, 'panos', f'{key}.jpg')
        assert os.path.isfile(img_path), f'Image not found {img_path}'
        floor_z     = -pano_data['camera_height']
        ceiling_z   = pano_data['ceiling_height'] - pano_data['camera_height']
        vertices    = np.array(pano_data[args.geometry]['vertices'])
        theta       = np.arctan2(-vertices[:,0], vertices[:,1])
        ceiling_phi = np.arctan2(ceiling_z, np.sqrt((vertices**2).sum(1)))
        floor_phi   = np.arctan2(floor_z, np.sqrt((vertices**2).sum(1)))
        coor_x      = (theta + np.pi) / (2.0*np.pi) * (args.width - 1)
        coor_y0     = (1 - (ceiling_phi + np.pi/2.0)/np.pi) * (args.width/2 - 1)
        coor_y1     = (1 - (floor_phi + np.pi/2.0)/np.pi) * (args.width/2 - 1)
        img = Image.open(img_path).resize((args.width, args.width//2), Image.LANCZOS)
        img.save(os.path.join(args.outdir, split, 'img', f'{scene_id}_{key}.jpg'), quality=95, optimize=True)
        with open(os.path.join(args.outdir, split, 'label_cor', f'{scene_id}_{key}.txt'), 'w') as f:
            for i in range(len(coor_x)):
                f.write(f'{coor_x[i]:.1f} {coor_y0[i]:.1f}\n')
                f.write(f'{coor_x[i]:.1f} {coor_y1[i]:.1f}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--partition', default='zind_partition.json')
    parser.add_argument('--indir', default='data/')
    parser.add_argument('--outdir', default='zind_horizonnet/')
    parser.add_argument('--width', default=1024, type=int)
    parser.add_argument('--geometry', default='layout_visible')
    parser.add_argument('--is_primary', default=1, choices=[-1,0,1], type=int,
                        help='-1 dont care / 0 false only / 1 true only')
    parser.add_argument('--is_inside', default=1, choices=[-1,0,1], type=int,
                        help='-1 dont care / 0 false only / 1 true only')
    parser.add_argument('--is_ceiling_flat', default=1, choices=[-1,0,1], type=int,
                        help='-1 dont care / 0 false only / 1 true only')
    parser.add_argument('--num_workers', default=10, type=int)
    args = parser.parse_args()

    assert os.path.isfile(args.partition), f'Partition not found: {args.partition}'
    assert os.path.exists(args.indir), f'Dir not found: {args.indir}'

    partition = json.load(open(args.partition))

    for split, scene_id_lst in partition.items():
        os.makedirs(os.path.join(args.outdir, split, 'img'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, split, 'label_cor'), exist_ok=True)
        print(f'Processing split: {split}')
        #show_statistic(args.indir, scene_id_lst)
        if args.num_workers > 1:
            run_partial = functools.partial(run, split=split, args=args)
            with multiprocessing.Pool(args.num_workers) as pool:
                list(tqdm(pool.imap(run_partial, scene_id_lst)))
        else:
            for scene_id in tqdm(scene_id_lst):
                run(split, scene_id, args)


import os
import argparse
from zipfile import ZipFile
from tqdm import tqdm
import imageio

'''
Assume datas is extracted by `misc/structured3d_extract_zip.py`.
That is to said, assuming following structure:
- {in_root}/scene_xxxxx
    - rgb/
        - *png
    - layout/
        - *txt

The reorganized structure as follow:
- {out_train_root}
    - img/
        - scene_xxxxx_*png (softlink)
    - label_cor/
        - scene_xxxxx_*txt (softlink)
- {out_valid_root} ...
- {out_test_root} ...
'''
TRAIN_SCENE = ['scene_%05d' % i for i in range(0, 3000)]
VALID_SCENE = ['scene_%05d' % i for i in range(3000, 3250)]
TEST_SCENE = ['scene_%05d' % i for i in range(3250, 3500)]

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', required=True)
parser.add_argument('--out_train_root', default='data/st3d_train_full_raw_light')
parser.add_argument('--out_valid_root', default='data/st3d_valid_full_raw_light')
parser.add_argument('--out_test_root', default='data/st3d_test_full_raw_light')
args = parser.parse_args()

def prepare_dataset(scene_ids, out_dir):
    root_img = os.path.join(out_dir, 'img')
    root_cor = os.path.join(out_dir, 'label_cor')
    os.makedirs(root_img, exist_ok=True)
    os.makedirs(root_cor, exist_ok=True)
    for scene_id in tqdm(scene_ids):
        source_img_root = os.path.join(args.in_root, scene_id, 'rgb')
        source_cor_root = os.path.join(args.in_root, scene_id, 'layout')
        for fname in os.listdir(source_cor_root):
            room_id = fname.split('_')[0]
            source_img_path = os.path.join(args.in_root, scene_id, 'rgb', room_id + '_rgb_rawlight.png')
            source_cor_path = os.path.join(args.in_root, scene_id, 'layout', room_id + '_layout.txt')
            target_img_path = os.path.join(root_img, '%s_%s.png' % (scene_id, room_id))
            target_cor_path = os.path.join(root_cor, '%s_%s.txt' % (scene_id, room_id))
            assert os.path.isfile(source_img_path)
            assert os.path.isfile(source_cor_path)
            os.symlink(source_img_path, target_img_path)
            os.symlink(source_cor_path, target_cor_path)

prepare_dataset(TRAIN_SCENE, args.out_train_root)
prepare_dataset(VALID_SCENE, args.out_valid_root)
prepare_dataset(TEST_SCENE, args.out_test_root)

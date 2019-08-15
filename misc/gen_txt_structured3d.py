'''
Help generate txt for train.py
Please contact https://github.com/bertjiazheng/Structured3D for dataset.
'''

import os
import glob
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root', required=True,
                    help='path to the dataset directory')
parser.add_argument('--train_txt', required=True,
                    help='path to save txt for train')
parser.add_argument('--valid_txt', required=True,
                    help='path to save txt for valid')
parser.add_argument('--test_txt', required=True,
                    help='path to save txt for test')
args = parser.parse_args()

train_scene = ['scene_%05d' % i for i in range(0, 3000)]
valid_scene = ['scene_%05d' % i for i in range(3000, 3250)]
test_scene = ['scene_%05d' % i for i in range(3250, 3500)]

# Simple check: all directories exist
for path in train_scene + valid_scene + test_scene:
    assert os.path.isdir(os.path.join(args.root, path)), '%s not found' % path

def gen_pairs(scene_id_lst):
    pairs = []
    for scene_id in scene_id_lst:
        for fname in os.listdir(os.path.join(args.root, scene_id, 'rgb')):
            room_id = os.path.split(fname)[1].split('_')[0]

            img_k = os.path.join(os.path.join(scene_id, 'rgb', fname))
            layout_k = os.path.join(os.path.join(scene_id, 'layout', room_id + '_layout.txt'))
            assert os.path.isfile(os.path.join(args.root, img_k))
            assert os.path.isfile(os.path.join(args.root, layout_k))
            pairs.append((img_k, layout_k))
    return pairs

with open(args.train_txt, 'w') as f:
    pairs = gen_pairs(train_scene)
    f.write('\n'.join([' '.join(p) for p in pairs]))

with open(args.valid_txt, 'w') as f:
    pairs = gen_pairs(valid_scene)
    f.write('\n'.join([' '.join(p) for p in pairs]))

with open(args.test_txt, 'w') as f:
    pairs = gen_pairs(test_scene)
    f.write('\n'.join([' '.join(p) for p in pairs]))

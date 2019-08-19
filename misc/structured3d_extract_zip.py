import os
import argparse
from zipfile import ZipFile
from tqdm import tqdm
import imageio

'''
Zipfile format assumption:
Structured3D
-- [scene_xxxxx]
    -- other something
    -- 2D_rendering
        -- [image_id]
            -- panorama
                -- camera_xyz.txt
                -- layout.txt
                -- [empty|simple|full]
                    -- depth.png
                    -- rgb_rawlight.png
                    -- rgb_coldlight.png
                    -- rgb_warmlight.png
                    -- other something

Output format
outdir
-- [scene_xxxxx]
    -- img
    -- layout
'''

parser = argparse.ArgumentParser()
parser.add_argument('--zippath', required=True)
parser.add_argument('--style', default='full')
parser.add_argument('--outdir', default='structured3d')
args = parser.parse_args()

path_format = 'Structured3D/%s/2D_rendering/%s/panorama/%s'

with ZipFile(args.zippath) as zipf:
    id_set = set()
    for path in zipf.namelist():
        assert path.startswith('Structured3D')
        if path.endswith('camera_xyz.txt'):
            path_lst = path.split('/')
            scene_id = path_lst[1]
            image_id = path_lst[3]
            id_set.add((scene_id, image_id))

    for scene_id, image_id in tqdm(id_set):
        path_img = path_format % (scene_id, image_id, '%s/rgb_rawlight.png' % args.style)
        path_layout = path_format % (scene_id, image_id, 'layout.txt')

        os.makedirs(os.path.join(args.outdir, scene_id, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, scene_id, 'layout'), exist_ok=True)

        with zipf.open(path_img) as f:
            rgb = imageio.imread(f)[..., :3]
            imageio.imwrite(os.path.join(args.outdir, scene_id, 'rgb', image_id + '_rgb_rawlight.png'), rgb)
        with zipf.open(path_layout) as f:
            with open(os.path.join(args.outdir, scene_id, 'layout', image_id + '_layout.txt'), 'w') as fo:
                fo.write(f.read().decode())

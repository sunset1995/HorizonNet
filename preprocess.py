'''
This script preprocess the given 360 panorama image under euqirectangular projection
and dump them to the given directory for further layout prediction and visualization.
The script will:
    - extract and dump the vanishing points
    - rotate the equirect image to align with the detected VP
    - extract the VP aligned line segments (for further layout prediction model)
The dump files:
    - `*_VP.txt` is the vanishg points
    - `*_aligned_rgb.png` is the VP aligned RGB image
    - `*_aligned_line.png` is the VP aligned line segments images

Author: Cheng Sun
Email : chengsun@gapp.nthu.edu.tw
'''

import os
import glob
import argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial

from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama

def process_image(i_path, output_dir, q_error, refine_iter, rgbonly):
    print('Processing', i_path, flush=True)

    # Load and cat input image
    img_ori = np.array(Image.open(i_path).resize((1024, 512), Image.BICUBIC))[..., :3]

    # VP detection and line segment extraction
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_ori,
                                                    qError=q_error,
                                                    refineIter=refine_iter)
    panoEdge = (panoEdge > 0)

    # Align images with VP
    i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

    # Dump results
    basename = os.path.splitext(os.path.basename(i_path))[0]
    if rgbonly:
        path = os.path.join(output_dir, '%s.png' % basename)
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path)
    else:
        path_VP = os.path.join(output_dir, '%s_VP.txt' % basename)
        path_i_img = os.path.join(output_dir, '%s_aligned_rgb.png' % basename)
        path_l_img = os.path.join(output_dir, '%s_aligned_line.png' % basename)

        with open(path_VP, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path_i_img)
        Image.fromarray((l_img * 255).astype(np.uint8)).save(path_l_img)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # I/O related arguments
    parser.add_argument('--img_glob', required=True,
                        help='Example: "data/images/*jpg" NOTE: Remember to quote your glob path.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--rgbonly', action='store_true',
                        help='Add this if use are preparing customer dataset')
    # Preprocessing related arguments
    parser.add_argument('--q_error', default=0.7, type=float)
    parser.add_argument('--refine_iter', default=3, type=int)
    # Code Parallelization Related
    parser.add_argument('--core_utilization', default=0.5, type=float,
                        help='Percentage of CPU cores to be used for parallel processing between 0.1 to 1. Actual number of used cores will be rounded to the nearest integer.')
    args = parser.parse_args()

    paths = sorted(glob.glob(args.img_glob))
    if len(paths) == 0:
        print('No images found')
        return

    for path in paths:
        try:
            # Check if the path exists and is a file
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found")
            
        except FileNotFoundError as e:
            # Handle the exception (e.g., log the error, raise further, etc.)
            print(f"Error: {e}")
    
    # Check target directory
    if not os.path.isdir(args.output_dir):
        print(f'Output directory {args.output_dir} not existed. Create one.')
        os.makedirs(args.output_dir)

    # Determine the number of available CPU cores
    num_cores = cpu_count()

    # Check if core utilization value is higher than 1.0 and adjust it down to 1.0 (Use all available cores)
    if args.core_utilization > 1.0:
        core_utilization = 1.0
    else:
        core_utilization = args.core_utilization

    # Determine the number of available CPU cores to be utilized (Default = 50% and a minimum of 1 core)
    num_cores_to_use = max(1, int(num_cores * core_utilization))

    # Process each input image in parallel using a specified number of cores
    with Pool(processes=num_cores_to_use) as pool:
        process_func = partial(process_image, output_dir=args.output_dir,
                                q_error=args.q_error, refine_iter=args.refine_iter,
                                rgbonly=args.rgbonly)
        pool.map(process_func, paths)

if __name__ == '__main__':
    main()
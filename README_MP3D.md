# Results on MatterportLayout dataset

References:
- [3D Manhattan Room Layout Reconstruction from a Single 360 Image](https://arxiv.org/abs/1910.04099)
- [Matterport3DLayoutAnnotation github](https://github.com/ericsujw/Matterport3DLayoutAnnotation)
- [LayoutNetv2 github](https://github.com/zouchuhang/LayoutNetv2)
- [DuLa-Net github](https://github.com/SunDaDenny/DuLa-Net)

## Dataset preparation
- Please refer to [Matterport3DLayoutAnnotation](https://github.com/ericsujw/Matterport3DLayoutAnnotation) to prepare the source datas.
    - Put all the rgb under `{ROOT}/image_up/`.
    - Download the annotation to `{ROOT}/label_data/` (originally json format).
    - Download the data split into `{ROOT}/mp3d_[train|val|test].txt`.
- Use below code to convert original ground-truth json into txt. (Remember to update the uppercase variables)
    ```python
    import os
    import glob
    import json
    import numpy as np

    IN_GLOB = 'label_data/*json'
    OUT_DIR = 'label_cor'
    os.makedirs(OUT_DIR, exist_ok=True)

    for p in glob.glob(IN_GLOB):
        gt = json.load(open(p))
        assert gt['cameraHeight'] == 1.6
        us = np.array([pts['coords'][0] for pts in gt['layoutPoints']['points']])
        us = us * 1024
        cs = np.array([pts['xyz'] for pts in gt['layoutPoints']['points']])
        cs = np.sqrt((cs**2)[:, [0, 2]].sum(1))

        vf = np.arctan2(-1.6, cs)
        vc = np.arctan2(-1.6 + gt['layoutHeight'], cs)
        vf = (-vf / np.pi + 0.5) * 512
        vc = (-vc / np.pi + 0.5) * 512

        cor_x = np.repeat(us, 2)
        cor_y = np.stack([vc, vf], -1).reshape(-1)
        cor_xy = np.stack([cor_x, cor_y], -1)

        out_path = os.path.join(OUT_DIR, os.path.split(p)[-1][:-4] + 'txt')
        with open(out_path, 'w') as f:
            for x, y in cor_xy:
                f.write('%.2f %.2f\n' % (x, y))
    ```
- Use below code to organize the data for training and evaluation. (Remember to update the uppercase variables)
    ```python
    import os
    from shutil import copy2

    IMG_ROOT = 'image_up'
    TXT_ROOT = 'label_cor'
    OUT_ROOT = 'mp3d_layout'
    TRAIN_TXT = 'mp3d_train.txt'
    VALID_TXT = 'mp3d_val.txt'
    TEST_TXT = 'mp3d_test.txt'

    def go(txt, split):
        out_img_root = os.path.join(OUT_ROOT, split, 'img')
        out_txt_root = os.path.join(OUT_ROOT, split, 'label_cor')
        os.makedirs(out_img_root, exist_ok=True)
        os.makedirs(out_txt_root, exist_ok=True)

        with open(txt) as f:
            ks = ['_'.join(l.strip().split()) for l in f]

        for k in ks:
            copy2(os.path.join(IMG_ROOT, k + '.png'), out_img_root)
            copy2(os.path.join(TXT_ROOT, k + '_label.txt'), out_txt_root)
            os.rename(os.path.join(out_txt_root, k + '_label.txt'), os.path.join(out_txt_root, k + '.txt'))


    go(TRAIN_TXT, 'train')
    go(VALID_TXT, 'valid')
    go(TEST_TXT, 'test')
    ```
- So now, you should have a `mp3d_layout` directory with below structure for HorizonNet to train.
    ```
    mp3d_layout/
    |--train/
    |  |--img/
    |  |  |--*.png
    |  |--label_cor/
    |  |  |--*.txt
    |--valid/
    |  |--img/
    |  |  |--*.png
    |  |--label_cor/
    |  |  |--*.txt
    |--test/
    |  |--img/
    |  |  |--*.png
    |  |--label_cor/
    |  |  |--*.txt
    ```

## Training
**Work in progress**
<!-- ```bash
python train.py --train_root_dir data/st3d_train_full_raw_light/ --valid_root_dir data/st3d_valid_full_raw_light/ --id resnet50_rnn__st3d --lr 3e-4 --batch_size_train 24 --epochs 50
```
See `python train.py -h` for more detail or [README.md](https://github.com/sunset1995/HorizonNet/blob/master/README.md) for more detail.

Download the trained model: [resnet50_rnn__st3d.pth](https://drive.google.com/open?id=16v1nhL9C2VZX-qQpikCsS6LiMJn3q6gO).
-->

## Testing
**Work in progress**
<!-- Generating layout for testing set:
```bash
python inference.py --pth ckpt/resnet50_rnn__st3d.pth --img_glob "data/st3d_test_full_raw_light/img/*" --output_dir tmp/ --visualize
```
- `--output_dir`: a directory you want to dump the extracted layout
- `--visualize`: visualize raw output (without post-processing) from HorizonNet.


Quantitativly evaluate:
```bash
python eval_general.py --dt_glob "./tmp/*json" --gt_glob "data/st3d_test_full_raw_light/label_cor/*"
``` -->

## Results
**Work in progress**
<!-- :clipboard: Below is the quantitative result on Structured3D testing set.

| # of corners | instances | 3D IoU | 2D IoU |
| :----------: | :-------: | :----: | :----: |
| 4            | 1067      | `94.14`  | `95.50` |
| 6            | 290       | `90.34`  | `91.54` |
| 8            | 130       | `87.98`  | `89.43` |
| 10+          | 202       | `79.95`  | `81.10` |
| overall      | 1693      | `91.31`  | `92.63` |

#### More Visual Results -->

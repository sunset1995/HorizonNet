import os
import argparse
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter
import shutil 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET
from dataset import PanoCorBonDataset
from misc.utils import adjust_learning_rate, save_model, load_trained_model
from inference import inference
from eval_general import test_general

class AugDataLoader():
    def __init__(self,aug_loader) -> None:
        self.aug_loader = aug_loader
        self.iter_loader = iter(self.aug_loader)
    
    def get_data(self):
        try:
            return next(self.iter_loader)
        except:
            self.iter_loader = iter(self.aug_loader)
            return next(self.iter_loader)
        
def save_checkpoint(state, is_best, checkpoint_dir, epoch):
    filename = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, f'best_model_{epoch}.pth.tar'))

def flatten_rnn_parameters(model):
    for module in model.modules():
        if isinstance(module, nn.RNNBase):  # Checks for all types of RNN layers
            module.flatten_parameters()

def feed_forward(net, x, y_bon, y_cor):
    flatten_rnn_parameters(net.module if hasattr(net, 'module') else net)
    x = x.to(device)
    y_bon = y_bon.to(device)
    y_cor = y_cor.to(device)
    losses = {}

    with autocast():
        y_bon_, y_cor_ = net(x)
        losses['bon'] = F.l1_loss(y_bon_, y_bon)
        losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor)
    
    losses['total'] = losses['bon'] + losses['cor']

    return losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', required=True,
                        help='experiment id to name checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--logs', default='./logs',
                        help='folder to logging')
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint.'
                             '(finetuning)')
    # Model related
    parser.add_argument('--backbone', default='resnet50',
                        choices=ENCODER_RESNET + ENCODER_DENSENET,
                        help='backbone of the network')
    parser.add_argument('--no_rnn', action='store_true',
                        help='whether to remove rnn or not')
    # Dataset related arguments
    parser.add_argument('--train_root_dir', default='/media/public_dataset/Stanford2D3D/Stanford2D3D_layout/train',
                        help='root directory to training dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--train_aug_root_dir', default=None,
                         help='root directory to training dataset. '
                         'should contains img, label_cor subdirectories')
    parser.add_argument('--valid_root_dir', default='/media/public_dataset/Stanford2D3D/Stanford2D3D_layout/test',
                        help='root directory to validation dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--no_flip', action='store_true',
                        help='disable left-right flip augmentation')
    parser.add_argument('--no_rotate', action='store_true',
                        help='disable horizontal rotate augmentation')
    parser.add_argument('--no_gamma', action='store_true',
                        help='disable gamma augmentation')
    parser.add_argument('--no_pano_stretch', action='store_true',
                        help='disable pano stretch')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='numbers of workers for dataloaders')
    # optimization related arguments
    parser.add_argument('--freeze_earlier_blocks', default=-1, type=int)
    parser.add_argument('--batch_size_train', default=8, type=int,
                        help='training mini-batch size')
    parser.add_argument('--batch_size_valid', default=2, type=int,
                        help='validation mini-batch size')
    parser.add_argument('--epochs', default=300, type=int,
                        help='epochs to train')
    parser.add_argument('--optim', default='Adam',
                        help='optimizer to use. only support SGD and Adam')
    parser.add_argument('--lr', default=3*1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--warmup_lr', default=1e-6, type=float,
                        help='starting learning rate for warm up')
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='numbers of warmup epochs')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='factor for L2 regularization')
    parser.add_argument('--bn_momentum', type=float)
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='enable data parallelism on multiple GPUs')
    parser.add_argument('--device', default='0',
                        help='Comma-separated list of GPUs to use (e.g., 0 or 0,1)')
    parser.add_argument('--seed', default=594277, type=int,
                        help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='iterations frequency to display')
    parser.add_argument('--save_every', type=int, default=25,
                        help='epochs frequency to save state_dict')
    args = parser.parse_args()

    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

    # Create dataloader
    if args.train_aug_root_dir:
        train_batch_size, train_aug_batch_size = args.batch_size_train//2, args.batch_size_train//2
        print(args.train_aug_root_dir)
        dataset_train_aug = PanoCorBonDataset(
            root_dir=args.train_aug_root_dir,
            flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
            stretch=not args.no_pano_stretch)
        loader_train_aug = DataLoader(dataset_train_aug, train_aug_batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=args.num_workers,
                            pin_memory=not args.no_cuda,
                            worker_init_fn=lambda x: np.random.seed())
        print(f'train batch size: {train_batch_size}, train augmentation batch size: {train_aug_batch_size}')
        print(f'training augmentation dataset contains {len(loader_train_aug.dataset)} images!!!')
    else:
        train_batch_size = args.batch_size_train

    train_batch_size = args.batch_size_train
    
    dataset_train = PanoCorBonDataset(
    root_dir=args.train_root_dir,
    flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
    stretch=not args.no_pano_stretch)
    loader_train = DataLoader(dataset_train, train_batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=lambda x: np.random.seed())
    print(f'training dataset contains {len(loader_train.dataset)} images!!!')
    
    if args.valid_root_dir:
        dataset_valid = PanoCorBonDataset(
            root_dir=args.valid_root_dir, return_cor=True,
            flip=False, rotate=False, gamma=False,
            stretch=False, valid= True)
        print(f'valid dataset contains {len(dataset_valid)} images!!!')

    # Create model
    if args.pth is not None:
        print('Finetune model is given.')
        print('Ignore --backbone and --no_rnn')
        net = load_trained_model(HorizonNet, args.pth).to(device)
        flatten_rnn_parameters(net.module if hasattr(net, 'module') else net)
    else:
        net = HorizonNet(args.backbone, not args.no_rnn).to(device)
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for Data Parallelism")
        net = nn.DataParallel(net)
    elif not args.no_cuda:
        net = net.to(device)  
    else:
        print("Using CPU")

    flatten_rnn_parameters(net.module if hasattr(net, 'module') else net)

    assert -1 <= args.freeze_earlier_blocks and args.freeze_earlier_blocks <= 4
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            print('Freeze block%d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

    if args.bn_momentum:
        for m in net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = args.bn_momentum

    # Create optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    scaler = GradScaler()
    # Create tensorboard for monitoring training
    tb_path = os.path.join(args.logs, args.id)
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    # Init variable
    args.warmup_iters = args.warmup_epochs * len(loader_train)
    args.max_iters = args.epochs * len(loader_train)
    args.running_lr = args.warmup_lr if args.warmup_epochs > 0 else args.lr
    args.cur_iter = 0
    args.best_valid_score = 0
    
    if args.train_aug_root_dir:
        aug_loader = AugDataLoader(loader_train_aug)    
    else:
        aug_loader = None

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):

        # Train phase
        net.train()
        flatten_rnn_parameters(net.module if hasattr(net, 'module') else net)
        if args.freeze_earlier_blocks != -1:
            b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
            blocks = [b0, b1, b2, b3, b4]
            for i in range(args.freeze_earlier_blocks + 1):
                for m in blocks[i]:
                    m.eval()
        iterator_train = iter(loader_train)
        for _ in trange(len(loader_train),
                        desc='Train ep%s' % ith_epoch, position=1):
            # Set learning rate
            adjust_learning_rate(optimizer, args)

            args.cur_iter += 1
            x, y_bon, y_cor = next(iterator_train)
            
            if aug_loader is not None:
                x_aug, y_bon_aug, y_cor_aug = aug_loader.get_data()
                x = torch.cat((x,x_aug),axis=0)
                y_bon = torch.cat((y_bon,y_bon_aug),axis=0)
                y_cor = torch.cat((y_cor,y_cor_aug),axis=0)
                
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                losses = feed_forward(net, x, y_bon, y_cor)
                loss = losses['total']

            # backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            for k, v in losses.items():
                k = 'train/%s' % k
                tb_writer.add_scalar(k, v.item(), args.cur_iter)
            tb_writer.add_scalar('train/lr', args.running_lr, args.cur_iter)
            loss = losses['total']


        # Valid phase
        net.eval()
        if args.valid_root_dir:
            valid_loss = {}
            for jth in trange(len(dataset_valid),
                            desc='Valid ep%d' % ith_epoch, position=2):
                x, y_bon, y_cor, gt_cor_id = dataset_valid[jth]
                x, y_bon, y_cor = x[None], y_bon[None], y_cor[None]
                with torch.no_grad():
                    losses = feed_forward(net, x, y_bon, y_cor)

                    # True eval result instead of training objective
                    true_eval = dict([
                        (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
                        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
                    ])
                    try:
                        dt_cor_id = inference(net, x, device, force_raw=True)[0]
                        dt_cor_id[:, 0] *= 1024
                        dt_cor_id[:, 1] *= 512
                    except:
                        dt_cor_id = np.array([
                            [k//2 * 1024, 256 - ((k%2)*2 - 1) * 120]
                            for k in range(8)
                        ])
                    test_general(dt_cor_id, gt_cor_id, 1024, 512, true_eval)
                    losses['2DIoU'] = torch.FloatTensor([true_eval['overall']['2DIoU']])
                    losses['3DIoU'] = torch.FloatTensor([true_eval['overall']['3DIoU']])
                    losses['rmse'] = torch.FloatTensor([true_eval['overall']['rmse']])
                    losses['delta_1'] = torch.FloatTensor([true_eval['overall']['delta_1']])

                for k, v in losses.items():
                    valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)

            for k, v in valid_loss.items():
                k = 'valid/%s' % k
                tb_writer.add_scalar(k, v / len(dataset_valid), ith_epoch)

            # Save best validation loss model
            now_valid_score = valid_loss['3DIoU'] / len(dataset_valid)
            print('Ep%3d %.4f vs. Best %.4f' % (ith_epoch, now_valid_score, args.best_valid_score))

            is_best = now_valid_score > args.best_valid_score
            if is_best:
                args.best_valid_score = now_valid_score

            # Prepare the dictionary to save
            save_dict = {'epoch': ith_epoch,
                         'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'best_valid_score': args.best_valid_score
                         }
            if hasattr(net, 'module'):
                save_dict['backbone'] = getattr(net.module, 'backbone', None) 
            else:
                save_dict['backbone'] = getattr(net, 'backbone', None)

            save_checkpoint(save_dict, is_best, os.path.join(args.ckpt, args.id), ith_epoch)

        # Periodically save model
        if ith_epoch % args.save_every == 0:
            save_model(net,
                       os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch),
                       args)

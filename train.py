import os
import argparse
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from model import HorizonNet, ENCODER_RESNET, ENCODER_DENSENET
from dataset import PanoCorBonDataset
from misc.utils import group_weight, adjust_learning_rate, save_model, load_trained_model


def feed_forward(net, x, y_bon, y_cor):
    x = x.to(device)
    y_bon = y_bon.to(device)
    y_cor = y_cor.to(device)
    losses = {}

    y_bon_, y_cor_ = net(x)
    losses['bon'] = F.l1_loss(y_bon_, y_bon)
    losses['cor'] = F.binary_cross_entropy_with_logits(y_cor_, y_cor)
    losses['total'] = losses['bon'] + losses['cor']

    # For model selection
    with torch.no_grad():
        nobrain_baseline_bon = np.pi / 4
        nobrain_baseline_cor = 0.2
        score_bon = 1 - (y_bon_ - y_bon).abs().mean() / nobrain_baseline_bon
        score_cor = 1 - (torch.sigmoid(y_cor_) - y_cor).abs().mean() / nobrain_baseline_cor
        losses['score'] = (score_bon + score_cor) / 2

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
    parser.add_argument('--train_root_dir', default='data/layoutnet_dataset/train',
                        help='root directory to training dataset. '
                             'should contains img, label_cor subdirectories')
    parser.add_argument('--valid_root_dir', default='data/layoutnet_dataset/valid',
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
    parser.add_argument('--num_workers', default=6, type=int,
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
    parser.add_argument('--lr', default=1e-4, type=float,
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
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--seed', default=594277, type=int,
                        help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='iterations frequency to display')
    parser.add_argument('--save_every', type=int, default=25,
                        help='epochs frequency to save state_dict')
    args = parser.parse_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

    # Create dataloader
    dataset_train = PanoCorBonDataset(
        root_dir=args.train_root_dir,
        flip=not args.no_flip, rotate=not args.no_rotate, gamma=not args.no_gamma,
        stretch=not args.no_pano_stretch)
    loader_train = DataLoader(dataset_train, args.batch_size_train,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=lambda x: np.random.seed())
    if args.valid_root_dir:
        dataset_valid = PanoCorBonDataset(
            root_dir=args.valid_root_dir,
            flip=False, rotate=False, gamma=False,
            stretch=False)
        loader_valid = DataLoader(dataset_valid, args.batch_size_valid,
                                  shuffle=False, drop_last=False,
                                  num_workers=args.num_workers,
                                  pin_memory=not args.no_cuda)

    # Create model
    if args.pth is not None:
        print('Finetune model is given.')
        print('Ignore --backbone and --no_rnn')
        net = load_trained_model(HorizonNet, args.pth).to(device)
    else:
        net = HorizonNet(args.backbone, not args.no_rnn).to(device)

    assert -1 <= args.freeze_earlier_blocks and args.freeze_earlier_blocks <= 4
    if args.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(args.freeze_earlier_blocks + 1):
            print('Freeze block%d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

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

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):

        # Train phase
        net.train()
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

            losses = feed_forward(net, x, y_bon, y_cor)
            for k, v in losses.items():
                k = 'train/%s' % k
                tb_writer.add_scalar(k, v.item(), args.cur_iter)
            tb_writer.add_scalar('train/lr', args.running_lr, args.cur_iter)
            loss = losses['total']

            # backprop
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type='inf')
            optimizer.step()

        # Valid phase
        net.eval()
        if args.valid_root_dir:
            iterator_valid = iter(loader_valid)
            valid_loss = {}
            for _ in trange(len(loader_valid),
                            desc='Valid ep%d' % ith_epoch, position=2):
                x, y_bon, y_cor = next(iterator_valid)
                with torch.no_grad():
                    losses = feed_forward(net, x, y_bon, y_cor)
                for k, v in losses.items():
                    valid_loss[k] = valid_loss.get(k, 0) + v.item() * x.size(0)

            for k, v in valid_loss.items():
                k = 'valid/%s' % k
                tb_writer.add_scalar(k, v / len(dataset_valid), ith_epoch)

            # Save best validation loss model
            if valid_loss['score'] > args.best_valid_score:
                args.best_valid_score = valid_loss['score'] / len(dataset_valid)
                save_model(net,
                           os.path.join(args.ckpt, args.id, 'best_valid.pth'),
                           args)

        # Periodically save model
        if ith_epoch % args.save_every == 0:
            save_model(net,
                       os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch),
                       args)

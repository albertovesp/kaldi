# Copyright 2018 Zili Huangi
#           2019 Desh Raj

# Apache 2.0

import os
import torch
import argparse
import random
import torch.optim.lr_scheduler as lr_scheduler
from utils import train, train_mtl, validate, validate_mtl, save_checkpoint, record_info, prepare_utt2feat, compute_mean_std
import shutil
from dataprocess import SPKID_Dataset, SPKID_Dataset_MTL
from models.tdnn import tdnn_sid_xvector, tdnn_sid_xvector_1, tdnn_sid_xvector_2, tdnn_sid_xvector_multi
from models.metrics import *
from models.focal_loss import *
import numpy as np
import sys
import socket

print(socket.gethostname())

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(
    description='Pytorch implementation of x-vector training')
parser.add_argument('exp_dir', type=str,
                    help='path of experiment')
parser.add_argument('--epochs', default=3, type=int,
                    help='number of total epochs to run')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--initialize', default=0, type=int,
                    help='use checkpoint to initialize model parameters')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--loss', default='ce', type=str,
                    help='loss function (default: ce)')
parser.add_argument('--metric', default='none', type=str,
                    help='metric function')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer (default: sgd)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--min_lr', default=1e-4, type=float, help='minimum learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--arch', default='tdnn', type=str,
                    help='model architecture')
parser.add_argument('--feat_dim', default=23, type=int,
                    help='number of features for each frame')
parser.add_argument('--embed_dim', default=512, type=int,
                    help='dimension of embedding')
parser.add_argument('--num_classes', default=5139, type=int,
                    help='number of nnet output dimensions')
parser.add_argument('--train_num_egs', default=149, type=int,
                    help='number of training egs')
parser.add_argument('--valid_num_egs', default=3, type=int,
                    help='number of validation egs')
parser.add_argument('--data_dir', default="data/swbd_sre_combined_no_sil", type=str,
                    help='data directory')
parser.add_argument('--clean_data_dir', type=str, help='corresponding clean directory'
                    'for multitask objective in RIR embeddings')
parser.add_argument('--train_egs_dir', default="exp/xvector_nnet_1a/egs", type=str,
                    help='path of training egs')
parser.add_argument('--valid_egs_dir', default="exp/xvector_nnet_1a/egs", type=str,
                    help='path of validation egs')
parser.add_argument('--multi_gpu', default=0, type=int,
                    help='training with multiple gpus')
parser.add_argument('--clip_value', default=5, type=float,
                    help='clip the gradient value')
parser.add_argument('--sphere_m', default=2, type=int,
                    help='parameter for sphere loss')
parser.add_argument('--use_relu', default=1, type=int,
                    help='whether to use RELU before the last linear layer')
parser.add_argument('--seed', default=7, type=int,
                    help='random seed')
parser.add_argument('--num_workers', default=0, type=int,
                    help='number of workers for data loading (default: 0)')
parser.add_argument('--use_tfb', dest='use_tfboard',
                    help='whether use tensorboard',
                    action='store_true')
parser.add_argument('--use_multi', dest='use_multi', help='whether to use multitask'
                    'objective', action='store_true')

best_acc, best_iter = 0, -1

def main():
    global args, best_acc
    args = parser.parse_args()
    print(args)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.loss == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=2)
    else:
        raise ValueError("Loss type not defined.")

    if args.metric == 'none':
        metric_fc = NormalProduct(args.embed_dim, args.num_classes, args.use_relu)
    elif args.metric == 'add_margin':
        metric_fc = AddMarginProduct(args.embed_dim, args.num_classes, s=30, m=0.35)
    elif args.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(args.embed_dim, args.num_classes, s=30, m=0.5, easy_margin=False)
    elif args.metric == 'sphere':
        metric_fc = SphereProduct(args.embed_dim, args.num_classes, m=args.sphere_m)
    else:
        raise ValueError("Metric function not defined.")

    # prepare utt2feat
    utt2feat = prepare_utt2feat(args.data_dir)
    if(use_multi):
        utt2feat_clean = prepare_utt2feat(args.clean_data_dir)

    # load mean and std
    if os.path.exists("{}/mean.npy".format(args.train_egs_dir)) and os.path.exists("{}/std.npy".format(args.train_egs_dir)):
        mean, std = np.load("{}/mean.npy".format(args.train_egs_dir)), np.load("{}/std.npy".format(args.train_egs_dir))
        shutil.copyfile("{}/mean.npy".format(args.train_egs_dir), "{}/mean.npy".format(args.exp_dir))
        shutil.copyfile("{}/std.npy".format(args.train_egs_dir), "{}/std.npy".format(args.exp_dir))
    else:
        mean, std = compute_mean_std(utt2feat, 5000)
        np.save("{}/mean.npy".format(args.train_egs_dir), mean)
        np.save("{}/std.npy".format(args.train_egs_dir), std)
        shutil.copyfile("{}/mean.npy".format(args.train_egs_dir), "{}/mean.npy".format(args.exp_dir))
        shutil.copyfile("{}/std.npy".format(args.train_egs_dir), "{}/std.npy".format(args.exp_dir))
    print("mean", mean, "std", std)

    # model
    if args.arch == 'tdnn':
        if(!use_multi):
            model = tdnn_sid_xvector(args.feat_dim)
        else:
            model = tdnn_sid_xvector_multi(args.feat_dim)
    elif args.arch == 'tdnn1':
        model = tdnn_sid_xvector_1(args.feat_dim) 
    elif args.arch == 'tdnn2':
        model = tdnn_sid_xvector_2(args.feat_dim) 
    else:
        raise ValueError("Model type not defined.")

    print(model)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
        metric_fc = torch.nn.DataParallel(metric_fc)
        print("Training with {} GPUs".format(torch.cuda.device_count()))
    model, metric_fc = model.to(device), metric_fc.to(device)

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer type not defined.")

    start_epoch, start_subfile = 1, 1
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.initialize:
                start_epoch = checkpoint['epoch']
                start_subfile = checkpoint['subfile'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            metric_fc.load_state_dict(checkpoint['metric_fc'])
            print("=> loaded checkpoint '{}' (epoch {} subfile {})"
                  .format(args.resume, checkpoint['epoch'], checkpoint['subfile']))
            print("=> best acc {}".format(best_acc))
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(args.resume))

    # define learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, verbose=True, min_lr=args.min_lr)

    # create the validation dataset and dataloader
    valset_list = []
    for subfile in range(1, 1 + args.valid_num_egs):
        valid_egs_filename = "{}/valid_egs.{}.scp".format(args.valid_egs_dir, subfile)
        if (use_multi):
            valset = SPKID_Dataset_MTL(valid_egs_filename, mean, std, utt2feat, utt2feat_clean)
        else:
            valset = SPKID_Dataset(valid_egs_filename, mean, std, utt2feat)
        valset_list.append(valset)
    sys.stdout.flush()

    # use tensorboard to monitor the loss
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("{}/log".format(args.exp_dir))
    else:
        logger = None

    # train
    iterations = 0
    for epoch in range(1, 1 + args.epochs):
        for subfile in range(1, 1 + args.train_num_egs):
            iterations += 1
            if (epoch < start_epoch) or (epoch == start_epoch and subfile < start_subfile):
                continue
            # training
            egs_filename = "{}/egs.{}.scp".format(args.train_egs_dir, subfile)
            if (use_multi):
                trainset = SPKID_Dataset_MTL(egs_filename, mean, std, utt2feat, utt2feat_clean)
            else:
                trainset = SPKID_Dataset(egs_filename, mean, std, utt2feat)
            trainloader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, 
                    batch_size=args.batch_size, pin_memory=True, shuffle=True)
            if(use_multi):
                train_info = train_mtl(trainloader, model, device, criterion, metric_fc, optimizer, args)
            else:
                train_info = train(trainloader, model, device, criterion, metric_fc, optimizer, args)
            print("Train -- Iter: {:04d}, LR: {:.6f}, Loss: {:.4f}, Acc@1: {:.3f}, Acc@5: {:.3f}".format(
                iterations, optimizer.param_groups[0]['lr'], train_info['loss'], train_info['top1'], train_info['top5']))
            sys.stdout.flush()

            # validation
            if(use_multi):
                dev_info = validate_mtl(valset_list, model, device, criterion, metric_fc, args)
            else:
                dev_info = validate(valset_list, model, device, criterion, metric_fc, args)
            print('Valid Loss: {:.4f}, Acc@1: {:.3f}, Acc@5: {:.3f}'.format(dev_info['loss'], dev_info['top1'], dev_info['top5']))
            sys.stdout.flush()
            val_acc = dev_info['top1']

            scheduler.step(val_acc)

            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if is_best:
                best_iter = iterations

            save_checkpoint({
                'epoch': epoch,
                'subfile': subfile,
                'state_dict': model.state_dict(),
                'metric_fc': metric_fc.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, "{}/model/checkpoint.pth.tar".format(args.exp_dir))
            if is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'subfile': subfile,
                    'state_dict': model.state_dict(),
                    'metric_fc': metric_fc.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, "{}/model/modelbest.pth.tar".format(args.exp_dir))

            # save the model every 10 subfiles
            if iterations % 10 == 0:
                shutil.copyfile("{}/model/checkpoint.pth.tar".format(args.exp_dir), '{}/model/iterations_{}.pth.tar'.format(args.exp_dir, iterations))
            
            # record training and validation information
            if args.use_tfboard:
                record_info(train_info, dev_info, iterations, logger)

    if args.use_tfboard:
        logger.close()

    print("Best Iteration {}, Best Acc {}".format(best_iter, best_acc))
    print("Finish")
            
if __name__ == "__main__":
    main()

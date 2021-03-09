#!/usr/bin/python3
#coding=utf-8

import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)



import sys
import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
from net  import GCPANet
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt
from net import FCRM

TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum

BASE_LR = 1e-3
MAX_LR = 0.1
FIND_LR = False #True
IMAGE_GROUP = 0


def train(Dataset, Network1, Network2):
    ## dataset GCPANet
    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train_' + str(IMAGE_GROUP), batch=16, lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    ## dataset FCRNet
    cfg2   = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train_' + str(IMAGE_GROUP), batch=8, lr=1e-4, momen=0.9, decay=5e-4, epoch=30)
    data2  = Dataset.Data(cfg2)
    loader2= DataLoader(data2, batch_size=cfg2.batch, shuffle=True, num_workers=8)
    ## network
    net    = Network1(cfg)
    net2   = Network2(cfg2)
    ##print('net=', net)
    net.cuda()
    net2.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    # optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    optimizer   = torch.optim.Adam([{'params':base}, {'params':head}], lr=cfg.lr, weight_decay=cfg.decay)

    for name, param in net2.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    # optimizer2  = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg2.lr, momentum=cfg2.momen, weight_decay=cfg2.decay, nesterov=True)
    optimizer2 = torch.optim.Adam([{'params': base}, {'params': head}], lr=cfg2.lr, weight_decay=cfg2.decay)

    sw          = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size  = len(loader)
    db_size2 = len(loader2)
    print('db_size2=', db_size2)
    if FIND_LR:
        lr_finder = LRFinder(net, optimizer, criterion=None)
        lr_finder.range_test(loader, end_lr=50, num_iter=100, step_mode="exp")
        plt.ion()
        lr_finder.plot()
        import pdb; pdb.set_trace()
    for epoch_g in range(5):
        if epoch_g == 0: mode_refine = 'train_0'
        elif epoch_g == 1: mode_refine = 'train_0_2'
        elif epoch_g == 2: mode_refine = 'train_0_2_4'
        elif epoch_g == 3:
            mode_refine = 'train_0_2_4_6'
        elif epoch_g == 4:
            mode_refine = 'train_0_2_4_6_8'
        cfg2 = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_refine, batch=8, lr=1e-4,
                                  momen=0.9, decay=5e-4, epoch=10)
        data2 = Dataset.Data(cfg2)
        loader2 = DataLoader(data2, batch_size=cfg2.batch, shuffle=True, num_workers=8)
        db_size2 = len(loader2)
        scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[10, 20], gamma=0.1)
        i = 0
        # for image, mask, coarse, (H, W), name in loader2:
        #     i += 1
        #     print('i=', i)
        #     print('name=', name)
        prefetcher = DataPrefetcher(loader)
        for epoch in range(30):
            prefetcher = DataPrefetcher(loader2)
            batch_idx = -1
            image, mask, coarse = prefetcher.next()
            image_num = 0
            while image is not None:
                # niter = epoch * db_size2 + batch_idx
                # lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, 30 * db_size2 * 5, niter, ratio=1.)
                # # print('lr=', lr, 'momentum', momentum, 'db_size=', db_size)
                # optimizer2.param_groups[0]['lr'] = 0.1 * lr  # for backbone
                # if epoch_g == 0:
                #     optimizer2.param_groups[1]['lr'] = lr
                # else:
                #     optimizer2.param_groups[1]['lr'] = lr * 0.5
                # optimizer2.momentum = momentum
                lr = scheduler2.get_lr()[0]
                optimizer2.param_groups[0]['lr'] = 0.1 * lr
                optimizer2.param_groups[1]['lr'] = lr
                batch_idx += 1
                global_step += 1
                #print('coarse=', coarse)
                # coarse = cv2.resize(coarse.cpu().numpy(), dsize=(9, 9), interpolation=cv2.INTER_LINEAR)
                #             # coarse = torch.from_numpy(coarse)
                if torch.is_tensor(image) == False:

                    print('image_num=', image_num, 'mask=', mask)
                out2, out3, out4, out5 = net2(image, coarse)
                train_img = torch.mul(torch.sigmoid(out2), image)
                gt_img = torch.mul(image, mask)
                #print('out2=', out2)
                loss2 = F.binary_cross_entropy_with_logits(out2, mask)
                loss3 = F.binary_cross_entropy_with_logits(out3, mask)
                loss4 = F.binary_cross_entropy_with_logits(out4, mask)
                loss5 = F.binary_cross_entropy_with_logits(out5, mask)
                loss6 = F.binary_cross_entropy_with_logits(train_img, gt_img)
                # print('loss6=', loss6)
                loss = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4 + loss6*0.8
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

                sw.add_scalar('lr', optimizer2.param_groups[0]['lr'], global_step=global_step)
                sw.add_scalars('loss',
                               {'loss2': loss2.item(), 'loss3': loss3.item(), 'loss4': loss4.item(), 'loss5': loss5.item(),
                                'loss6': loss6.item(), 'loss': loss.item()}, global_step=global_step)
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f | loss6=%.6f' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, epoch_g+1 ,optimizer2.param_groups[0]['lr'],
                    loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item())
                    print(msg)
                    logger.info(msg)
                image, mask, coarse = prefetcher.next()
                image_num += 1
            scheduler2.step()

            if (epoch + 1) % 10 == 0:
                torch.save(net2.state_dict(), cfg.savepath + '/model-refine' + str(epoch_g) + '-' + str(epoch + 1) + '.pt')

        #test dataset
        if epoch_g == 0: refine_test = 'test_1'
        elif epoch_g == 1: refine_test = 'test_3'
        elif epoch_g == 2: refine_test = 'test_5'
        elif epoch_g == 3:
            refine_test = 'test_7'
        elif epoch_g == 4:
            refine_test = 'test_9'
        cfg_test  = Dataset.Config(datapath='./data/DUTS', mode=refine_test)
        data_test = Dataset.Data(cfg_test)
        loader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=8)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.05)
        #test network
        net_test = net2
        net_test.train(False)
        # net_test.cuda()
        net_test.eval()

        with torch.no_grad():
            i = 0
            for image, mask, coarse, (H, W), name in loader_test:
                image, coarse          = image.cuda().float(), coarse.cuda().float()
                out2, out3, out4, out5 = net_test(image, coarse)
                # print('H=', H, 'W=', W, 'name=', name)
                out2 = F.interpolate(out2, size=(H,W), mode='bilinear')
                pred = (torch.sigmoid(out2[0, 0]) * 255).cpu().numpy()
                head = './data/DUTS/coarse_refine_' + str(epoch_g)
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0], np.uint8(pred))
                # print(name[0])
                # cv2.imwrite('./data/DUTS/coarse/' + name[0], np.uint8(pred))
                name = name[0].split('.')
                if i == 0:
                    cv2.imwrite('./data/DUTS/log_coarse/' + name[0] + '_' + str(epoch_g) + '_refine.png', np.uint8(pred))
                cv2.imwrite('./data/DUTS/coarse/' + name[0] + '_RBD.png', np.uint8(pred))
                i += 1

        ## dataset GCPANet
        if epoch_g == 0: mode_sod = 'train_0_1'
        elif epoch_g == 1: mode_sod = 'train_0_1_3'
        elif epoch_g == 2: mode_sod = 'train_0_1_3_5'
        elif epoch_g == 3:
            mode_sod = 'train_0_1_3_5_7'
        elif epoch_g == 4:
            mode_sod = 'train_0_1_3_5_7_9'
        cfg = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode=mode_sod, batch=8, lr=1e-4, momen=0.9,
                                 decay=5e-4, epoch=30)
        data = Dataset.Data(cfg)
        loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
        db_size = len(loader)
        prefetcher = DataPrefetcher(loader)
        prefetcher = DataPrefetcher(loader)
        prefetcher = DataPrefetcher(loader)

        #training
        global_step = 0
        for epoch in range(cfg.epoch):
            prefetcher = DataPrefetcher(loader)
            batch_idx = -1
            image, mask, coarse = prefetcher.next()
            while image is not None:
                # niter = epoch * db_size + batch_idx
                # lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size*5, niter, ratio=1.)
                # optimizer.param_groups[0]['lr'] = 0.1 * lr #for backbone
                # if epoch_g == 0:
                #     optimizer.param_groups[1]['lr'] = lr
                # else:
                #     optimizer.param_groups[1]['lr'] = 0.5 * lr
                # optimizer.momentum = momentum
                lr = scheduler2.get_lr()[0]
                optimizer2.param_groups[0]['lr'] = 0.1 * lr
                optimizer2.param_groups[1]['lr'] = lr
                batch_idx += 1
                global_step += 1
                out2, out3, out4, out5 = net(image)
                loss2                  = F.binary_cross_entropy_with_logits(out2, coarse)
                loss3                  = F.binary_cross_entropy_with_logits(out3, coarse)
                loss4                  = F.binary_cross_entropy_with_logits(out4, coarse)
                loss5                  = F.binary_cross_entropy_with_logits(out5, coarse)
                loss                   = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
                sw.add_scalars('loss', {'loss2':loss2.item(), 'loss3':loss3.item(), 'loss4':loss4.item(), 'loss5':loss5.item(), 'loss':loss.item()}, global_step=global_step)
                if batch_idx % 10 == 0:
                    msg = '%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f'%(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, epoch_g+1, optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item())
                    print(msg)
                    logger.info(msg)
                image, mask, coarse = prefetcher.next()
            scheduler.step()

            if (epoch+1)%10 == 0 or (epoch+1)==cfg.epoch:
                torch.save(net.state_dict(), cfg.savepath+'/model-sod-' + str(epoch_g) + '-' + str(epoch+1) + '.pt')

        # test dataset
        if epoch_g == 0: sod_test = 'test_2'
        elif epoch_g == 1: sod_test = 'test_4'
        elif epoch_g == 2: sod_test = 'test_6'
        elif epoch_g == 3:
            sod_test = 'test_8'
        elif epoch_g == 4:
            sod_test = 'test_9'
        cfg_test = Dataset.Config(datapath='./data/DUTS', mode=sod_test)
        data_test = Dataset.Data(cfg_test)
        loader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=8)
        # test network
        net_test = net
        net_test.train(False)
        net_test.cuda()
        net_test.eval()
        with torch.no_grad():
            i = 0
            for image, mask, coarse, (H, W), name in loader_test:
                image                  = image.cuda().float()
                out2, out3, out4, out5 = net_test(image)
                out2 = F.interpolate(out2, size=(H,W), mode='bilinear')
                pred = (torch.sigmoid(out2[0, 0]) * 255).cpu().numpy()
                # print(name[0])
                name = name[0].split('.')
                cv2.imwrite('./data/DUTS/coarse/' + name[0] + '_RBD.png', np.uint8(pred))
                if i == 0:
                    cv2.imwrite('./data/DUTS/log_coarse/' + name[0] + '_' + str(epoch_g) + '_sod.png', np.uint8(pred))
                i += 1

if __name__=='__main__':
    train(dataset, GCPANet, FCRM)

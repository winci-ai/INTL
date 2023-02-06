import builtins
import math
import os
import random
import numpy as np
import time
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from cfg import get_cfg
from methods import get_method
from src.transform import Augmentations, ImageNetTransform, MultiCrops
from src.meter import AverageMeter, ProgressMeter
import wandb
import torchvision.datasets as datasets

def main():
    cfg = get_cfg()

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    print('world-size:',cfg.world_size)
    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node:',ngpus_per_node)
    if cfg.multiprocessing_distributed:
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(cfg.gpu, ngpus_per_node, cfg)

def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu
    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass
    cfg.wandb = 'intl_train'
    if cfg.gpu == 0 or cfg.gpu is None:
        wandb.init(project=cfg.wandb, name = cfg.env_name, config=cfg)
        
    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(cfg.arch))
    model = get_method(cfg.method)(cfg)

    cfg.base_lr = cfg.lr * cfg.bs / 256

    if cfg.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if cfg.gpu is not None:
            print('gpu:',cfg.gpu)
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            cfg.bs = int(cfg.bs / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:       
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    print(model) 

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)

    optimizer = torch.optim.SGD([{'params': param_biases, 'weight_decay': 0,},
                                {'params': param_weights, 'weight_decay': cfg.weight_decay}],
                                lr=0, momentum=0.9)

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    cfg = MultiCrops(cfg)

    # Data loading code 
    assert len(cfg.nmb_crops) == len(cfg.gaussian_prob)
    assert len(cfg.nmb_crops) == len(cfg.solarization_prob)
    assert len(cfg.nmb_crops) == len(cfg.crops_size)
    assert len(cfg.nmb_crops) == len(cfg.min_scale_crops)
    assert len(cfg.nmb_crops) == len(cfg.max_scale_crops)

    transforms = []
    for i in range(len(cfg.nmb_crops)):
        for _ in range(cfg.nmb_crops[i]):
            transforms.append(ImageNetTransform(crop_size = cfg.crops_size[i], min_scale = cfg.min_scale_crops[i],
                                max_scale = cfg.max_scale_crops[i], gaussian_prob = cfg.gaussian_prob[i], 
                                solarization_prob = cfg.solarization_prob[i])  )
                                
    traindir = os.path.join(cfg.data_path, 'train')
    train_dataset = datasets.ImageFolder(traindir, Augmentations(transforms))

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.bs, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    cfg.total_steps = cfg.epochs * len(train_loader)
    cfg.warmup_steps = cfg.warmup_eps * len(train_loader)

    print(cfg)  

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        loss = train(train_loader, model, optimizer, scaler, epoch, cfg)

        if cfg.gpu == 0 or cfg.gpu is None:
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], 
                        "m": model.module.m, "loss": loss, "ep": epoch})
 
        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                and cfg.rank % ngpus_per_node == 0):
            save_pth = './ckpt'
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            state = {'epoch': epoch + 1,
                    'arch': cfg.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),}
            torch.save(state,'{}/ckpt_{}.pth.tar'.format(save_pth, cfg.env_name))

            if (epoch + 1) % 100 == 0:
                torch.save(state,'{}/ckpt_{}_{:04d}.pth.tar'.format(save_pth, cfg.env_name, epoch))
      
def train(train_loader, model, optimizer, scaler, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    loss_ep = []
    # train for one epoch
    for step, (samples, _) in enumerate(train_loader): 
        # measure data loading time
        data_time.update(time.time() - end)
        cur_steps = step + epoch * len(train_loader)
        adjust_learning_rate(optimizer, cfg, cur_steps)
        adjust_momentum(model, cfg, cur_steps)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model(samples)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_ep.append(loss.item())
        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % cfg.print_freq == 0:
            progress.display(step)

    return np.mean(loss_ep)

def adjust_learning_rate(optimizer, cfg, step):
    if step < cfg.warmup_steps:
        lr = cfg.base_lr * (step + 1) / cfg.warmup_steps
    else:
        step -= cfg.warmup_steps
        max_steps = cfg.total_steps - cfg.warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * (step + 1) / max_steps))
        end_lr = cfg.base_lr * 0.001
        lr = cfg.base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_momentum(model, cfg, step):
    if cfg.method == 'ins_m':
        model.module.m = 1. - 0.5 * (1. + math.cos(math.pi * step / cfg.total_steps)) * (1. - cfg.m)

if __name__ == '__main__':
    main()

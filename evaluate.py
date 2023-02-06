
import builtins
import math
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from cfg import get_cfg
import wandb
from torch.optim.lr_scheduler import MultiStepLR
from src.meter import AverageMeter, ProgressMeter

best_acc1 = 0

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
    if cfg.train_percent in {1, 10}:
        cfg.train_files = open('./src/percent/{}percent.txt'.format(cfg.train_percent), 'r').readlines()
 
    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:
        cfg.world_size = ngpus_per_node * cfg.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        main_worker(cfg.gpu, ngpus_per_node, cfg)

def main_worker(gpu, ngpus_per_node, cfg):
    global best_acc1
    cfg.gpu = gpu
    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    cfg.wandb = 'intl_eval'
    if cfg.gpu == 0 or cfg.gpu is None:
        wandb.init(project=cfg.wandb, name = cfg.env_name, config=cfg)

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(cfg.arch))
    model = models.__dict__[cfg.arch]()
    if cfg.dataset == 'in100':
        model.fc = nn.Linear(512, 100)

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if cfg.weights == 'freeze':
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    classifier_parameters, model_parameters = [], []
    for name, param in model.named_parameters():
        if name in {'fc.weight', 'fc.bias'}:
            classifier_parameters.append(param)
        else:
            model_parameters.append(param)

    cfg.lr_classifier = cfg.lr_classifier * cfg.bs / 256
    print("=> base classifier learning rate: ", cfg.lr_classifier)
    cfg.lr_backbone = cfg.lr_backbone * cfg.bs / 256
    print("=> base backbone learning rate: ", cfg.lr_backbone)

    param_groups = [dict(params=classifier_parameters, lr=cfg.lr_classifier)]
    if cfg.weights == 'finetune':
        param_groups.append(dict(params=model_parameters, lr=cfg.lr_backbone))
    optimizer = torch.optim.SGD(param_groups, 0, momentum=0.9, weight_decay=cfg.weight_decay)

    if cfg.schedule == 'step':
        scheduler = MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    # load from pre-trained, before DistributedDataParallel constructor
    if cfg.pretrained:
        if os.path.isfile(cfg.pretrained):
            print("=> loading checkpoint '{}'".format(cfg.pretrained))
            checkpoint = torch.load(cfg.pretrained, map_location="cpu")

            # rename ins pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
                    # remove prefix
                    state_dict[k[len("module.backbone."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            cfg.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}' (epoch {})".
                format(cfg.pretrained, checkpoint['epoch']))

            del checkpoint, state_dict
        else:
            print("=> no checkpoint found at '{}'".format(cfg.pretrained))

    if cfg.distributed:
        if cfg.gpu is not None:
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
        if cfg.arch.startswith('alexnet') or cfg.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
    if cfg.weights == 'freeze':
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
    print(model)

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
            best_acc1 = checkpoint['best_acc1']
            if cfg.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(cfg.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(cfg.data_path, 'train')
    valdir = os.path.join(cfg.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if cfg.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in cfg.train_files:
            fname = fname.strip()
            cls = fname.split('_')[0]
            pth = os.path.join(traindir, cls, fname)
            train_dataset.samples.append(
                (pth, train_dataset.class_to_idx[cls]))

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.bs, shuffle=(train_sampler is None),
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=cfg.workers, pin_memory=True)
    
    if cfg.evaluate:
        validate(val_loader, model, criterion, cfg)
        return

    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        if cfg.schedule == 'cos':
            adjust_learning_rate(optimizer, epoch, cfg)
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        if cfg.schedule == 'step':
            scheduler.step()

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, cfg)

        if cfg.gpu == 0 or cfg.gpu is None:
            wandb.log({"top1": acc1, "top5": acc5,"epoch": epoch,})
       
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not cfg.multiprocessing_distributed or (cfg.multiprocessing_distributed
                and cfg.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, cfg)

def train(train_loader, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if cfg.weights == 'finetune':
        model.train()
    elif cfg.weights == 'freeze':
        model.eval()
    else:
        assert False

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            images = images.cuda(cfg.gpu, non_blocking=True)
        target = target.cuda(cfg.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            target = target.cuda(cfg.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def save_checkpoint(state, is_best, cfg):
    filename = str(cfg.env_name)+'_lincls_ckpt.pth.tar' 
    torch.save(state, filename)
    if is_best:
        best_file = str(cfg.env_name) + '_lincls_best.pth.tar'
        shutil.copyfile(filename, best_file)

def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate based on schedule"""
    q =  0.5 * (1. + math.cos(math.pi * epoch / cfg.epochs))
    optimizer.param_groups[0]['lr'] = cfg.lr_classifier * q
    if cfg.weights == 'finetune':
        optimizer.param_groups[1]['lr'] = cfg.lr_backbone * q

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()

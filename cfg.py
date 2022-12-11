import argparse
from torchvision import models
from methods import METHOD_LIST

DS_LIST = ["in100","imagenet"]

def get_cfg():
    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    """ generates configuration from user input in console """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

    parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')

    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument(
        "--bs", type=int, default=512, help="train bs",
    )
    parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='initial (base) learning rate for train', dest='lr')

    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
                    
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to checkpoint for evaluation(default: none)')

    parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')

    parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')

    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

    parser.add_argument('--multiprocessing-distributed','--md', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument(
        "--method", type=str, choices=METHOD_LIST, default="ins", help="loss type",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="MID",
        help="name of the run for wandb project",
    )
    parser.add_argument(
        "--iters", type=int, default=4, help="ITN_iterations",
    )

    parser.add_argument(
        "--warmup_eps",
        type=int,
        default=2,
        help="epochs of learning rate warmup",
    )
    
    # data augmentation
    parser.add_argument("--nmb_crops", type=int, default=[1, 1], nargs="+",
                    help="list of number of crops (example: [1, 1, 6])")
                    
    parser.add_argument("--crops_size", type=int, default=[224, 224], nargs="+",
                    help="crops resolutions (example: [224, 224, 96])")

    parser.add_argument("--min_scale_crops", type=float, default=[0.08, 0.08], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.2, 0.14, 0.05])")

    parser.add_argument("--max_scale_crops", type=float, default=[1, 1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1.,0.95.,0.84.])")

    parser.add_argument("--gaussian_prob", type=float, default=[1.0, 0.1], nargs="+",
                    help="gaussian_prob (example: [1.0, 0.1, 0])")

    parser.add_argument("--solarization_prob", type=float, default=[0.0, 0.2], nargs="+",
                    help="gaussian_prob (example: [0.0, 0.2, 0])")
    
    parser.add_argument('--multicrop', type=int, default=0, help='multicrop')

    parser.add_argument(
        "--w_eps", type=float, default=0, help="eps for stability for whitening"
    )
    parser.add_argument(
        "--projection_layers", type=int, default=3, help="number of FC layers in projection"
    )
    parser.add_argument(
        "--projection_size", type=int, default=8192, help="size of FC layers in projection"
    )
    parser.add_argument("--emb", type=int, default=8192, help="embedding size")

    parser.add_argument(
        "--m", type=float, default=0.999, help="itn_momentum"
    )

    parser.add_argument("--dataset", type=str, choices=DS_LIST, default="imagenet")

    parser.add_argument("--data_path", type=str, default='data/ImageNet/')

    parser.add_argument("--axis", 
        type=int, 
        choices=[0,1],
        default=0, 
        help='0 for channel whitening, 1 for batch whitening')

    ###lincls argument
    parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')

    parser.add_argument('--lars', action='store_true',
                    help='Use lars optimizer')

    parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
    
    parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')

    parser.add_argument('--lr-classifier', default=0.1, type=float, metavar='LR',
                    help='classifier base learning rate')


    return parser.parse_args()

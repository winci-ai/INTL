import argparse
from torchvision import models

METHOD_LIST = ["mid","midm"]


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
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument(
        "--bs", type=int, default=256, help="train bs",
    )
    parser.add_argument('--m', default=0.999, type=float, metavar='L',
                        help='itn_momentum')
                        
    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                    metavar='LR', help='initial (base) learning rate for train', dest='lr')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
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
        "--method", type=str, choices=METHOD_LIST, default="mid", help="loss type",
    )

    parser.add_argument(
        "--env_name",
        type=str,
        default="MID_cifar",
        help="name of the run for wandb project",
    )

    parser.add_argument(
        "--nmb_crops",
        type=int,
        default=2,
        help="number of samples (d) generated from each image",
    )

    parser.add_argument("--solarization_prob", type=float, default=[0.0, 0.2], nargs="+",
                    help="gaussian_prob (example: [0.0, 0.2])")

    parser.add_argument(
        "--iters", type=int, default=4, help="ITN_iterations",
    )

    parser.add_argument(
        "--warmup_eps",
        type=int,
        default=2,
        help="epochs of learning rate warmup",
    )

    parser.add_argument("--knn", type=int, default=5, help="k in k-nn classifier")

    parser.add_argument(
        "--w_eps", type=float, default=0, help="eps for stability for whitening"
    )
    parser.add_argument(
        "--projection_layers", type=int, default=3, help="number of FC layers in projection"
    )
    parser.add_argument(
        "--projection_size", type=int, default=2048, help="size of FC layers in projection"
    )
    parser.add_argument("--emb", type=int, default=2048, help="embedding size")

    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10")

    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument(
        "--eval_every", type=int, default=5, help="how often to evaluate"
    )
    parser.add_argument("--axis", 
        type=int, 
        choices=[0,1],
        default=0, 
        help='0 for channel whitening, 1 for batch whitening')



    return parser.parse_args()

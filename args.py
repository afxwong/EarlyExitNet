import argparse
import datetime
import os
import logging

def modify_args(args):
    args.splits = ['train', 'val']
    args.log_level = logging.getLevelName(args.log_level.upper())

    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    elif args.data == 'imagenette':
        args.num_classes = 10

    if not hasattr(args, "save_path") or args.save_path is None:
        args.save_path = os.path.join("models", args.arch, args.data, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if args.data.startswith('cifar'):
        args.image_size = (32, 32)
    elif args.data == 'imagenette':
        args.image_size = (224, 224)
    return args


model_names = ['vgg11_bn', 'resnet50', 'densenet121']

arg_parser = argparse.ArgumentParser(
    description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save-path', default=None,
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--use_gpu', default=True, type=bool, help='Use CPU if False')
exp_group.add_argument('--log-level', default="INFO", type=str, help='Logging level')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='imagenette',
                        choices=['cifar10', 'cifar100', 'imagenette'],
                        help='data to work on')
data_group.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')

# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('-b', '--batch-size', default=32, type=int, help='batch size for train dataloader')
optim_group.add_argument('-tb', '--test-batch-size', default=32, type=int, help='batch size for test dataloader')
optim_group.add_argument('--classifier-epochs', default=30, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')

# inference related
optim_group = arg_parser.add_argument_group('inference', 'inference setting')
optim_group.add_argument('--alpha', default=0.5, type=float, metavar='N',
                        help='alpha value for training classification layers')
optim_group.add_argument('--gate-epochs', default=30, type=int, metavar='N',
                         help='manual epoch number (useful on restarts)')
# optim_group.add_argument('--val_budget', type=float,
#                          help='average inference budget per sample, scans range if not provided')
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import wide_models
import mini_loader
import numpy as np
import pprint
import torch.nn.functional as F
import random
import pandas as pd

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--data', metavar='DIR', default='MY_mini_data',
                    help='path to dataset') ##downloaded
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-number', default=64, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.04, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-c', '--checkpoint',
                    default='TransMatch_5w5s15q_100u_25e',
                    type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_ft_checkpoint)')
parser.add_argument('--unlabelnumber', default=100, type=int, help='The number of unlabeled images per class')
parser.add_argument('--model', default='pretrained_model_on_base_class.pth.tar',
                    type=str, metavar='PATH',
                    help='path to model (default: none)')  ##downloaded
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_false',
                    help='evaluate model on validation set')
parser.add_argument('--random', action='store_true',
                    help='whether use random novel weights')
parser.add_argument('--num-way', default=5, type=int, help='N-way, number of novel classes')
parser.add_argument('--num-sample', default=5, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_false', help='whether only test on novel classes')
parser.add_argument('--T', default=0.5, type=float, help='temperature when guessing labels in MixMatch')
parser.add_argument('--alpha', default=0.75, type=float, help='parameters in Beta-distribution in MixMatch')
parser.add_argument('--lambda-u', default=5, type=float, help='coefficients for the consistency loss in MixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float, help='coefficient for exponential moving average')
parser.add_argument('--gamma', default=0.2, type=float, help='learning rate decay rate')
parser.add_argument('--step_size', default=100, type=int, help='learning rate decay step')
parser.add_argument('--extend', action='store_false',
                    help='whether to extend the dataset repeatly (to make it larger for training convenience)')
parser.add_argument('--querynumber', default=15, type=int, help='query number')
parser.add_argument('--rampup', default=0, type=int)
parser.add_argument('--aug', default=10, type=int, help='augmentation times during imprinting')
parser.add_argument('--distractor', action='store_true',
                    help='whether use distractor classes for unlabeled data')
parser.add_argument('--distractor_class', default=3, type=int, help='number of distractor classes')
_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def main():
    global args
    final_result = []
    imprinting_result = []
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pprint(vars(args))

    ##ema: exponential moving average for model parameters
    def create_model(ema=False):
        model = wide_models.Net(num_classes=args.num_way).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    for seed in range(600):  ##repeat 600 test experiments
        model = create_model()
        ema_model = create_model(ema=True)

        ################Loading the pretrained model on base-class data (start)#######################

        print('==> Reading from model checkpoint..')
        assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
        checkpoint = torch.load(args.model)

        need_dict = {k: v for k, v in checkpoint['state_dict'].items() if k != 'classifier.fc.weight'}
        model_dict = model.state_dict()
        model_dict.update(need_dict)
        model.load_state_dict(model_dict)

        ema_model_dict = ema_model.state_dict()
        ema_model_dict.update(need_dict)
        ema_model.load_state_dict(ema_model_dict)
        ################Loading the pretrained model on base-class data (end)#######################
        print("=> loaded model checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))
        cudnn.benchmark = True

        #################### Data loading code (start)###############################################
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        random.seed(seed)
        fixed_class = np.sort((random.sample(range(80, 100), args.num_way)))  ## randomly choose 5 classes
        train_idx = []
        unlabel_idx = []
        val_idx = []
        data_csv = pd.read_csv('MY_mini_data/novel_data.txt')
        # print(fixed_class)
        for i in fixed_class:  ## randomly choose labeled and unlabeled data from the 5 classes
            temp_index = random.sample(list(data_csv[data_csv['label'] == i]['idx']), 600)  ## prevent the bug from random seed
            train_idx += temp_index[0:args.num_sample]
            val_idx += temp_index[args.num_sample:(args.num_sample + args.querynumber)] ## query images
            unlabel_idx += temp_index[(args.num_sample + args.querynumber):(
                        args.num_sample + args.querynumber + args.unlabelnumber)]

        if args.distractor:  ## if unlabeled data also comes from distractor classes
            distract_class = random.sample([x for x in range(80, 100) if x not in fixed_class], args.distractor_class)
            for i in distract_class:
                temp_dis_index = random.sample(list(data_csv[data_csv['label'] == i]['idx']), 600)
                unlabel_idx += temp_dis_index[0:args.unlabelnumber]

        novel_transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(80),
            transforms.ToTensor(),
            normalize]) if not args.aug else transforms.Compose([
            transforms.Resize(92),
            transforms.RandomCrop(80),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        novel_dataset = mini_loader.novel_ImageLoader(
            root=args.data, give_idx=train_idx, transform=novel_transform, unlabeled=False, aug=args.aug)

        novel_loader = torch.utils.data.DataLoader(
            novel_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        train_labeled_dataset = mini_loader.novel_ImageLoader(
            root=args.data, give_idx=train_idx, transform=transforms.Compose([
                transforms.Resize(92),
                transforms.RandomCrop(80),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), unlabeled=False, extend=args.extend)

        train_labeled_loader = torch.utils.data.DataLoader(
            train_labeled_dataset, batch_size=args.batch_size,  # sampler=train_dataset.get_balanced_sampler(),
            num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

        train_unlabeled_dataset = mini_loader.novel_ImageLoader(
            root=args.data, give_idx=unlabel_idx, transform=transforms.Compose([
                transforms.Resize(92),
                transforms.RandomCrop(80),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), unlabeled=True)
        train_unlabeled_loader = torch.utils.data.DataLoader(
            train_unlabeled_dataset, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)

        val_dataset = mini_loader.novel_ImageLoader(root=args.data, give_idx=val_idx, transform=transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(80),
            transforms.ToTensor(),
            normalize,
        ]), unlabeled=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        #################### Data loading code (end)###############################################

        ################ imprint weights for the classifier (start) ##########################################
        imprint(novel_loader, model)
        ################ imprint weights for the classifier (end) ##########################################

        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        print('length', len(val_dataset))
        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        train_criterion = SemiLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
        title = 'Impriningt + FT'

        logger = Logger(os.path.join(args.checkpoint, 'log' + str(seed) + '.txt'), title=title)
        logger.set_names(['Learning_Rate', 'Train_Loss', 'Train_Loss_X', 'Train_Loss_U', 'Valid_Loss', 'Valid_Acc'])

        #################See the results of Imprinting (start)########################
        if args.evaluate:
            print('Do a quick evaluation')
            imprinting_loss, imprinting_acc = validate(val_loader, model, criterion)
            imprinting_result.append(imprinting_acc)
        #################See the results of Imprinting (end)########################

        ################Fine-tuning with unlabeled data by MixMatch#####################
        for epoch in range(args.start_epoch, args.epochs):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
            #
            train_loss, train_loss_x, train_loss_u = train(train_labeled_loader, train_unlabeled_loader, model,
                                                           train_criterion, optimizer, ema_optimizer, epoch)

            ## evaluate on validation set
            test_loss, test_acc = validate(val_loader, model, criterion)
            ## append logger file
            logger.append([lr, train_loss, train_loss_x, train_loss_u, test_loss, test_acc])
        final_result.append(test_acc)
        logger.close()
        logger.plot()
        #each log+X.txt and log+X.png represents the Xth test result
        savefig(os.path.join(args.checkpoint, 'log' + str(seed) + '.png'))
        ##final_result.csv saves 600 experiment results for TransMatch
        ##imprinting_result.csv saves 600 experiment results for Imprinting
        pd.DataFrame(final_result).to_csv(os.path.join(args.checkpoint, 'final_result.csv'))
        pd.DataFrame(imprinting_result).to_csv(os.path.join(args.checkpoint, 'imprinting_result.csv'))
        


################Fine-tuning with unlabeled data by TransMatch#####################

###########END############


def imprint(novel_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Imprinting', max=len(novel_loader))
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input = input.cuda()
            # compute output
            output = model.extract(input)

            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                batch=batch_idx + 1,
                size=len(novel_loader),
                data=data_time.val,
                bt=batch_time.val,
                total=bar.elapsed_td,
                eta=bar.eta_td
            )
            bar.next()
        bar.finish()
    new_weight = torch.zeros(args.num_way, 256)
    for i in range(args.num_way):
        tmp = output_stack[target_stack == i].mean(0) if not args.random else torch.randn(256)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = new_weight.cuda()
    model.classifier.fc = nn.Linear(256, args.num_way, bias=False)
    model.classifier.fc.weight.data = weight


# training by MixMatch
def train(labeled_trainloader, unlabeled_trainloader, model, criterion, optimizer, ema_optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training  ', max=len(labeled_trainloader) + len(unlabeled_trainloader))
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    for batch_idx in range(args.batch_number):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()
        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)
        ## (targets_x)
        targets_x = torch.zeros(batch_size, args.num_way).scatter_(1, targets_x.long().view(-1, 1), 1)
        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        with torch.no_grad():
            ## compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
        ## mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)
        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        ## interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))
        ## put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        ## get the loss term for MixMatch; here is criterion is Semiloss
        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.batch_number)

        loss = Lx + w * Lu
        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        model.weight_norm()
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} '.format(
            batch=batch_idx + 1,
            size=args.batch_number,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            # w=ws.avg,
        )
        bar.next()
    bar.finish()
    ema_optimizer.step(bn=True)
    return (losses.avg, losses_x.avg, losses_u.avg,)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing   ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input = input.cuda()
            # print(target)
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)


def linear_rampup(current):
    if args.rampup == 0:
        return 1.0
    else:
        current = np.clip(current / args.rampup, 0.0, 1.0)
        return float(current)


class SemiLoss(object):  # The loss term for MixMatch
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):  # Exponential moving average for model parameters
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = wide_models.Net(num_classes=args.num_way).cuda()
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.data.copy_(param.data)

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    main()

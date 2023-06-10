import torch
import argparse
import util
import os
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable

import loader
import models
import config
from util import AverageStorage, ProgressMeter, accuracy, mixup_data, mixup_criterion, rand_bbox


def test(test_loader, model, criterion, device):
    loss_meter = AverageStorage('Loss', ':4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Testing',
                             meters=[loss_meter, acc_meter])
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), labels.size(0))

        progress.display(i)

    return loss_meter, acc_meter

def train(train_loader, model, criterion, optimizer, device):
    loss_meter = AverageStorage('Loss', ':.4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_meter, acc_meter

def train_augmix(train_loader, model, criterion, optimizer, device):
    s_losses = AverageStorage('s_Loss', ':.4e')
    c_losses = AverageStorage('c_Loss', ':.4e')
    losses = AverageStorage('Loss', ':.4e')
    acc_meter = AverageStorage('Acc@1', ':.2%')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[s_losses, c_losses, losses, acc_meter])

    model.train()
    loss_ema = 0.
    for i, samples in enumerate(train_loader):        
        optimizer.zero_grad()
        inputs, labels = samples
        labels = labels.to(device)

        images_all = torch.cat(inputs, 0).to(device)
        logits_all = model(images_all)

        logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, inputs[0].size(0))

        # Cross-entropy is only computed on clean images
        loss = criterion(logits_clean, labels)

        p_clean, p_aug1, p_aug2 = F.softmax(
            logits_clean, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
            logits_aug2, dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        consist_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                        F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        s_losses.update(loss.item(), inputs[0].size(0))
        c_losses.update(consist_loss.item(), inputs[0].size(0))
        loss += 12 * consist_loss
        losses.update(loss.item(), inputs[0].size(0))

        acc= accuracy(logits_clean, labels)
        acc_meter.update(acc.item(), labels.size(0))

        loss.backward()
        optimizer.step()
        loss_ema = loss_ema * 0.9 + float(loss) * 0.1
        progress.display(i)


    return losses, acc_meter

def train_mixup(train_loader, model, criterion, optimizer, device):
    loss_meter = AverageStorage('Loss', ':.4e')
    acc1_meter = AverageStorage('Acc@1', ':.2%')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc1_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, device)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        acc1 = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_meter, acc1_meter

def train_cutout(train_loader, model, criterion, optimizer, device):
    loss_meter = AverageStorage('Loss', ':.4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_meter, acc_meter

def train_cutmix(train_loader, model, criterion, optimizer, device):
    loss_meter = AverageStorage('Loss', ':.4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        r = np.random.rand(1)
        if r < 0.5:
            # generate mixed sample
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(inputs.size()[0]).cuda()
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, target_a) * lam + criterion(outputs, target_b) * (1. - lam)
        else:
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_meter, acc_meter

def train_autoaug(train_loader, model, criterion, optimizer, device):
    loss_meter = AverageStorage('Loss', ':.4e')
    acc_meter = AverageStorage('Acc', ':.2%')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_meter, acc_meter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--dataset_name', type=str, default='cifar10')
    parser.add_argument('--SOTA', type=str, default='basic')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 模型、日志保存路径
    basic_dir = util.get_basic_dir(args.model_name, args.dataset_name, args.SOTA)
    log_dir = os.path.join(basic_dir, 'runs')
    model_path = os.path.join(basic_dir, 'model_ori.pth')

    # 数据集
    train_loader = loader.load_data(args.dataset_name, data_type='train', SOTA=args.SOTA)
    test_loader = loader.load_data(args.dataset_name, data_type='test')
    data_config = config.get_data_config(args.dataset_name)

    # 模型
    in_channels, num_classes = data_config['in_channels'], data_config['num_classes']
    model = models.load_model(args.model_name, in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.num_epochs)

    writer = SummaryWriter(log_dir)


    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()
    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(config.num_epochs)):
        if args.SOTA == 'basic':
            loss, acc = train(train_loader, model, criterion, optimizer, device)
        elif args.SOTA == 'augmix':
            loss, acc = train_augmix(train_loader, model, criterion, optimizer, device)
        elif args.SOTA == 'mixup':
            loss, acc = train_mixup(train_loader, model, criterion, optimizer, device)
        elif args.SOTA == 'cutout':
            loss, acc = train_cutout(train_loader, model, criterion, optimizer, device)
        elif args.SOTA == 'cutmix':
            loss, acc = train_cutmix(train_loader, model, criterion, optimizer, device)
        elif args.SOTA == 'autoaug':
            loss, acc = train_autoaug(train_loader, model, criterion, optimizer, device)
        writer.add_scalar(tag='training loss', scalar_value=loss.avg, global_step=epoch)
        writer.add_scalar(tag='training acc1', scalar_value=acc.avg, global_step=epoch)
        loss, acc = test(test_loader, model, criterion, device)
        writer.add_scalar(tag='test loss', scalar_value=loss.avg, global_step=epoch)
        writer.add_scalar(tag='test acc1', scalar_value=acc.avg, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc.avg:
            best_acc = acc.avg
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)

        scheduler.step()

    print('COMPLETE !!!')
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)


if __name__ == '__main__':
    main()
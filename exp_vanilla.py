from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import dataset
import os
import argparse
import csv
import pickle
from torch.utils.data.sampler import SubsetRandomSampler

from networks import vgg
from utils.miscellaneous import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable
import glob

def compute_mean(data):
    data_x = data['x']
    data_y = data['y']
    start_flag = 0
    for x, y in zip(data_x, data_y):
        if y == 0:
            if start_flag:
                x_list = np.concatenate((x_list, x[np.newaxis, ]), 0)

            else:
                x_list = x[np.newaxis, ]
                start_flag = 1
    # mean value of empty sample
    return np.mean(x_list, axis=0)

parser = argparse.ArgumentParser()
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='mixup_default', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument('--input_data_path', type=str, default='./data')
opt = parser.parse_args()

torch.manual_seed(opt.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
base_learning_rate = 0.1

# Data
flag = 0
for data_file in glob.glob(r'{}/*.pkl'.format(opt.input_data_path)):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        x_mean = compute_mean(data)
        if not flag:
            csifile = data['x'] - x_mean
            targetfile = data['y']
            flag = 1
        else:
            csifile = np.concatenate((csifile, data['x'] - x_mean))
            targetfile = np.concatenate((targetfile, data['y']))

data = dataset.CSISet(csifile, targetfile)

random_seed = 42
shuffle = False
# Creating data indices for training and validation splits:
dataset_size = len(data)
indices = list(range(dataset_size))
validation_split = 0.2
split = int(np.floor(validation_split * dataset_size))
print(split)
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
trainloader = dataset.CSILoader(data, opt,sampler=train_sampler)
testloader = dataset.CSILoader(data, opt,sampler=valid_sampler)



print('==> Building model..')
# net = VGG('VGG19')
net = vgg.VGG('VGG11')
# net = ResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()

# result_folder = './results/'
# if not os.path.exists(result_folder):
#     os.makedirs(result_folder)
#
# logname = result_folder + net.__class__.__name__ + '_' + opt.sess + '_' + str(opt.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=opt.decay)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.float().cuda(), targets.long().cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss / batch_idx, 100. * correct / total

def test(epoch):
    global best_acc, batch_idx
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.float().cuda(), targets.long().cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.0*float(correct)/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return test_loss / batch_idx, 100. * correct / total


def checkpoint(acc, epoch):
    # Save checkpoint.
    if epoch+1 % 10 == 0:
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7.' + opt.sess + '_' + str(opt.seed))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# if not os.path.exists(logname):
#     with open(logname, 'w') as logfile:
#         logwriter = csv.writer(logfile, delimiter=',')
#         logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])
#


if __name__ == '__main__':
    for epoch in range(start_epoch, 100):
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        # with open(logname, 'a') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',')
        #     logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
    print("best test acc:{}".format(best_acc))
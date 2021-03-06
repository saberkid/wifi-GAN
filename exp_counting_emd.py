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
from sklearn.metrics import confusion_matrix
from networks import vgg
from networks import resnet1D
from utils.miscellaneous import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable
import glob
import re

data_x_train = []
data_x_test = []
data_y_train = []
data_y_test = []
IMF_S = 4

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='cpslab', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--base_lr', default=0.01, type=float, help='base learning rate')
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs")
parser.add_argument('--input_data_path', type=str, default='data/counting')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint/ckpt.cpslab_20')
parser.add_argument('--test_mode', type=bool, default=False)
parser.add_argument('--mixup', type=bool, default=False)
best_acc = 0
use_cuda = torch.cuda.is_available()
opt = parser.parse_args()
torch.manual_seed(opt.seed)
data_path = opt.input_data_path


def merge_ndarray(arr1, arr2):
    if not len(arr1):
        return arr2
    else:
        return np.concatenate((arr1, arr2), axis=0 )

def get_mean_empty(data):
    list_empty = []
    for i in range(len(data['x'])):
        if data['y'][i] == 0:
            list_empty.append(data['x'][i])
    array_empty = np.array(list_empty)

    return np.mean(array_empty, axis=0)


for data_file in glob.glob(r'{}/*.pkl'.format(data_path)):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
        rd = int(re.findall(r'\d+', data_file)[-1])
        # csi_mean_empty = get_mean_empty(data)
        # data['x'] -= csi_mean_empty
        data['x'] = data['x'][:, :, :8, :]

        if rd in [3,7,10,13, 16, 19]:
            data_x_test = merge_ndarray(data_x_test, data['x'])
            data_y_test = merge_ndarray(data_y_test, data['y'])
        else:
            data_x_train = merge_ndarray(data_x_train, data['x'])
            data_y_train = merge_ndarray(data_y_train, data['y'])

unique_train, counts_train = np.unique(data_y_train, return_counts=True)
label_counts_train = dict(zip(unique_train, counts_train))
unique_test, counts_test = np.unique(data_y_test, return_counts=True)
label_counts_test = dict(zip(unique_test, counts_test))
print(data_x_test.shape)
print('-------------Training Set Stats---------------')
print(label_counts_train)
print('------------Testing Set Stats------------')
print(label_counts_test)

# Creating data indices for training and validation splits:
if not opt.test_mode:
    data_train = dataset.CSISet(data_x_train, data_y_train, imf_s=IMF_S, imf_selection=True)
    trainloader = dataset.CSILoader(data_train, opt, shuffle=True)
data_test = dataset.CSISet(data_x_test, data_y_test, imf_s=IMF_S, imf_selection=True)
testloader = dataset.CSILoader(data_test, opt, shuffle=True)




print('==> Building model..')
# net = VGG('VGG19')
#net = vgg.VGG('VGG11', in_channels=64, num_classes=4 ,linear_in=1536)
net = resnet1D.ResNetCSI(num_classes=4, in_channels=data_x_test.shape[2] * IMF_S)
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
optimizer = optim.SGD(net.parameters(), lr=opt.base_lr, momentum=0.9, weight_decay=opt.decay)



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

        if opt.mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, opt.alpha, use_cuda)
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
        else:
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
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return (train_loss / batch_idx, 100. * correct / total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_all = []
    target_all = []
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

        pred_all = merge_ndarray(pred_all, predicted.cpu())
        target_all = merge_ndarray(target_all, targets.data.cpu())

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    checkpoint(acc, epoch)

    #Confusion Mat
    print(confusion_matrix(pred_all, target_all))
    return (test_loss/batch_idx, 100.*correct/total)

def checkpoint(acc, epoch):
    # Save checkpoint.
    if (epoch+1) % 10 == 0:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net, './checkpoint/ckpt.' + opt.sess + '_' + str(epoch + 1))

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = opt.base_lr
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (opt.base_lr - 0.1) * epoch / 10.
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
    if opt.test_mode:
        net = torch.load(opt.checkpoint_path)
        test(0)
    else:
        for epoch in range(opt.epoch):
            adjust_learning_rate(optimizer, epoch)
            train_loss, train_acc = train(epoch)
            test_loss, test_acc = test(epoch)
            # with open(logname, 'a') as logfile:
            #     logwriter = csv.writer(logfile, delimiter=',')
            #     logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        print("best test acc:{}".format(best_acc))



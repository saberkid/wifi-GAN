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
import cv2
from networks import vgg, LeNet
from utils.miscellaneous import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable

label_dict = {'bed': 0, 'fall': 1, 'pickup' : 2, 'run' : 3, 'sitdown' : 4, 'standup' : 5, 'walk' : 6}
label_count_dict_tr = {0: 0, 1: 0, 2:  0, 3: 0, 4: 0, 5: 0, 6: 0}
label_count_dict_va = {0: 0, 1: 0, 2:  0, 3: 0, 4: 0, 5: 0, 6: 0}
data_path = 'data/activity'
trim = 4000
train_val_split = 0.8
downsampling_rate = 5
window_len = 1000
train_size = 64
data_x_train = []
data_x_test = []
data_y_train = []
data_y_test = []
for dir in os.listdir(data_path):
    label_y = label_dict[dir]
    class_total = len(os.listdir(data_path + '/' + dir))
    sample_i = 0
    for file in os.listdir(data_path + '/' + dir):
        data_file = data_path + '/' + dir + '/' + file
        img = cv2.imread(data_file)
        #print(img.shape)
        if sample_i < class_total * train_val_split:
            data_x_train.append(img)
            data_y_train.append(label_y)
        else:
            data_x_test.append(img)
            data_y_test.append(label_y)
        sample_i += 1




data_x_train = np.asarray(data_x_train)
data_x_test = np.asarray(data_x_test)
print(data_x_train.shape)
print(data_x_test.shape)
# data_x_train = data_x_train.swapaxes(2, 3)
# data_x_test = data_x_test.swapaxes(2, 3)
#data_x = data_x.reshape(-1, 100, 150, 3)
data_y_train = np.asarray(data_y_train)
data_y_test = np.asarray(data_y_test)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='mixup_default', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--base_lr', default=0.01, type=float, help='base learning rate')
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs")
parser.add_argument('--input_data_path', type=str, default='./data')

best_acc = 0
use_cuda = torch.cuda.is_available()
opt = parser.parse_args()
torch.manual_seed(opt.seed)

shuffle = True
# Creating data indices for training and validation splits:
data_train = dataset.CSISet(data_x_train, data_y_train)
data_test = dataset.CSISet(data_x_test, data_y_test)
# dataset_size = len(data)
# indices = list(range(dataset_size))
# split = 0.8
# split = int(np.floor(split * dataset_size))
# print(split)
# if shuffle:
#     np.random.seed(opt.seed)
#     np.random.shuffle(indices)
#train_indices, val_indices = indices[split:], indices[:split]
# train_indices, val_indices = indices[:split], indices[split:]

trainloader = dataset.CSILoader(data_train, opt, shuffle=True)
testloader = dataset.CSILoader(data_test, opt, shuffle=True)


print('==> Building model..')
net = vgg.VGG('VGG11', linear_in=2048)
# net = ResNet18()
#net = LeNet.LeNet(in_channel=3, linear_in=9216)
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
    return train_loss / batch_idx, 100. * correct / total

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
            % (test_loss/(batch_idx+1), 100.* float(correct)/total, correct, total))

    # Save checkpoint.
    acc = 100.*float(correct)/total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)

    #Confusion Mat
    print(confusion_matrix(pred_all, target_all))
    return (test_loss/batch_idx, 100.*correct/total)

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
    for epoch in range(opt.epoch):
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        # with open(logname, 'a') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',')
        #     logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
    print("best test acc:{}".format(best_acc))



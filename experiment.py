import argparse

from torch.utils.data.sampler import SubsetRandomSampler

import dataset
import os
from networks.wgan import WGan
import numpy as np
os.makedirs("output", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'test_d'])

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--c_dim", type=int, default=6, help="num of classes")
parser.add_argument("--lambda_gp", type=int, default=5, help="lambda for gradient penalty")
parser.add_argument("--lambda_cls", type=int, default=1, help="lambda for cls")
parser.add_argument("--lambda_rec", type=int, default=10, help="lambda for reconstruction")
parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: generator learning rate")
parser.add_argument("--d_lr", type=float, default=0.0001, help="adam: discriminator learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=10, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument('--test_iters', type=int, default=10, help='test model from this step')

parser.add_argument('--model_save_dir', type=str, default='checkpoint')
parser.add_argument('--model_save_epoch', type=int, default=10, help='epochs to save a checkpoint')
parser.add_argument('--result_dir', type=str, default='output')
parser.add_argument('--input_data_path', type=str, default='./csiset_5952_56_48_16.npy')
parser.add_argument('--input_label_path', type=str, default='./target.npy')
opt = parser.parse_args()

if __name__ == '__main__':
    csifile = np.load(opt.input_data_path)
    targetfile = np.load(opt.input_label_path)
    # csifile = np.load('csiset_test.npy')
    # targetfile = np.load('target_test.npy')
    #input_shape = (56, 10, 10)
    data = dataset.CSISet(csifile, targetfile)

    random_seed = 42
    shuffle = True
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
    dataloader = {'train': dataset.CSILoader(data, opt,sampler=train_sampler), 'val': dataset.CSILoader(data, opt,sampler=valid_sampler)}
    csigan = WGan(opt, dataloader)

    if opt.mode == 'train':
        csigan.train()
    elif opt.mode == 'test':
        csigan.test()
    elif opt.mode == 'test_d':
        csigan.test_d()


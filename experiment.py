import argparse
import dataset
import os
from networks.wgan import WGan
import numpy as np
os.makedirs("output", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
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
parser.add_argument('--result_dir', type=str, default='output')
opt = parser.parse_args()

if __name__ == '__main__':
    csifile = np.load('csiset_5952_56_48_16.npy')
    targetfile = np.load('target.npy')
    # csifile = np.load('csiset_test.npy')
    # targetfile = np.load('target_test.npy')
    #input_shape = (56, 10, 10)
    data = dataset.CSISet(csifile, targetfile)
    dataloader = dataset.CSILoader(data, opt)
    csigan = WGan(opt, dataloader)

    if opt.mode == 'train':
        csigan.train()
    elif opt.mode == 'test':
        csigan.test()


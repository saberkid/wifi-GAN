import numpy as np
from torch.autograd import Variable
from networks.discriminator2 import Discriminator
from networks.generator2 import Generator
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.nn as nn
import os


class WGan():
    def __init__(self, opt, dataloader):
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator and discriminator

        self.opt = opt
        self.G = Generator()
        self.D = Discriminator()
        self.dataloader = dataloader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if cuda:
            self.G.cuda()
            self.D.cuda()
        # Loss weight for gradient penalty
        self.lambda_gp = 10
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.c_dim = 6
        self.g_lr = opt.g_lr
        self.d_lr = opt.d_lr
        self.n_critic = opt.n_critic
        self.test_iters = opt.test_iters
        self.model_save_dir = opt.model_save_dir
        self.result_dir = opt.result_dir
        # TODO rewrite in opt
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.model_save_epoch = 10

        self.lr_update_step = 1000
        self._init_optimizer()

    # Optimizers
    def _init_optimizer(self):
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.opt.b1, self.opt.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.opt.b1, self.opt.b2))

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

    def create_labels(self, c_org, c_dim=6):
        """Generate target domain labels for debugging and testing."""

        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def train(self):
        for epoch in range(self.opt.n_epochs):
            for i, (batch_imgs, batch_labels) in enumerate(self.dataloader):

                X = Variable(batch_imgs).float().to(self.device)
                # Generate target domain labels randomly.
                batch_labels = batch_labels.long().to(self.device)
                rand_idx = torch.randperm(batch_labels.size(0))
                batch_labels_trg = batch_labels[rand_idx].to(self.device)
                #print(batch_labels_trg)


                c_org = self.label2onehot(batch_labels, self.c_dim)
                c_trg = self.label2onehot(batch_labels_trg, self.c_dim)
                c_org = c_org.to(self.device)
                c_trg = c_trg.to(self.device)
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Compute loss with real images.
                out_src, out_cls = self.D(X)
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, batch_labels)

                # Compute loss with fake images.
                x_fake = self.G(X, c_trg)
                out_src, out_cls = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(X.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * X.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                #print(d_loss_real)
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.optimizer_D.step()
                # Train the generator every n_critic steps
                if i % self.opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Original-to-target domain.
                    x_fake = self.G(X, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, batch_labels_trg)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(X - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.opt.n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                    )

                #Save model checkpoints.
            if (epoch + 1) % self.model_save_epoch == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(epoch + 1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(epoch + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                #batches_done += self.opt.n_critic

                # Decay learning rates.
                # if (i + 1) % self.lr_update_step == 0 and (i + 1) > (self.num_iters - self.num_iters_decay):
                #     self.g_lr -= (self.g_lr / float(self.num_iters_decay))
                #     self.d_lr -= (self.d_lr / float(self.num_iters_decay))
                #     print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(self.g_lr, self.d_lr))
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        self.dataloader.batch_size = 1

        with torch.no_grad():
            x_fake_list = []
            label_list = []
            for i, (x_real, c_org) in enumerate(self.dataloader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim)

                # Translate images.
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg).numpy())
                    label_list.append(c_trg.numpy())

                # Save the translated images.

            x_fake_list = np.asarray(x_fake_list)
            label_list = np.asarray(label_list)
            print(x_fake_list.shape)
            print(label_list.shape)
            result_path_data = os.path.join(self.result_dir, 'output_data.pkl')
            result_path_label = os.path.join(self.result_dir, 'output_label.pkl')
            np.save(result_path_data, x_fake_list)
            np.save(result_path_label, label_list)
            print('Saved real and fake images into {}...'.format(self.result_dir))
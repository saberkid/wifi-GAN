import numpy as np

from torchvision.utils import save_image
from torch.autograd import Variable
from networks.discriminator import Discriminator
from networks.generator import Generator

import torch.autograd as autograd
import torch


class WGan():
    def __init__(self, opt, input_shape, dataloader):
        cuda = True if torch.cuda.is_available() else False
        # Initialize generator and discriminator

        self.opt = opt
        self.generator = Generator(input_shape, self.opt)
        self.discriminator = Discriminator(input_shape)
        self.dataloader = dataloader
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
        # Loss weight for gradient penalty
        self.lambda_gp = 10
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self._init_optimizer()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

    # Optimizers
    def _init_optimizer(self):
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

    def compute_gradient_penalty(self, D, real_samples, fake_samples):

        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    # ----------
    #  Training
    # ----------
    def train(self):
        batches_done = 0
        for epoch in range(self.opt.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.opt.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    self.optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.opt.n_epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                    )

                    # if batches_done % self.opt.sample_interval == 0:
                    #     save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                    batches_done += self.opt.n_critic


            torch.save(self.generator.state_dict(), 'params_generator_%d.pkl'%epoch)
            torch.save(self.discriminator.state_dict(), 'params_disciminator_%d.pkl'%epoch)
            # model_object.load_state_dict(torch.load('params.pkl'))
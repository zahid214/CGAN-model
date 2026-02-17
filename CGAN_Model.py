import argparse
import os
import numpy as np
import math
import csv
import lpips

import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

# Make folders
os.makedirs("images", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Prepare CSV file
csv_file = "logs/training_log.csv"
fieldnames = ['epoch', 'batch', 'd_loss', 'g_loss', 'd_score', 'g_score', 'psnr', 'ssim', 'msssim', 'lpips', 'mos']

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument('--data', type=str, default=r"E:\GAN-implementations\data-10\Real-data")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
lpips_fn = None

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    lpips_fn = lpips.LPIPS(net='alex').cuda()

# # Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

# ===== Custom Dataset =====
transform = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)])
dataset = datasets.ImageFolder(root=opt.data, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    epoch_msssim = 0.0
    epoch_lpips = 0.0
    num_batches = 0
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        # )

        # batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     sample_image(n_row=10, batches_done=batches_done)

        # Scores (Discriminator outputs)
        d_score = discriminator(real_imgs, labels).mean().item()
        g_score = discriminator(gen_imgs, labels).mean().item()

        # ---------------------
        #  Metrics (PSNR, SSIM) per batch
        # ---------------------
        real_norm = (real_imgs - real_imgs.min()) / (real_imgs.max() - real_imgs.min() + 1e-8)
        gen_norm  = (gen_imgs  - gen_imgs.min())  / (gen_imgs.max()  - gen_imgs.min()  + 1e-8)
        real_np = real_imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
        gen_np = gen_imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
        real_np = (real_np - real_np.min()) / (real_np.max() - real_np.min() + 1e-8)
        gen_np = (gen_np - gen_np.min()) / (gen_np.max() - gen_np.min() + 1e-8)

        psnr_val = psnr_metric(real_np, gen_np, data_range=1.0)
        ssim_val = ssim_metric(real_np, gen_np, data_range=1.0, channel_axis=-1)
        # msssim_val = ms_ssim(gen_norm, real_norm, data_range=1.0, size_average=True).item()
        msssim_val = ms_ssim(gen_norm, real_norm, data_range=1.0, size_average=True, win_size=3).item()
        real_lp = real_norm * 2 - 1; gen_lp  = gen_norm  * 2 - 1
        lpips_val = lpips_fn(gen_lp, real_lp).mean().item()

        epoch_psnr += psnr_val
        epoch_ssim += ssim_val
        epoch_msssim += msssim_val
        epoch_lpips += lpips_val
        num_batches += 1

        print(
            f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
            f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
            f"[D score: {d_score:.4f}] [G score: {g_score:.4f}] "
            f"[PSNR: {psnr_val:.4f}] [SSIM: {ssim_val:.4f}]"
        )

    avg_psnr = epoch_psnr / num_batches
    avg_ssim = epoch_ssim / num_batches
    avg_msssim = epoch_msssim / num_batches
    avg_lpips = epoch_lpips / num_batches
    proxy_mos = (0.4 * avg_ssim + 0.3 * avg_msssim + 0.3 * (1 - avg_lpips))

    # ---------------------
    # Save CSV per batch
    # ---------------------
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            'epoch': epoch,
            'batch': i,
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_score': d_score,
            'g_score': g_score,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'msssim': avg_msssim,
            'lpips': avg_lpips,
            'mos': proxy_mos})

    # ---------------------
    # Save ONE image at the end of each epoch (from last batch)
    # ---------------------
    # save_path = f"images/epoch_{epoch}.png"
    # save_image(gen_imgs.data[0], save_path, normalize=True)
    # print(f"✅ Epoch {epoch} finished — image saved at {save_path}")


    # Save only one generated image per epoch (first sample in the batch)
    for im in range(len(gen_imgs)):
        # single_img = gen_imgs[im].unsqueeze(im)   # shape (1, C, H, W)
        single_img = gen_imgs[im]   # shape (1, C, H, W)
        save_path = f"images/epoch_{epoch}_{im}.png"
        save_image(single_img, save_path, normalize=True)
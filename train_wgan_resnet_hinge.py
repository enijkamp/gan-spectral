import argparse
import logging
import sys
import os
import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.utils as vutils

import inception_score_v3 as is_v3


class NetG(nn.Module):
    def __init__(self, z_dim, size=128):
        super(NetG, self).__init__()
        self.z_dim = z_dim
        self.size = size

        self.dense = nn.Linear(self.z_dim, 4 * 4 * size)
        self.final = nn.Conv2d(size, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)

        self.model = nn.Sequential(
            ResBlockGenerator(size, size),
            ResBlockGenerator(size, size),
            ResBlockGenerator(size, size),
            nn.BatchNorm2d(size),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.size, 4, 4))


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, upsampling=True):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if upsampling:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                spectral_norm(self.conv1),
                nn.ReLU(),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.ReLU(),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class NetD(nn.Module):
    def __init__(self, size=128):
        super(NetD, self).__init__()
        self.size = size
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(3, size),
                ResBlockDiscriminator(size, size, stride=2),
                ResBlockDiscriminator(size, size),
                ResBlockDiscriminator(size, size),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(size, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = spectral_norm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, self.size))


def get_exp_id():
    return os.path.splitext(os.path.basename(__file__))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def create_logger(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger('')
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


torch.cuda.set_device(0)

output_dir = get_output_dir(get_exp_id())
logger = create_logger(output_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--gamma', type=float, default=1.00)
parser.add_argument('--n_disc', type=int, default=5)
parser.add_argument('--nz', type=int, default=128)

args = parser.parse_args()

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Resize(32),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
    batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

net_d = NetD().cuda()
net_g = NetG(args.nz).cuda()

logger.info(args)
logger.info(net_d)
logger.info(net_g)

train_flag = lambda: [net.train() for net in [net_d, net_g]]
eval_flag = lambda: [net.eval() for net in [net_d, net_g]]
grad_norm = lambda net: torch.sqrt(sum(torch.sum(p.grad**2) for p in net.parameters()))
zero_grad = lambda: [net.zero_grad() for net in [net_d, net_g]]

optim_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optim_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
schedule_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=args.gamma)
schedule_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.gamma)

logger.info(('{:>14}'*8).format('epoch', 'E(real)', 'E(fake)', 'loss(D)', 'loss(G)', 'grad(D)', 'grad(G)', 'incept_v3'))

z_fixed = torch.tensor(torch.randn(64, args.nz)).cuda()

for epoch in range(500):
    loss_d_s, loss_g_s, grad_d_s, grad_g_s, e_real_s, e_fake_s = [], [], [], [], [], []

    train_flag()
    for i, (x, _) in enumerate(loader):

        for _ in range(args.n_disc):

            zero_grad()
            z = torch.randn(args.batch_size, args.nz).cuda()
            x_hat = net_g(z)
            x = x.cuda()

            e_real = torch.mean(F.relu(1.0 - net_d(x)))
            e_fake = torch.mean(F.relu(1.0 + net_d(x_hat.detach())))
            loss_d = e_real + e_fake
            loss_d.backward()
            optim_d.step()

            loss_d_s.append(loss_d.data.item())
            grad_d_s.append(grad_norm(net_d).data.item())
            e_real_s.append(e_real.data.item())
            e_fake_s.append(e_fake.data.item())

        zero_grad()
        loss_g = -torch.mean(net_d(x_hat))
        loss_g.backward()
        optim_g.step()

        loss_g_s.append(loss_g.data.item())
        grad_g_s.append(grad_norm(net_g).data.item())

    schedule_d.step()
    schedule_g.step()


    eval_flag()

    num_samples = 2000
    noise_z = torch.FloatTensor(args.batch_size, args.nz)
    new_noise = lambda: noise_z.normal_().cuda()
    gen_samples = torch.cat([net_g(new_noise()).detach().cpu() for _ in range(int(num_samples / 100))])
    incept_v3 = is_v3.inception_score(gen_samples, resize=True, splits=1)[0]

    vutils.save_image(net_g(z_fixed).data, '{}/{}_samples.png'.format(output_dir, epoch), normalize=True)

    logger.info(('{:>14}' + '{:>14.3f}'*7).format(epoch, np.mean(e_real_s), np.mean(e_fake_s), np.mean(loss_d_s), np.mean(loss_g_s), np.mean(grad_d_s), np.mean(grad_g_s), incept_v3))

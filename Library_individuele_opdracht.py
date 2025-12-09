# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 15:56:39 2025

@author: Marti
"""

import zipfile
import io
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import os
import requests
import re

def normalize_path(path):
    parts = path.split("/")
    normalized = []
    for p in parts:
        if not normalized or p != normalized[-1]:
            normalized.append(p)
    return "/".join(normalized)

def load_real_dataset(zip_path):
    images = {"train": [], "test": []}
    with zipfile.ZipFile(zip_path, "r") as z:
        for fname in z.namelist():
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            norm = normalize_path(fname)
            if "/train/" in norm:
                split = "train"
            elif "/test/" in norm:
                split = "test"
            else:
                continue
            with z.open(fname) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                images[split].append(img)
    return images

def load_anomaly_dataset(zip_path):
    imgs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for fname in z.namelist():
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                with z.open(fname) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    imgs.append(img)
    return imgs

class ImageListDataset(Dataset):
    def __init__(self, pil_images, transform=None):
        self.data = pil_images
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img

def get_transforms(image_size=128):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5])
    ])

def make_dataloader(image_list, batch=32, shuffle=True, size=128):
    ds = ImageListDataset(image_list, transform=get_transforms(size))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

def mse_loss(x, recon):
    return torch.mean((x - recon) ** 2, dim=[1,2,3])

def compute_anomaly_score(x, recon):
    return mse_loss(x, recon)

class ConvAE(nn.Module):
    def __init__(self, zdim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Conv2d(128,zdim,4,2,1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(zdim,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

class DenoiseAE(nn.Module):
    def __init__(self, zdim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Conv2d(128,zdim,4,2,1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(zdim,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

class VAE(nn.Module):
    def __init__(self, zdim=128, img_size=128):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,img_size,img_size)
            h = self.enc_conv(dummy)
            self.flatten_dim = h.view(1,-1).shape[1]

        self.fc_mu = nn.Linear(self.flatten_dim, zdim)
        self.fc_logvar = nn.Linear(self.flatten_dim, zdim)
        self.fc_dec = nn.Linear(zdim, self.flatten_dim)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh()
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc_dec(z)
        spatial_dim = int((self.flatten_dim / 128) ** 0.5)
        h = h.view(h.size(0), 128, spatial_dim, spatial_dim)
        return self.dec_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class UNetAE(nn.Module):
    def __init__(self, zdim=128):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3,32,3,1,1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32,64,3,2,1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(64,128,3,2,1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv2d(128,zdim,3,2,1), nn.ReLU())
        self.dec4 = nn.Sequential(nn.ConvTranspose2d(zdim,128,4,2,1), nn.ReLU())
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Conv2d(32,3,3,1,1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        z  = self.enc4(e3)
        d4 = self.dec4(z)
        d3 = self.dec3(d4 + e3)
        d2 = self.dec2(d3 + e2)
        d1 = self.dec1(d2 + e1)
        return d1


class Encoder(nn.Module):
    def __init__(self, zdim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1,3,128,128)
            h = self.conv(dummy)
            self.flatten_dim = h.view(1,-1).shape[1]
        self.fc_mu = nn.Linear(self.flatten_dim, zdim)
        self.fc_logvar = nn.Linear(self.flatten_dim, zdim)
    def forward(self,x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, zdim=128, flatten_dim=8192):
        super().__init__()
        self.fc = nn.Linear(zdim, flatten_dim)
        channels = 128
        spatial = int((flatten_dim / channels) ** 0.5)
        self.channels = channels
        self.spatial = spatial
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(channels,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh()
        )
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.channels, self.spatial, self.spatial)
        return self.deconv(h)

class VAEGAN(nn.Module):
    def __init__(self, zdim=128):
        super().__init__()
        self.encoder = Encoder(zdim=zdim)
        self.decoder = Decoder(zdim=zdim, flatten_dim=self.encoder.flatten_dim)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparam(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,3,stride,1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride!=1 or in_channels!=out_channels:
            self.short = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.short = nn.Identity()
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.short(x)
        return self.relu(out)

class ResNetAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            ResBlock(3,32,2),
            ResBlock(32,64,2),
            ResBlock(64,128,2),
            ResBlock(128,256,2)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh()
        )
    def forward(self,x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon


AnoVAEGAN = VAEGAN

def download_mediafire(url, output_path):
    if os.path.exists(output_path):
        print("Already downloaded.")
        return 

    print(f"Downloading: {output_path} ...")
    response = requests.get(url)
    html = response.text
    import re
    match = re.search(r'href="(https://download[^"]+)"', html)
    if not match:
        raise ValueError("Could not find Mediafire download link. URL may be incorrect.")

    real_url = match.group(1)
    file_data = requests.get(real_url).content
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(file_data)
    print(f"✓ Downloaded: {output_path}")


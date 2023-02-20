import sys
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import glob

# Note - you must have torchvision installed for this example
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from skimage.measure import regionprops
from torchvision.transforms.functional import crop
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pytorch_lightning import loggers as pl_loggers
import torchvision
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import convolve, sobel
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pythae
from pythae.trainers import BaseTrainerConfig, BaseTrainer


class LitAutoEncoderTorch(pl.LightningModule):
    def __init__(self, model, batch_size=1, learning_rate=1e-4, params=None):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.params = params
        self.model = model
        self.PYTHAE_FLAG = issubclass(self.model.__class__, pythae.models.BaseAE)
        if self.PYTHAE_FLAG:
            self.pythae_flag()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        # self.loss_fn = torch.nn.BCELoss()
        
    def pythae_flag(self):
        self.model = self.model.to(self.device)
        self.model.train()
        
        
    def decoder(self, z):
        return self.model.decoder(z)

    def encoder(self, img):
        return self.model.encoder(img)

    def decode(self, z):
        return self.model.decode(z)

    def encode(self, img):
        return self.model.encode(img)

    def forward(self, x):
        if self.PYTHAE_FLAG:
            return self.model.forward({"data": x})["recon_x"]
        return self.model.forward(x)

    def recon(self, x):
        if self.PYTHAE_FLAG:
            return self.forward(x)
        return self.model.recon(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    # return self.recon(batch)

    def get_loss(self, batch):
        if self.PYTHAE_FLAG:
            return self.model.forward({"data": batch})["loss"]
        # self.curr_device = real_img.device

        results = self.get_results(batch)
        recons = self.recon(batch)

        loss = self.model.loss_function(*results, recons=recons, input=batch)
        return loss["loss"]

    def get_results(self, batch):
        return self.model.forward({"data": batch})

    def test_step(self, batch, batch_idx):
        test_loss = self.get_loss(batch)
        self.log("test_loss", test_loss, on_epoch=True)
        return test_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.get_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.get_loss(batch)
        results = self.get_results(batch)

        self.log("train_loss", loss)
        # self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

        self.logger.experiment.add_image(
            "input", torchvision.utils.make_grid(batch), batch_idx
        )
        if self.PYTHAE_FLAG:
            self.logger.experiment.add_image(
                "output",
                torchvision.utils.make_grid(results["recon_x"]),
                batch_idx,
            )
        else:
            self.logger.experiment.add_image(
                "output",
                torchvision.utils.make_grid(self.model.output_from_results(*results)),
                batch_idx,
            )
        return loss

    # def _training_step(self, inputs, batch_idx):
    #     vq_loss, output, perplexity = self.forward(inputs)
    #     # output = x_recon
    #     # loss = self.loss_fn(output, inputs)

    #     # vq_loss, data_recon, perplexity = model(inputs)
    #     # recon_error = F.mse_loss(output, inputs)
    #     recon_error = self.loss_fn(output, inputs)
    #     loss = recon_error + vq_loss  # Missing variance bit
    #     self.log("train_loss", loss)
    #     # tensorboard = self.logger.experiment
    #     # self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)

    #     # torchvision.utils.make_grid(output)
    #     self.logger.experiment.add_image(
    #         "input", torchvision.utils.make_grid(inputs), batch_idx
    #     )
    #     # self.logger.experiment.add_embedding(
    #     #     "input_image", torchvision.utils.make_grid(transformer_image(inputs)), batch_idx)
    #     self.logger.experiment.add_image(
    #         "output", torchvision.utils.make_grid(output), batch_idx
    #     )
    #     # self.logger.experiment.add_embedding(
    #     #     "output_image", torchvision.utils.make_grid(transformer_image(output)), batch_idx)

    #     # tensorboard.add_image("input", transforms.ToPILImage()(output[batch_idx]), batch_idx)
    #     # tensorboard.add_image("output", transforms.ToPILImage()(output[batch_idx]), batch_idx)
    #     return loss

    def get_embedding(self):
        return self.model.get_embedding()

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)

    def get_model(self):
        return self.model

    @property
    def num_training_steps(self) -> int:
        return 100

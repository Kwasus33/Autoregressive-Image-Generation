from img_generation.utils import EarlyStopping, visualize_val_reconstruction

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm


NUM_EMBEDDINGS = 1024


class ConvBlock(nn.Module):
    def __init__(self, input_channels, hidden_dim_ch):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim_ch = hidden_dim_ch

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dim_ch, kernel_size=(4, 4), stride=2, padding=1),  # 28x28 -> 14x14
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.hidden_dim_ch, out_channels=2 * self.hidden_dim_ch, kernel_size=(4, 4), stride=2, padding=1),  # 14x14 -> 7x7
            nn.GroupNorm(num_groups=8, num_channels=2 * self.hidden_dim_ch),
        )

    def forward(self, x):
        return self.conv_layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, hidden_dim_ch, output_channels):
        super().__init__()

        self.hidden_dim_ch = hidden_dim_ch
        self.output_channels = output_channels

        # overlapping kernels in deconvolution could create checkerboard artifacts - https://distill.pub/2016/deconv-checkerboard/
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2 * self.hidden_dim_ch, out_channels=self.hidden_dim_ch, kernel_size=(4, 4), stride=2, padding=1
            ),  # 7x7 -> 14x14
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim_ch),
            nn.SiLU(),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim_ch, out_channels=self.output_channels, kernel_size=(4, 4), stride=2, padding=1
            ),  # 14x14-> 28x28
        )

    def forward(self, x):
        return self.de_conv(x)


""" Conditional VAE implementation """


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim_ch, latent_dim, num_classes, class_embed_dim):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim_ch = hidden_dim_ch
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        self.conv_block = ConvBlock(input_channels, hidden_dim_ch)
        self.act = nn.SiLU()
        self.flatten = nn.Flatten()
        self.label_embed = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.class_embed_dim)
        self.fc_mean = nn.Linear(
            in_features=2 * self.hidden_dim_ch * 7 * 7 + self.class_embed_dim,
            out_features=self.latent_dim,
        )  # optimal latent_dim is around out_features divided by 8 or 16 or 32
        self.fc_logvar = nn.Linear(
            in_features=2 * self.hidden_dim_ch * 7 * 7 + self.class_embed_dim,
            out_features=self.latent_dim,
        )

    def forward(self, x, y):
        x = self.conv_block(x)
        x = self.flatten(self.act(x))
        y_embed = self.label_embed(y)
        h = torch.cat([x, y_embed], dim=1)  # dim=0 is batch_size, dim=1 are flatten params
        mean = self.fc_mean(h)
        log_var = self.fc_logvar(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim_ch, output_channels, num_classes, class_embed_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim_ch = hidden_dim_ch
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        self.label_embed = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.class_embed_dim)
        self.fc = nn.Linear(
            in_features=self.latent_dim + self.class_embed_dim,
            out_features=2 * self.hidden_dim_ch * 7 * 7,
        )
        self.de_conv_block = DeconvBlock(hidden_dim_ch, output_channels)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.SiLU()

    def forward(self, z, y):
        y_embed = self.label_embed(y)
        z = torch.cat([z, y_embed], dim=1)
        z = self.fc(z)
        # ^^^ tested with and without act after linear layer - left without as features from latent space are from normal distro - wrong activation may too much influence the negative ones
        z = z.view([-1, 2 * self.hidden_dim_ch, 7, 7])
        x_hat = self.de_conv_block(z)
        return self.sigmoid(x_hat)


class VAE(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_dim_channels,
        latent_dim,
        num_classes,
        class_embed_dim,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = Encoder(
            input_channels=input_channels,
            hidden_dim_ch=hidden_dim_channels,
            latent_dim=latent_dim,
            num_classes=num_classes,
            class_embed_dim=class_embed_dim,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim_ch=hidden_dim_channels,
            output_channels=input_channels,
            num_classes=num_classes,
            class_embed_dim=class_embed_dim,
        )

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # exp( log var ) --> var, sqrt(var) --> std
        eps = torch.randn_like(std)
        # scales sampled normal distro (eps) by std + adds mean of processed img - that way we get normal distro shifted to normal-like distro of img data
        return mean + std * eps

    def forward(self, x, y):
        mean, logvar = self.encoder(x, y)
        z = self.reparametrize(mean=mean, logvar=logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mean, logvar


def vae_loss_function(x, x_hat, mean, log_var, beta=0.5):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    total_loss = reproduction_loss + beta * KLD
    # ^^^ Beta-VAE - https://apxml.com/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/beta-vaes
    return total_loss, reproduction_loss


def train_vae(
    model: nn.Module,
    loaders: dict[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    early_stopping: EarlyStopping,
    num_epochs: int,
    warmup_epochs: int,
    loss_beta: float,
    device: torch.device,
    visualize_val: bool = True,
):
    total_loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
    reconst_loss_metric = torchmetrics.aggregation.MeanMetric().to(device)  # reconstruction loss
    val_losses = []
    beta = loss_beta

    for epoch in range(1, num_epochs + 1):
        # KL Annealing - https://apxml.com/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/advanced-optimization-vaes
        # --> tested but didn't give better results
        # if epoch < warmup_epochs:
        #     beta = loss_beta * (epoch / warmup_epochs)
        # else:
        #     beta = loss_beta

        if epoch % 10 == 0:
            val_real = []  # real imgs
            val_reconst = []  # reconstructed imgs

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for x, y in tqdm(loaders[phase]):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    x_hat, mean, logvar = model(x, y)
                    total_loss, repr_loss = criterion(x, x_hat, mean, logvar, beta)

                    total_loss_metric(total_loss)
                    reconst_loss_metric(repr_loss)

                    if phase == "train":
                        total_loss.backward()
                        optimizer.step()
                    if phase == "val" and epoch % 10 == 0:
                        val_real.append(x)
                        val_reconst.append(x_hat)

            current_lr = lr_scheduler.get_last_lr()[0]
            total_mean_loss = total_loss_metric.compute()
            # total_mean_loss /= len(x)
            reconst_mean_loss = reconst_loss_metric.compute()
            # reconst_mean_loss /= len(x)

            print(
                f"(Epoch {epoch}/[{phase}]) -> Total Loss:\t{total_mean_loss:.3f} -> Reconstruction Loss:\t{reconst_mean_loss:.3f} -> lr: {current_lr}"
            )

            if phase == "val":
                early_stopping(reconst_mean_loss, model)
                val_losses.append(float(total_mean_loss.cpu()))
                if visualize_val and epoch % 10 == 0:
                    visualize_val_reconstruction(torch.cat(val_real).cpu(), torch.cat(val_reconst).cpu())

            total_loss_metric.reset()
            reconst_loss_metric.reset()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        lr_scheduler.step()

    return early_stopping, val_losses


""" VQ-VAE Implementation based on: https://huggingface.co/blog/ariG23498/understand-vq """


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):  # commitment cost - beta, the commitment loss weight
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z):
        b, c, h, w = z.shape
        z_channel_last = z.permute(0, 2, 3, 1)  # (b, h, w, c)
        z_flattened = z_channel_last.reshape(b * h * w, self.embedding_dim)

        # Calculate distances between z (predicted embeddig) and the codebook (dictionary of known embeddings) embeddings |a-b|²
        distances = (
            torch.sum(z_flattened**2, dim=-1, keepdim=True)  # a²
            + torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True)  # b²
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())  # -2ab
        )
        encoding_indices = torch.argmin(distances, dim=-1)  # Get the index with the smallest distance

        z_q = self.embedding(encoding_indices)  # Get the quantized vector
        z_q = z_q.reshape(b, h, w, self.embedding_dim)  # from flatten embedding vector to --> img-like tensor (b, h, w, c)
        z_q = z_q.permute(0, 3, 1, 2)  # (b, c, h, w)

        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)  # Calculate the commitment loss
        z_q = z + (z_q - z).detach()  # Straight-through estimator trick for gradient backpropagation

        return z_q, loss, encoding_indices


class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_embeddings):
        """
        latent_dim_channels = 2 * hidden_dim
        """
        super(VQVAE, self).__init__()

        self.encoder = ConvBlock(input_channels=in_channels, hidden_dim_ch=hidden_dim)  # encoder lattent_dim would is 2*hidden_dim
        self.vq_layer = VQEmbedding(num_embeddings, 2 * hidden_dim)  # Vector Quantization
        self.decoder = DeconvBlock(hidden_dim_ch=hidden_dim, output_channels=in_channels)
        self.normalize = nn.Sigmoid()
        # self.normalize = nn.Tanh()  # Output values in range [-1, 1] - used if input values are normalized to [-1, 1]

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq_layer(z_e)
        return z_q, vq_loss, encoding_indices

    def forward(self, x):
        z_q, vq_loss, _ = self.encode(x)
        x_recon = self.normalize(self.decoder(z_q))
        return x_recon, vq_loss


def vqvae_loss(recon_x, x, vq_loss):
    recon_loss = F.mse_loss(recon_x, x)
    return recon_loss + vq_loss


def train_vq_vae(
    model: nn.Module,
    loaders: dict[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    early_stopping: EarlyStopping,
    num_epochs: int,
    device: torch.device,
    visualize_val: bool = True,
):
    loss_metric = torchmetrics.aggregation.MeanMetric().to(device)
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        if epoch % 10 == 0:
            val_real = []  # real imgs
            val_reconst = []  # reconstructed imgs

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            for x, y in tqdm(loaders[phase]):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    x_hat, vq_loss = model(x)  # x_hat is reconstructed x
                    loss = criterion(x_hat, x, vq_loss)

                    loss_metric(loss)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    if phase == "val" and epoch % 10 == 0:
                        val_real.append(x)
                        val_reconst.append(x_hat)

            current_lr = lr_scheduler.get_last_lr()[0]
            mean_loss = loss_metric.compute()
            # mean_loss /= len(x)

            print(f"(Epoch {epoch}/[{phase}]) -> Total Loss:\t{mean_loss:.3f} -> lr: {current_lr}")

            if phase == "val":
                early_stopping(mean_loss, model)
                val_losses.append(float(mean_loss.cpu()))
                if visualize_val and epoch % 10 == 0:
                    visualize_val_reconstruction(torch.cat(val_real).cpu(), torch.cat(val_reconst).cpu())

            loss_metric.reset()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        lr_scheduler.step()

    return early_stopping, val_losses

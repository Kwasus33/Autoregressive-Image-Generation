from img_generation.utils import EarlyStopping, visualize_val_reconstruction

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchmetrics

from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(
        self, input_channels, hidden_dim_ch, latent_dim, num_classes, class_embed_dim
    ):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_dim_ch = hidden_dim_ch
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.hidden_dim_ch,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),  # 32x32 -> 16x16 [(32+1+1-4)//2 + 1]
            # tested also with BatchNorm in Encoder and no normalization in Decoder
            # (BatchNorm in decoder means using real img normalization params to random noise ==> applying patterns learned on structured data to random data ==> VERY POOR generations, tested :)
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim_ch),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=self.hidden_dim_ch,
                out_channels=2 * self.hidden_dim_ch,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=8, num_channels=2 * self.hidden_dim_ch),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=2 * self.hidden_dim_ch,
                out_channels=4 * self.hidden_dim_ch,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),  # 8x8 -> 4x4
            nn.GroupNorm(num_groups=8, num_channels=4 * self.hidden_dim_ch),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.label_embed = nn.Embedding(
            num_embeddings=self.num_classes, embedding_dim=self.class_embed_dim
        )
        self.fc_mean = nn.Linear(
            in_features=4 * self.hidden_dim_ch * 4 * 4 + self.class_embed_dim,
            out_features=self.latent_dim,
        )  # optimal latent_dim is around out_features divided by 8 or 16 or 32
        self.fc_logvar = nn.Linear(
            in_features=4 * self.hidden_dim_ch * 4 * 4 + self.class_embed_dim,
            out_features=self.latent_dim,
        )

    def forward(self, x, y):
        x = self.conv_layers(x)
        y_embed = self.label_embed(y)
        h = torch.cat(
            [x, y_embed], dim=1
        )  # dim=0 is batch_size, dim=1 are flatten params
        mean = self.fc_mean(h)
        log_var = self.fc_logvar(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(
        self, latent_dim, hidden_dim_ch, output_channels, num_classes, class_embed_dim
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim_ch = hidden_dim_ch
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim

        self.label_embed = nn.Embedding(
            num_embeddings=self.num_classes, embedding_dim=self.class_embed_dim
        )
        self.fc = nn.Linear(
            in_features=self.latent_dim + self.class_embed_dim,
            out_features=4 * self.hidden_dim_ch * 4 * 4,
        )

        # overlapping kernels in deconvolution could create checkerboard artifacts - https://distill.pub/2016/deconv-checkerboard/
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=4 * self.hidden_dim_ch,
                out_channels=2 * self.hidden_dim_ch,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),  # 4x4 -> 8x8
            nn.GroupNorm(num_groups=8, num_channels=2 * self.hidden_dim_ch),
            nn.SiLU(),
            nn.ConvTranspose2d(
                in_channels=2 * self.hidden_dim_ch,
                out_channels=self.hidden_dim_ch,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),  # 8x8-> 16x16
            nn.GroupNorm(num_groups=8, num_channels=self.hidden_dim_ch),
            nn.SiLU(),
            nn.ConvTranspose2d(
                in_channels=self.hidden_dim_ch,
                out_channels=self.output_channels,
                kernel_size=(4, 4),
                stride=2,
                padding=1,
            ),  # 16x16 -> 32x32
            nn.Sigmoid(),
        )

        self.act = nn.SiLU()

    def forward(self, z, y):
        y_embed = self.label_embed(y)
        z = torch.cat([z, y_embed], dim=1)
        z = self.fc(
            z
        )  # tested with and without act after linear layer - left without as features from latent space are from normal distro - wrong activation may too much influence the negative ones
        z = z.view([-1, 4 * self.hidden_dim_ch, 4, 4])
        x_hat = self.de_conv(z)
        return x_hat


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
        return (
            mean + std * eps
        )  # scales sampled normal distro (eps) by std + adds mean of processed img - that way we get normal distro shifted to normal-like distro of img data

    def forward(self, x, y):
        mean, logvar = self.encoder(x, y)
        z = self.reparametrize(mean=mean, logvar=logvar)
        x_hat = self.decoder(z, y)
        return x_hat, mean, logvar


def vae_loss_function(x, x_hat, mean, log_var, beta=0.5):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    total_loss = (
        reproduction_loss + beta * KLD
    )  # Beta-VAE - https://apxml.com/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/beta-vaes
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
    reconst_loss_metric = torchmetrics.aggregation.MeanMetric().to(
        device
    )  # reconstruction loss
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
            reconst_mean_loss = reconst_loss_metric.compute()

            print(
                f"(Epoch {epoch}/[{phase}]) -> Total Loss:\t{total_mean_loss:.3f} -> Reconstruction Loss:\t{reconst_mean_loss:.3f} -> lr: {current_lr}"
            )

            if phase == "val":
                early_stopping(reconst_mean_loss, model)
                val_losses.append(float(total_mean_loss.cpu()))
                if visualize_val and epoch % 10 == 0:
                    visualize_val_reconstruction(
                        torch.cat(val_real).cpu(), torch.cat(val_reconst).cpu()
                    )

            total_loss_metric.reset()
            reconst_loss_metric.reset()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        lr_scheduler.step()

    return early_stopping, val_losses

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchmetrics

from matplotlib import pyplot as plt

from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def visualize_val_reconstruction(input_imgs, reconst_imgs):
    imgs = torch.stack([input_imgs[:20], reconst_imgs[:20]], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=False)
    grid = grid.permute(1, 2, 0)
    if len(input_imgs) == 4:
        plt.figure(figsize=(15, 10))
    else:
        plt.figure(figsize=(15, 15))
    plt.title("Reconstructions")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


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

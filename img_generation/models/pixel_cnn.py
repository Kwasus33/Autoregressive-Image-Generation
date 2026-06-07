import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from tqdm import tqdm

from img_generation.utils import visualize_val_reconstruction

"""
Model based on UvA Deep Learning Course implementation: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html
"""


class MaskedConvolution(nn.Module):
    def __init__(self, c_in, c_out, mask, **kwargs):
        """
        Implements a convolution with mask applied on its weights.
        Inputs:
            c_in - Number of input channels
            c_out - Number of output channels
            mask - Tensor of shape [kernel_size_H, kernel_size_W] with 0s where
                   the convolution should be masked, and 1s otherwise.
            kwargs - Additional arguments for the convolution
        """
        super().__init__()

        kernel_size = (mask.shape[0], mask.shape[1])
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = tuple([dilation * (kernel_size[i] - 1) // 2 for i in range(2)])

        self.conv = nn.Conv2d(c_in, c_out, kernel_size, padding=padding, **kwargs)

        # Mask as buffer => it is no parameter but still a tensor of the module (must be moved with the devices)
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        self.conv.weight.data *= self.mask  # Ensures zero's at masked positions
        return self.conv(x)


class VerticalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels below. For efficiency, we could also reduce the kernel
        # size in height, but for simplicity, we stick with masking here.
        mask = torch.ones(kernel_size, kernel_size)
        mask[kernel_size // 2 + 1 :, :] = 0

        # For the very first convolution, we will also mask the center row
        if mask_center:
            mask[kernel_size // 2, :] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class HorizontalStackConvolution(MaskedConvolution):
    def __init__(self, c_in, c_out, kernel_size=3, mask_center=False, **kwargs):
        # Mask out all pixels on the left. Note that our kernel has a size of 1
        # in height because we only look at the pixel in the same row.
        mask = torch.ones(1, kernel_size)
        mask[0, kernel_size // 2 + 1 :] = 0

        # for first convolution, the center pixel is masked
        if mask_center:
            mask[0, kernel_size // 2] = 0

        super().__init__(c_in, c_out, mask, **kwargs)


class GatedMaskedConv(nn.Module):
    def __init__(self, c_in, **kwargs):
        """
        Gated Convolution block implemented the computation graph shown above.
        """
        super().__init__()
        self.conv_vert = VerticalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_horiz = HorizontalStackConvolution(c_in, c_out=2 * c_in, **kwargs)
        self.conv_vert_to_horiz = nn.Conv2d(2 * c_in, 2 * c_in, kernel_size=1, padding=0)
        self.conv_horiz_1x1 = nn.Conv2d(c_in, c_in, kernel_size=1, padding=0)

    def forward(self, v_stack, h_stack):
        # Vertical stack (left)
        v_stack_feat = self.conv_vert(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        # Horizontal stack (right)
        h_stack_feat = self.conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + self.conv_vert_to_horiz(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horiz_1x1(h_stack_feat)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out


class PixelCNN(pl.LightningModule):
    def __init__(self, c_in, c_hidden, num_classes, on_tokens=False):
        super().__init__()
        self.save_hyperparameters()
        # operations based on number of classes - works for 256 classes of pixel values as well as 16 classes of token values
        self.num_classes = num_classes
        self.on_tokens = on_tokens

        if self.on_tokens:
            self.embedding = nn.Embedding(num_classes + 1, c_hidden, padding_idx=num_classes)  # adds PAD token - used as token mask in Masked Conv
            conv_c_in = c_hidden
        else:
            conv_c_in = c_in

        # initial convolutions skip the center pixel - mask_center=True
        self.conv_vstack = VerticalStackConvolution(conv_c_in, c_hidden, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(conv_c_in, c_hidden, mask_center=True)
        # convolution block of PixelCNN (dilation used instead of downscaling)
        self.conv_layers = nn.ModuleList(
            [
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=2),
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=4),
                GatedMaskedConv(c_hidden),
                GatedMaskedConv(c_hidden, dilation=2),
                GatedMaskedConv(c_hidden),
            ]
        )

        # Output classification convolution (1x1) - e.g. predict value of 8bit pixel 0..255 (256 classes), predict value of token idx [0..K-1]
        out_channels = num_classes if self.on_tokens else c_in * num_classes
        self.conv_out = nn.Conv2d(c_hidden, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Forward image through model and return logits for each pixel.
        Inputs:
            x - Image tensor with integer values between 0 and num_classes-1 (e.g. 0..255 for pixels).
        """
        # At first input must be denormalized (0..255) as loss function campares predicted pixel value 0..255 to it's orignal value from input !
        if self.on_tokens:
            if x.dim() == 4:
                x = x.squeeze(1)  # gets rid of channel dim (redundant for tokens - kept only to unify operations on pixels and tokens)
            # if it's tokens input is x.shape=[B, H, W] and is tranformed to [B, in_channels=c_hidden(vec embedding size), H, W] to go through conv
            x = self.embedding(x).permute(0, 3, 1, 2).contiguous()
            # permute only swaps indexes, contiguous makes data inline in memory (inline allignment required by conv)
        else:
            x = (x.float() / float(self.num_classes - 1)) * 2 - 1

        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)

        out = self.conv_out(F.elu(h_stack))  # 1x1 classification conv, applied ELU before 1x1 conv for non-linearity on residual connection
        if not self.on_tokens:
            # [B, Classes_preds, Ch, H, W] for imgs, [B, Ch=num_embedding_classes, H, W] for tokens
            out = out.reshape(out.shape[0], self.num_classes, out.shape[1] // self.num_classes, out.shape[2], out.shape[3])
        return out

    def calc_likelihood(self, x):
        # Forward pass with bpd likelihood calculation
        pred = self.forward(x)
        nll = F.cross_entropy(pred, x, reduction="none")
        dims = [1, 2] if self.on_tokens else [1, 2, 3]
        bpd = nll.mean(dim=dims) * np.log2(np.exp(1))
        return bpd.mean(), pred

    @torch.no_grad()
    def sample(self, img_shape, img=None, device=torch.device("cpu")):
        """
        Sampling function for the autoregressive model.
        Inputs:
            img_shape - Shape of the image to generate (B,C,H,W)
            img (optional) - If given, this tensor will be used as
                             a starting image. The pixels to fill
                             should be -1 in the input tensor.
        """
        if img is None:
            pad = self.num_classes if self.on_tokens else -1
            img = torch.zeros(img_shape, dtype=torch.long).to(device) + pad  # empty img filled with '-1' or 'pad_token'

        for h in tqdm(range(img_shape[2]), leave=False):
            for w in range(img_shape[3]):
                for c in range(img_shape[1]):
                    # skip if not to be filled (-1)
                    if (img[:, c, h, w] != pad).all().item():
                        continue
                    # only the upper part of the image is passed, as all other parts will be skipped by the masked convolutions
                    pred = self.forward(img[:, :, : h + 1, :])
                    if self.on_tokens:
                        probs = F.softmax(pred[:, :, h, w], dim=-1)  # [B, Classes, H, W]
                    else:
                        probs = F.softmax(pred[:, :, c, h, w], dim=-1)  # [B, Classes, Channels, H, W]
                    img[:, c, h, w] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        return img.squeeze(1) if self.on_tokens else img  # ([B, Ch=1, H, W] --> [B, H, W]) if on_tokens else  [B, Ch, H, W]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    # Methods called {phase}_step are needed for pl.Trainer to automate training
    def training_step(self, batch, batch_idx):
        loss, pred = self.calc_likelihood(batch[0])
        self.log("train_bpd", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.calc_likelihood(batch[0])
        self.log("val_bpd", loss)
        print(f"\nValidation bpd: {loss:4.3f}")
        if not self.on_tokens and batch_idx == 0:
            pred_pixels = pred.argmax(dim=1)
            pred_pixels = pred_pixels.float() / self.num_classes
            real_imgs = batch[0] / self.num_classes
            visualize_val_reconstruction(real_imgs.cpu(), pred_pixels.cpu())

    def test_step(self, batch, batch_idx):
        loss, pred = self.calc_likelihood(batch[0])
        self.log("test_bpd", loss)


def train_pixel_cnn(epochs: int, dataloaders: DataLoader, device: torch.device, model_path: str, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=model_path,
        # accelerator="auto",
        accelerator="gpu" if str(device).startswith("cuda") else "mps" if str(device).startswith("mps") else "cpu",
        devices=1,
        max_epochs=epochs,
        logger=CSVLogger(model_path, name="logs_pixelcnn"),
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=1),
        ],
    )

    result = None
    pretrained_filename = os.path.join(model_path, "PixelCNN.ckpt")

    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = PixelCNN.load_from_checkpoint(pretrained_filename)
        ckpt = torch.load(pretrained_filename, map_location=device)
        result = ckpt.get("result", None)
    else:
        model = PixelCNN(**kwargs)
        trainer.fit(model, dataloaders["train"], dataloaders["val"])

    model = model.to(device)

    if result is None:  # best model is tested on val, test datsets
        val_result = trainer.test(model, dataloaders["val"], verbose=False)
        test_result = trainer.test(model, dataloaders["test"], verbose=False)
        result = {"test": test_result, "val": val_result}

    return model, result

import torch
import torch.nn as nn


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

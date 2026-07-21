import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ResidualBlock1D(nn.Module):
    def __init__(self,  in_channels: int, out_channels: int,
        stride: int = 1, dropout: float = 0.0,):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + identity
        x = F.gelu(x)

        return x
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        maximum_length: int = 2048,
    ):
        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.zeros(1, maximum_length, embedding_dim)
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]

        if sequence_length > self.position_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the maximum "
                f"supported length {self.position_embedding.shape[1]}."
            )

        return x + self.position_embedding[:, :sequence_length]
    

class SpectralCNNTransformer(nn.Module):
    """
    CNN + residual blocks + Transformer backbone.

    Input:
        (batch_size, 1, spectrum_length)

    Output:
        (batch_size, embedding_dim)
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_feedforward_dim: int = 256,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        maximum_spectrum_length: int = 2048,
    ):
        super().__init__()

        if transformer_dim % transformer_heads != 0:
            raise ValueError(
                "transformer_dim must be divisible by transformer_heads."
            )

        # Preserve the original spectral resolution in the stem.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                cnn_channels[0],
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.GELU(),
        )

        blocks = []
        current_channels = cnn_channels[0]

        for block_index, output_channels in enumerate(cnn_channels):
            # Only one mild downsampling operation.
            stride = 2 if block_index == 1 else 1

            blocks.append(
                ResidualBlock1D(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    stride=stride,
                    dropout=dropout,
                )
            )

            blocks.append(
                ResidualBlock1D(
                    in_channels=output_channels,
                    out_channels=output_channels,
                    stride=1,
                    dropout=dropout,
                )
            )

            current_channels = output_channels

        self.residual_blocks = nn.Sequential(*blocks)

        # Project CNN channels to the Transformer token dimension.
        self.token_projection = nn.Conv1d(
            current_channels,
            transformer_dim,
            kernel_size=1,
        )

        self.positional_encoding = LearnablePositionalEncoding(
            embedding_dim=transformer_dim,
            maximum_length=maximum_spectrum_length,
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=transformer_layers,
        )

        self.final_norm = nn.LayerNorm(transformer_dim)

        self.embedding_head = nn.Sequential(
            nn.Linear(transformer_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the full wavelength-token representation.

        Shape:
            (batch_size, reduced_spectrum_length, transformer_dim)
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (B, L) or (B, C, L), "
                f"but received {tuple(x.shape)}."
            )

        x = self.stem(x)
        x = self.residual_blocks(x)
        x = self.token_projection(x)

        # Conv1d output: (B, channels, length)
        # Transformer input: (B, length, channels)
        x = x.transpose(1, 2)

        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.final_norm(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_features(x)

        # Mean pooling retains information from every wavelength token.
        pooled = tokens.mean(dim=1)
        embedding = self.embedding_head(pooled)

        # Do not normalize here. SimSiam normalizes inside its loss, and
        # unnormalized embeddings are more flexible for downstream analysis.
        return embedding
    
# backbone = SpectralCNNTransformer(
#     in_channels=1,
#     cnn_channels=(32, 64, 128),
#     transformer_dim=128,
#     transformer_heads=4,
#     transformer_layers=2,
#     transformer_feedforward_dim=256,
#     embedding_dim=256,
#     dropout=0.1,
# )

# for MgNe or other smaller datasets
# backbone = SpectralCNNTransformer(
#     cnn_channels=(16, 32, 64),
#     transformer_dim=64,
#     transformer_heads=4,
#     transformer_layers=2,
#     transformer_feedforward_dim=128,
#     embedding_dim=128,
# )



# from lightly.models.modules import (
#     SimSiamPredictionHead,
#     SimSiamProjectionHead,
# )


# ### SimSiam option
# class SpectralSimSiam(pl.LightningModule):
#     def __init__(
#         self,
#         backbone_embedding_dim: int = 256,
#         projection_hidden_dim: int = 512,
#         projection_output_dim: int = 128,
#         prediction_hidden_dim: int = 64,
#         learning_rate: float = 3e-4,
#         weight_decay: float = 1e-4,
#         maximum_epochs: int = 100,
#     ):
#         super().__init__()

#         self.save_hyperparameters()

#         self.backbone = SpectralCNNTransformer(
#             in_channels=1,
#             cnn_channels=(32, 64, 128),
#             transformer_dim=128,
#             transformer_heads=4,
#             transformer_layers=2,
#             transformer_feedforward_dim=256,
#             embedding_dim=backbone_embedding_dim,
#             dropout=0.1,
#         )

#         self.projection_head = SimSiamProjectionHead(
#             backbone_embedding_dim,
#             projection_hidden_dim,
#             projection_output_dim,
#         )

#         self.prediction_head = SimSiamPredictionHead(
#             projection_output_dim,
#             prediction_hidden_dim,
#             projection_output_dim,
#         )

#     @staticmethod
#     def negative_cosine_similarity(
#         prediction: torch.Tensor,
#         target: torch.Tensor,
#     ) -> torch.Tensor:
#         prediction = F.normalize(prediction, dim=1)
#         target = F.normalize(target, dim=1)

#         return -(prediction * target).sum(dim=1).mean()

#     def encode(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Return the scientifically useful backbone embedding.

#         This is the representation to use for UMAP and HDBSCAN.
#         """
#         return self.backbone(x)

#     def forward(
#         self,
#         x: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         embedding = self.backbone(x)
#         projection = self.projection_head(embedding)
#         prediction = self.prediction_head(projection)

#         return embedding, projection, prediction

#     def training_step(
#         self,
#         batch: tuple[torch.Tensor, torch.Tensor],
#         batch_idx: int,
#     ) -> torch.Tensor:
#         view_1, view_2 = batch

#         _, projection_1, prediction_1 = self(view_1)
#         _, projection_2, prediction_2 = self(view_2)

#         loss_12 = self.negative_cosine_similarity(
#             prediction_1,
#             projection_2.detach(),
#         )

#         loss_21 = self.negative_cosine_similarity(
#             prediction_2,
#             projection_1.detach(),
#         )

#         loss = 0.5 * (loss_12 + loss_21)

#         self.log(
#             "train_loss",
#             loss,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             batch_size=view_1.shape[0],
#         )

#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(),
#             lr=self.hparams.learning_rate,
#             weight_decay=self.hparams.weight_decay,
#         )

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=self.hparams.maximum_epochs,
#             eta_min=1e-6,
#         )

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "epoch",
#             },
#         }
    

### VICReg option
class VICRegLoss(nn.Module):
    """
    VICReg loss:

    - Invariance: corresponding augmented views should be similar.
    - Variance: each representation dimension should retain variation.
    - Covariance: different representation dimensions should be decorrelated.
    """

    def __init__(
        self,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
        variance_target: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()

        self.invariance_weight = invariance_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_target = variance_target
        self.eps = eps

    @staticmethod
    def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
        """
        Return all off-diagonal elements of a square matrix.
        """
        n, m = matrix.shape

        if n != m:
            raise ValueError(
                f"Expected a square matrix, received shape {matrix.shape}."
            )

        return (
            matrix.flatten()[:-1]
            .view(n - 1, n + 1)[:, 1:]
            .flatten()
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if z1.shape != z2.shape:
            raise ValueError(
                f"Representation shapes must match: "
                f"{z1.shape} versus {z2.shape}."
            )

        batch_size, feature_dim = z1.shape

        if batch_size < 2:
            raise ValueError(
                "VICReg requires at least two samples per batch."
            )

        # 1. Invariance term
        invariance_loss = F.mse_loss(z1, z2)

        # Center each feature dimension across the batch.
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        # 2. Variance term
        std_z1 = torch.sqrt(
            z1_centered.var(dim=0, unbiased=False) + self.eps
        )
        std_z2 = torch.sqrt(
            z2_centered.var(dim=0, unbiased=False) + self.eps
        )

        variance_loss = 0.5 * (
            F.relu(self.variance_target - std_z1).mean()
            + F.relu(self.variance_target - std_z2).mean()
        )

        # 3. Covariance term
        covariance_z1 = (
            z1_centered.T @ z1_centered
        ) / (batch_size - 1)

        covariance_z2 = (
            z2_centered.T @ z2_centered
        ) / (batch_size - 1)

        covariance_loss = (
            self.off_diagonal(covariance_z1).pow(2).sum()
            / feature_dim
            +
            self.off_diagonal(covariance_z2).pow(2).sum()
            / feature_dim
        )

        total_loss = (
            self.invariance_weight * invariance_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )

        components = {
            "invariance_loss": invariance_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
        }

        return total_loss, components
    
class VICRegProjectionHead(nn.Module):
    """
    MLP projection head used only during self-supervised training.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralVICReg(pl.LightningModule):
    def __init__(
        self,
        backbone_embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        projection_output_dim: int = 256,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        maximum_epochs: int = 100,
        invariance_weight: float = 25.0,
        variance_weight: float = 25.0,
        covariance_weight: float = 1.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.backbone = SpectralCNNTransformer(
            in_channels=1,
            cnn_channels=(32, 64, 128),
            transformer_dim=128,
            transformer_heads=4,
            transformer_layers=2,
            transformer_feedforward_dim=256,
            embedding_dim=backbone_embedding_dim,
            dropout=0.1,
        )

        self.projection_head = VICRegProjectionHead(
            input_dim=backbone_embedding_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )

        self.criterion = VICRegLoss(
            invariance_weight=invariance_weight,
            variance_weight=variance_weight,
            covariance_weight=covariance_weight,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backbone representation used for UMAP and HDBSCAN.
        """
        return self.backbone(x)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(x)
        projection = self.projection_head(embedding)

        return embedding, projection

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        view_1, view_2 = batch

        _, projection_1 = self(view_1)
        _, projection_2 = self(view_2)

        loss, components = self.criterion(
            projection_1,
            projection_2,
        )

        batch_size = view_1.shape[0]

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.log(
            "invariance_loss",
            components["invariance_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "variance_loss",
            components["variance_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            "covariance_loss",
            components["covariance_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.maximum_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
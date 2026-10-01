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
            nn.Conv1d(in_channels, cnn_channels[0], kernel_size=5, stride=1,
                padding=2, bias=False,),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.GELU())

        blocks = []
        current_channels = cnn_channels[0]

        for block_index, output_channels in enumerate(cnn_channels):
            # Only one mild downsampling operation.
            stride = 2 if block_index == 1 else 1

            blocks.append(ResidualBlock1D(
                    in_channels=current_channels,
                    out_channels=output_channels,
                    stride=stride,
                    dropout=dropout,)
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

### SimSiam option
class NegativeCosineSimilarity(nn.Module):
    """
    SimSiam loss.

    The predictor output from one view is matched to the stop-gradient
    projection from the other view. The loss is evaluated symmetrically.
    """

    @staticmethod
    def forward(
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        prediction = F.normalize(prediction, dim=1)
        target = F.normalize(target.detach(), dim=1)

        return -(prediction * target).sum(dim=1).mean()


class SimSiamProjectionHead(nn.Module):
    """
    Three-layer SimSiam projection MLP.

    Batch normalization is used after every linear layer. Following the
    original SimSiam design, the final BatchNorm layer has no affine
    parameters.
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
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiamPredictionHead(nn.Module):
    """
    Two-layer bottleneck predictor used only during pretraining.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralSimSiam(pl.LightningModule):
    """
    SimSiam model for one-dimensional spectra.

    Expected training batch:
        view_1, view_2

    Each view may have shape:
        (batch_size, spectrum_length)
    or:
        (batch_size, 1, spectrum_length)

    For downstream UMAP, HDBSCAN, or classification, use encode(), which
    returns the backbone embedding before the SimSiam projection head.
    """

    def __init__(
        self,
        backbone_embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        projection_output_dim: int = 256,
        prediction_hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        maximum_epochs: int = 100,
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

        self.projection_head = SimSiamProjectionHead(
            input_dim=backbone_embedding_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
        )

        self.prediction_head = SimSiamPredictionHead(
            input_dim=projection_output_dim,
            hidden_dim=prediction_hidden_dim,
            output_dim=projection_output_dim,
        )

        self.criterion = NegativeCosineSimilarity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the backbone representation for downstream analysis.
        """
        return self.backbone(x)

    def forward(self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
            embedding: backbone representation
            projection: projector output
            prediction: predictor output
        """
        embedding = self.backbone(x)
        projection = self.projection_head(embedding)
        prediction = self.prediction_head(projection)

        return embedding, projection, prediction

    def shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        view_1, view_2 = batch

        _, projection_1, prediction_1 = self(view_1)
        _, projection_2, prediction_2 = self(view_2)

        # Cross-view matching with stop-gradient targets.
        loss_12 = self.criterion(prediction_1, projection_2)
        loss_21 = self.criterion(prediction_2, projection_1)
        loss = 0.5 * (loss_12 + loss_21)

        batch_size = view_1.shape[0]

        self.log(
            f"{stage}_loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_cosine_similarity",
            -loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self.shared_step(batch, stage="train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self.shared_step(batch, stage="val")

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

        return {"optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# Example:
# model = SpectralSimSiam(
#     backbone_embedding_dim=256,
#     projection_hidden_dim=512,
#     projection_output_dim=256,
#     prediction_hidden_dim=128,
#     learning_rate=3e-4,
#     weight_decay=1e-4,
#     maximum_epochs=100,
# )
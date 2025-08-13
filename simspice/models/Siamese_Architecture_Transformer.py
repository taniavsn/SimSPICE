import torch
from torch import nn
import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl  # type: ignore
from lightly.models.modules import SimSiamPredictionHead, \
                                    SimSiamProjectionHead  # type: ignore
from lightly.loss import NTXentLoss  # type: ignore


class Transformer1DBackbone(nn.Module):
    def __init__(self, max_length=512, d_model=64, nhead=4,
                 num_layers=3, output_dim=128):
        super(Transformer1DBackbone, self).__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)

        # Create max_length positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (B, C=1, L)
        x = x.permute(0, 2, 1)  # (B, L, C=1)

        # Project input to d_model
        x = self.input_proj(x)  # (B, L, D)

        # Add positional encoding up to input length
        pos_enc = self.pos_encoding[:x.size(1), :]  # (L, D)
        x = x + pos_enc  # broadcasting (B, L, D) + (L, D)

        x = self.transformer(x)  # (B, L, D)
        x = x.permute(0, 2, 1)   # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        x = F.relu(self.fc(x))  # (B, output_dim)
        return F.normalize(x, dim=1, eps=1e-8)


class SimSiam(pl.LightningModule):
    """
    Parameters:
    - output_dim (int): Dimension of the output layer.
    - backbone_output_dim (int): Dimension of the backbone's output.
    - hidden_layer_dim (int): Dimension of the hidden layer.
    """

    def __init__(self, output_dim=64, backbone_output_dim=128,
                 hidden_layer_dim=128):
        super().__init__()
        self.backbone = Transformer1DBackbone(output_dim=backbone_output_dim)

        # projection head: map data representations into a space that 
        # facilitates comparison and learning
        self.projection_head = SimSiamProjectionHead(
            backbone_output_dim, hidden_layer_dim, output_dim
        )

        # prediction head: produce the final output from the learned features
        self.prediction_head = SimSiamPredictionHead(
            output_dim, hidden_layer_dim, output_dim
        )
        self.criterion = NTXentLoss(temperature=0.07)

    def forward(self, x):
        f = self.backbone(x)
        # z are the embeddings, meaning the data represented
        #  in a lower dimension space.
        z = self.projection_head(f)
        p = self.prediction_head(z)
        return z, p

    def training_step(self, batch):
        (x0, x1) = batch

        # Forward both views
        z0, p0 = self.forward(x0)  # z0: projection, p0: prediction
        z1, p1 = self.forward(x1)

        # Stop-gradient: prevent gradient from flowing into z1 when 
        # comparing p0→z1, and vice versa
        loss = 0.5 * (
            self.criterion(p0, z1.detach()) +
            self.criterion(p1, z0.detach())
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


def run_model(checkpoint, dataset):
    """
    checkpoint: the wandb checkpoint to load the model.
    dataset: a SproutDataset object, with the augmentation_type set to None.
            ex: SproutDataset(dataset_path="C:\\Users\\tania\\Documents\\SPICE
            \\SPROUTS\\spectra_train_mini.nc", augmentation_type=None)
    """
    loaded_model = SimSiam.load_from_checkpoint(checkpoint)
    loaded_model.eval()
    outputs = []
    with torch.no_grad():  # Disable gradient computation for inference
        for i in tqdm.tqdm(range(dataset.__len__())):
            spec = dataset.__getitem__(i).unsqueeze(0)
            # Move tensor to the same device as the model
            device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
            loaded_model = loaded_model.to(device)
            spec = spec.to(device)

            outputs.append(loaded_model(spec)[0].cpu().numpy())
    return outputs

import torch
from torch import nn
import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl # type: ignore
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead # type: ignore
from lightly.loss import NTXentLoss # type: ignore


class Siamese1DNet_backbone(nn.Module):
    def __init__(self, output_dim=128):
        super(Siamese1DNet_backbone, self).__init__()
                
        # Shared feature extraction network
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layers for embeddings
        self.fc1 = nn.LazyLinear(output_dim)  # keep the backbone complex enough
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) 
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))   
        return F.normalize(x, dim=1)


class SimSiam(pl.LightningModule):
    """        
    Parameters:
    - output_dim (int): Dimension of the output layer.
    - backbone_output_dim (int): Dimension of the backbone's output.
    - hidden_layer_dim (int): Dimension of the hidden layer.
    """
    def __init__(self, output_dim=64, backbone_output_dim=128, hidden_layer_dim=128):
        super().__init__()
        self.backbone = Siamese1DNet_backbone(output_dim=backbone_output_dim)

        # projection head: map data representations into a space that facilitates comparison and learning
        self.projection_head = SimSiamProjectionHead(backbone_output_dim, hidden_layer_dim, output_dim) 

        # prediction head: produce the final output from the learned features
        self.prediction_head = SimSiamPredictionHead(output_dim, hidden_layer_dim, output_dim)
        self.criterion =  NTXentLoss(temperature=0.07)

    def forward(self, x):
        f = self.backbone(x)
        z = self.projection_head(f)  # z are the embeddings, meaning the data represented in a lower dimension space.
        p = self.prediction_head(z)
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        # print('\n', x0.shape, x1.shape)

        #### Wrap loss function? loss_fn would be self.criterion
        # loss_fn = NTXentLoss()
        # loss_fn = SelfSupervisedLoss(loss_fn)
        # loss = loss_fn(z0,z1)
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


# class CustomSimSiam(SimSiam):
#     def __init__(self, output_dim=128, backbone_output_dim=128, hidden_layer_dim=128, **kwargs):
#         # Call the parent class constructor with the new dimensions
#         super().__init__(
#             output_dim=output_dim,
#             backbone_output_dim=backbone_output_dim,
#             hidden_layer_dim=hidden_layer_dim,
#             **kwargs
#         )


def run_model(checkpoint, dataset):
    '''
    checkpoint: the wandb checkpoint to load the model.
    dataset: a SproutDataset object, with the augmentation_type set to None. 
            ex: SproutDataset(dataset_path="C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\spectra_train_mini.nc", augmentation_type=None)
    '''
    loaded_model = SimSiam.load_from_checkpoint(checkpoint)
    loaded_model.eval()
    outputs = []
    with torch.no_grad():  # Disable gradient computation for inference
        for i in tqdm.tqdm(range (dataset.__len__())):
            spec = dataset.__getitem__(i).unsqueeze(0)
            # Move tensor to the same device as the model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_model = loaded_model.to(device)
            spec = spec.to(device)

            outputs.append(loaded_model(spec)[0].cpu().numpy())
    return outputs

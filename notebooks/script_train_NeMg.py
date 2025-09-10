import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)
parent_dir1 = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.append(parent_dir1)

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch import nn
import torch.nn.functional as F
from simspice.data.SproutDataset_NeMg import SproutDataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import numpy as np
import matplotlib.pyplot as plt
from cuml.cluster.hdbscan import HDBSCAN

import matplotlib.pyplot as plt
from lightly.loss import NTXentLoss

import simspice.utils.inverse_mapping_functions as imf
import simspice.models.Siamese_Architecture_Resnet as SA
import wandb
#import umap.umap_ as umap
import tqdm
from datetime import datetime

plt.rcParams['image.origin'] = 'lower'

BATCH_SIZE = 128

simspice = "/d0/tvaresano/SimSPICE/"
torch.cuda.set_device(2)

dataset_path = simspice+"spectra_train_NeMg_10files.nc"
dataset = SproutDataset(dataset_path=dataset_path, augmentation_type='single', 
                        csv_files=simspice+'L2_names.csv',
                        type_distrib_shift='uniform', type_distrib_gain='uniform',
                        log_space=False, normalize_intensity=False, 
                        shift_range=(-0.05, 0.05), gain_range=(0.1, 5))

dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)


id = 'gain01-5_10file_shift005'

# Train model
model = SA.SimSiam(output_dim=32, backbone_output_dim=64, hidden_layer_dim=64)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

wandb_logger = WandbLogger(project="Resnet50_SimSiam_single32_NeMg", 
                           name=f"{id}_{datetime.today().strftime('%Y-%m-%d')}", id=id, log_model=True)

trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator, logger=wandb_logger)
trainer.fit(model=model, train_dataloaders=dataloader)


##
#  Run model
dataset_none = SproutDataset(dataset_path=dataset_path, augmentation_type=None, csv_files=simspice+'L2_names.csv',
                                           log_space=False, normalize_intensity=False)

# Move model to device ONCE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Use DataLoader for batching + parallel workers
loader = DataLoader(dataset_none, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

outputs = []
with torch.no_grad():
    for spec in tqdm.tqdm(loader):
        spec = spec.to(device)
        batch_out = model(spec)[0]  # shape: (batch, ...)
        outputs.append(batch_out.cpu())

# Concatenate all batches into one tensor
outputs = torch.cat(outputs, dim=0).numpy()

stacked_outputs = np.stack(outputs).squeeze()
np.save(simspice+f'notebooks/jobs/model_outputs/MgNe_32_resnet50_simsiam/outputs{id}.npy', stacked_outputs)
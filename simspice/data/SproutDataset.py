import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from simspice.utils.Augmentation import Augmentation
from simspice.data.Sprout_ML import Sprout_ML
from torch.utils.data import Dataset
import numpy as np
import xarray as xr # type: ignore
import pandas as pd
import torch



BATCH_SIZE = 32

# zarr array (load in __init__ and becomes real when used) -- later

class SproutDataset(Dataset, Sprout_ML):
    def __init__(
        self,
        csv_files="L2_names.csv",
        file_dir=r"C:\Users\tania\Documents\SPICE\SPROUTS\datasets_deepL",
        dataset_path=r"C:\Users\tania\Documents\SPICE\SPROUTS\spectra_train.nc",
        augmentation_type="double",
        log_space=False,
        mu_doppler=0.0,
        sigma_shift=0.1,
        sigma_gain=0.1,
        shift_range=(-0.4, 0.4),
        gain_range=(0.8, 1.2),
        background_range=(-0.02, 0.02),
        type_distrib_gain="uniform",
        type_distrib_shift="Gaussian",
        normalize_intensity=True,
        normalization_method="area",
        normalization_eps=1e-8,
        add_noise=True,
    ):
        self.file_names = pd.read_csv(
            os.path.join(file_dir, csv_files)
        )

        self.file_dir = file_dir
        self.all_spectra = xr.open_dataset(dataset_path)

        self.augmentation_type = augmentation_type

        self.augmenter = Augmentation(
            mu_doppler=mu_doppler,
            sigma_shift=sigma_shift,
            sigma_gain=sigma_gain,
            shift_range=shift_range,
            gain_range=gain_range,
            type_distrib_gain=type_distrib_gain,
            type_distrib_shift=type_distrib_shift,
            normalize_intensity=normalize_intensity,
            normalization_method=normalization_method,
            normalization_eps=normalization_eps,
            log_space=log_space,
            add_noise=add_noise,
            add_background=True,
            background_range=background_range )

    def __len__(self):
        return self.all_spectra.sizes["index"]

    def __getitem__(self, index):
        row = self.all_spectra.isel(index=index)
        wavelength = np.asarray(row["wvl"].values, dtype=np.float32,).copy()
        original_flux = np.asarray(row["flux"].values, dtype=np.float32,).copy()
        original_flux = self.augmenter.preprocess(original_flux, wavelength=wavelength)

        original_tensor = torch.from_numpy(original_flux).unsqueeze(0)

        if self.augmentation_type is None:
            return original_tensor

        augmentation_type = self.augmentation_type.lower()

        if augmentation_type == "double":
            augmented_1, _ = (self.augmenter.run_all_augmentations(row))
            augmented_2, _ = (self.augmenter.run_all_augmentations(row))

            return (torch.from_numpy(augmented_1).unsqueeze(0),
                torch.from_numpy(augmented_2).unsqueeze(0))

        if augmentation_type == "single":
            augmented, _ = (self.augmenter.run_all_augmentations(row))

            return (original_tensor, torch.from_numpy(augmented).unsqueeze(0) )

        raise ValueError(
            "augmentation_type must be None, "
            "'single', or 'double'.")
        


        

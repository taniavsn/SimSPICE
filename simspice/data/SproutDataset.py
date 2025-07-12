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

class SproutDataset(Dataset, Sprout_ML):  # offspring of both classes
    
    def __init__(self, csv_files: str='L2_names.csv', file_dir: str='C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\datasets_deepL\\', 
                 dataset_path = "C:\\Users\\tania\\Documents\\SPICE\\SPROUTS\\spectra_train.nc",
                 batch_size: int=BATCH_SIZE, augmentation_type: str='single', log_space=False, 
                 mu_doppler=0, sigma = 1, num_hits = 2,
                 shift_range=(-0.4, 0.4), gain_range=(0.1, 3), 
                 type_distrib_gain='Gaussian', type_distrib_shift='Gaussian', normalize_intensity=False):
        '''
        file_dir (str): path to the folder containing the images
        augmentation (str): whether to just return the original spectrum (None)
                                    perform single augmentation ('single') 
                                    or perform double augmentation ('double').
                                    No augmentation returns a single spectrum, single or double augmentation returns two.
        log10space (Bool): decides wether the output spectra will be in log space or not.
        '''
        # super().__init__(self, filename)
        self.file_names = pd.read_csv(os.path.join(file_dir, csv_files))  # get all the file names 
        self.batch_size = batch_size
        self.file_dir = file_dir

        # Get all spectra
        self.all_spectra = xr.open_dataset(dataset_path)

        # set up augmentations
        self.augmentation_type = augmentation_type
        self.log_space = log_space

        self.normalize_intensity = normalize_intensity 

        # GCR parameters
        self.num_hits = num_hits

        # Shift parameters
        self.shift_range = shift_range
        self.type_distrib_shift = type_distrib_shift
        self.mu_doppler = mu_doppler 

        # Gain parameters
        self.sigma = sigma
        self.type_distrib_gain = type_distrib_gain
        self.gain_range = gain_range
        

    def __len__(self) -> int: # how many spectra do we have?
        '''
        Length of the dataset
        '''
        return len(self.all_spectra['index'])
    

    def __getitem__(self, index):
        '''
        returns two spectra, with a flag that makes it return two augmented spectra (based on an original) or the original and an augmented version.
        Args:
            idx (int): index of the spectrum to retrieve
        '''
        aug = Augmentation(mu_doppler=self.mu_doppler, sigma = self.sigma, num_hits = self.num_hits,
                            shift_range=self.shift_range, gain_range=self.gain_range, 
                            type_distrib_gain=self.type_distrib_gain, type_distrib_shift=self.type_distrib_shift, 
                            normalize_intensity=self.normalize_intensity, log_space=self.log_space)
        
        row = self.all_spectra.isel(index=index)

        spectrum = row['flux'].values
        # wvl_array = row['wvl'].values
        # mask = row['mask'].values
        if self.log_space:
            spectrum = np.nan_to_num(np.log10(spectrum), nan=0, posinf=0, neginf=0)

        ## initialize the augmentation : return either the original spectrum and 1 augmentation, or 2 augmentations (of a single item) 
        if self.augmentation_type is None: 
            return torch.Tensor(spectrum)[None, :] #adds 1 extra dimension (channels) 

        elif self.augmentation_type.lower() == 'double':
            spec_aug_1, mask_aug_1 = aug.run_all_augmentations(row)
            spec_aug_2, mask_aug_2 = aug.run_all_augmentations(row)
            return torch.Tensor(spec_aug_1)[None, :], torch.Tensor(spec_aug_2)[None, :] #, mask_aug_1, mask_aug_2, wvl_array

        elif self.augmentation_type.lower() == 'single':
            spec_aug_1, mask_aug_1 = aug.run_all_augmentations(row)
            return torch.Tensor(spectrum)[None, :], torch.Tensor(spec_aug_1)[None, :] #, mask, mask_aug_1, wvl_array
        


        

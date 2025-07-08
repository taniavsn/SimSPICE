import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings 
import random
from scipy.stats import poisson
warnings.filterwarnings("ignore")

class Augmentation():

    def __init__(self, 
                 mu_doppler=0, sigma = 1, num_hits = 2,
                 shift_range=(-0.4, 0.4), gain_range=(0.1, 3), 
                 type_distrib_gain='Gaussian', type_distrib_shift='Gaussian', 
                 normalize_intensity=False, log_space=True): 
        '''
        mu_doppler: mean of distribution of the simulated doppler shift (should be 0)
        shift_range: range of doppler shift values

        num_hits: number of occurences of GCRs on an item (spectrum)

        gain_range = range of gain values
        sigma: standard deviation of the distribution of gain, if Gaussian chosen.

        normalize_intensity: Bool
        log_space: Bool

        '''
        self.normalize_intensity = normalize_intensity 
        self.log_space = log_space

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

        self.methods_list = [method for method in dir(Augmentation) if callable(getattr(Augmentation, method)) and method.startswith('add')]
        
        


    def add_shift_spectrum(self, spectrum):
        """
        Shifts a spectrum along a wavelength array.

        Parameters:
        spectrum: Dataset row object (all_spectra.isel(index=index))
        shift (float): The shift in wavelength units (positive for redshift, negative for blueshift).

        Returns:
        np.ndarray: Shifted spectrum aligned with the original wavelength array.
        """
        # Interpolation function for the original spectrum
        wavelength = spectrum['wvl'].values
        flux = spectrum['flux'].values
        mask = spectrum['mask'].values
        if self.type_distrib_shift == 'uniform':
            shift = random.uniform(self.shift_range[0], self.shift_range[1]) 
        else:
            shift = random.gauss(self.mu_doppler, self.sigma) 
        interp_func = interp1d(wavelength + shift, flux, bounds_error=False, fill_value=0)
        shifted_spectrum = interp_func(wavelength)
        interp_func_mask = interp1d(wavelength + shift, mask, bounds_error=False, fill_value=0)
        shifted_mask = interp_func_mask(wavelength)
        
        return shifted_spectrum, shifted_mask


    def add_gain_spectrum(self, spectrum):
        '''
        Applies a gain to the spectrum
        The gain is chosen randomly from either a uniform or gaussian distribution

        spectrum: Dataset row object (all_spectra.isel(index=index))
        '''
        if isinstance(spectrum, np.ndarray):
            if self.type_distrib_gain == 'uniform':
                return spectrum*random.uniform(self.gain_range[0], self.gain_range[1]) 
            else:
                return spectrum*random.gauss((self.gain_range[1] - self.gain_range[0])/2, self.sigma)  # Guassian distribution
        else:
            if self.type_distrib_gain == 'uniform':
                return spectrum['flux'].values*random.uniform(self.gain_range[0], self.gain_range[1]) 
            else:
                return spectrum['flux'].values*random.gauss((self.gain_range[1] - self.gain_range[0])/2, self.sigma)  # Guassian distribution
    


    def add_photon_noise(self, spectrum):  # scipy poisson?
        '''
        Applies a Poisson-distributed noise to the spectrum

        spectrum: Dataset row object (all_spectra.isel(index=index))
        '''
        # return poisson.pmf(self.k, self.mu, self.loc)
        if isinstance(spectrum, np.ndarray):
            spectrum[spectrum < 0] = 0
            return spectrum+np.random.poisson(np.nan_to_num(spectrum))/8
        else:
            spectrum = spectrum['flux'].values
            spectrum[spectrum < 0] = 0
            return spectrum+np.random.poisson(np.nan_to_num(spectrum['flux'].values))/8



    def add_GCR_noise(self, spectrum):
        '''
        Applies random GCR spikes to the data at a specified number of points.
        
        Parameters:
        - spectrum: Dataset row object (all_spectra.isel(index=index))
        - num_hits: The number of pixels in the spectrum that will get a GCR spike.
        '''
        if isinstance(spectrum, np.ndarray):
            intensity_range = (np.max(spectrum) * 2, np.max(spectrum) * 2.5)
            noisy_spectrum = spectrum.copy().astype(float)
            num_points = len(spectrum)
        else:
            intensity_range = (np.max(spectrum['flux'].values) * 1.5, np.max(spectrum['flux'].values) * 2)
            noisy_spectrum = spectrum['flux'].values.copy().astype(float)
            num_points = len(spectrum['flux'].values)
        
        valid_indices = np.arange(5, num_points - 5)  # Exclude five first and last indices
    
        # Randomly select indices for GCR hits
        hit_indices = np.random.choice(valid_indices, size=self.num_hits, replace=False)
        
        # Generate random spike intensities within the specified range
        spikes = np.random.uniform(intensity_range[0], intensity_range[1], size=self.num_hits)
        # Apply spikes at the selected indices
        noisy_spectrum[hit_indices] += spikes
        
        return noisy_spectrum
    

    def run_all_augmentations(self, spectrum):
        '''
        applies augmentation methods successively in this order: 
        1. Shift
        2. Gain
        3. Photon noise
        4. GCR noise
        Returns augmented spectrum and shifted mask
        '''
        if self.normalize_intensity:
            if isinstance(spectrum, np.ndarray):
                spectrum = spectrum/np.sum(spectrum)
            else:
                spectrum['flux'].values = spectrum['flux'].values/np.sum(spectrum['flux'].values)
        if self.log_space:
            if isinstance(spectrum, np.ndarray):
                spectrum = np.abs(np.nan_to_num(np.log(spectrum),nan=0, posinf=0, neginf=0))
            else:
                spectrum['flux'].values = np.abs(np.nan_to_num(np.log(spectrum['flux'].values),nan=0, posinf=0, neginf=0))
        spectrum, mask = self.add_shift_spectrum(spectrum)
        spectrum = self.add_gain_spectrum(spectrum)
        spectrum = self.add_photon_noise(spectrum)
        # spectrum = self.add_GCR_noise(spectrum)
        
        return spectrum, mask


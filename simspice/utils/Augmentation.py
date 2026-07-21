import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings 
import random
from scipy.stats import poisson
warnings.filterwarnings("ignore")

import random
import numpy as np
from scipy.interpolate import interp1d


class Augmentation:
    def __init__(
        self,
        mu_doppler=0.0,
        sigma_shift=0.1,
        sigma_gain=0.1,
        shift_range=(-0.4, 0.4),
        gain_range=(0.8, 1.2),
        type_distrib_gain="uniform",
        type_distrib_shift="Gaussian",
        add_noise=True,
        add_background=True,
        background_range=(-0.02, 0.02),
        normalize_intensity=False,
        log_space=False,
    ):
        self.normalize_intensity = normalize_intensity
        self.log_space = log_space
        self.add_noise = add_noise
        self.add_background = add_background

        self.shift_range = shift_range
        self.type_distrib_shift = type_distrib_shift
        self.mu_doppler = mu_doppler
        self.sigma_shift = sigma_shift

        self.type_distrib_gain = type_distrib_gain
        self.gain_range = gain_range
        self.sigma_gain = sigma_gain

        self.background_range = background_range


    ## ADD DOPPLER SHIFT
    def add_shift_spectrum(self, wavelength, flux, mask=None):
        if self.type_distrib_shift.lower() == "uniform":
            shift = random.uniform(self.shift_range[0], self.shift_range[1])
        else:
            shift = random.gauss(self.mu_doppler, self.sigma_shift)
            shift = np.clip(shift, self.shift_range[0], self.shift_range[1] )

        flux_interpolator = interp1d( wavelength + shift, flux, kind="linear", bounds_error=False, fill_value=0.0)
        shifted_flux = flux_interpolator(wavelength)
        shifted_mask = None

        if mask is not None:
            mask_interpolator = interp1d(
                wavelength + shift,
                mask.astype(float),
                kind="nearest",
                bounds_error=False,
                fill_value=0.0, )

            shifted_mask = (mask_interpolator(wavelength) >= 0.5)

        return shifted_flux, shifted_mask

    ## ADD GAIN
    def add_gain_spectrum(self, spectrum):
        if self.type_distrib_gain.lower() == "uniform":
            gain = random.uniform(self.gain_range[0], self.gain_range[1])
        else:
            gain_mean = 0.5 * (self.gain_range[0] + self.gain_range[1])
            gain = random.gauss(gain_mean, self.sigma_gain)
            gain = np.clip(gain, self.gain_range[0], self.gain_range[1])

        return spectrum * gain

    ## ADD BACKGROUND
    def add_background_spectrum(self, spectrum):
        reference = np.nanpercentile(np.abs(spectrum), 99)
        if not np.isfinite(reference) or reference <= 0:
            return spectrum
        background_fraction = random.uniform(self.background_range[0], self.background_range[1])
        return (spectrum + background_fraction * reference)


    ## ADD NOISE
    def add_photon_noise(self, spectrum):
        spectrum = np.asarray(spectrum, dtype=np.float32,).copy()

        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        spectrum = np.clip(spectrum, a_min=0.0, a_max=None)

        return (spectrum + np.random.poisson(spectrum) / 8.0)

    def preprocess(self, flux):
        flux = np.asarray(flux, dtype=np.float32).copy()

        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_intensity:
            scale = np.nanpercentile(np.abs(flux), 99)

            if np.isfinite(scale) and scale > 0:
                flux = flux / scale
            else:
                flux = np.zeros_like(flux)

        if self.log_space:
            # Safe for normalized, nonnegative intensities.
            flux = np.log1p(np.clip(flux, 0.0, None))

        return flux

    def run_all_augmentations(self, spectrum):
        wavelength = np.asarray(spectrum["wvl"].values, dtype=np.float64)
        flux = np.asarray(spectrum["flux"].values, dtype=np.float32).copy()
        mask = None
        if "mask" in spectrum:
            mask = np.asarray(spectrum["mask"].values, dtype=bool).copy()

        # Apply physical changes in linear intensity space.
        flux, mask = self.add_shift_spectrum(
            wavelength=wavelength, flux=flux,
            mask=mask)

        flux = self.add_gain_spectrum(flux)

        if self.add_background:
            flux = self.add_background_spectrum(flux)

        if self.add_noise:
            flux = self.add_photon_noise(flux)

        # Normalize/log only after physical augmentations.
        flux = self.preprocess(flux)

        return flux.astype(np.float32), mask


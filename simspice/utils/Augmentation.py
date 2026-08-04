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
        normalization_method="area",
        normalization_eps=1e-8,
        log_space=False,
    ):
        self.normalize_intensity = normalize_intensity
        self.normalization_method = normalization_method
        self.normalization_eps = normalization_eps
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
    def add_photon_noise(self, spectrum, count_scale=100.0):
        spectrum = np.asarray(spectrum, dtype=np.float32).copy()
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        positive_signal = np.clip(spectrum, 0.0, None)
        expected_counts = (positive_signal * count_scale)
        noisy_signal = (np.random.poisson(expected_counts) / count_scale)
        return noisy_signal.astype(np.float32)

    ## Normalize by integrated spectra
    def normalize_spectrum(self, flux, wavelength=None):
        """
        Normalize a spectrum by its integrated area.

        The spectral dimension must be the final axis.
        """
        flux = np.asarray(flux, dtype=np.float32)
        positive_flux = np.clip(flux, 0.0, None)

        if self.normalization_method.lower() == "area":
            if wavelength is None:
                raise ValueError("wavelength must be provided for area normalization.")

            wavelength = np.asarray(wavelength, dtype=np.float32)
            if wavelength.shape[-1] != flux.shape[-1]:
                raise ValueError(
                    f"Wavelength has {wavelength.shape[-1]} bins, "
                    f"but flux has {flux.shape[-1]} bins."
                )

            scale = np.trapezoid(positive_flux, x=wavelength, axis=-1)

        elif self.normalization_method.lower() == "sum":
            # Useful for concatenated, evenly sampled spectral windows.
            scale = np.sum(positive_flux, axis=-1)

        elif self.normalization_method.lower() == "percentile":
            scale = np.nanpercentile( np.abs(flux), 99, axis=-1)

        else:
            raise ValueError(
                f"Unknown normalization method: "
                f"{self.normalization_method!r}"
            )

        scale = np.asarray(scale)

        # Restore the spectral axis for broadcasting.
        scale = np.expand_dims(scale, axis=-1)

        valid = np.isfinite(scale) & (scale > self.normalization_eps)

        return np.divide(
            positive_flux,
            scale,
            out=np.zeros_like(positive_flux),
            where=valid,
        )

    def preprocess(self, flux, wavelength=None):
        flux = np.asarray(flux, dtype=np.float32).copy()
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_intensity:
            flux = self.normalize_spectrum(flux, wavelength=wavelength)

        if self.log_space:
            flux = np.log1p(np.clip(flux, 0.0, None))

        return flux.astype(np.float32)

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
        flux = self.preprocess(flux, wavelength=wavelength)

        return flux.astype(np.float32), mask




"""Spectral augmentations and deterministic gain-invariant window encoding."""
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
        window_lengths=None,
        apply_gain=True,
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
        # Shared by several datasets: leave gain enabled for legacy pipelines,
        # but set apply_gain=False for the ONeS two-channel dataset.
        self.apply_gain = apply_gain
        self.window_lengths = tuple(window_lengths) if window_lengths is not None else None

        self.background_range = background_range


    ## ADD DOPPLER SHIFT
    def add_shift_spectrum(self, wavelength, flux, mask=None):
        """Shift spectra in wavelength without interpolating across window gaps.

        The same shift is used in every window. For legacy single-window
        datasets, window_lengths=None keeps the old full-array behavior.
        """
        if self.type_distrib_shift.lower() == "uniform":
            shift = random.uniform(self.shift_range[0], self.shift_range[1])
        else:
            shift = random.gauss(self.mu_doppler, self.sigma_shift)
            shift = np.clip(shift, self.shift_range[0], self.shift_range[1])

        wavelength = np.asarray(wavelength, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)

        if self.window_lengths is None:
            boundaries = [(0, flux.size)]
        else:
            if sum(self.window_lengths) != flux.size:
                raise ValueError('window_lengths must sum to the spectrum length')
            ends = np.cumsum((0,) + self.window_lengths)
            boundaries = list(zip(ends[:-1], ends[1:]))

        shifted_flux = np.zeros_like(flux)
        shifted_mask = np.zeros_like(flux, dtype=bool) if mask is not None else None

        for lo, hi in boundaries:
            wvl = wavelength[lo:hi]
            shifted_flux[lo:hi] = interp1d(
                wvl + shift, flux[lo:hi], kind="linear",
                bounds_error=False, fill_value=0.0,
            )(wvl)
            if mask is not None:
                shifted_mask[lo:hi] = interp1d(
                    wvl + shift, mask[lo:hi].astype(float), kind="nearest",
                    bounds_error=False, fill_value=0.0,
                )(wvl) >= 0.5

        return shifted_flux, shifted_mask

    ## ADD GAIN (disabled by the ONeS dataset)
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
            # Interpolated NetCDF masks can be fractional; 0.1 is not valid.
            mask = np.asarray(spectrum["mask"].values, dtype=float) >= 0.5

        # Apply physical changes in linear intensity space.
        flux, mask = self.add_shift_spectrum(wavelength=wavelength, flux=flux, mask=mask)

        if self.apply_gain:
            flux = self.add_gain_spectrum(flux)

        if self.add_background:
            flux = self.add_background_spectrum(flux)

        if self.add_noise:
            flux = self.add_photon_noise(flux)

        # Keep padded bins out of the input even after background/noise.
        if mask is not None:
            flux = np.where(mask, flux, 0.0)

        # Legacy preprocessing is disabled in the ONeS dataset: both views and
        # inference use the exact same gain-invariant representation there.
        flux = self.preprocess(flux, wavelength=wavelength)

        return flux.astype(np.float32), mask





def encode_window_invariant(flux, mask=None, window_lengths=(41, 40, 40)):
    """Return (2, L) gain-invariant ONeS features and scalar log-intensity m.

    For window w, intensity I_w = sum(max(flux, 0)) on valid samples.
    Channel 0: max(flux, 0) / I_w on valid samples, else 0.
    Channel 1: log(I_w) - mean_w(log(I_w)), constant within window.
    m: mean_w(log(I_w)); save as a separate intensity label.

    For any common positive gain g, encoded features are unchanged and
    m(g * flux) = m(flux) + log(g), to floating-point precision.

    Each window must have a strictly positive total. Invalid zero-signal
    spectra should be filtered from the dataset instead of adding epsilons,
    which would spoil exact scale invariance.
    """
    flux = np.asarray(flux, dtype=np.float64)
    lengths = tuple(int(n) for n in window_lengths)
    if flux.ndim != 1 or sum(lengths) != flux.size or any(n < 1 for n in lengths):
        raise ValueError(f'Expected 1-D spectrum of length {sum(lengths)}; got {flux.shape}')

    if mask is None:
        valid = np.isfinite(flux)
    else:
        mask = np.asarray(mask)
        if mask.shape != flux.shape:
            raise ValueError('mask and flux must have matching shapes')
        valid = (mask >= 0.5) & np.isfinite(flux)

    positive = np.where(valid, np.maximum(flux, 0.0), 0.0)
    blocks = np.split(positive, np.cumsum(lengths)[:-1])
    totals = np.array([block.sum(dtype=np.float64) for block in blocks])
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError(
            f'Zero or invalid positive flux total in window(s) '
            f'{np.where(~np.isfinite(totals) | (totals <= 0))[0].tolist()}. '
            'Exclude these spectra from training/inference.'
        )

    log_totals = np.log(totals)
    m = log_totals.mean()
    rel = log_totals - m

    shape_channel = np.concatenate([block / totals[i] for i, block in enumerate(blocks)])
    relative_channel = np.concatenate([
        np.full(lengths[i], rel[i], dtype=np.float64) for i in range(len(lengths))
    ])
    
    two_channels = np.stack([shape_channel, relative_channel], axis=0)
    return two_channels.astype(np.float32), np.float32(m)

"""Dataset for gain-invariant representation of three ONeS spectral windows."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from simspice.utils.Augmentation_ONeS import Augmentation, encode_window_invariant

WINDOW_LENGTHS = (41, 40, 40)  # O III/Mg IX, S V/O IV, Ne VIII in NetCDF order


class SproutDataset(Dataset):
    def __init__(
        self,
        csv_files="L2_names.csv",
        file_dir=".",
        dataset_path="/d0/tvaresano/SimSPICE/spectra_Jan2023_ONeS.nc",
        augmentation_type="double",
        log_space=False,
        mu_doppler=0.0,
        sigma_shift=0.1,
        sigma_gain=0.1,              # deprecated: ignored
        shift_range=(-0.4, 0.4),
        gain_range=(0.8, 1.2),       # deprecated: ignored
        type_distrib_gain="uniform", # deprecated: ignored
        type_distrib_shift="Gaussian",
        normalize_intensity=False,
        normalization_method="area", # deprecated for ONeS
        normalization_eps=1e-8,      # deprecated for ONeS
        add_noise=True,
        background_range=(-0.02, 0.02),
        return_intensity_label=False,
        add_background=True,
        max_augmentation_attempts=12,
    ):
        if normalize_intensity or log_space:
            raise ValueError(
                'ONeS uses per-window two-channel normalization. '
                'Set normalize_intensity=False and log_space=False.'
            )

        self.file_names = pd.read_csv(os.path.join(file_dir, csv_files))
        self.file_dir = file_dir
        self.all_spectra = xr.open_dataset(dataset_path)
        self.augmentation_type = augmentation_type
        self.return_intensity_label = return_intensity_label
        self.window_lengths = WINDOW_LENGTHS
        self.max_augmentation_attempts = int(max_augmentation_attempts)

        if self.max_augmentation_attempts < 1:
            raise ValueError("max_augmentation_attempts must be >= 1")

        if self.all_spectra.sizes["wvl"] != sum(self.window_lengths):
            raise ValueError('Expected 121 ONeS wavelength bins (41 + 40 + 40).')

        self.augmenter = Augmentation(
            mu_doppler=mu_doppler,
            sigma_shift=sigma_shift,
            sigma_gain=sigma_gain,
            shift_range=shift_range,
            gain_range=gain_range,
            type_distrib_gain=type_distrib_gain,
            type_distrib_shift=type_distrib_shift,
            normalize_intensity=False,
            log_space=False,
            add_noise=add_noise,
            add_background=add_background,
            background_range=background_range,
            window_lengths=self.window_lengths,
            apply_gain=False,
        )

    def __len__(self):
        return self.all_spectra.sizes["index"]

    def _encode(self, flux, mask=None):
        features, m = encode_window_invariant(flux, mask=mask, window_lengths=self.window_lengths)
        return torch.from_numpy(features), torch.tensor(m, dtype=torch.float32)

    def _encode_augmented(self, row, index):
        """
        Retry stochastic augmentations that remove all
        positive flux in a spectral window.

        Avoid adding epsilon to zero totals because that
        would break strict gain invariance.
        """

        last_error = None
        for _ in range(self.max_augmentation_attempts):
            augmented, aug_mask = (self.augmenter.run_all_augmentations(row))
            try:
                view, _ = self._encode(augmented, mask=aug_mask)
                return view

            except ValueError as exc:

                if "Zero or invalid positive flux total in window(s)" not in str(exc):
                    raise

                last_error = exc

        raise ValueError(f"Spectrum index {index}: all "
            f"{self.max_augmentation_attempts} augmentation "
            "attempts produced a zero/invalid intensity "
            "window. Try add_background=False, reduce "
            "noise or shifts, or filter low-signal spectra. "
            f"Last error: {last_error}") from last_error


    def get_intensity_label(self, index):
        """Mean log-window-total m of the original, unaugmented spectrum."""
        row = self.all_spectra.isel(index=index)
        mask = row["mask"].values if "mask" in row else None
        _, m = self._encode(row["flux"].values, mask=mask)
        return m

    def __getitem__(self, index):
        row = self.all_spectra.isel(index=index)
        mask = row["mask"].values if "mask" in row else None

        # This very same transform is used for deterministic inference and
        # every training view, with no full-spectrum normalization or log1p.
        original_tensor, m = self._encode(row["flux"].values, mask=mask)

        if self.augmentation_type is None:
            return (original_tensor, m) if self.return_intensity_label else original_tensor

        mode = self.augmentation_type.lower()

        if mode == "double":
            view_1 = self._encode_augmented(row, index)
            view_2 = self._encode_augmented(row, index)

        elif mode == "single":
            view_1 = original_tensor
            view_2 = self._encode_augmented(row, index)

        else:
            raise ValueError("augmentation_type must be None, 'single', or 'double'.")

        return (view_1, view_2, m) if self.return_intensity_label else (view_1, view_2)

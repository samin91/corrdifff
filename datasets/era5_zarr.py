# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import xarray as xr
import zarr

from datasets.base import ChannelMetadata, DownscalingDataset


class Era5ZarrDataset(DownscalingDataset):
    """Reader for the climbench ERA5 regular-grid zarr datasets.

    Expects `data_path` to be a directory containing `input_<split>.zarr`
    (coarse conditioning fields) and `target_<split>.zarr` (high-resolution
    fields), both on a regular lat/lon grid and sharing the same `time`
    coordinate. `input_variables`/`output_variables` select which data
    variables (channels) to read from each store; the same variable list can
    be used for both when training a multi-channel downscaling model (e.g.
    experiment 3), or a single shared variable when training a
    single-channel model (experiments 1 and 2).
    """

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables: List[str],
        output_variables: List[str],
        invariant_variables: Union[List[str], None] = None,
        split: str = "train",
    ):
        if invariant_variables:
            raise NotImplementedError(
                "Era5ZarrDataset has no invariant fields available."
            )

        data_path = Path(data_path)
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)

        input_coords = xr.open_zarr(data_path / f"input_{split}.zarr")
        target_coords = xr.open_zarr(data_path / f"target_{split}.zarr")
        self._time = input_coords["time"].values
        self._lat = target_coords["lat"].values
        self._lon = target_coords["lon"].values
        self._img_shape = (target_coords.sizes["lat"], target_coords.sizes["lon"])

        self._input_group = zarr.open(str(data_path / f"input_{split}.zarr"), mode="r")
        self._target_group = zarr.open(str(data_path / f"target_{split}.zarr"), mode="r")

        with open(stats_path) as f:
            stats = json.load(f)
        self.input_mean, self.input_std = _load_stats(
            stats, "input", self.input_variables
        )
        self.output_mean, self.output_std = _load_stats(
            stats, "output", self.output_variables
        )

    def __len__(self):
        return self._time.shape[0]

    def __getitem__(self, idx):
        x = np.stack(
            [self._input_group[v][idx] for v in self.input_variables], axis=0
        ).astype(np.float32)
        y = np.stack(
            [self._target_group[v][idx] for v in self.output_variables], axis=0
        ).astype(np.float32)

        x = _bicubic_upsample(x, self._img_shape)

        x = self.normalize_input(x)
        y = self.normalize_output(y)
        return (y, x)

    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        return self._lon

    def latitude(self) -> np.ndarray:
        """Get latitude values from the dataset."""
        return self._lat

    def input_channels(self) -> List[ChannelMetadata]:
        """Metadata for the input channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for the output channels. A list of ChannelMetadata, one for each channel"""
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        """Get time values from the dataset."""
        return list(self._time)

    def image_shape(self) -> Tuple[int, int]:
        """Get the (height, width) of the data (same for input and output)."""
        return self._img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from physical units to normalized data."""
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        """Convert input from normalized data to physical units."""
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from physical units to normalized data."""
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        """Convert output from normalized data to physical units."""
        return x * self.output_std + self.output_mean


def _bicubic_upsample(x: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Upsample a (C, H, W) array to (C, *shape) with bicubic interpolation."""
    x = x.transpose(1, 2, 0)  # H, W, C
    x = cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    if x.ndim == 2:  # cv2 drops the channel dim when C == 1
        x = x[:, :, None]
    return x.transpose(2, 0, 1)  # C, H, W


def _load_stats(
    stats: dict, group: str, variables: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.array([stats[group][v]["mean"] for v in variables], dtype=np.float32)
    std = np.array([stats[group][v]["std"] for v in variables], dtype=np.float32)
    return mean[:, None, None], std[:, None, None]

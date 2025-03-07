import os
import datetime
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset

file_cache = {}

class LazyH5Dataset(Dataset):

    def __init__(self, file_path, variable_names=("VELOX", "VELOY", "VELOZ"), transform=None):
        """
        Args:
            file_path (str or list[str]): Path(s) to the h5 file(s).
            variable_names (tuple[str]): Names of simulation variables stored in FIELDS.
            transform (callable, optional): Transformation to apply on each sample.
        """
        # Accept single file or list.
        if isinstance(file_path, (str, bytes, os.PathLike)):
            self.file_paths = [file_path]
        else:
            self.file_paths = file_path

        self.variable_names = variable_names
        self.transform = transform

        # Build index mapping across file(s) using number of time steps (columns)
        self.file_sample_counts = []
        self.cumulative_counts = []
        cum = 0
        for fp in self.file_paths:
            with h5py.File(fp, 'r', swmr=True, libver="latest") as f:
                ds = f["DATASET/FIELDS"][self.variable_names[0]]["value"]
                n_time_steps = ds.shape[1]  # Number of time steps.
            self.file_sample_counts.append(n_time_steps)
            cum += n_time_steps
            self.cumulative_counts.append(cum)

        # Extract minimal metadata (mesh shape and time) from the first file.
        self.mesh_shape, self.time = self._extract_metadata(self.file_paths[0])
        self.num_channels = len(variable_names)

    def _extract_metadata(self, fp):

        with h5py.File(fp, 'r', swmr=True, libver="latest") as f:
            ds_group = f["DATASET"]
            xyz = np.array(ds_group["xyz"])  # Shape: (N, 3)
            nx = len(np.unique(xyz[:, 0]))
            ny = len(np.unique(xyz[:, 1]))
            nz = len(np.unique(xyz[:, 2]))
            mesh_shape = (nx, ny, nz)
            # Extract time vector from VARIABLES group.
            if "VARIABLES" in ds_group and "time" in ds_group["VARIABLES"]:
                time = np.array(ds_group["VARIABLES"]["time"])
            else:
                time = None
        return mesh_shape, time

    def __len__(self):
        return self.cumulative_counts[-1]

    def _get_file_and_local_index(self, global_idx):
        for file_idx, cum_count in enumerate(self.cumulative_counts):
            if global_idx < cum_count:
                local_idx = global_idx if file_idx == 0 else global_idx - self.cumulative_counts[file_idx - 1]
                return file_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")
    

    def __getitem__(self, idx):
        global file_cache
        file_idx, local_idx = self._get_file_and_local_index(idx)
        fp = self.file_paths[file_idx]
        if fp not in file_cache:
            file_cache[fp] = h5py.File(fp, 'r', swmr=True, libver="latest")
            file_cache[fp].id.get_access_plist().set_chunk_cache(1024**2 * 512, 1000, 0.9)
            print(f"Opened {fp} with SWMR mode.")

        f = file_cache[fp]
        sample = {var: f["DATASET/FIELDS"][var]["value"][:, local_idx] for var in self.variable_names}
        data_np = np.stack([sample[var] for var in self.variable_names], axis=0)
        data = torch.from_numpy(data_np).float().reshape(self.num_channels, *self.mesh_shape)

        if self.transform:
            data = self.transform(data)
        return data
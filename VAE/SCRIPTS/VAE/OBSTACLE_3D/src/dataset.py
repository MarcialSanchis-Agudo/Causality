import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import deque


class MultiFileLazyH5Dataset(Dataset):
    def __init__(self, filepaths, cache_size=20, batch_size=8):
        self.filepaths = filepaths
        self.cumulative_snapshots = []
        self.total_snapshots = 0
        self.fields_names = ["VELOX", "VELOY", "VELOZ"]
        self.file_handles = {}
        self.cache_size = cache_size  
        self.batch_size = batch_size  
        self.cache = deque(maxlen=cache_size)  

        for fp in filepaths:
            with h5py.File(fp, 'r', swmr=True, libver="latest") as f:
                ds_group = f["DATASET/FIELDS"]
                field_shape = ds_group[self.fields_names[0]]["value"].shape
                self.num_channels = len(self.fields_names)

                snapshot_count = field_shape[1]
                self.total_snapshots += snapshot_count
                self.cumulative_snapshots.append(self.total_snapshots)

                # 🔥 Definir `mesh_shape` en el primer archivo
                if not hasattr(self, "mesh_shape"):
                    xyz = np.array(f["DATASET/xyz"][:])
                    nx = len(np.unique(xyz[:, 0]))
                    ny = len(np.unique(xyz[:, 1]))
                    nz = len(np.unique(xyz[:, 2]))
                    self.mesh_shape = (nx, ny, nz)

        print(f"Total snapshots: {self.total_snapshots}")
        print(f"Mesh shape: {self.mesh_shape}")

    def __len__(self):
        return self.total_snapshots

    def _get_file_handle(self, filepath):
        """Evita abrir múltiples veces el mismo archivo HDF5."""
        if filepath not in self.file_handles:
            print(f"📂 Abriendo archivo HDF5: {filepath}")
            self.file_handles[filepath] = h5py.File(filepath, 'r', swmr=True, libver="latest",
                                                    rdcc_nbytes=1024 * 1024 * 1024,  
                                                    rdcc_nslots=2_000_000)  
        return self.file_handles[filepath]

    def __getitem__(self, index):
        """Carga un snapshot del archivo HDF5, optimizando el acceso a disco."""
        file_idx = next(i for i, cum_snap in enumerate(self.cumulative_snapshots) if index < cum_snap)
        local_index = index if file_idx == 0 else index - self.cumulative_snapshots[file_idx - 1]
        filepath = self.filepaths[file_idx]

        for cached_file, cached_idx, cached_data in self.cache:
            if cached_file == filepath and cached_idx <= local_index < cached_idx + self.batch_size:
                offset = local_index - cached_idx
                return cached_data[:, offset]  

        start_idx = local_index - (local_index % self.batch_size)  
        end_idx = min(start_idx + self.batch_size, self.total_snapshots)

        f = self._get_file_handle(filepath)
        ds_group = f["DATASET/FIELDS"]
        sample_data = np.zeros((len(self.fields_names), end_idx - start_idx, *self.mesh_shape), dtype=np.float32)

        for i, field_name in enumerate(self.fields_names):
            ds = ds_group[field_name]["value"]
            sample_data[i] = ds[:, start_idx:end_idx].reshape((self.batch_size, *self.mesh_shape))  

        self.cache.append((filepath, start_idx, torch.tensor(sample_data, dtype=torch.float32)))

        offset = local_index - start_idx
        return self.cache[-1][2][:, offset]
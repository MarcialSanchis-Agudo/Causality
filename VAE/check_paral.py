import numpy as np
import h5py
import os
from scipy.interpolate import griddata
import torch
import pyAlya
import pyLOM
import mat73
from mpi4py import MPI

# Set environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function for interpolation on both CPU or GPU
def interpolate_chunk(x_chunk, y_chunk, z_chunk, u_chunk, xyz, method='linear', device='cpu'):
    try:
        if 'cuda' in device and torch.cuda.is_available():
            torch.cuda.set_device(device)
        
        if 'cuda' in device:
            x_chunk = torch.tensor(x_chunk, dtype=torch.float32).to(device)
            y_chunk = torch.tensor(y_chunk, dtype=torch.float32).to(device)
            z_chunk = torch.tensor(z_chunk, dtype=torch.float32).to(device)
            u_chunk = torch.tensor(u_chunk, dtype=torch.float32).to(device)
            xyz = torch.tensor(xyz, dtype=torch.float32).to(device)

            result = griddata(
                (x_chunk.cpu().numpy(), y_chunk.cpu().numpy(), z_chunk.cpu().numpy()), 
                u_chunk.cpu().numpy(), xyz.cpu().numpy(), method=method
            )
        else:
            result = griddata(
                (x_chunk, y_chunk, z_chunk), 
                u_chunk, xyz, method=method
            )
        return result
    
    except RuntimeError as e:
        print(f"Error in process on {device}: {e}")
        if 'cuda' in device:
            print(f"Falling back to CPU for device {device}.")
            return interpolate_chunk(
                x_chunk, y_chunk, z_chunk, u_chunk, xyz, method='linear', device='cpu'
            )

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank} starting...")
    
    if rank == 0:
        print('------------ LOADING DATA ----------------')
        datapath = '/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/Set_18/U.mat'
        file = mat73.loadmat(datapath)
        print('Keys in the loaded .mat file:', file.keys())
        
        print('------------ DATA ANALYSIS ----------------')
        print('-------- U --------')
        x = np.array(file['x'], dtype=np.float32)
        y = np.array(file['y'], dtype=np.float32)
        z = np.array(file['z'], dtype=np.float32)
        time = np.array(file['t'], dtype=np.float32)
        u_x = np.array(file['U'], dtype=np.float32)

        print('Data loaded', x.shape, y.shape, z.shape, flush=True)

        # Reshape and unique calculations
        xs = np.unique(x)
        ys = np.unique(y)
        zs = np.unique(z)

        # Reshape u_x to 4D array
        u = u_x.reshape((len(np.unique(time)), len(xs), len(ys), len(zs)))
        u = u[:, 0:288:2, 0:96:2, 0:144:2]

        # Prepare data for all ranks
        x = x[0:288:2, 0:96:2, 0:144:2]
        y = y[0:288:2, 0:96:2, 0:144:2]
        z = z[0:288:2, 0:96:2, 0:144:2]

        u_x = u.reshape((len(time), -1)).T
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()

        # Broadcasting necessary data to all processes
        comm.Bcast(xs, root=0)
        comm.Bcast(ys, root=0)
        comm.Bcast(zs, root=0)
        comm.Bcast(u_x, root=0)

    else:
        # Initialize arrays for non-root ranks
        xs = np.empty(0, dtype=np.float32)
        ys = np.empty(0, dtype=np.float32)
        zs = np.empty(0, dtype=np.float32)
        u_x = np.empty(0, dtype=np.float32)

        # Receive unique values
        comm.Bcast(xs, root=0)
        comm.Bcast(ys, root=0)
        comm.Bcast(zs, root=0)
        comm.Bcast(u_x, root=0)

        print(f"Rank {rank}: Received unique values xs, ys, zs sizes: {len(xs)}, {len(ys)}, {len(zs)}")

    # Check if xs, ys, and zs have been correctly populated before using them
    if xs.size == 0 or ys.size == 0 or zs.size == 0:
        print(f"Rank {rank}: Warning! One or more unique arrays are empty.")
        return  # Early exit if arrays are empty

    # Now you can safely calculate min and max
    minx, maxx = np.min(xs), np.max(xs)
    miny, maxy = np.min(ys), np.max(ys)
    minz, maxz = np.min(zs), np.max(zs)

    print(f"Rank {rank}: minx = {minx}, maxx = {maxx}")
    print(f"Rank {rank}: miny = {miny}, maxy = {maxy}")
    print(f"Rank {rank}: minz = {minz}, maxz = {maxz}")

    nx, ny, nz = len(xs), len(ys), len(zs)

    p1 = np.array([minx, miny, minz], dtype=np.float32)
    p2 = np.array([maxx, miny, minz], dtype=np.float32)
    p4 = np.array([minx, maxy, minz], dtype=np.float32)
    p5 = np.array([minx, miny, maxz], dtype=np.float32)

    print('MESH', nx, ny, nz)
    mesh = pyAlya.MeshSOD2D.cube(p1, p2, p4, p5, nx, ny, nz)
    print('Mesh created', mesh.xyz.shape, flush=True)

    np.savez('coords.npz', xyz=mesh.xyz)

    print('x shape:', x.shape, 'y shape:', y.shape, 'z shape:', z.shape, flush=True)

    # Preparing for interpolation
    num_cores = min(size, 64)
    chunk_size = len(x) // num_cores

    x_chunks = np.array_split(x, num_cores)
    y_chunks = np.array_split(y, num_cores)
    z_chunks = np.array_split(z, num_cores)
    u_chunks = np.array_split(u_x, num_cores, axis=0)

    available_gpus = torch.cuda.device_count()
    
    if available_gpus > 0:
        devices = [f'cuda:{i}' for i in range(available_gpus)]
        print(f"Using GPUs: {devices}")
        
        # Warmup GPU initialization before multiprocessing
        for device in devices:
            torch.cuda.init() 
            torch.tensor([1.0], device=device)
    else:
        devices = ['cpu']
        print("Using CPU as no GPUs are available.")
    
    device_assignments = devices * (num_cores // len(devices)) + devices[:num_cores % len(devices)]

    print(f'Interpolating using {num_cores} cores with device distribution...')
    
    results = []
    for i in range(num_cores):
        result = interpolate_chunk(x_chunks[i], y_chunks[i], z_chunks[i], u_chunks[i], mesh.xyz, method='linear', device=device_assignments[i])
        results.append(result)

    u_xi = np.concatenate(results, axis=0)
    print('ux interpolated', u_xi.shape, flush=True)

    # Save the results on rank 0
    if rank == 0:
        m = pyLOM.Mesh.from_pyAlya(mesh)
        p = pyLOM.PartitionTable.from_pyAlya(mesh.partition_table, has_master=False)
        d = pyLOM.Dataset(ptable=p, mesh=m, time=time)
        d.add_variable('velox', True, 1, u_xi)
        d.save('obstacle_3D_full_para.h5', nopartition=True)

if __name__ == "__main__":
    main()

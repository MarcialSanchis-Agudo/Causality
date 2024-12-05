##############################333 PYLOM MESH ########################
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import interpn
import h5py
import sys
import os
from scipy.interpolate import griddata
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc
import torch
import pyAlya
import pyLOM
import mat73
import mpi4py
mpi4py.rc.recv_mprobe = False

task_running = threading.Event()


def gpu_interpolate(x_unique, y_unique, z_unique, u_chunk, xyz, method='linear', device_id=0):
    # Move data to GPU
    with cp.cuda.Device(device_id):

        x_unique = cp.asarray(x_unique).astype(cp.float32)
        y_unique = cp.asarray(y_unique).astype(cp.float32)
        z_unique = cp.asarray(z_unique).astype(cp.float32)
        u_chunk = cp.asarray(u_chunk).astype(cp.float32)
        xyz = cp.asarray(xyz).astype(cp.float32)
        points = (x_unique, y_unique, z_unique)

        u_chunk_interp = u_chunk.reshape(u_chunk.shape[0], len(x_unique), len(y_unique), len(z_unique))

        interpolated = cp.empty((u_chunk.shape[0], xyz.shape[0])).astype(cp.float32)
        for i in range(u_chunk.shape[0]):
            interpolated[i] = interpn((points), u_chunk_interp[i], xyz, method=method)

        return cp.asnumpy(interpolated.T)


def gpu_parallel_interpolation(x_unique, y_unique, z_unique, u_chunk, xyz, method='linear'):
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    chunks = np.array_split(u_chunk, n_gpus)

    results = [None] * n_gpus

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        interpolations = {executor.submit(gpu_interpolate,x_unique, y_unique, z_unique, chunk, xyz, method, i): i for i, chunk in enumerate(chunks)}
        
        for interpolation in as_completed(interpolations):
            i = interpolations[interpolation]
            results[i] = interpolation.result()
    
    return np.concatenate(results, axis=1)


def interpolate_cpu(x, y, z,u_chunk, xyz, method='linear'):

    # Interpolate value for each snapshot
    interpolated = griddata((x, y ,z), u_chunk.T, xyz, method=method)

    print("Interpolated shape:", interpolated.shape)
    print("Interpolated values:", interpolated)
    print("Interp. Max:", np.max(interpolated))
    print("Interp. Min:", np.min(interpolated))

    return interpolated

def gpu_dummy_task(device_id=0):
    with cp.cuda.Device(device_id):
        print(f"GPU {device_id} tasks started")
        while True:
            task_running.wait()
            a = cp.random.random((10000, 10000), dtype=cp.float32)
            b = cp.random.random((10000, 10000), dtype=cp.float32)
            c = cp.matmul(a, b)

            if not task_running.is_set():
                break
        
        print(f"GPU {device_id} tasks finished")


def release_gpu_memory():
    for i in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(i):
            cp.get_default_memory_pool().free_all_blocks()
            print(f"Memory on GPU {i} has been freed.")


def main():
    # Load and create the data 

    n_gpus = cp.cuda.runtime.getDeviceCount()
    threads = []
    for i in range(n_gpus):
        thread = threading.Thread(target=gpu_dummy_task, args=(i,))
        threads.append(thread)
        thread.start()
    
    for _ in range(n_gpus):
        task_running.set()


    print('------------ LOADING DATA ----------------')
    datapath_U = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_183_233_N5e/DATASET35e/U.mat'
    datapath_V = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_183_233_N5e/DATASET35e/V.mat'
    datapath_W = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_183_233_N5e/DATASET35e/W.mat'
        

    file_U = mat73.loadmat(datapath_U)
    file_V = mat73.loadmat(datapath_V)
    file_W = mat73.loadmat(datapath_W)
    print('Keys in the loaded .mat file:', file_U.keys())

    print('------------ DATA ANALYSIS ---------------')
    x = np.array(file_U['x'], dtype=np.float32)
    y = np.array(file_U['y'], dtype=np.float32)
    z = np.array(file_U['z'], dtype=np.float32)
    time = np.array(file_U['t'], dtype=np.float32)
    u_x = np.array(file_U['U'], dtype=np.float32)
    u_y = np.array(file_V['V'], dtype=np.float32)
    u_z = np.array(file_W['W'], dtype=np.float32)

    print('Data loaded:', 'x:', x.shape, 'y:', y.shape, 'z:', z.shape, 'time:', time.shape, 'u_x:', u_x.shape, 'u_y:', u_y.shape, 'u_z:', u_z.shape, flush=True)

    # Process and reshape data
    U = u_x.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
    U = U[:,0:288,0:96,0:144]
    V = u_y.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
    V = V[:,0:288,0:96,0:144]
    W = u_z.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
    W = W[:,0:288,0:96,0:144]
    x = x.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
    y = y.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
    z = z.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
    x = x[0:288,0:96,0:144]
    y = y[0:288,0:96,0:144]
    z = z[0:288,0:96,0:144]
    u_x = U.reshape((len(time),-1))
    u_y = V.reshape((len(time),-1))
    u_z = W.reshape((len(time),-1))
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    del U, V, W
    gc.collect()

    # time = time[:20]
    # u_x = u_x[:20, :]

    print("U shape:", u_x.shape)
    print("V shape:", u_y.shape)
    print("W shape:", u_z.shape)

    # Get X, Y and Z unique values
    xs = np.unique(x)
    ys = np.unique(y)
    zs = np.unique(z)
    
    # Calculate min and max
    min_x, max_x, n_x = np.min(xs), np.max(xs), len(xs)
    min_y, max_y, n_y = np.min(ys), np.max(ys), len(ys)
    min_z, max_z, n_z = np.min(zs), np.max(zs), len(zs)

    print(f"X: min_x = {min_x}, max_x = {max_x}")
    print(f"Y: min_y = {min_y}, max_y = {max_y}")
    print(f"Z: min_z = {min_z}, max_z = {max_z}")

    # Create cube mesh
    p1 = np.array([min_x, min_y, min_z], dtype=np.float32)
    p2 = np.array([max_x, min_y, min_z], dtype=np.float32)
    p4 = np.array([min_x, max_y, min_z], dtype=np.float32)
    p5 = np.array([min_x, min_y, max_z], dtype=np.float32)

    
    print(f"Creating MESH: {n_x} x {n_y} x {n_z}")
    mesh = pyAlya.MeshSOD2D.cube(p1, p2, p4, p5, n_x, n_y, n_z)
    print('Mesh created', mesh.xyz.shape, flush=True)

   # Adjust values to max and min value (avoid precision errors) 
    mesh.xyz[:, 0] = np.clip(mesh.xyz[:, 0], xs.min(), xs.max())
    mesh.xyz[:, 1] = np.clip(mesh.xyz[:, 1], ys.min(), ys.max())
    mesh.xyz[:, 2] = np.clip(mesh.xyz[:, 2], zs.min(), zs.max())

    # Prepare for interpolation
    xyz = mesh.xyz
    print('XYZ mesh:', xyz)

    task_running.clear()
    for thread in threads:
        thread.join()
    
    release_gpu_memory()

    del n_x, n_y, n_z, min_x, min_y, min_z, max_x, max_y, max_z, p1, p2, p4, p5
    gc.collect()

    # Interpolate in each rank using the scattered chunk
    if torch.cuda.is_available():
        print('Starting interpolation using GPU')
        u_xi_combined = gpu_parallel_interpolation(xs, ys, zs, u_x, xyz, method='linear')
        release_gpu_memory()
        u_yi_combined = gpu_parallel_interpolation(xs, ys, zs, u_y, xyz, method='linear')
        release_gpu_memory()
        u_zi_combined = gpu_parallel_interpolation(xs, ys, zs, u_z, xyz, method='linear')
        release_gpu_memory()
    else:
        print('Starting interpolation using CPU - No GPU backend found')
        u_xi_combined = interpolate_cpu(x, y, z, u_x, xyz)
        u_yi_combined = interpolate_cpu(x, y, z, u_y, xyz)
        u_zi_combined = interpolate_cpu(x, y, z, u_z, xyz)
        
    print("After interpolation, U field has data:", u_xi_combined)
    print("Final combined shape for U:", u_xi_combined.shape)
    print("After interpolation, V field has data:", u_yi_combined)
    print("Final combined shape for V:", u_yi_combined.shape)
    print("After interpolation, W field has data:", u_zi_combined)
    print("Final combined shape for W:", u_zi_combined.shape)   

    p = pyLOM.PartitionTable.from_pyQvarsi(mesh.partition_table, has_master=True)
    print(f"Partition table created: {p}")
    print(f"Partition table: points = {p._points}, elements = {p._elements}")
    print(f"Partition table has master: {p._master}")
    
    m = pyLOM.Mesh.from_pyQvarsi(mesh, ptable=p)
    print("Mesh processed", m)
        
    d = pyLOM.Dataset(xyz    = m.xyz,
                      ptable = p,
                      order  = m.pointOrder,
                      point  = True,
                      vars   = {'time':{'idim':0, 'value': time}},
                      VELOX  = {'ndim':1, 'value':u_xi_combined},
                      VELOY  = {'ndim':1, 'value':u_yi_combined},
                      VELOZ  = {'ndim':1, 'value':u_zi_combined})

    print(f"Dataset created", d)
    print("Ready to save data.")
    m.save('/mimer/NOBACKUP/groups/kthmech/carlos/Datasets_3D/Obstacle_VAE/obstacle_3D_35.h5', nopartition=True)
    print("Mesh saved")
    d.save('/mimer/NOBACKUP/groups/kthmech/carlos/Datasets_3D/Obstacle_VAE/obstacle_3D_35.h5', nopartition=True)
    print(" HDF5 file saved.")


if __name__ == "__main__":
    main()
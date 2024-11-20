##############################333 PYLOM MESH ########################
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import interpn
import h5py
import sys
import os
from scipy.interpolate import griddata
import torch
import torch.nn.functional as F
import pyAlya
import pyLOM
import mat73
from mpi4py import MPI


def gpu_interpolate(x_unique, y_unique, z_unique, u_chunk, xyz, method='linear'):
    # Move data to GPU

    # gridpoints = cp.asarray(gridpoints).astype(cp.float32)
    x_unique = cp.asarray(x_unique).astype(cp.float32)
    y_unique = cp.asarray(y_unique).astype(cp.float32)
    z_unique = cp.asarray(z_unique).astype(cp.float32)

    u_chunk = cp.asarray(u_chunk).astype(cp.float32)
    xyz = cp.asarray(xyz).astype(cp.float32)

    n_snaps = u_chunk.shape[0]
    # x_unique = cp.unique(x)
    # y_unique = cp.unique(y)
    # z_unique = cp.unique(z)
    points = (x_unique, y_unique ,z_unique)
    u_chunk_interp = u_chunk.reshape(n_snaps, len(x_unique), len(y_unique), len(z_unique))

    # print("XYZ_GPU", xyz)
    # print("POINTS_GPU", points)
    # print("X_GPU", x)
    # print("Y_GPU", y)
    # print("Z_GPU", z)

    interpolated = cp.empty((n_snaps, xyz.shape[0])).astype(cp.float32)
    for i in range(n_snaps):
        interpolated[i] = interpn((points), u_chunk_interp[i], xyz, method=method)

    
    # interpolated = griddata((x.get(), y.get() ,z.get()), u_chunk.T.get(), xyz.get(), method=method)
    
    return cp.asnumpy(interpolated.T)


def interpolate_cpu(x, y, z,u_chunk, xyz, method='linear'):

    n_snaps = u_chunk.shape[0]
    # u_chunk = u_chunk.reshape(n_snaps, len(xs), len(ys), len(zs))
    # print("U_x after reshape:", u_chunk.shape)

    # Interpolate value for each snapshot
    interpolated = griddata((x, y ,z), u_chunk.T, xyz, method=method)
    # interpolated = np.empty((n_snaps, xyz.shape[0])).astype(np.float32)
    # for i in range(n_snaps):
    #     interpolated[i] = griddata(grid_points, u_chunk[i], xyz, method=method)

    print("Interpolated shape:", interpolated.shape)
    print("Interpolated values:", interpolated)
    print("Interp. Max:", np.max(interpolated))
    print("Interp. Min:", np.min(interpolated))

    return interpolated


# def interpolate_chunk(x, y, z, u_chunk, xyz, method='linear'):
#     
#     result = interpolate_cpu(x, y, z, u_chunk, xyz, method=method)
#     
#     return result


def interpolate_chunk(x, y, z, x_unique, y_unique, z_unique, u_chunk, xyz, device, method='linear',):
    device = torch.device(device)
    try:
        if torch.cuda.is_available():
            # Use the CUDA enabled GPU if it is available to perform the interpolation
            result = gpu_interpolate(x_unique, y_unique, z_unique, u_chunk, xyz, method=method)
        else:
            # Fallback to CPU-based interpolation using scipy
            result = interpolate_cpu(x, y, z, u_chunk, xyz, method=method)
        
        return result
    
    except RuntimeError as e:
        print(f"Error in process on {device}: {e}")
        if device.type == 'cuda' :
            print(f"Falling back to CPU for device {device}.")
            return interpolate_cpu(x, y, z, u_chunk, xyz, method=method)


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}') if torch.cuda.is_available() else torch.device('cpu')

    print(f'Process {rank}/{nprocs} started on device {device}.')

    comm.Barrier()

    # Load anc create the data only for rank 0 as it is only necessary to do this process once; 
    if rank == 0:
        print('------------ LOADING DATA ----------------')
        datapath = '/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/Set_18/U.mat'
        file = mat73.loadmat(datapath)
        print('Keys in the loaded .mat file:', file.keys())

        print('------------ DATA ANALYSIS ---------------')
        x = np.array(file['x'], dtype=np.float32)
        y = np.array(file['y'], dtype=np.float32)
        z = np.array(file['z'], dtype=np.float32)
        time = np.array(file['t'], dtype=np.float32)
        u_x = np.array(file['U'], dtype=np.float32)

        print('Data loaded:', 'x:', x.shape, 'y:', y.shape, 'z:', z.shape, 'time:', time.shape, 'u_x:', u_x.shape, flush=True)

        # Unique values for grid
        
        u = u_x.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        u = u[:,0:288,0:96,0:144]
        x = x.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        y = y.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        z = z.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        x = x[0:288,0:96,0:144]
        y = y[0:288,0:96,0:144]
        z = z[0:288,0:96,0:144]
        u_x = u.reshape((len(time),-1))
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        
        xs = np.unique(x)
        ys = np.unique(y)
        zs = np.unique(z)

        time = time[:300]
        u_x = u_x[:300, :]
        # grid_points = np.array(np.meshgrid(xs, ys, zs, indexing='ij'))
        # grid_points = grid_points.reshape(3, -1).T
        # print("Grid", grid_points)
        # print("Grid shape", grid_points.shape)

        print("U_x shape:", u_x.shape)
        # Calculate min and max
        
        sendbuf = u_x
        u_x_size_1 = u_x.shape[1]

        max_elements_per_block = 2**31 // np.dtype(np.float32).itemsize
        num_blocks = (sendbuf.size // max_elements_per_block) + 1
        sendbuf_blocks = np.array_split(sendbuf, num_blocks)

        print(f"Num of blocks for interpolating data: {num_blocks}")
        print(f'Sendbuf: has data:{sendbuf}')
        print(f'Sendbuf shape: {sendbuf.shape}')

    else:
        xs, ys, zs = None, None, None
        x, y, z = None, None, None
        u_x_size_1 = None
        sendbuf_blocks = None
        num_blocks = None
        time = None

    # Broadcast data to all ranks
    xs = comm.bcast(xs, root=0)
    ys = comm.bcast(ys, root=0)
    zs = comm.bcast(zs, root=0)
    x = comm.bcast(x, root=0)
    y = comm.bcast(y, root=0)
    z = comm.bcast(z, root=0)
    u_x_size_1 = comm.bcast(u_x_size_1, root=0)
    num_blocks = comm.bcast(num_blocks, root=0)
    time = comm.bcast(time, root=0)


    min_x, max_x, n_x = np.min(xs), np.max(xs), len(xs)
    min_y, max_y, n_y = np.min(ys), np.max(ys), len(ys)
    min_z, max_z, n_z = np.min(zs), np.max(zs), len(zs)

    print(f"Rank {rank}: min_x = {min_x}, max_x = {max_x}")
    print(f"Rank {rank}: min_y = {min_y}, max_y = {max_y}")
    print(f"Rank {rank}: min_z = {min_z}, max_z = {max_z}")

    p1 = np.array([min_x, min_y, min_z], dtype=np.float32)
    p2 = np.array([max_x, min_y, min_z], dtype=np.float32)
    p4 = np.array([min_x, max_y, min_z], dtype=np.float32)
    p5 = np.array([min_x, min_y, max_z], dtype=np.float32)

    # Create cube mesh
    if rank == 0:
        print(f"Creating MESH: {n_x} x {n_y} x {n_z}")
    mesh = pyAlya.MeshSOD2D.cube(p1, p2, p4, p5, n_x, n_y, n_z)
    if rank == 0:
        print('Mesh created', mesh.xyz.shape, flush=True)

    mesh.xyz[:, 0] = np.clip(mesh.xyz[:, 0], xs.min(), xs.max())
    mesh.xyz[:, 1] = np.clip(mesh.xyz[:, 1], ys.min(), ys.max())
    mesh.xyz[:, 2] = np.clip(mesh.xyz[:, 2], zs.min(), zs.max())

    np.savez('coords.npz', xyz=mesh.xyz)

    # Prepare for interpolation
    xyz = mesh.xyz
    if rank == 0:
        print('XYZ mesh:', xyz)

    comm.Barrier()

    gathered_data = []
    for i in range(num_blocks):
        if rank == 0:
            print(f'------------ Processing block {i} ----------------')
            sendbuf_block = sendbuf_blocks[i]
        else:
            sendbuf_block = None
            
        if rank == 0:
            ave, res = divmod(sendbuf_block.shape[0], nprocs)
            count = [ave + 1 if p < res else ave for p in range(nprocs)]
            count = np.array(count, dtype=np.int32)
            # Displacement
            displ = [sum(count[:p]) for p in range(nprocs)]
            displ = np.array(displ, dtype=np.int32)
            count_elements = np.array(count * u_x_size_1)
            displ_elements = np.array(displ * u_x_size_1)
        else:
            count = displ = count_elements = displ_elements = None


        # comm.Bcast(count, root=0)
        count = comm.bcast(count, root=0)
        displ = comm.bcast(displ, root=0)
        count_elements = comm.bcast(count_elements, root=0)
        displ_elements = comm.bcast(displ_elements, root=0)

        snaps_per_process = count[rank]
        recvbuf = np.empty((snaps_per_process, u_x_size_1), dtype=np.float32)

        # Scatter the data
        comm.Scatterv([sendbuf_block, count_elements, displ_elements, MPI.FLOAT], recvbuf, root=0)

        print(f'Block {i}: After Scatter, process{rank} has data:', recvbuf)

        recvbuf_2D = recvbuf.reshape((snaps_per_process, u_x_size_1))

        cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

        # Interpolate in each rank using the scattered chunk
        u_xi_local = interpolate_chunk(x, y, z, xs, ys, zs, recvbuf_2D, xyz, device=device)
    
        # Gather interpolated data from all ranks
        sendbuf2 = u_xi_local
        if rank == 0:
            recvbuf2 = np.empty((sum(count), u_x_size_1), dtype = np.float32)
        else:
            recvbuf2 = None

        print(f"Block {i}: Rank {rank} - Sendbuf size:", sendbuf2.shape)
        if rank == 0:
            print(f"Block {i}: Counts per rank:", count_elements)
            print(f"Block {i}: Displacements for gather:", displ_elements)

        print(f"Block {i}: Rank {rank} sending {sendbuf2[rank]} rows with displacements {displ_elements[rank]}.")
    
        comm.Gatherv(sendbuf2, [recvbuf2, count_elements, displ_elements, MPI.FLOAT], root=0)

        if rank == 0:
            print(f"Block {i}: After Gatherv, process 0 has data:", recvbuf2)
        
        recvbuf2 = comm.bcast(recvbuf2)
        gathered_data.append(recvbuf2)

    # Save results on rank 0
    comm.Barrier()

    # if rank == 0:
    u_xi_combined = np.concatenate(gathered_data, axis=0)
    print("After gathering all data, interpolated vel has data:", u_xi_combined)
    print("Final combined shape for the velocity:", u_xi_combined.shape)
    u_xi_combined = u_xi_combined.T
    print("Final combined shape for the velocity after transposing:", u_xi_combined.shape)
    print("Final velocity after transposing:", u_xi_combined)       
    # u_xi_combined = recvbuf2.reshape((-1, u_x.shape[1]))
    # print('ux interpolated', u_xi_combined.shape, flush=True)
    # print(f"Rank {rank} started processing mesh")
    # u_xi_blocks = np.array_split(u_xi_combined, num_blocks, axis=1)
    # print("U Blocks splitted:", u_xi_blocks)
    
    # u_xi_combined = comm.bcast(u_xi_combined, root=0)
    comm.Barrier()
    # if rank == 0:
    m = pyLOM.Mesh.from_pyQvarsi(mesh)
    if rank == 0:
        print("Mesh processed", m)
        
    # print(f"Rank {rank} creating partition data")
    p = pyLOM.PartitionTable.from_pyQvarsi(mesh.partition_table, has_master=True)
    if rank == 0:
        print(f"Partition table created: {p}")
    print(f"Rank {rank} partition table: points = {p._points}, elements = {p._elements}")
    # print(f"Rank {rank} creating dataset object")
    d = pyLOM.Dataset(xyz=m.xyz, ptable=p, order=m.pointOrder, point=True,
                      vars={'time':{'idim':0, 'value': time}},
                      VELOX = {'ndim':1, 'value':u_xi_combined})
    if rank == 0:
        print(f"Dataset created", d)
        # d.add_variable('time', 1, time)
        # d.add_variable('velox', 1, u_xi_combined)
        # print("Velocity variable created", d)
    if rank == 0:
        print("Ready to save data.")
    d.save('obstacle_3D_full_para_PRUEBA_1.h5', nopartition=True)
    if rank == 0:        
        print(" HDF5 file saved.")

    comm.Barrier()

    # f = h5py.File('test_save.h5', 'w', driver='mpio', comm=comm)
    # dset = f.create_dataset('example_dataset', data=u_xi_combined)
    # dset[rank] = dset

    # f.close()

    # print("Archivo test_save.h5 guardado exitosamente.")
    
    # print(f"Rank {rank} started writing data")
    
    comm.Barrier()
    comm.Disconnect()


if __name__ == "__main__":
    main()
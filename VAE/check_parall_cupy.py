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
import gc


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
        datapath_U = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_183_233_N5e/DATASET33e/U.mat'
        datapath_V = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_183_233_N5e/DATASET33e/V.mat'
        datapath_W = '/mimer/NOBACKUP/groups/kthmech/abhvis/Projects/UFLOW_data/obstacleNOTRIP_T_183_233_N5e/DATASET33e/W.mat'
        
        with h5py.File(datapath_U, 'r') as file_U, h5py.File(datapath_V, 'r') as file_V, h5py.File(datapath_W, 'r') as file_W:
        
            print('Keys in the loaded U.mat file:', file_U.keys())
            print('Keys in the loaded V.mat file:', file_V.keys())
            print('Keys in the loaded W.mat file:', file_W.keys())

            print('------------ DATA ANALYSIS ---------------')
            # x = np.array(file_U['x'], dtype=np.float32)
            # y = np.array(file_U['y'], dtype=np.float32)
            # z = np.array(file_U['z'], dtype=np.float32)
            # time = np.array(file_U['t'], dtype=np.float32)
            # U_x = np.array((file_U['U'][()])[:], dtype=np.float32)
            # U_y = np.array(file_V['V'][()], dtype=np.float32)
            # U_z = np.array(file_W['W'][()], dtype=np.float32)
            
            x = np.array(file_U['x'], dtype=np.float32)
            y = np.array(file_U['y'], dtype=np.float32)
            z = np.array(file_U['z'], dtype=np.float32)
            time = np.array(file_U['t'], dtype=np.float32)

            refs_U = file_U['U'][:, 0]
            refs_V = file_V['V'][:, 0]
            refs_W = file_W['W'][:, 0]

            # num_refs_U = len(refs_U)
            # num_refs_V = len(refs_V)
            # num_refs_W = len(refs_W)

            # data_shape_U = file_U[refs_U[0]][()].shape
            # data_shape_V = file_V[refs_V[0]][()].shape
            # data_shape_W = file_W[refs_W[0]][()].shape

            # total_length_U = data_shape_U[1]
            # total_length_V = data_shape_V[1]
            # total_length_W = data_shape_W[1]

            # U_x = np.empty((num_refs_U, total_length_U), dtype=np.float32)
            # U_y = np.empty((num_refs_V, total_length_V), dtype=np.float32)
            # U_z = np.empty((num_refs_W, total_length_W), dtype=np.float32)

            # for i, ref in enumerate(refs_U):
            #     U_x[i, :] = file_U[ref][0, :]
            # for i, ref in enumerate(refs_V):
            #     U_y[i, :] = file_V[ref][0, :]
            # for i, ref in enumerate(refs_W):
            #     U_z[i, :] = file_W[ref][0, :]         


            U_x = np.array([file_U[ref][0, :] for ref in refs_U], dtype=np.float32)
            U_y = np.array([file_V[ref][0, :] for ref in refs_V], dtype=np.float32)
            U_z = np.array([file_W[ref][0, :] for ref in refs_W], dtype=np.float32)       

            print('Data loaded:', 'x:', x.shape, 'y:', y.shape, 'z:', z.shape, 'time:', time.shape, 'u_x:', U_x.shape, 'u_y:', U_y.shape, 'u_z:', U_z.shape, flush=True)
            print('U_x:', U_x)
            print('U_y:', U_y)
            print('U_z:', U_z)
        
        u_x = U_x.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        u_y = U_y.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        u_z = U_z.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        u_x = u_x[:,0:288,0:96,0:144]
        u_y = u_y[:,0:288,0:96,0:144]
        u_z = u_z[:,0:288,0:96,0:144]
        x = x.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        y = y.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        z = z.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
        x = x[0:288,0:96,0:144]
        y = y[0:288,0:96,0:144]
        z = z[0:288,0:96,0:144]
        u_x = u_x.reshape((len(time),-1))
        u_y = u_y.reshape((len(time),-1))
        u_z = u_z.reshape((len(time),-1))
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        
        xs = np.unique(x)
        ys = np.unique(y)
        zs = np.unique(z)

        print("U_x shape:", u_x.shape)
        # Calculate min and max
        
        sendbuf_x = u_x
        sendbuf_y = u_y
        sendbuf_z = u_z
        u_x_size_1 = u_x.shape[1]

        max_elements_per_block = 2**31 // np.dtype(np.float32).itemsize
        num_blocks = (sendbuf_x.size // max_elements_per_block) + 1
        sendbuf_blocks_x = np.array_split(sendbuf_x, num_blocks)
        sendbuf_blocks_y = np.array_split(sendbuf_y, num_blocks)
        sendbuf_blocks_z = np.array_split(sendbuf_z, num_blocks)

        print(f"Num of blocks for interpolating data: {num_blocks}")
        # print(f'Sendbuf: has data:{sendbuf_x}')
        # print(f'Sendbuf shape: {sendbuf_x.shape}')

        del U_x, u_x, file_U, U_y, u_y, file_V, U_z, u_z, file_W
        gc.collect()

    else:
        xs, ys, zs = None, None, None
        x, y, z = None, None, None
        u_x_size_1 = None
        sendbuf_blocks_x = None
        sendbuf_blocks_y = None
        sendbuf_blocks_z = None
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

    del max_x, min_x, n_x, max_y, min_y, n_y, max_z, min_z, n_z, p1, p2, p4, p5
    gc.collect()

    gathered_data_x = []
    gathered_data_y = []
    gathered_data_z = []
    for i in range(num_blocks):
        if rank == 0:
            print(f'------------ Processing block {i} ----------------')
            sendbuf_block_x = sendbuf_blocks_x[i]
            sendbuf_block_y = sendbuf_blocks_y[i]
            sendbuf_block_z = sendbuf_blocks_z[i]
            
        else:
            sendbuf_block_x = None
            sendbuf_block_y = None
            sendbuf_block_z = None
            
        if rank == 0:
            # Prepare dimensions for scattering (and gathering) velocity fields 
            # X
            ave_x, res_x = divmod(sendbuf_block_x.shape[0], nprocs)
            count_x = [ave_x + 1 if p < res_x else ave_x for p in range(nprocs)]
            count_x = np.array(count_x, dtype=np.int32)
            # Y
            ave_y, res_y = divmod(sendbuf_block_y.shape[0], nprocs)
            count_y = [ave_y + 1 if p < res_y else ave_y for p in range(nprocs)]
            count_y = np.array(count_y, dtype=np.int32)
            # Z
            ave_z, res_z = divmod(sendbuf_block_z.shape[0], nprocs)
            count_z = [ave_z + 1 if p < res_z else ave_z for p in range(nprocs)]
            count_z = np.array(count_z, dtype=np.int32)
            # Displacements
            # X
            displ_x = [sum(count_x[:p]) for p in range(nprocs)]
            displ_x = np.array(displ_x, dtype=np.int32)
            count_elements_x = np.array(count_x * u_x_size_1)
            displ_elements_x = np.array(displ_x * u_x_size_1)
            # Y
            displ_y = [sum(count_y[:p]) for p in range(nprocs)]
            displ_y = np.array(displ_y, dtype=np.int32)
            count_elements_y = np.array(count_y * u_x_size_1)
            displ_elements_y = np.array(displ_y * u_x_size_1)
            # Z
            displ_z = [sum(count_z[:p]) for p in range(nprocs)]
            displ_z = np.array(displ_z, dtype=np.int32)
            count_elements_z = np.array(count_z * u_x_size_1)
            displ_elements_z = np.array(displ_z * u_x_size_1)            
        else:
            count_x = displ_x = count_elements_x = displ_elements_x = None
            count_y = displ_y = count_elements_y = displ_elements_y = None
            count_z = displ_z = count_elements_z = displ_elements_z = None

        # Broadcast counts, displacements, number of count elements and number of displacement elements for each dimension
        # X
        count_x = comm.bcast(count_x, root=0)
        displ_x = comm.bcast(displ_x, root=0)
        count_elements_x = comm.bcast(count_elements_x, root=0)
        displ_elements_x = comm.bcast(displ_elements_x, root=0)
        # Y
        count_y = comm.bcast(count_y, root=0)
        displ_y = comm.bcast(displ_y, root=0)
        count_elements_y = comm.bcast(count_elements_y, root=0)
        displ_elements_y = comm.bcast(displ_elements_y, root=0)
        # Z
        count_z = comm.bcast(count_z, root=0)
        displ_z = comm.bcast(displ_z, root=0)
        count_elements_z = comm.bcast(count_elements_z, root=0)
        displ_elements_z = comm.bcast(displ_elements_z, root=0)
        # Split velocity fields according to individual snapshots
        # X
        snaps_per_process_x = count_x[rank]
        recvbuf_x = np.empty((snaps_per_process_x, u_x_size_1), dtype=np.float32)
        # Y
        snaps_per_process_y = count_y[rank]
        recvbuf_y = np.empty((snaps_per_process_y, u_x_size_1), dtype=np.float32)
        # Z
        snaps_per_process_z = count_z[rank]
        recvbuf_z = np.empty((snaps_per_process_z, u_x_size_1), dtype=np.float32)

        # Scatter the data
        # X
        comm.Scatterv([sendbuf_block_x, count_elements_x, displ_elements_x, MPI.FLOAT], recvbuf_x, root=0)
        # Y
        comm.Scatterv([sendbuf_block_y, count_elements_y, displ_elements_y, MPI.FLOAT], recvbuf_y, root=0)
        # Z
        comm.Scatterv([sendbuf_block_z, count_elements_z, displ_elements_z, MPI.FLOAT], recvbuf_z, root=0)

        print(f'Block {i}: After Scatter, process{rank} has data for u_x:', recvbuf_x)
        print(f'Block {i}: After Scatter, process{rank} has data for u_y:', recvbuf_y)
        print(f'Block {i}: After Scatter, process{rank} has data for u_z:', recvbuf_z)

        recvbuf_2D_x = recvbuf_x.reshape((snaps_per_process_x, u_x_size_1))
        recvbuf_2D_y = recvbuf_y.reshape((snaps_per_process_y, u_x_size_1))
        recvbuf_2D_z = recvbuf_z.reshape((snaps_per_process_z, u_x_size_1))

        cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

        # Interpolate in each rank using the scattered chunk
        # X
        u_xi_local = interpolate_chunk(x, y, z, xs, ys, zs, recvbuf_2D_x, xyz, device=device)
        u_yi_local = interpolate_chunk(x, y, z, xs, ys, zs, recvbuf_2D_y, xyz, device=device)
        u_zi_local = interpolate_chunk(x, y, z, xs, ys, zs, recvbuf_2D_z, xyz, device=device)
    
        # Prepare interpolated data from all ranks to be gathered
        # Send the interpolated values
        sendbuf2_x = u_xi_local
        sendbuf2_y = u_yi_local
        sendbuf2_z = u_zi_local
        # Create the reception buffer
        if rank == 0:
            recvbuf2_x = np.empty((sum(count_x), u_x_size_1), dtype = np.float32)
            recvbuf2_y = np.empty((sum(count_y), u_x_size_1), dtype = np.float32)
            recvbuf2_z = np.empty((sum(count_z), u_x_size_1), dtype = np.float32)
        else:
            recvbuf2_x = None
            recvbuf2_y = None
            recvbuf2_z = None

        print(f"Block {i}: Rank {rank} - Sendbuf size for u_x:", sendbuf2_x.shape)
        print(f"Block {i}: Rank {rank} - Sendbuf size for u_y:", sendbuf2_y.shape)
        print(f"Block {i}: Rank {rank} - Sendbuf size for u_z:", sendbuf2_z.shape)
        if rank == 0:
            print(f"Block {i}: Counts per rank - u_x:", count_elements_x)
            print(f"Block {i}: Counts per rank - u_y:", count_elements_y)
            print(f"Block {i}: Counts per rank - u_z:", count_elements_z)
            print(f"Block {i}: Displacements for gather - u_x:", displ_elements_x)
            print(f"Block {i}: Displacements for gather - u_y:", displ_elements_y)
            print(f"Block {i}: Displacements for gather - u_z:", displ_elements_z)

        print(f"Block {i}: Rank {rank} sending for u_x - {sendbuf2_x[rank]} rows with displacements {displ_elements_x[rank]}.")
        print(f"Block {i}: Rank {rank} sending for u_y - {sendbuf2_y[rank]} rows with displacements {displ_elements_y[rank]}.")
        print(f"Block {i}: Rank {rank} sending for u_z - {sendbuf2_z[rank]} rows with displacements {displ_elements_z[rank]}.")

        # Gather interpolated data from all ranks
        # X
        comm.Gatherv(sendbuf2_x, [recvbuf2_x, count_elements_x, displ_elements_x, MPI.FLOAT], root=0)
        # Y
        comm.Gatherv(sendbuf2_y, [recvbuf2_y, count_elements_y, displ_elements_y, MPI.FLOAT], root=0)
        # Z
        comm.Gatherv(sendbuf2_z, [recvbuf2_z, count_elements_z, displ_elements_z, MPI.FLOAT], root=0)

        if rank == 0:
            print(f"Block {i}: After Gatherv, for u_x, process 0 has data:", recvbuf2_x)
            print(f"Block {i}: After Gatherv, for u_y, process 0 has data:", recvbuf2_y)
            print(f"Block {i}: After Gatherv, for u_z, process 0 has data:", recvbuf2_z)
        
        # Broadcast gathered data to all the other ranks
        recvbuf2_x = comm.bcast(recvbuf2_x)
        recvbuf2_y = comm.bcast(recvbuf2_y)
        recvbuf2_z = comm.bcast(recvbuf2_z)
        gathered_data_x.append(recvbuf2_x)
        gathered_data_y.append(recvbuf2_y)
        gathered_data_z.append(recvbuf2_z)

    # Save results on rank 0
    comm.Barrier()

    # if rank == 0:
    u_xi_combined = np.concatenate(gathered_data_x, axis=0)
    print("After gathering all data, interpolated u_x has data:", u_xi_combined)
    print("Final combined shape for the u_x velocity:", u_xi_combined.shape)
    u_xi_combined = u_xi_combined.T
    print("Final combined shape for the u_x velocity after transposing:", u_xi_combined.shape)
    print("Final u_x velocity after transposing:", u_xi_combined)
    # Y
    u_yi_combined = np.concatenate(gathered_data_y, axis=0)
    print("After gathering all data, interpolated u_y has data:", u_yi_combined)
    print("Final combined shape for the u_y velocity:", u_yi_combined.shape)
    u_yi_combined = u_yi_combined.T
    print("Final combined shape for the u_y velocity after transposing:", u_yi_combined.shape)
    print("Final u_y velocity after transposing:", u_yi_combined)
    # Z
    u_zi_combined = np.concatenate(gathered_data_z, axis=0)
    print("After gathering all data, interpolated u_z has data:", u_zi_combined)
    print("Final combined shape for the u_z velocity:", u_zi_combined.shape)
    u_zi_combined = u_zi_combined.T
    print("Final combined shape for the u_z velocity after transposing:", u_zi_combined.shape)
    print("Final u_z velocity after transposing:", u_zi_combined)

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
                      VELOX = {'ndim':1, 'value':u_xi_combined},
                      VELOY = {'ndim':1, 'value':u_yi_combined},
                      VELOZ = {'ndim':1, 'value':u_zi_combined})
    if rank == 0:
        print(f"Dataset created", d)
        # d.add_variable('time', 1, time)
        # d.add_variable('velox', 1, u_xi_combined)
        # print("Velocity variable created", d)
    if rank == 0:
        print("Ready to save data.")
    d.save('obstacle_3D_parall_FULL.h5', nopartition=True)
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
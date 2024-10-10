# import numpy as np
# import matplotlib.pyplot as plt
# import mat73

# print('------------ LOADING DATA ----------------')
# datapath = '/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/Set_18/U.mat'

# U_dict = mat73.loadmat(datapath)
# #V_dict = mat73.loadmat('V.mat')
# #W_dict = mat73.loadmat('W.mat')
# print('------------ DATA ANALYSIS ----------------')
# U = np.array(U_dict['U']) 
# print('-------- U --------')
# x = U_dict['x']
# y = U_dict['y']
# z = U_dict['z']
# t = U_dict['t']
# print('------- SHAPES ----------',U.shape,np.unique(x).shape,np.unique(y).shape,np.unique(z).shape,np.unique(t).shape)

# plt.figure()
# plt.plot(z[::2]/np.max(z[:]), 'b', linewidth=2, label='z-W')
# plt.plot(y[::2]/np.max(y[:]), 'r', linewidth=2, label='y-H')
# plt.plot(x[::2]/np.max(x[:]), 'g', linewidth=2, label='x-D')
# plt.xlim([0, 8e3])

# plt.legend()
# plt.savefig('mesh_obs.pdf')
# plt.close()


##############################333 PYLOM MESH ########################
import numpy as np
import h5py
import sys
import os
# module_path = os.path.abspath(os.path.join('..', '/mimer/NOBACKUP/groups/kthmech/sanchis/scratch/pyalya'))

# if module_path not in sys.path:
#     sys.path.append(module_path)
# print(module_path)
import pyAlya

import pyLOM
from scipy.interpolate import griddata
import mat73

print('------------ LOADING DATA ----------------')
datapath = '/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/Set_18/U.mat'

file = mat73.loadmat(datapath)
print('Keys in the loaded .mat file:', file.keys())
#V_dict = mat73.loadmat('V.mat')
#W_dict = mat73.loadmat('W.mat')
print('------------ DATA ANALYSIS ----------------')
print('-------- U --------')
x =  np.array(file['x'])
y =  np.array(file['y'])
z =  np.array(file['z'])
time = np.array(file['t'])
#order = np.argsort(np.array(file['global'])-1)
#xyz   = np.array(file['xyz'])
u_x   = np.array(file['U'])
#u_y   = np.array(file['u_y'])
#u_z   = np.array(file['u_z'])
print('data loaded',x.shape,y.shape,z.shape,flush=True)

xs = np.unique(x)
ys = np.unique(y)
zs = np.unique(z)

u = u_x.reshape((np.unique(time)).shape[0],(np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
# u = u[:,0:288:4,0:96:4,0:144:2]
x = x.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
y = y.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
z = z.reshape((np.unique(x)).shape[0],  (np.unique(y)).shape[0], (np.unique(z)).shape[0])
# x = x[0:288,0:96:4,0:144:2]
# y = y[0:288:4,0:96:4,0:144:2]
# z = z[0:288:4,0:96:4,0:144:2]
u_x = u.reshape((len(time),-1)).T
x = x.reshape(-1)
y = y.reshape(-1)
z = z.reshape(-1)
xs = np.unique(x)
ys = np.unique(y)
zs = np.unique(z)
# xs = x
# ys = y
# zs = z

minx, maxx, nx = np.min(xs), np.max(xs), int(xs.shape[0])
miny, maxy, ny = np.min(ys), np.max(ys), int(ys.shape[0])
minz, maxz, nz = np.min(zs), np.max(zs), int(zs.shape[0])

p1 = np.array([minx,miny,minz])
p2 = np.array([maxx,miny,minz])
p4 = np.array([minx,maxy,minz])
p5 = np.array([minx,miny,maxz])

print('MESH',nx,ny,nz)
mesh = pyAlya.MeshSOD2D.cube(p1,p2,p4,p5,nx,ny,nz)
print('mesh created', mesh.xyz.shape, flush=True)
np.savez('coords.npz', xyz=mesh.xyz)

print('x shape:', x.shape, 'dtype:', x.dtype)
print('y shape:', y.shape, 'dtype:', y.dtype)
print('z shape:', z.shape, 'dtype:', z.dtype)
print('u_x shape:', u_x.shape, 'dtype:', u_x.dtype)
print('mesh shape:', mesh.xyz.shape, 'dtype:', mesh.xyz.dtype)
u_xi = griddata((x,y,z), u_x, mesh.xyz, method='linear')
#u_yi = griddata(xyz, u_y, (mesh.xyz[:,0], mesh.xyz[:,1], mesh.xyz[:,2]), method='linear')
#u_zi = griddata(xyz, u_z, (mesh.xyz[:,0], mesh.xyz[:,1], mesh.xyz[:,2]), method='linear')
print('ux interpolated', u_xi.shape, flush=True)

m = pyLOM.Mesh.from_pyAlya(mesh)
p = pyLOM.PartitionTable.from_pyAlya(mesh.partition_table,has_master=False)
d = pyLOM.Dataset(ptable=p, mesh=m, time=time)

d.add_variable('velox', True, 1, u_xi)
#d.add_variable('veloy', True, 1, u_y)
#d.add_variable('veloz', True, 1, u_z)
d.save('obstacle_3D_full.h5', nopartition=True)
"""
#######################333 h5 #############3
import h5py
import numpy as np

print('------------ LOADING DATA ----------------')

# Path to the .h5 file
datapath = '/mimer/NOBACKUP/groups/deepmechalvis/marcial/data/clipped-3D-set2-10k.h5'

# Load the .h5 file using h5py
with h5py.File(datapath, 'r') as file:
    print('Keys in the loaded .h5 file:', list(file.keys()))
    
    # Loop through the keys to print datasets and their shapes
    for key in file.keys():
        print(f"Key: {key}")
        
        # Check if the item is a dataset and print its shape
        dataset = file[key]
        if isinstance(dataset, h5py.Dataset):
            print(f"Shape of {key}: {dataset.shape}")
            print(f"Type of {key}: {dataset.dtype}")

print('------------ DATA ANALYSIS ----------------')
"""
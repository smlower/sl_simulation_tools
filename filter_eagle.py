import h5py
import numpy as np
import pandas as pd
import sys, os
import tqdm
import glob
try:
    import read_eagle
except:
    print('Need to install read_eagle routine from https://github.com/jchelly/read_eagle')
    sys.exit()

try:
    from mpi4py import MPI
except:
    print('Need to instal mpi4py')
    sys.exit()

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

#############
# Line arguments
#############
path_to_snapshots = sys.argv[1]
galaxy_list_path = sys.argv[2]
output_path = sys.argv[3]
galaxy_number = sys.argv[4]
####################

IDs = pd.read_csv(galaxy_list_path)
group = IDs['GroupNumber'][int(galaxy_number)]
subgroup = IDs['SubGroupNumber'][int(galaxy_number)]
print('Read in galaxy ID')

snap_dir = path_to_snapshots
snapshot_0 = glob.glob(snap_dir+'/snap_*_z*.0.hdf5')
snap = read_eagle.EagleSnapshot(snapshot_0)


#find box size
try:
    start = snap_dir.find('RefL') + 4
    end = snap_dir.find('N', start)
    box_size = float(snap_dir[start:end])
except:
    print('Need to input boxsize')
    sys.exit()


snap.select_region(0, box_size * 0.6777, 0, box_size * 0.6777, 0, box_size * 0.6777)
snap.split_selection(comm_rank, comm_size)
print('Selected region')
output_file = h5py.File(output_path+'/galaxy_'+str(galaxy)+'.hdf5', 'w')
input1 = h5py.File(snap_dir+'/'+fname, 'r')
attrs = ['Config',
 'Constants',
 'HashTable',
 'Header',
 'Parameters',
 'RuntimePars',
 'Units']

for key in attrs:

    output_file.copy(input1[key], key)

input1.close()

print('Copied Header and others')

gas_attr = snap.datasets(0)
print('Found gas attributes')
star_attr = snap.datasets(4)
print('Found star attributes')
gas_groups = snap.read_dataset(0, 'GroupNumber')
print('got gas groups')
gas_subgroups = snap.read_dataset(0, 'SubGroupNumber')
print('got gas subgroups')
star_groups = snap.read_dataset(4, 'GroupNumber')
print('got star groups')
star_subgroups = snap.read_dataset(4, 'SubGroupNumber')
print('got star subgroups')

mask_g = np.logical_and(gas_groups == group, gas_subgroups == subgroup)
print('got gas mask')
mask_s = np.logical_and(star_groups == group, star_subgroups == subgroup)
print('got star mask')
output_file.create_group('PartType0')
output_file.create_group('PartType4')

print('going into gas attributes')
for attr in tqdm.tqdm(gas_attr):
    output_file['PartType0'][attr[1:]] = snap.read_dataset(0, attr)[mask_g]
    if attr[1:] == 'Mass':
        output_file['PartType0']['Masses'] = snap.read_dataset(0, attr)[mask_g]
print('going into star attributes')
for attr in tqdm.tqdm(star_attr):
    output_file['PartType4'][attr[1:]] = snap.read_dataset(4, attr)[mask_s]
    if attr[1:] == 'Mass':
        output_file['PartType4']['Masses'] = snap.read_dataset(4, attr)[mask_s]
output_file.close()




re_out = h5py.File(output_path+'/galaxy_'+str(galaxy)+'.hdf5', 'r+')

slist = len(re_out['PartType4']['Coordinates'])
glist = len(re_out['PartType0']['Coordinates'])
re_out['Header'].attrs.modify('NumPart_ThisFile', np.array([glist, 0, 0, 0, slist, 0]))
re_out['Header'].attrs.modify('NumPart_Total', np.array([glist, 0, 0, 0, slist, 0]))
re_out['Header'].attrs.modify('NumFilesPerSnapshot', 1)
re_out.close()

    


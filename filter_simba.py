import h5py
import caesar
import sys
import glob
import numpy as np
import tqdm


###########
# Line arguments
###########
snapshot_path = sys.argv[1]
snap_num = sys.argv[2]
galaxy = sys.argv[3]
output_path = sys.argv[4]
##############

ds = snapshot_path
caesar_file = glob.glob(snapshot_path'/output/Groups/caesar_0'+"{:03d}".format(snap_num)+'*.hdf5')
obj = caesar.load(caesar_file[0])

#here, you can index through a list of selected galaxies, or just use the input galaxy 
#galaxy_num = pd.read_csv('/read/some/selection/file)[int(galaxy)]

glist = obj.galaxies[int(galaxy)].glist
slist = obj.galaxies[int(galaxy)].slist


with h5py.File(ds+'/snapshot_0'+str(snap_num)+'.hdf5', 'r') as input_file, h5py.File(output_path+'galaxy_'+str(galaxy)+'.hdf5', 'w') as output_file:
    output_file.copy(input_file['Header'], 'Header')
    print('starting with gas attributes now')
    output_file.create_group('PartType0')
    for k in tqdm.tqdm(input_file['PartType0']):
        output_file['PartType0'][k] = input_file['PartType0'][k][:][glist]
    print('moving to star attributes now')
    output_file.create_group('PartType4')
    for k in tqdm.tqdm(input_file['PartType4']):
        output_file['PartType4'][k] = input_file['PartType4'][k][:][slist]


print('done copying attributes, going to edit header now')
outfile_reload = output_path+'galaxy_'+str(galaxy)+'.hdf5'

re_out = h5py.File(outfile_reload)                                                  
re_out['Header'].attrs.modify('NumPart_ThisFile', np.array([len(glist), 0, 0, 0, len(slist), 0]))  
re_out['Header'].attrs.modify('NumPart_Total', np.array([len(glist), 0, 0, 0, len(slist), 0]))  

re_out.close()

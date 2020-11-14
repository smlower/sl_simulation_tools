#purpose: to set up slurm files and model *.py files from the
#positions written by caesar_cosmology_npzgen.py for a cosmological
#simulation.  This is written for the University of Florida's
#HiPerGator2 cluster.

import numpy as np
from subprocess import call
import sys

nnodes=1


#################
# Edit these !!!
snap_redshift = float(z)
npzfile = '/orange/narayanan/s.lower/TNG/position_npzs/tng_snap33_pos.npz' 
model_dir_base = '/orange/narayanan/s.lower/TNG/pd_runs/'
hydro_dir = '/orange/narayanan/s.lower/TNG/filtered_snapshots/'
hydro_dir_remote = hydro_dir
model_run_name='TNG_m100'
#################






COSMOFLAG=0 #flag for setting if the gadget snapshots are broken up into multiples or not and follow a nomenclature snapshot_000.0.hdf5
FILTERFLAG = 1 #flag for setting if the gadget snapshots are filtered or not, and follow a nomenclature snap305_galaxy1800_filtered.hdf5


SPHGR_COORDINATE_REWRITE = True


#===============================================

if (COSMOFLAG == 1) and (FILTERFLAG == 1):
    raise ValueError("COSMOFLAG AND FILTER FLAG CAN'T BOTH BE SET")


data = np.load(npzfile,allow_pickle=True)
pos = data['pos'][()] #positions dictionary
#ngalaxies is the dict that says how many galaxies each snapshot has, in case it's less than NGALAXIES_MAX
ngalaxies = data['ngalaxies'][()]




for snap in [snap_num]:
    
    model_dir = model_dir_base+'/snap{:03d}'.format(snap)
    model_dir_remote = model_dir
    
    tcmb = 2.73*(1.+snap_redshift)

    NGALAXIES = ngalaxies['snap'+str(snap)]
    for nh in range(NGALAXIES):
        try:
            xpos = pos['galaxy'+str(nh)]['snap'+str(snap)][0]
        except: continue
        
        ypos = pos['galaxy'+str(nh)]['snap'+str(snap)][1]
        zpos = pos['galaxy'+str(nh)]['snap'+str(snap)][2]
        
        cmd = "./cosmology_setup_all_cluster.hipergator.sh "+str(nnodes)+' '+model_dir+' '+hydro_dir+' '+model_run_name+' '+str(COSMOFLAG)+' '+str(FILTERFLAG)+' '+model_dir_remote+' '+hydro_dir_remote+' '+str(xpos)+' '+str(ypos)+' '+str(zpos)+' '+str(nh)+' '+str(snap)+' '+str(tcmb)
        call(cmd,shell=True)

        

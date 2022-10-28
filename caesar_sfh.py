import sys, os
from tqdm.auto import tqdm
import fsps
import yt
import caesar
import numpy as np
import pickle
from multiprocessing import Pool

snapshot = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/snapshot_087.hdf5'
csfile = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/Groups/caesar_0087_z5.024.hdf5'
outfile = f'/orange/narayanan/s.lower/simba/sfh_m25n512_z5.pickle'
 
print('Loading yt snapshot')
ds = yt.load(snapshot)

print('Quick loading caesar')
obj = caesar.load(csfile)
obj.yt_dataset = ds
dd = obj.yt_dataset.all_data()
print('Loading particle data')
scalefactor = dd[("PartType4", "StellarFormationTime")]
stellar_masses = dd[("PartType4", "Masses")]
stellar_metals = dd[("PartType4", 'metallicity')]


print('Loading fsps')
fsps_ssp = fsps.StellarPopulation(sfh=0,
                zcontinuous=1,
                imf_type=2,
                zred=0.0, add_dust_emission=False)
solar_Z = 0.0142


# Compute the age of all the star particles from the provided scale factor at creation                               
formation_z = (1.0 / scalefactor) - 1.0
yt_cosmo = yt.utilities.cosmology.Cosmology(hubble_constant=0.68, omega_lambda = 0.7, omega_matter = 0.3)
stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")

# Age of the universe right now                                                                                     
simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")

print(f'simtime: {simtime:.1f}')
x = 0
final_massfrac, final_formation_times, final_formation_masses = [], [], []


#get the galaxies from the caesar file
ids = []
for i in obj.galaxies:
    ids.append(i.GroupID)



def get_sfh(galaxy):
    this_galaxy_stellar_ages = stellar_ages[obj.galaxies[ids[galaxy]].slist]
    this_galaxy_stellar_masses = stellar_masses[obj.galaxies[ids[galaxy]].slist]
    this_galaxy_stellar_metals = stellar_metals[obj.galaxies[ids[galaxy]].slist]
    this_galaxy_formation_masses = []
    for age, metallicity, mass in zip(this_galaxy_stellar_ages, this_galaxy_stellar_metals, this_galaxy_stellar_masses):
        mass = mass.in_units('Msun')
        fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
        mass_remaining = fsps_ssp.stellar_mass
        initial_mass = np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)
        massform = mass / initial_mass
        this_galaxy_formation_masses.append(massform)
    this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
    this_galaxy_formation_times = np.array(simtime - this_galaxy_stellar_ages, dtype=float)
    return this_galaxy_formation_times, this_galaxy_formation_masses

with Pool(16) as p:
    out1, out2 = zip(*tqdm(p.imap(get_sfh, range(len(ids))), total=len(ids)))
    final_formation_times = out1
    final_formation_masses = out2

with open(outfile, 'wb') as f:
    pickle.dump({
        'id':ids, 
        'massform':final_formation_masses,
        'tform':final_formation_times, # these objs are lists of arrays -- each element in list corresponds to a galaxy, each element in array corresponds to a star
    },f)
        
# so to get a SFH you would do a binned_statistic: sum of massform / bin width, binned by tform
        

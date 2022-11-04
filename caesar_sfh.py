import sys, os
from tqdm.auto import tqdm
import fsps
import yt
import caesar
import numpy as np
import pickle
from multiprocessing import Pool


#====================================================
#MODIFIABLE HEADER
#====================================================

COSMOLOGICAL = False
AREPO = True #only really impacts idealized galaxy simulations. 

snapshot = '/orange/narayanan/desika.narayanan/arepo_runs/idealized/aliphate_tests/mw_mr_sfe1_sfrdens1000_althresh10/output/snapshot_999.hdf5'
csfile = '/orange/narayanan/desika.narayanan/gizmo_runs/simba/m25n512/output/Groups/caesar_0087_z5.024.hdf5' #meaningless for idealized
outfile = f'/orange/narayanan/desika.narayanan/arepo_runs/idealized/aliphate_tests/mw_mr_sfe1_sfrdens1000_althresh10/output/sfh.pickle'
 

idealized_galaxy_center = [300,300,300] #code units.  meaningless if cosmological.

#====================================================


print('Loading yt snapshot')
ds = yt.load(snapshot)


if AREPO: 
    def _newstars(pfilter,data):
        filter = data[(pfilter.filtered_type, "GFM_StellarFormationTime")] > 0
        return filter
        
    yt.add_particle_filter("newstars",function=_newstars,filtered_type='PartType4')
    ds.add_particle_filter("newstars")

if COSMOLOGICAL: 
    print('Quick loading caesar')
    obj = caesar.load(csfile)
    obj.yt_dataset = ds
    dd = obj.yt_dataset.all_data()
    print('Loading particle data')
    if AREPO == False: 
        scalefactor = dd[("PartType4", "StellarFormationTime")]
    else:
        scalefactor = data[("newstars","GFM_StellarFormationTime")].value
    # Compute the age of all the star particles from the provided scale factor at creation
    formation_z = (1.0 / scalefactor) - 1.0
    yt_cosmo = yt.utilities.cosmology.Cosmology(hubble_constant=0.68, omega_lambda = 0.7, omega_matter = 0.3)
    stellar_formation_times = yt_cosmo.t_from_z(formation_z).in_units("Gyr")
    
    # Age of the universe right now
    simtime = yt_cosmo.t_from_z(ds.current_redshift).in_units("Gyr")
    stellar_ages = (simtime - stellar_formation_times).in_units("Gyr")
else:
    dd = ds.all_data()
    #code ripped from powderday arepo and gadget front ends
    simtime = ds.current_time.in_units('Gyr')
    simtime = simtime.value
    if AREPO:
        print("------------------------------------------------------------------")
        print("WARNING WARNING WARNING:")
        print("Assuming units in stellar ages are s*kpc/km")
        print("if this is not true - please edit right under this warning message")
        print("------------------------------------------------------------------")
        age = simtime-(ds.arr(dd[("newstars","GFM_StellarFormationTime")],'s*kpc/km').in_units('Gyr')).value
        # make the minimum age 1 million years
        age[np.where(age < 1.e-3)[0]] = 1.e-3
        stellar_ages = age
    else:
        age = simtime-ds.arr(dd[('PartType4', 'StellarFormationTime')],'Gyr').value
        # make the minimum age 1 million years
        age[np.where(age < 1.e-3)[0]] = 1.e-3
        stellar_ages = age
    

if AREPO == False:
    stellar_masses = dd[("PartType4", "Masses")]
    stellar_metals = dd[("PartType4", 'metallicity')]
else:
    stellar_masses = dd[("newstars", "Masses")]
    stellar_metals = dd[('newstars', 'GFM_Metallicity')]

print('Loading fsps')
fsps_ssp = fsps.StellarPopulation(sfh=0,
                zcontinuous=1,
                imf_type=2,
                zred=0.0, add_dust_emission=False)
solar_Z = 0.0142

print(f'simtime: {simtime:.1f}')
x = 0
final_massfrac, final_formation_times, final_formation_masses = [], [], []

ids = []
if COSMOLOGICAL:

    #get the galaxies from the caesar file

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

else:

    this_galaxy_stellar_ages = stellar_ages
    this_galaxy_stellar_masses = stellar_masses
    this_galaxy_stellar_metals = stellar_metals
    this_galaxy_formation_masses = []


    for i in tqdm(range(len(this_galaxy_stellar_ages))):
        age = this_galaxy_stellar_ages[i]
        metallicity = this_galaxy_stellar_metals[i]
        mass = this_galaxy_stellar_masses[i]
    #for age, metallicity, mass in tqdm(zip(this_galaxy_stellar_ages, this_galaxy_stellar_metals, this_galaxy_stellar_masses)):
        mass = mass.in_units('Msun')
        fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
        mass_remaining = fsps_ssp.stellar_mass
        initial_mass = np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining)
        massform = mass / initial_mass
        this_galaxy_formation_masses.append(massform)
    this_galaxy_formation_masses = np.array(this_galaxy_formation_masses)
    this_galaxy_formation_times = np.array(simtime - this_galaxy_stellar_ages, dtype=float)
    final_formation_masses = this_galaxy_formation_masses
    final_formation_times = this_galaxy_formation_times

with open(outfile, 'wb') as f:
    pickle.dump({
        'id':ids, 
        'massform':final_formation_masses,
        'tform':final_formation_times, # these objs are lists of arrays -- each element in list corresponds to a galaxy, each element in array corresponds to a star
    },f)
        
# so to get a SFH you would do a binned_statistic: sum of massform / bin width, binned by tform
        

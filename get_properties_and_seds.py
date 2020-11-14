import yt, h5py
import numpy as np
import pandas as pd
import fsps
from hyperion.model import ModelOutput
from sedpy.observate import load_filters
import glob,tqdm
import astropy.constants as constants
import astropy.units as u
from astropy.cosmology import Planck15
import sys


def ssfr_relation(mstar):
    return (10**-13) * (10**mstar)

def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

###########
# Line arguments
###########
pd_dir = sys.argv[1]
snap_dir = sys.argv[2]
z = sys.argv[3]
sim_type = sys.argv[4] #to differentiate between simba with manual dust and TNG/Eagle with dtm
obs_frame = sys.argv[5] #if True, will redshift the rest frame SED into observer frame
#############



galex = ['galex_FUV', 'galex_NUV']
hst_wfc3_uv  = ['wfc3_uvis_f275w', 'wfc3_uvis_f336w', 'wfc3_uvis_f475w','wfc3_uvis_f555w', 'wfc3_uvis_f606w', 'wfc3_uvis_f814w']
sdss = ['sdss_i0']
hst_wfc3_ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']
spitzer_irac = ['spitzer_irac_ch1']
spitzer_mips = ['spitzer_mips_24']
herschel_pacs = ['herschel_pacs_70', 'herschel_pacs_100', 'herschel_pacs_160']
herschel_spire = ['herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
jwst_miri = ['jwst_f560w', 'jwst_f770w', 'jwst_f1000w', 'jwst_f1130w', 'jwst_f1280w', 'jwst_f1500w', 'jwst_f1800w']
jwst_nircam = ['jwst_f070w', 'jwst_f090w', 'jwst_f115w', 'jwst_f150w', 'jwst_f200w', 'jwst_f277w']
alma = ['alma_band6']
scuba = ['scuba_850']
print('loading filters')
filternames = (galex + hst_wfc3_uv +  hst_wfc3_ir + jwst_miri + jwst_nircam + spitzer_irac + spitzer_mips + herschel_pacs + herschel_spire + alma + scuba)
filters_unsorted = load_filters(filternames)
waves_unsorted = [x.wave_mean for x in filters_unsorted]
filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]

print('Loading SP')
fsps_ssp = fsps.StellarPopulation(sfh=0,
                zcontinuous=1,
                imf_type=2,
                zred=0.0, add_dust_emission=False)
solar_Z = 0.0196


print('loading snapshots')
snaps = glob.glob(snap_dir+'/galaxy*.hdf5')


print('initializing lists')
flux_Jy = []
fluxe_Jy = []
stellar_mass_Msun = []
dust_mass_Msun = []
sfr100 = []
metallicity_logzsol = []
gal_count = []
filter_list = []

pd_list = glob.glob(pd_dir+'/*.rtout.sed')
snap_num = int(pd_list[0].split('snap')[2].split('.')[0])
print('loading galaxy list')
galaxy_list = []
for i in pd_list:
    galaxy_list.append(int(i.split('.')[2].split('galaxy')[1]))

for galaxy in tqdm.tqdm(galaxy_list):
    
    m = ModelOutput(pd_dir+'/snap'+str(snap_num)+'.galaxy'+str(galaxy)+'.rtout.sed')
    ds = yt.load(snaps_dir+'/galaxy_'+str(galaxy)+'.hdf5', 'r') 
    wave, flx = m.get_sed(inclination=0, aperture=-1)
    
    gal_count.append(galaxy)
    
    #get mock photometry
    wave  = np.asarray(wave)*u.micron 
    if obs_frame:
        wav = wave[::-1].to(u.AA)*(1 + float(z))
    else:
        wav = wave[::-1].to(u.AA)
    flux = np.asarray(flx)[::-1]*u.erg/u.s
    if float(z) == 0.0:
        dl = (10*u.pc).to(u.cm)
    else:
        dl = Planck15.luminosity_distance(float(z)).to('cm')
    flux /= (4.*3.14*dl**2.)
    nu = constants.c.cgs/(wav.to(u.cm))
    nu = nu.to(u.Hz)
    flux /= nu
    flux = flux.to(u.Jy)

    flx = []
    flxe = []
    for i in range(len(filters)):
        if filters[i].name == 'alma_band6':
            if float(z) == 0.0:
                #in rest frame, pd SEDs end at 1e7 micron, which is about halfway through ALMA band 6, so we have to cheat the flux a bit for z=0
                flx.append(flux[find_nearest(wav.value,1.0e7)].value)
                flxe.append(0.03 * flx[i])
            else:
                flux_range = flux[find_nearest(wav.value,filters[i].wavelength[0]):find_nearest(wav.value,filters[i].wavelength[-1])].value
                wav_range = wav[find_nearest(wav.value,filters[i].wavelength[0]):find_nearest(wav.value,filters[i].wavelength[-1])].value
                trans = np.interp(wav_range, filters[i].wavelength, filters[i].transmission)
                a = np.trapz(wav_range * trans * flux_range, wav_range, axis=-1)
                b = np.trapz(wav_range * trans, wav_range)
                flx.append(a/b)
                flxe.append(0.03* flx[i])
        else:
            flux_range = flux[find_nearest(wav.value,filters[i].wavelength[0]):find_nearest(wav.value,filters[i].wavelength[-1])].value
            wav_range = wav[find_nearest(wav.value,filters[i].wavelength[0]):find_nearest(wav.value,filters[i].wavelength[-1])].value
            trans = np.interp(wav_range, filters[i].wavelength, filters[i].transmission)
            a = np.trapz(wav_range * trans * flux_range, wav_range, axis=-1)
            b = np.trapz(wav_range * trans, wav_range)
            flx.append(a/b)
            flxe.append(0.03* flx[i])

    flux_Jy.append(np.asarray(flx))
    fluxe_Jy.append(np.asarray(flxe))
    filter_list.append([x.name for x in filters])

    #get properties: M*, SFR, Z, Mdust
    print('getting properties')
    ad = ds.all_data()
    star_masses = ad[('PartType4', 'Masses')].in_units('Msun')
    stellar_mass_Msun.append(np.sum(star_masses.value))
    print('got stellar mass')
    scalefactor = ad[("PartType4", "StellarFormationTime")]
    star_metal = ad[("PartType4", 'metallicity')]
    metallicity_logzsol.append(np.log10((np.sum(star_metal * star_masses.value) / np.sum(star_masses.value)) / solar_Z))
    print('got stellar metallicity')
    formation_z = (1.0 / scalefactor) - 1.0
    stellar_formation_age = ds.cosmology.t_from_z(formation_z).in_units("Gyr")
    simtime = ds.cosmology.t_from_z(ds.current_redshift).in_units("Gyr")
    stellar_ages = (simtime - stellar_formation_age).in_units("Gyr")
    star_masses = np.array(ds['PartType4']['Masses'])
    star_metal = np.array(ds['PartType4']['Metallicity'])
    w50 = np.where(stellar_ages.in_units('Myr').value < 100)[0]
    if len(w50) == 0:
        print('no stars born within the last 100 Myr. Setting sfr according to sSFR relation')
        sfr100.append(ssfr_relation(np.log10(np.sum(star_masses.value))))
        #depending on what you want to do, you can set this == 0 instead of using the sSFR relation for galaxies with 0 SFR
    else:
        initial_mass = 0.0
        print('getting initial SP mass')
        for star in tqdm.tqdm(w50):
            current_mass = star_masses[star]
            fsps_ssp.params["logzsol"] = np.log10(star_metal[star] / solar_Z)
            mass_remaining = fsps_ssp.stellar_mass
            initial_mass += current_mass / np.interp(np.log10(stellar_ages[star]*1.e9),fsps_ssp.ssp_ages,mass_remaining)  
        sfr_50myr = initial_mass/100.e6
        sfr100.append(sfr_50myr)
    
    print('got SFR')
    
    if sim_type == 'simba':
        dust_masses = ad.ds.arr(ad[("PartType0", "Dust_Masses")].value, 'code_mass')                                                                       
        dust_mass_Msun.append(np.sum(dust_masses.in_units('Msun').value))
    else: #assuming TNG or Eagle
        metal_mass = (ad[("PartType0", "Masses")])*(ad["PartType0", "Metallicity"].value)
        dust_masses = metal_mass * 0.4
        dust_mass_Msun.append(np.sum(dust_masses.in_units('Msun').value))
    print('got dust mass')


if obs_frame:
    seds_name = snap_dir+'/ml_SEDs_z'+z+'_obs_frame.pkl'
else:
    seds_name = snap_dir+'/ml_SEDs_z'+z+'_rest_frame.pkl'
print('saving SEDs as ',seds_name)
sed_data = {'ID' : gal_count, 'Filters' : filter_list, 'Flux [Jy]' : flux_Jy, 'Flux Err': fluxe_Jy}
s = pd.DataFrame(sed_data, index=np.arange(len(gal_count)))
s.to_pickle(seds_name)

props_name= snap_dir+'/ml_props_z'+z+'.pkl'
print('saving properties as ',props_name)
prop_data = {'ID' : gal_count, 'stellar_mass' : stellar_mass_Msun, 'dust_mass' : dust_mass_Msun, 'sfr' : sfr100, 'metallicity' : metallicity_logzsol}
t = pd.DataFrame(prop_data, index = np.arange(len(gal_count)))
t.to_pickle(props_name)


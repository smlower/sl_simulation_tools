import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib
matplotlib.rcParams.update({
    "savefig.facecolor": "w",
    "figure.facecolor" : 'w',
    "figure.figsize" : (10,8),
    "text.color": "k",
    "legend.fontsize" : 20,
    "font.size" : 30,
    "axes.edgecolor": "k",
    "axes.labelcolor": "k",
    "axes.linewidth": 3,
    "xtick.color": "k",
    "ytick.color": "k",
    "xtick.labelsize" : 25,
    "ytick.labelsize" : 25,
    "ytick.major.size" : 12,
    "xtick.major.size" : 12,
    "ytick.major.width" : 2,
    "xtick.major.width" : 2,
    "font.family": 'STIXGeneral',
    "mathtext.fontset" : "cm"
})

binwidth = 3 #Myr


#this function is given to scipy.stats.binned_statistics and acts on the particle masses per bin to give total(Msun) / timebin
def get_massform(massform):
        return np.sum(massform) / (binwidth * 1e6)

def get_galaxy_SFH(file_):

    galaxy = 19 #fixed for this example since test file is for one galaxy only
    dat = pd.read_pickle(file_)
    massform = np.array(dat['massform'][np.where(np.asarray(dat['id'])==galaxy_id)[0][0]])
    tform =  np.array(dat['tform'][np.where(np.asarray(dat['id'])==galaxy_id)[0][0]])*1000 #convert from Gyr to Myr
    t_H = np.max(tform)

    bins = np.arange(100, t_H, binwidth) #can use whatever bin size/start/end that fit your problem

    sfrs, bins, binnumber = scipy.stats.binned_statistic(tform, massform, statistic=get_massform, bins=bins)
    sfrs[np.isnan(sfrs)] = 0
    bincenters = 0.5*(bins[:-1]+bins[1:])
    sfh = sfrs
    return bincenters, sfh


test_file = '/orange/narayanan/s.lower/simba/m25n256_dm/zooms/galaxy_properties/sfh_m25zoom_run19_galaxy1_within_120_kpc_0.8452787482129269 Gyr.pickle'

time, sfh = get_galaxy_sfh(test_file)


plt.plot(time, sfh)
plt.ylabel('SFR [M$_{\odot}/$yr]')
plt.xlabel('t$_H$ [Myr]')

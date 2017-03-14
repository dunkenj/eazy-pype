"""
K. Duncan - duncan@strw.leidenuniv.nl
"""

import os, re, sys
import array
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid

# Scipy extras
from scipy.integrate import simps, cumtrapz, trapz
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
from scipy.signal import argrelmax

# Astropy
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, Row, Column, vstack
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.utils.console import ProgressBar
import h5py

# Scikit Learn (machine learning algorithms)
from sklearn.externals import joblib

# Set cosmology (for volume priors)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--params", type=str,
                    help = "Parameter file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()
quiet = args.quiet



if __name__ == "__main__":
    params_root = re.split(".py", args.params)[0]
    if os.path.isfile(params_root+".pyc"):
        os.remove(params_root+".pyc")

    import importlib
    try:
        pipe_params = importlib.import_module(params_root)
        print('Successfully loaded "{0}" as params'.format(args.params))
        reload(pipe_params)
    except:
        print('Failed to load "{0}" as params'.format(args.params))
        raise

    try:
        photometry = Table.read('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog), 
                                format = pipe_params.photometry_format)
    except:
        raise
        
        
    subsize = pipe_params.block_size
    nsteps = int(len(photometry)/subsize)+1       


    st_paths = Table.read('templates/Pickles.spectra_all.param', format='ascii.no_header')
    st_names = []
            
    for j in range(len(st_paths)):
        head, tail = os.path.split(st_paths[j]['col2'])    
        tname, _ = os.path.splitext(tail)
        st_names.append(tname)
    st_names = np.array(st_names)    
    

    """
    Open HDF5 files for each
    """
    folder = '{0}/full'.format(pipe_params.working_folder)
    hdf_hb = h5py.File('{0}/pz_all_hb.hdf'.format(folder), 'w')

    with np.load('{0}/1/HB_pz.npz'.format(folder)) as data:
        pz = data['pz']
        zgrid = data['zgrid']

    HB_pz = hdf_hb.create_dataset("Pz", (len(photometry), len(zgrid)), dtype='f')
    z = hdf_hb.create_dataset("zgrid", data=zgrid)
    
    bar = ProgressBar(nsteps)
    for i in range(nsteps):
    
        pzarr = []
        zouts = []
        
        chis = []
        nfilts = []
        
        folder = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
        phot = Table.read('{0}/{1}.cat'.format(folder, i+1), format='ascii.commented_header')
        
        """
        HB Fits
        """
        sub_cat = Table.read('{0}/HB.{1}.cat'.format(folder, i+1), format='ascii.commented_header')

        with np.load('{0}/HB_pz.npz'.format(folder)) as data:
            pz = data['pz']
            zgrid = data['zgrid']
        
        HB_pz[int(i*subsize):int((1+i)*subsize), :] = pz        
        
        za = zgrid[np.argmax(pz, axis=1)]
        
        sub_cat['za_hb'] = za
        
        """
        Individual Sets
        """
        
        for itx, template in enumerate(pipe_params.templates):
            #print(template)
            
            """ Load Values/Arrays/Catalogs """
            basename='{0}.{1}'.format(template, i+1)
            #pz, zgrid = getPz('{0}/{1}'.format(folder, basename))
            zout = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')
            
            sub_cat['za_{0}'.format(template)] = zout['z_{0}'.format(pipe_params.fitting_mode[itx])]
            sub_cat['zm_{0}'.format(template)] = zout['z_m1']
            sub_cat['zpeak_{0}'.format(template)] = zout['z_peak']
            chi_r = zout['chi_{0}'.format(pipe_params.fitting_mode[itx])]/(zout['nfilt'] - 1)
            chi_r[zout['chi_{0}'.format(pipe_params.fitting_mode[itx])] == -99.] = -99.
            sub_cat['chi_r_{0}'.format(template)] = chi_r
            sub_cat['l68_{0}'.format(template)] = zout['l68']
            sub_cat['u68_{0}'.format(template)] = zout['u68']
            sub_cat['nfilt_{0}'.format(template)] = zout['nfilt']
        

        
        chis = np.array(chis)
        nfilts = np.array(nfilts)
        
        """
        Stellar Fits
        """
        basename='{0}.{1}'.format('pickles', i+1)
        stellar = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')
        star_best = np.zeros(len(stellar), dtype='S6')
        star_best[stellar['temp_1'] >= -90] = st_names[stellar['temp_1']-1][stellar['temp_1'] >= -90]
        
        schi = stellar['chi_1']/(stellar['nfilt']-1)
        schi[stellar['chi_1'] == -99.] = -99.
        sub_cat['chi_r_stellar'] = schi
        
        sub_cat['stellar_type'] = star_best
        
        if i == 0:
            full_cat = Table.copy(sub_cat)
        else:
            full_cat = vstack([full_cat, sub_cat])
        bar.update()
  
    folder = '{0}/full'.format(pipe_params.working_folder)
    path = '{0}/photoz_all_merged.fits'.format(folder)
    if os.path.isfile(path):
        os.remove(path)
    full_cat.write(path, format='fits')
  
    chis = np.array([full_cat['chi_r_eazy'], full_cat['chi_r_cosmos'], full_cat['chi_r_atlas']])
    chi_best = np.min(chis, axis=0)
    
    
    hdf_hb.close()
    
    
  

import numpy as np
import re
import array
import glob
import matplotlib.pyplot as plt
import pyfits
from scipy.constants import golden
from matplotlib.ticker import NullFormatter
from astropy.stats import sigma_clipped_stats, bootstrap
from astropy.stats import median_absolute_deviation as mad

import os

def getSEDs(basepath, temp_filt='photz.tempfilt'):

    ### Load EAZY stuffs ###
    with open(temp_filt,'rb') as f:
        s = array.array('i')
        s.fromfile(f,4)

        NFILT = s[0]
        NTEMP = s[1]
        NZ = s[2]
        NOBJ = s[3]

        tempfilt = array.array('d')
        tempfilt.fromfile(f,NZ*NTEMP*NFILT)
        tempfilt = np.reshape(tempfilt,(NZ,NTEMP,NFILT))
        
        lc = array.array('d')
        lc.fromfile(f,NFILT)
        lc = np.array(lc)

        zgrid = array.array('d')
        zgrid.fromfile(f,NZ)
        zgrid = np.array(zgrid)

        fnu = array.array('d')
        fnu.fromfile(f,NFILT*NOBJ)
        fnu = np.reshape(fnu,(NOBJ,NFILT))

        efnu = array.array('d')
        efnu.fromfile(f,NFILT*NOBJ)
        efnu = np.reshape(efnu,(NOBJ,NFILT))

    #Coeffs (OBS_SED)
    with open(basepath+'.coeff','rb') as f:
        s = array.array('i')
        s.fromfile(f,4)

        NFILT = s[0]
        NTEMP = s[1]
        NZ = s[2]
        NOBJ = s[3]

        coeffs = array.array('d')
        coeffs.fromfile(f,NTEMP*NOBJ)
        coeffs = np.reshape(coeffs,(NOBJ,NTEMP))

        izbest = array.array('i')
        izbest.fromfile(f,NOBJ)
        izbest = np.array(izbest)

        tnorm = array.array('d')
        tnorm.fromfile(f,NTEMP)
        tnorm = np.array(tnorm)

    #Full templates (TEMP_SED)
    with open(basepath+'.temp_sed','rb') as f:
        s = array.array('i')
        s.fromfile(f,3)

        NTEMP = s[0]
        NTEMPL = s[1]
        NZ = s[2]
        
        templam = array.array('d')
        templam.fromfile(f,NTEMPL)
        templam = np.array(templam)

        temp_seds = array.array('d')
        temp_seds.fromfile(f,NTEMPL*NTEMP)
        temp_seds = np.reshape(temp_seds,(NTEMP,NTEMPL))
        
        da = array.array('d')
        da.fromfile(f,NZ)
        da = np.array(da)

        db = array.array('d')
        db.fromfile(f,NZ)
        db = np.array(db)


    # PofZ
    with open(basepath+'.pz','rb') as f:
        s = array.array('i')
        s.fromfile(f,2)
        
        NZ = s[0]
        NOBJ = s[1]

        pz = array.array('d')
        pz.fromfile(f,NZ*NOBJ)
        pz = np.reshape(pz,(NOBJ,NZ))


    #f=pyfits.open('/home/ppxkd/work/data/EAZY/training/run1/gsd4e_tf_h_111215_eazy_erb_spec.data_merged_specz.fits')
    #tbdata=f[1].data
    fit_seds = (coeffs[:,:,None] * tempfilt[izbest]).sum(1)
    #fit_seds2 = np.dot(coeffs[:10], tempfilt[izbest][:10])

    obs_seds = fnu

    return fnu, efnu, fit_seds, lc

def plot_sed(base, gal_index, savepath=None, fig_format='pdf'):
    """ Plot fitted SEDs and residual from stellar mass code outputs
        
    Parameters
    ----------
       
        gal_index : ind,
            Index of galaxy (within catalog) to plot SED of.

        savepath : string, (optional)
            Output path to save figure to.
        
        format : string, default = 'pdf'
            Matplotlib output format
       
       
    Returns
    -------
    
        Figure
    
    """
    fnu, efnu, fit_seds, wl = getSEDs(base, base+'.tempfilt')
    flux = fnu
    fluxerr = efnu
    fit_flux = fit_seds
    fwhm = 0.1*wl

    print(wl)

    translate = np.loadtxt(base+'.translate', dtype='str')
    fnames = translate[:,0]
    eazy_fno = translate[:,1]

    isflux = [filt.endswith('_flux') for filt in fnames]

    fnames = fnames[np.array(isflux)]
    eazy_fno = eazy_fno[np.array(isflux)]    
    eazy_fi = [int(re.split('F', fn)[1])-1 for fn in eazy_fno]
    
    print(fnames[eazy_fi])
    
    Fig, Axes = plt.subplots(2, figsize=(7,5.5), sharex=True,
                             gridspec_kw={'height_ratios':[2.,1]})
    Ax = Axes[0]
    Bx = Axes[1]
       
    Ax.errorbar(wl/1e4, flux[gal_index,:], yerr=fluxerr[gal_index,:], xerr=0.5*fwhm/1e4, 
                fmt='o', capsize=0, elinewidth=2, color='firebrick', label='Obs. Flux')
    Ax.plot(wl/1e4, fit_flux[gal_index,:], 'x', mew=2, color='steelblue', label='Model Flux')
    Ax.set_yscale('log')
    Ax.set_xscale('log')
    Ax.set_ylabel(r'$F_{\nu}$ $[\mu \rm{Jy}]$')
    
    Bx.plot([0.01, 10], [0,0], color='k', lw='2')
    
    residual = (fit_flux[gal_index,:]-flux[gal_index,:])/fluxerr[gal_index,:]
    can_plot = (np.abs(residual) < 3.5) * (fluxerr[gal_index,:] >= 0.)
    upper = (residual >=3.5)
    lower = (residual <= -3.5)
    
    for ixn, name in enumerate(fnames):
        Ax.text(wl[ixn]/1.0e4, flux[gal_index][ixn]*1.1, re.split('_flux', name)[0])
   
    try:
        obs = (fluxerr[gal_index,:] >= 0)
        color_a = np.zeros(len(flux[gal_index][obs]))
        color_b = np.zeros(len(flux[gal_index][obs]))
        color_a[1:] = -2.5*np.log10(flux[gal_index,obs][:-1]/flux[gal_index,obs][1:])
        color_b[:-1] = -2.5*np.log10(flux[gal_index,obs][:-1]/flux[gal_index,obs][1:])
        
        drop = np.logical_and(color_a < -1., color_b > 1.)
        spike = np.logical_and(color_a > 2, color_b < -2.)
        
        criteria1 = np.logical_or(drop, spike)
        criteria2 = np.logical_or(color_a < -4, color_b > 4)
        
        anomalous_b = np.logical_or(criteria1,criteria2)
        Ax.plot(wl[obs][anomalous_b]/1e4, flux[gal_index,obs][anomalous_b], '+', color='gold', label='Identified Outlier')
               
        
    except:
        pass
    
    leg = Ax.legend(loc='lower left', prop={'size':9}, numpoints=1)
    
    Bx.plot(wl[can_plot]/1e4, residual[can_plot], 'o',
            mew=0, color='steelblue')
    Bx.plot(wl[upper]/1e4, np.ones(np.sum(upper))*3.3, '^', ms=10,
        mew=0, color='steelblue')
    Bx.plot(wl[lower]/1e4, np.ones(np.sum(lower))*-3.3, 'v', ms=10,
        mew=0, color='steelblue')


    Bx.set_xticks([0.3, 0.4, 0.5, 0.8, 1.0, 2., 3., 4., 5.])
    Bx.set_xticklabels(['0.3', '0.4', '0.5', '0.8', '1', '2', '3', '4', '5'])
    Bx.set_xlabel(r'Wavelength $[\mu m]$')
    Bx.set_ylabel(r'$\frac{F_{m} - F_{o}}{\sigma_{o}}$', size=16)
    Bx.set_xlim([0.25,5.5])
    Bx.set_ylim([-3.5, 3.5])
    
    Bx.fill_between([0.01, 10], [5., 5.], [2., 2.], color='firebrick', alpha=0.3)
    Bx.fill_between([0.01, 10], [5., 5.], [3., 3.], color='firebrick', alpha=0.3)
    
    Bx.fill_between([0.01, 10], [-2., -2.], [-5., -5.], color='firebrick', alpha=0.3)
    Bx.fill_between([0.01, 10], [-3., -3.], [-5., -5.], color='firebrick', alpha=0.3)
    """
    output_string = '{0[0]} = {0[1]:.3f}{0[2]}' + \
                    '{0[3]} = {0[4]:.2f}{0[5]}' + \
                    '{0[9]} = {0[10]:.2f}{0[8]}' + \
                    '{0[6]} = {0[7]:.2f}'
                    
    output_values = ['z', results_cat['z'][gal_index], '\n',
                     '$\log_{10}(M_{best})$', results_cat['Mass_best'][gal_index], '\n',
                     '$\chi^2$', results_cat['chi_best'][gal_index], '\n',
                     '$\log_{10}(M_{50})$', results_cat['Mass_median'][gal_index]]
    
    Ax.text(4.5, Ax.get_ylim()[0] * 1.5, 
            output_string.format(output_values),
            verticalalignment='bottom',
            horizontalalignment='right')
    """
    Fig.tight_layout()
    Fig.subplots_adjust(hspace=0.05)
    
    if savepath != None:
        Fig.savefig(savepath, format=fig_format, bbox_inches='tight')
        plt.clf()
        plt.close(Fig)
    else:
        plt.show()

def calc_zeropoints(base, verbose=False):
    fnu, efnu, fit_seds, wl = getSEDs(base, base+'.tempfilt')

    #phot = Table.read(catalog, format='ascii.commented_header')
    #loadzp = np.loadtxt('/home/duncan/code/eazy-photoz/inputs/'+base+'.zeropoint',dtype='str')[:,1].astype('float')

    flux = fnu
    fluxerr = efnu
    fit_flux = fit_seds
    fwhm = 0.1*wl

    translate = np.loadtxt(base+'.translate', dtype='str')
    fnames = translate[:,0]
    eazy_fno = translate[:,1]

    isflux = [filt.endswith('_flux') for filt in fnames]

    fnames = fnames[np.array(isflux)]
    eazy_fno = eazy_fno[np.array(isflux)]

    medians = np.zeros(fnu.shape[1])
    scatter = np.zeros(fnu.shape[1])

    Nfilts = len(isflux)
    

    Fig, Ax = plt.subplots(5, int(Nfilts/5)-1, sharex=True, figsize = (12.*golden, 10))

    for i, ax in enumerate(Ax.flatten()[:fnu.shape[1]]):
        cut = ((fnu > 3*efnu) * (efnu > 0.) * (fnu < 100.))[:,i]
        ratio = (fit_seds[cut,i]-fnu[cut,i])/fit_seds[cut,i] + 1
        
        
        #ratio = (fit_seds[cut,i]-fnu[cut,i])/efnu[cut,i]
        c = np.invert(np.isnan(ratio))
        ratio = ratio[c]

        if np.sum(c) > 10:
            medians[i] = np.nanmedian(ratio)
            bootresult = bootstrap(ratio, 100, 
                                   samples=np.maximum(len(ratio)-1, int(0.1*len(ratio))), 
                                   bootfunc=np.nanmedian)
            scatter[i] = np.std(bootresult)
        
        
            hist, bins, ob = ax.hist(ratio, bins=101, range=(0.,2.), 
                                     histtype='stepfilled', normed=True)
            
            ax.text(1.5,1,'{0:.3f}'.format(medians[i]),size=10,
                    bbox=dict(boxstyle="round", fc="w", alpha=0.7, lw=0.))

            ax.set_xlim([0,2])
            ax.set_ylim([0,np.max(hist)*1.33])
            #ax.set_yscale('log')

        else:
            medians[i] = -99.
            scatter[i] = -99.

        if i % 9 == 0:
            ax.set_ylabel('Normalised counts')
        ax.set_xlabel(r'$F_{\rm{fit}}/F_{\rm{obs}}$')
            
        #ax.set_xticks([0.,0.5,1.,1.5])
        ax.set_title(fnames[i],x=0.5,y=0.8,size=9,
                     bbox=dict(boxstyle="round", fc="w", alpha=0.7, lw=0.))


        if verbose:
            print ('{0}: {1:.3f} +/- {2:.3f}'.format(fnames[i], medians[i], scatter[i]))

    Fig.subplots_adjust(left=0.05,right=0.98,bottom=0.065,top=0.98,wspace=0,hspace=0)
    #plt.show()
    
    c = np.isnan(medians)
    medians[c] = 99.
    scatter[c] = 99.

    output_path = base+'.zeropoint'

    with open(output_path,'w') as file:
        for i, med in enumerate(medians):
            if (np.abs(med-1) > 2.*np.abs(scatter[i])):
                file.write('{0}    {1:.3f} {2}'.format(eazy_fno[i], med, '\n'))

    return output_path, Fig, medians, scatter

if __name__ == "__main__":
    print('TESTS')


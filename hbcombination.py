"""
Implementation of the Hierarchical Bayes P(z) combination from Dahlen+ 2013

K. Duncan - duncan@strw.leidenuniv.nl
"""

import os, re
import array
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid

from sklearn.cross_validation import ShuffleSplit
# Scipy extras
from scipy.integrate import simps, cumtrapz, trapz
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.stats import ks_2samp

# Astropy
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.utils.console import ProgressBar

import priors
import pdf_calibration as pdf
import emcee

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--params", type=str,
                    help = "Parameter file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()
quiet = args.quiet
    
#Scikit Learn (machine learning algorithms)


# Set cosmology (for volume priors)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def maybemkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)    
        
def getPz(basepath):

    ### Load EAZY stuffs ###
    with open(basepath+'.tempfilt','rb') as f:
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

    #pz -= (pz.min()-1)

    pz = np.exp(-0.5*pz)

    norm = np.trapz(pz, zgrid, axis =1)
    pz /= norm[:, None]

    return pz, zgrid

def calc_68(pdf_y, pdf_x, photoz, quiet=False):

    NGALS = len(pdf_y)
    l68, u68, intdiff = [], [], []

    if not quiet:
        # Print some helpful things
        print '-'*55
        print '\t{0}\t{1}\t{2}\t{3}\t{4}'.format('#', 'z_p', 'l68', 'u68', 'z_p cumsum')
        print '-'*55

    # Cumulative trapz them
    gal_pdf_cums = cumtrapz(pdf_y, pdf_x, axis=1, initial=0 )

    for gal in range(NGALS):
        # Set galaxy properties 
        gal_pdf, gal_photoz = pdf_y[gal,:], photoz[gal]
        # Calculate the cumulative integral of the P(z) here, padded with an initial zero
        gal_pdf_cum = gal_pdf_cums[gal,:]
        # gal_pdf_cum = cumtrapz( gal_pdf, pdf_x, initial=0 )
        # Find the cumulative integral at the best-fit photometric redshift
        photoz_cumsum = gal_pdf_cum[ np.argmin( abs( pdf_x - gal_photoz ) ) ]
        # Find the indicies of the lower confidence limit
        l68_idx = np.argmin( abs( gal_pdf_cum - (photoz_cumsum - 0.3415) ) )
        # Set the remaining integrated probability to zero. If the lower limit hits the z=0.01
        #   ...bookend then add it to the upper limit so that the integral between the two limits
        #   ...should equal 0.683. Should.
        leftover_pz = 0.
        if l68_idx == 0:    leftover_pz = 0.3415 - photoz_cumsum
        # Find the upper confidence limit
        u68_idx = np.argmin( abs( gal_pdf_cum - (photoz_cumsum + 0.3415 + leftover_pz) ) )
        # Now check for an upper confidence limit bookend
        if (u68_idx == len(gal_pdf_cum) - 1) and (l68_idx > 0):
            leftover_pz = 0.3415 - (1. - photoz_cumsum)
            l68_idx = np.argmin( abs( gal_pdf_cum - (photoz_cumsum - 0.3415 - leftover_pz)) )
        # Save the confidence limits
        l68.append( pdf_x[l68_idx] ), u68.append( pdf_x[u68_idx] )
        intdiff.append( gal_pdf_cum[u68_idx] - gal_pdf_cum[l68_idx] )

        if not quiet:
            if gal % 1000 == 0:  print '\t{0}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.4f}'.format(gal+1, gal_photoz, l68[-1], u68[-1],photoz_cumsum, intdiff[-1])

    return (l68, u68, intdiff)

def calc_1sigma_frac(pz, zgrid, specz, quiet=False):
    pz_argmax = np.argmax(pz, axis=1)
    photoz = zgrid[pz_argmax]

    #initial_68 = calc_68(pofz_sample, zgrid, photoz_sample)
    interp_dz = 0.005
    interp_z            = np.arange( zgrid.min(), zgrid.max(), interp_dz)
    interp_pz_cube_fn   = interp1d( zgrid, pz, kind='linear', axis=1 )
    interp_pz_cube      = interp_pz_cube_fn( interp_z )


    # Get confidence limits
    z_lowers, z_uppers, intdiff = calc_68( interp_pz_cube, interp_z, photoz, quiet=quiet)
    # Convert to sane types
    z_uppers, z_lowers = np.array(z_uppers), np.array(z_lowers)


    # Calculate the percentage of 1-sigma spec-z matches
    in_pdf_1sigma = np.sum( (specz >= z_lowers) * (specz <= z_uppers), dtype=float) / len(specz)

    return in_pdf_1sigma


def calcStats(photoz, specz):
    cut = np.logical_and(photoz >= 0, specz >= 0.)
    print('NGD: {0}'.format(cut.sum()))
    dz = photoz - specz

    ol1 = (np.abs(dz)/(1+specz) > 0.2 )    
    nmad = 1.48 * np.median( np.abs(dz[cut] - np.median(dz[cut])) / (1+specz[cut]))
    ol2 = (np.abs(dz)/(1+specz) > 5*nmad )
    OLF1 = np.sum( ol1[cut] ) / float(len(dz[cut]))
    OLF2 = np.sum( ol2[cut] ) / float(len(dz[cut]))

    print('NMAD: {0:.4f}'.format(nmad))
    print('Bias: {0:.4f}'.format(np.nanmedian(dz[cut]/(1+specz[cut]))))
    print('Bias: {0:.4f}'.format(np.nanmedian(dz[cut])))
    print('OLF: Def1 = {0:.4f} Def2 = {1:0.4f}'.format(OLF1, OLF2))
    print('KS: {0}'.format(ks_2samp(specz[cut], photoz[cut])))
    print('\n')
    
    ol1_s, ol2_s = np.invert(ol1), np.invert(ol2)
    
    return ol1_s, ol2_s, np.nanmedian(dz[ol1_s]/(1+specz[ol1_s]))

def doplot(gal):
    Fig, Ax = plt.subplots(1, figsize=(5.5,3.5))

    rand = np.random.randint(pzarr.shape[0])

    Ax.plot(zgrid, pzarr[0, gal], '--', lw=1)
    Ax.plot(zgrid, pzarr[1, gal], '--', lw=1)
    Ax.plot(zgrid, pzarr[2, gal], '--', lw=1)

    Ax.plot(zgrid, pzarr_hb[gal], lw=2, color='k')
    
    Ax.plot(np.ones((2))*photom['z_spec'][gal], [0, Ax.get_ylim()[1]], color='0.5')
    Ax.set_xlim([0, 9.])
    
    plt.show()


def lnprior(alpha):
    if 0.1 < alpha < 100.:
        return 0.0
    return -np.inf

def lnlike(alpha, pz, zgrid, zspec):
    ci, bins = pdf.calc_ci_dist(pz**(1/alpha), zgrid, zspec)
    ci_dist = -1*np.log((ci[:80]-bins[:80])**2).sum()
    return ci_dist

def lnprob(alpha, pz, zgrid, zspec):
    lp = lnprior(alpha)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(alpha, pz, zgrid, zspec)

def fitalphas(pz, zgrid, zspec, alpha_start,
              nwalkers=10, nsamples=10, fburnin=0.1, nthreads = 10):
    """ Fit prior functional form to observed dataset with emcee
    
    """
    ndim = 1
    burnin = int(nsamples*fburnin)    
    # Set up the sampler.
    pos = [alpha_start + 0.5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                    args=(pz_train, zgrid, zspec_train),
                                    threads=nthreads)


    # Clear and run the production chain.
    sampler.run_mcmc(pos, nsamples, rstate0=np.random.get_state())
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))     
    return np.median(samples), sampler


def find_ci_cut(pz, zgrid):
    peak_p = np.max(pz)
    pz_c = np.max(pz)
    int_pz = 0.

    while int_pz < 0.8:
        pz_c *= 0.99
        cut = (pz < pz_c)
        pz_i = np.copy(pz)
        pz_i[cut] = 0.
        
        int_pz = np.trapz(pz_i, zgrid) / np.trapz(pz, zgrid)
        #print('{0} {1}'.format(int_pz, pz_c))
        
    return pz_c
        

def get_peak_z(pz, zgrid):
    pz_c = find_ci_cut(pz, zgrid)
    p80 = (pz > pz_c)
    
    pz_to_int = np.copy(pz)
    pz_to_int[np.invert(p80)] *= 0.
    
    lbounds = np.argwhere(np.diff(p80.astype('float')) == 1.).squeeze()
    ubounds = np.argwhere(np.diff(p80.astype('float')) == -1.).squeeze()
    lbounds = np.array(lbounds, ndmin=1)
    ubounds = np.array(ubounds, ndmin=1)
    
    zpeaks = []
    z_low = []
    z_high = []
    peak_areas = []
    
    if len(lbounds) >= 1 and len(ubounds) >= 1:
        if lbounds[0] > ubounds[0]: 
            lbounds = np.insert(lbounds, 0, 0)    
        
        for ipx in range(len(lbounds)):
            #print('{0} {1}'.format(lbounds, ubounds))
            lo = lbounds[ipx]
            try:
                up = ubounds[ipx]
            except:
                up = -1
                      
            area = np.trapz(pz_to_int[lo:up], zgrid[lo:up])
            peak_areas.append(area)
            
            zmz = np.trapz(pz_to_int[lo:up]*zgrid[lo:up], zgrid[lo:up]) / area
            z_low.append(zgrid[lo])
            z_high.append(zgrid[up])
            zpeaks.append(zmz)
            
        peak_areas = np.array(peak_areas)
        zpeaks = np.array(zpeaks)
        z_low = np.array(z_low)
        z_high = np.array(z_high)
        
        order = np.argsort(peak_areas)[::-1]
        #print(order)
    
        return zpeaks[order], z_low[order], z_high[order], peak_areas[order]
    
    elif len(lbounds) == 1 and len(ubounds) == 0:
        lo = lbounds[0]
        area = np.trapz(pz_to_int[lo:-1], zgrid[lo:-1])
        peak_areas.append(area)
        
        zmz = np.trapz(pz_to_int[lo:-1]*zgrid[lo:-1], zgrid[lo:-1]) / area
        zpeaks.append(zmz)
        z_low.append(zgrid[lo])
        z_high.append(zgrid[-1])
    
        return zpeaks, z_low, z_high, peak_areas


    elif len(lbounds) == 0 and len(ubounds) == 1:
        up = ubounds[0]
        area = np.trapz(pz_to_int[0:up], zgrid[0:up])
        peak_areas.append(area)
        
        zmz = np.trapz(pz_to_int[0:up]*zgrid[0:up], zgrid[0:up]) / area
        zpeaks.append(zmz)
        z_low.append(zgrid[0])
        z_high.append(zgrid[up])
   
        return zpeaks, z_low, z_high, peak_areas
        
    else:

        """
        try:
            area = np.trapz(pz_to_int, zgrid)
            zmz = np.trapz(pz_to_int*zgrid, zgrid) / area
            return zmz, 
        except:
            out = ''
            return -99., -99., -99., -99.
        """
        return -99., -99., -99., -99.

def pz_to_catalog(pz, zgrid, catalog, verbose=True):

    output = Table()       
    pri_peakz = np.zeros_like(catalog['z_spec'])
    pri_upper = np.zeros_like(catalog['z_spec'])
    pri_lower = np.zeros_like(catalog['z_spec'])
    pri_area = np.zeros_like(catalog['z_spec'])
    
    pri_peakz.name = 'z1_median'
    pri_upper.name = 'z1_max'
    pri_lower.name = 'z1_min'
    pri_area.name = 'z1_area'
    
    pri_peakz.format = '%.4f'
    pri_upper.format = '%.4f'
    pri_lower.format = '%.4f'
    pri_area.format = '%.3f'
    

    sec_peakz = np.zeros_like(catalog['z_spec'])
    sec_upper = np.zeros_like(catalog['z_spec'])
    sec_lower = np.zeros_like(catalog['z_spec'])
    sec_area = np.zeros_like(catalog['z_spec'])
    
    sec_peakz.name = 'z2_median'
    sec_upper.name = 'z2_max'
    sec_lower.name = 'z2_min'
    sec_area.name = 'z2_area'
    
    sec_peakz.format = '%.4f'
    sec_upper.format = '%.4f'
    sec_lower.format = '%.4f'
    sec_area.format = '%.3f'
            
    if verbose:
        bar = ProgressBar(len(pz))
    
    for i, pzi in enumerate(pz):
        peaks, l80s, u80s, areas = get_peak_z(pzi, zgrid)
        peaks = np.array(peaks, ndmin=1)
        l80s = np.array(l80s, ndmin=1)
        u80s = np.array(u80s, ndmin=1)
        areas = np.array(areas, ndmin=1)
        
        if np.isnan(peaks[0]):
            pri_peakz[i] = -99.
        else:
            pri_peakz[i] = peaks[0]
        
        pri_upper[i] = u80s[0]
        pri_lower[i] = l80s[0]
        pri_area[i] = areas[0]
        
        if len(peaks) > 1:
            sec_peakz[i] = peaks[1]
            sec_upper[i] = u80s[1]
            sec_lower[i] = l80s[1]
            sec_area[i] = areas[1]
        else:
            sec_peakz[i] = -99.
            sec_upper[i] = -99.
            sec_lower[i] = -99.
            sec_area[i] = -99.
    
    if verbose:
        bar.update()
            
    output.add_column(catalog['id'])
    output.add_column(pri_peakz)
    output.add_column(pri_lower)
    output.add_column(pri_upper)
    output.add_column(pri_area)

    output.add_column(sec_peakz)
    output.add_column(sec_lower)
    output.add_column(sec_upper)
    output.add_column(sec_area)        
    return output 
        
def HBpz(pzarr, zgrid, pzbad, beta=2., fbad_min=0.05, fbad_max=0.15, nbad = 6):
    fbad_range = np.linspace(fbad_min, fbad_max, nbad)
    pzarr_fbad = np.zeros((pzarr.shape[1], pzarr.shape[2], len(fbad_range)))
    
    for f, fbad in enumerate(fbad_range):
        #print fbad
        pzb = (fbad*pzbad) + (pzarr*(1-fbad))
        pzarr_fbad[:, :, f] = np.exp(np.sum(np.log(pzb)/beta, axis=0))


    pzarr_hb = np.trapz(pzarr_fbad, fbad_range, axis=2)
    pzarr_hb /= np.trapz(pzarr_hb, zgrid, axis=1)[:, None]
    return pzarr_hb   

def maybe_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)    
        
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

    #full_cat = Table.read('/data2/ken/bootes/mbrown_cat/Bootes_merged_Icorr_mbrown_ap4_specz.fits', format='fits')  
    #zspec = full_cat['z_spec']


    #photometry = Table.read('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog), 
    #                        format = pipe_params.photometry_format)

    #photom = photometry[photometry['z_spec'] >= 0.]

    photom = Table.read('{0}/testing/{1}'.format(pipe_params.working_folder, 'training_all.cat'),
                        format='ascii.commented_header')    


    AGN = (photom['AGN'] == 1)
    GAL = np.invert(AGN)
    
    z_max_name = ['z_a', 'z_1', 'z_1']
    titles = ['EAZY', 'ATLAS (Brown+)', 'XMM-COSMOS (Salvato+)']
    filt = 'I'
    
    for sbset in [GAL, AGN]:
        pzarr = []
        zouts = []
        alphas_median = []
        
        if (sbset == GAL).all():
            sbname = 'gal'
            alphas_init = [1.4, 2.5, 5.]
            fbad_max = 0.05
            fbad_min = 0.0
            lzc = 0.001
        else:
            sbname =  'agn'
            alphas_init = [5., 5., 5.]
            fbad_max = 0.5
            fbad_min = 0.3
            lzc = 0.0
            
        for itx, template in enumerate(pipe_params.templates):
            print(template)
            
            """ Load Values/Arrays/Catalogs """
            folder = '{0}/testing/all_specz/{1}'.format(pipe_params.working_folder, template)
            basename='training_all_with_zp.{0}'.format(template)
            pz, zgrid = getPz('{0}/{1}'.format(folder, basename))
            catalog = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')
            
            pz = pz[sbset]
            catalog = catalog[sbset]
            
            """ Split and Do Training """
            # ShuffleSplit into training and test
            rs = ShuffleSplit(len(catalog), 1, test_size = pipe_params.test_fraction)

            for i, (train_index, test_index) in enumerate(rs):
                zspec_train = catalog['z_spec'][train_index]
                zspec_test = catalog['z_spec'][test_index]
                
                pz_train = pz[train_index]
                pz_test = pz[test_index]
            
            ol1, ol2, bias = calcStats(catalog['z_peak'][train_index], catalog['z_spec'][train_index])
                
            # MCMC/Fit the best alpha to scale 
            print('Doing fits...')
            alphas_best, sampler = fitalphas(pz_train[ol1], zgrid, zspec_train[ol1], alphas_init[itx],
                                             nwalkers=10, nsamples=50, fburnin=0.1, nthreads = 10)
            
            print(alphas_best)
            bestfit = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
            print(bestfit)
            alphas_median.append(bestfit)
            
            alphas_best = bestfit
            
            
            pz_mod = pz_train**(1/alphas_best)
            mag_col = pipe_params.prior_colname
            bright = (photom[mag_col][sbset][train_index] >= 12.)*(photom[mag_col][sbset][train_index] < 20.)
            faint = (photom[mag_col][sbset][train_index] >= 20.)*(photom[mag_col][sbset][train_index] < 24.)

            ci_train_all, bins = pdf.calc_ci_dist(pz_mod, zgrid, zspec_train)
            ci_train_bright, bins = pdf.calc_ci_dist(pz_mod[bright], zgrid, zspec_train[bright])
            ci_train_faint, bins = pdf.calc_ci_dist(pz_mod[faint], zgrid, zspec_train[faint])

            
            ci_train_orig, bins = pdf.calc_ci_dist(pz_train, zgrid, zspec_train)
            
            ci_test_orig, bins = pdf.calc_ci_dist(pz_test, zgrid, zspec_test)
            ci_test_mod, bins = pdf.calc_ci_dist(pz_test**(1./alphas_best), zgrid, zspec_test)
            
            
            """
            Plots and Statistics for HB
            
            """
            Fig, Ax = plt.subplots(1, 2, figsize=(8.5, 4.5), sharey=True)
            
            Ax[0].plot(bins, ci_train_orig, '--', lw=2, color='firebrick', label='Train - Orig.')
            Ax[0].plot(bins, ci_train_all, lw=2, color='firebrick', label = 'Train - Smthd.')

            Ax[0].plot(bins, ci_test_orig, '--', lw=2, color='steelblue', label = 'Test - Orig.')
            Ax[0].plot(bins, ci_test_mod, lw=2, color='steelblue', label = 'Test - Smthd.')
            Ax[0].plot([0,1],[0,1], ':', color='k', lw=2)       
            
            leg1 = Ax[0].legend(loc='upper left', prop={'size':9})
            
            Ax[1].plot(bins, ci_train_all, 'k', label='All', lw=2)
            Ax[1].plot(bins, ci_train_faint, color='firebrick', label='Faint', lw=2)
            Ax[1].plot(bins, ci_train_bright, color='steelblue', label='Bright', lw=2)
            Ax[1].plot([0,1],[0,1], ':', color='k', lw=2)

            leg2 = Ax[1].legend(loc='upper left', prop={'size':9})

            for ax in Ax:
                ax.set_xlabel(r'$c$', size=12)
                ax.set_xlim([0., 1.])
                ax.set_ylim([0., 1.])
            Ax[0].set_ylabel(r'$\hat{F}(c)$', size=12)
            Fig.suptitle('{0} - PDF Calibration'.format(titles[itx]))            
            Fig.tight_layout()
            Fig.subplots_adjust(top=0.92)
            
            
            folder = '{0}/plots'.format(pipe_params.working_folder)
            maybemkdir(folder)
            plot_path = '{0}/pdf_calibration_{1}_{2}.pdf'.format(folder, sbname, template)
            Fig.savefig(plot_path, bbox_inches='tight', format='pdf')
            
            
            """
            Re-normalise and append
            """
            # Re-normalise PDFs        
            pz = pz**(1/alphas_best)
            pz /= np.trapz(pz, zgrid, axis=1)[:, None]
            
            pzarr.append(pz)
            zouts.append(catalog)
        

        zspec = zouts[0]['z_spec']
        pzarr = np.array(pzarr)


        """
        Define or Calculate U(z) - P(z) in case of bad fit:

        Options are 'flat', 'vol', 'mag'
            
        """
        
        if pipe_params.fbad_prior == 'flat':
            pzbad = np.ones_like(zgrid) # Flat prior
            pzbad /= np.trapz(pzbad, zgrid)
            
        elif pipe_params.fbad_prior == 'vol':
            pzbad = cosmo.differential_comoving_volume(zgrid)
            pzbad /= np.trapz(pzbad, zgrid) # Volume element prior (for our cosmology)

        elif pipe_params.fbad_prior == 'mag':
            prior_params = np.load('{0}/{1}_{2}_prior_coeff.npz'.format(pipe_params.working_folder, filt, sbname))
            z0t = prior_params['z0t']
            kmt1 = prior_params['kmt1']
            kmt2 = prior_params['kmt2']
            alpha = prior_params['alpha']  
            
            best = [z0t[0], kmt1[0], kmt2[0], alpha[0]]
            mags = np.array(photom[pipe_params.prior_colname])[sbset]
            
            pzbad = priors.pzl(zgrid, mags, *best, lzc=lzc)

            prior_params = np.load('{0}/{1}_{2}_prior_coeff.npz'.format(pipe_params.working_folder, filt, 'agn'))
            z0t = prior_params['z0t']
            kmt1 = prior_params['kmt1']
            kmt2 = prior_params['kmt2']
            alpha = prior_params['alpha']  
            
            best = [z0t[0], kmt1[0], kmt2[0], alpha[0]]
            mags = np.array(photom[pipe_params.prior_colname])[sbset]
            
            pzbad_agn = priors.pzl(zgrid, mags, *best, lzc=lzc)

        else:
            raise(ValueError('fbad parameter not recognised. Must be one of: flat/vol/mag'))

        pzbad_mag = np.copy(pzbad)

         

        rs = ShuffleSplit(len(catalog), 1, test_size = 0.8)
        for i, (train_index, test_index) in enumerate(rs):
            pz_train = pz[train_index]
            pz_test = pz[test_index]


        betas = np.linspace(1., 2.5, 21)
        ci_dists = np.zeros_like(betas)
        
        fbad_max = 1.
        
        for ia, beta in enumerate(betas):
            print(beta)
            if pipe_params.fbad_prior == 'mag':
                pzarr_hb = HBpz(pzarr[:,train_index], zgrid, pzbad[train_index], beta=beta, fbad_min=fbad_min, fbad_max=fbad_max)
            else:
                pzarr_hb = HBpz(pzarr[:,train_index], zgrid, pzbad, beta=beta, fbad_min=fbad_min, fbad_max=fbad_max)
                
            ci_hb, bins = pdf.calc_ci_dist(pzarr_hb, zgrid, zspec[train_index])
            ci_dists[ia] = -1*np.log(((ci_hb[:80]-bins[:80])**2).sum())
        
        fit = InterpolatedUnivariateSpline(betas, ci_dists, k=3)
        beta_fine = np.linspace(1., 3., 1001)
        beta_best = beta_fine[np.argmax(fit(beta_fine))]
        
        pzarr_hb = HBpz(pzarr, zgrid, pzbad, beta=beta_best, fbad_max = fbad_max, fbad_min=fbad_min)

        #coscat = pz_to_catalog(pzarr[1], zgrid, zouts[0])

        """
        Plots and Statistics for HB
        """

        ci_hb, bins = pdf.calc_ci_dist(pzarr_hb, zgrid, zspec)
        ci_eazy, bins = pdf.calc_ci_dist(pzarr[0], zgrid, zspec)
        ci_atlas, bins = pdf.calc_ci_dist(pzarr[1], zgrid, zspec)
        ci_cosmos, bins = pdf.calc_ci_dist(pzarr[2], zgrid, zspec)

        Fig, Ax = plt.subplots(1, 2, figsize=(8.5, 4.5), sharey=True)
        
        Ax[0].plot(bins, ci_eazy, '--', lw=2, color='firebrick', label='EAZY')
        Ax[0].plot(bins, ci_cosmos, '--', lw=2, color='steelblue', label='XMM-COSMOS')
        Ax[0].plot(bins, ci_atlas, '--', lw=2, color='olivedrab', label='ATLAS')
        Ax[0].plot(bins, ci_hb, lw=2, color='black', label = 'HB Combined')

        Ax[0].plot([0,1],[0,1], ':', color='k', lw=2)       
        
        leg1 = Ax[0].legend(loc='upper left', prop={'size':9})

        
        bright = np.array((photom['I_mag'] >= 12.)*(photom['I_mag'] < 20.))[sbset]
        faint = np.array((photom['I_mag'] >= 20.)*(photom['I_mag'] < 24.))[sbset]
        
        ci_hb_bright, bins = pdf.calc_ci_dist(pzarr_hb[bright], zgrid, zspec[bright])
        ci_hb_faint, bins = pdf.calc_ci_dist(pzarr_hb[faint], zgrid, zspec[faint])
     
        Ax[1].plot(bins, ci_hb, 'k', label='All', lw=2)
        Ax[1].plot(bins, ci_hb_faint, color='firebrick', label='Faint', lw=2)
        Ax[1].plot(bins, ci_hb_bright, color='steelblue', label='Bright', lw=2)
        Ax[1].plot([0,1],[0,1], ':', color='k', lw=2)

        leg2 = Ax[1].legend(loc='upper left', prop={'size':9})

        for ax in Ax:
            ax.set_xlabel(r'$c$', size=12)
            ax.set_xlim([0., 1.])
            ax.set_ylim([0., 1.])
        Ax[0].set_ylabel(r'$\hat{F}(c)$', size=12)
        Fig.suptitle('Hierarchical Bayesian Combination'.format(titles[itx]))            
        Fig.tight_layout()
        Fig.subplots_adjust(top=0.92)
        
        folder = '{0}/plots'.format(pipe_params.working_folder)
        plot_path = '{0}/pdf_calibration_HB_{1}.pdf'.format(folder, sbname)
        
        Fig.savefig(plot_path, bbox_inches='tight', format='pdf')

        #np.savez('temp_bak.npz', pzarr = pzarr, pzarr_hb = pzarr_hb)
        
        """
        Verify Stats are Improved as Expected
        """
        
        pz_argmax = np.argmax(pzarr_hb, axis=1)
        photoz = zgrid[pz_argmax]
        photomz = np.trapz(pzarr_hb*zgrid[None, :], zgrid, axis=1) 
        calcStats(photoz, zspec)
        calcStats(photomz, zspec)
        
        hbcat = pz_to_catalog(pzarr_hb, zgrid, zouts[0]) 
        ol1, ol2, _ = calcStats(hbcat['z1_median'], zspec)
        
        alternate = np.copy(hbcat['z1_median'])
        to_change = np.logical_and(np.invert(ol1), hbcat['z2_median'] >= 0.) 
        
        alternate[to_change] = hbcat['z2_median'][to_change]
        
        calcStats(alternate, zspec)

        catpath = '{0}/HB_{1}_calibration.cat'.format(folder, sbname)
        hbcat.write(catpath, format='ascii.commented_header')

        """
        t_suffix = ['a', '1', '1']
        best_fits = np.zeros((len(zspec), 3))
        reduced_chi = np.zeros((len(zspec), 3))
        for i, t in enumerate(t_suffix):
            rchi = zouts[i]['chi_{0}'.format(t)]/zouts[i]['nfilt']
            reduced_chi[:, i] = rchi
            best_fits[:,i] = zouts[i]['z_peak']
        
        weights = np.exp(-0.5*reduced_chi)
        weights /= weights.sum(1)[:,None]

        bf = np.ma.array(best_fits, mask=(reduced_chi > 4.))
        rc = np.ma.array(reduced_chi, mask=(reduced_chi > 4.))
        we = np.exp(-0.5*rc)
        weighted = np.average(best_fits, weights = we, axis=1)

        chi_sum = reduced_chi.sum(1)
        p95 = np.percentile(chi_sum, 95)
        """
        
        
        """
        Save Fitted Values to Dict
        """
        folder = '{0}/testing/all_specz'.format(pipe_params.working_folder)
        path = '{0}/HB_hyperparameters_{1}.npz'.format(folder, sbname)
        np.savez(path, alphas = alphas_median, beta = beta_best)


        def plot_pzs(idn):
            Fig, Ax = plt.subplots(1)
            
            Ax.plot(zgrid, pzarr[0][idn], label='EAZY')
            Ax.plot(zgrid, pzarr[1][idn], label='ATLAS')
            Ax.plot(zgrid, pzarr[2][idn], label='XMM-COSMOS')
        
            Ax.plot(zgrid, pzarr_hb[idn], lw=2, color='k')
            leg = Ax.legend(loc='upper right')
            
            stack = pzarr[0][idn]+pzarr[1][idn]+pzarr[2][idn]
            stack /= np.trapz(stack, zgrid)
            #Ax.plot(zgrid, stack, color='0.5', lw=2)
            
            ymin, ymax = Ax.get_ylim()
            Ax.plot(zgrid, pzbad[idn], color='0.5', lw=2)
            Ax.plot([zspec[idn], zspec[idn]], [0, ymax], '--', color='0.5')        
            peakz, zlo, zhi, area = get_peak_z(pzarr_hb[idn], zgrid)
            
            Ax.plot([peakz, peakz], [0, ymax], ':', color='olivedrab')

            pzc = find_ci_cut(pzarr_hb[idn], zgrid)
            pz_idx = (pzarr_hb[idn] > pzc)
            plt.fill_between(zgrid, pzarr_hb[idn], where=pz_idx,
                             color='0.3', alpha=0.3)
            
            Ax.set_xlim([0.0, 7.])
            #Ax.set_xscale('log')
            plt.show()





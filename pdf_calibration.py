import numpy as np
import array
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
from functools import partial
import multiprocessing as mp

# Scipy extras
from scipy.integrate import simps, cumtrapz, trapz
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp

# Astropy
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, Row, Column, join
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.stats import sigma_clipped_stats
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import astropy.units as u

import priors

def calcStats(photoz, specz, verbose=True):
    cut = np.logical_and(specz >= 0., photoz >= 0.)
    photoz = photoz[cut]
    specz = specz[cut]
    
    dz = photoz - specz
    
    
    
    sigma_all = np.sqrt( np.sum((dz/(1+specz))**2) / float(len(dz)))
    nmad = 1.48 * np.median( np.abs((dz - np.median(dz)) / (1+specz)))
    #nmad = 1.48 * np.median( np.abs(dz) / (1+specz))
    bias = np.median(dz/(1+specz))
    
    ol1 = (np.abs(dz)/(1+specz) > 0.2 )
    OLF1 = np.sum( ol1 ) / float(len(dz))
    sigma_ol1 = np.sqrt( np.sum((dz[np.invert(ol1)]/(1+specz[np.invert(ol1)]))**2) / float(len(dz[np.invert(ol1)])))
    
    ol2 = (np.abs(dz)/(1+specz) > 5*nmad )
    OLF2 = np.sum( ol2 ) / float(len(dz))
    sigma_ol2 = np.sqrt( np.sum((dz[np.invert(ol2)]/(1+specz[np.invert(ol2)]))**2) / float(len(dz[np.invert(ol2)])))
    KSscore = ks_2samp(specz, photoz)[0]
    
    if verbose:
        print('Sigma_all: {0:.3f}'.format(sigma_all))
        print('Sigma_NMAD: {0:.3f}'.format(nmad))
        print('Bias: {0:.3f}'.format(bias))
        
        print('OLF: Def1 = {0:.3f} Def2 = {1:0.3f}'.format(OLF1, OLF2))
        print('Sigma_OL: Def 1 = {0:.3f} Def2 = {1:0.3f}'.format(sigma_ol1, sigma_ol2))
        print('KS: {0:.3f}'.format(KSscore))

    return [sigma_all, nmad, bias, OLF1, sigma_ol1, OLF2, sigma_ol2, KSscore]

def func(a, b):
    return a + b

def hpd_ci_int(a , zgrid):
    ipz, i_zspec = a

    p_zspec = ipz[i_zspec]
    mask = (ipz > p_zspec)
    dz = np.diff(zgrid)[0]
    hpd_ci = np.sum(ipz[mask]*dz) / np.sum(ipz*dz)
    return hpd_ci #np.clip(hpd_ci, 0, 1)

def calc_HPDci_parallel(pz, zgrid, specz, nproc=4):
    interp_dz = 0.005
    interp_z            = np.arange( zgrid.min(), zgrid.max(), interp_dz)
    interp_pz_cube_fn   = interp1d( zgrid, pz, kind='linear', axis=1 )
    interp_pz_cube      = interp_pz_cube_fn( interp_z )

    i_zspec = np.argmin((np.abs(zspec[:,None] - zgrid[None,:])), axis=1)
    
    zdata = zip(interp_pz_cube, specz)
    pool = mp.Pool(nproc)
    CI = pool.map(partial(hpd_ci_int, zgrid=interp_z), zdata)
    pool.terminate()

    return CI

def calc_HPDciv(pz, zgrid, specz, dz = 0.005):
    dz = np.diff(zgrid[:2])

    i_zspec = np.argmin((np.abs(specz[:,None] - zgrid[None,:])), axis=1)
    pz_s = pz[np.arange(len(i_zspec)), i_zspec]
    mask = (pz < pz_s[:, None])
    ipz_masked = np.copy(pz)
    ipz_masked[mask] *= 0.

    CI = np.trapz(ipz_masked, zgrid, axis=1) / np.trapz(pz, zgrid, axis=1)

    return CI

def calc_HPDci(pz, zgrid, specz, dz = 0.005):
    interp_dz = dz
    interp_z            = np.arange( zgrid.min(), zgrid.max(), interp_dz)
    interp_pz_cube_fn   = interp1d( zgrid, pz, kind=2, axis=1 )
    interp_pz_cube      = interp_pz_cube_fn( interp_z )

    CI = np.zeros(len(specz))

    for gal, ipz in enumerate(interp_pz_cube):
        i_zspec = np.argmin(np.abs(specz[gal]-interp_z))
    
        p_zspec = ipz[i_zspec]
        mask = (ipz > p_zspec)
        
        hpd_ci = np.sum(ipz[mask]*interp_dz) / np.sum(ipz*interp_dz)
        CI[gal] = hpd_ci #np.clip(hpd_ci, 0, 1)
    
    #CI = np.minimum(CI, 1.)
    return CI
   
def calc_ci_dist(pz, zgrid, specz):
    ci_pdf = calc_HPDciv(pz, zgrid, specz)
    nbins = 100
    hist, bin_edges = np.histogram(ci_pdf, bins=nbins, range=(0,1), normed=True)
    cumhist = np.cumsum(hist)/nbins
    bin_max = 0.5*(bin_edges[:-1]+bin_edges[1:])
    
    return cumhist, bin_max

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
            if gal % 200 == 0:  print '\t{0}\t{1:1.3f}\t{2:1.3f}\t{3:1.3f}\t{4:1.3f}\t{5:1.4f}'.format(gal+1, gal_photoz, l68[-1], u68[-1],photoz_cumsum, intdiff[-1])

    return (l68, u68, intdiff)

def getpz(basepath, temp_filt='photz.tempfilt'):

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

        chiz = array.array('d')
        chiz.fromfile(f,NZ*NOBJ)
        chiz = np.reshape(chiz,(NOBJ,NZ))


    pofz = np.exp(-0.5*chiz)
    pofz /= np.trapz(pofz, zgrid)[:, None]
    
    return pofz, zgrid

def pz(z, m0, z0t, kmt1, kmt2, alpha):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    vol = cosmo.differential_comoving_volume(z).value / 39467332216.008575
    #vol = z**alpha
    cutoff = np.exp(-1*((z[None, :] / zmt[:, None])**alpha))
    px = vol[None,:]*cutoff
    
    zrange = np.append(0, np.logspace(-3, 1, 1000))
    zvol = cosmo.differential_comoving_volume(zrange).value / 39467332216.008575
    cutoff_all = np.exp(-1*((zrange[None,:] / zmt[:,None])**alpha))
    magnorm = zvol[None,:]*cutoff_all
    intgrl = np.trapz(magnorm, zrange, axis=1)
    
    return px/intgrl[:, None]

if __name__ == "__main__":
    
    phot = Table.read('/data2/ken/bootes/Bootes_merged_Icorr_2014a_all_ap4_mags.anom.testsample.cat', format='ascii.commented_header')
    base = '/data2/ken/bootes/v2/photz.eazy.zspec.n34.zp'
    pofz, zgrid = getpz(base, base+'.tempfilt')
    
    pz_argmax = np.argmax(pofz, axis=1)
    pz_peak = zgrid[pz_argmax]
    pz_median = np.trapz(pofz*zgrid, zgrid, axis=1)


    Iprior_c = np.load('bootes_I_prior_coeff.npz')
    z0t = Iprior_c['z0t'][0]
    kmt1 = Iprior_c['kmt1'][0]
    kmt2 = Iprior_c['kmt2'][0]
    alpha = Iprior_c['alpha'][0]
    
    prior_pz = pz(zgrid, phot['I_mag'], z0t, kmt1, kmt2, alpha)
    pofz_prior = pofz*prior_pz
    pofz_prior /= np.trapz(pofz_prior, zgrid, axis=1)[:, None]

    pz_peak_prior = zgrid[np.argmax(pofz_prior, axis=1)]
    pz_median_prior = np.trapz(pofz_prior*zgrid, zgrid, axis=1)

    zout = Table.read(base+'.zout', format='ascii.commented_header')
    photoz = zout['z_a']
    chi2=zout['chi_a']
    nfilt=zout['nfilt']
    specz = zout['z_spec']        

    cc = np.copy(pofz)
    cc[chi2 > np.percentile(chi2, 95)] = prior_pz[chi2 > np.percentile(chi2, 95)]

    pz_peak_prior = zgrid[np.argmax(cc, axis=1)]
    pz_median_prior = np.trapz(cc*zgrid, zgrid, axis=1)
    
    sample = (specz >= 0.)*(photoz >= 0.)*(chi2 < np.percentile(chi2, 98))*(phot['CLASS_STAR'] < 0.9)

    phot_sample = phot[sample]

    specz_sample = specz[sample]
    photoz_sample = photoz[sample]
    pofz_sample = pofz[sample]

    smoothing = np.linspace(0, 0.005, 6)
    alphas = np.linspace(0.9, 1.1, 11.)

    def modify_pz(pofz, zgrid, alpha, gwidth, offset):
        kernel = Gaussian1DKernel(gwidth / np.diff(zgrid)[0])
        
        if gwidth > 0:
            pofz_mod = np.array([convolve(pz, kernel) for pz in pofz**alpha])

        else:
            pofz_mod = pofz**alpha
        
        if np.abs(offset) > 0.:
            pofz_interp = interp1d( zgrid, pofz_mod, kind='linear', axis=1, fill_value=0., bounds_error=False)
            pofz_shift = pofz_interp( zgrid + offset )

        else:
            pofz_shift = pofz_mod        
                
        return pofz_shift
    
    def get_ci_stats(pofz, zgrid, specz):
        cumhist, bin_max = calc_ci_dist(pofz, zgrid, specz)
        edist = np.sum((cumhist[5:95]-bin_max[5:95])**2)
        return cumhist, bin_max, edist
    
    chist = np.zeros((len(smoothing), len(alphas), 100))
    fom = np.zeros((len(smoothing), len(alphas)))
    
    for ixs, gwidth in enumerate(smoothing):
        for ixa, alpha in enumerate(alphas):
            cumhist, bin_max = calc_ci_dist(modify_pz(pofz_sample, zgrid, alpha, gwidth, 0.), zgrid, specz_sample)
            edist = np.sum((cumhist[5:95]-bin_max[5:95])**2)

            chist[ixs, ixa] = cumhist
            fom[ixs, ixa] = edist
            print('Gaussian width = {0}, Alpha = {1}: {2:.3f}'.format(gwidth, alpha, edist)) 


    """    

    pofz_stack = np.nansum(pofz_sample, axis=0)/pofz_sample.shape[1]
    pofz_stack /= simps(pofz_stack, zgrid)

    cumhist, bin_max = calc_ci_dist(pofz_sample, zgrid, specz_sample)
    cumhist_mod, bin_max = calc_ci_dist(pofz_sample**1.2, zgrid, specz_sample)
    cumhist_gauss, bin_max = calc_ci_dist(np.array([convolve(pz, kernel) for pz in pofz_sample]),
                                          zgrid, specz_sample)

    pofz_interp = interp1d( zgrid, pofz_sample, kind='linear', axis=1, fill_value=0., bounds_error=False)
    pofz_shift = pofz_interp( zgrid + 0.02 )
    cumhist_shift, bin_max = calc_ci_dist(pofz_shift, zgrid, specz_sample)
            
    plt.plot(bin_max, cumhist)
    plt.plot(bin_max, cumhist_mod)
    plt.plot(bin_max, cumhist_shift)
    plt.plot(bin_max, cumhist_gauss)
    plt.plot(bin_max, bin_max, '--', color='k')
    plt.show()
    
    
    if c50 > 0.5:
        iterations = 0
        while c50 > 0.5:
            tmp_pz = np.copy(pofz_sample)
            interp_dz = 0.005
            
            iterations += 1

            for gal in range(tmp_pz.shape[0]):
                kernel = Gaussian1DKernel((iterations*0.001) / np.diff(zgrid)[0])
                tmp_pz[gal] = convolve(tmp_pz[gal],kernel)
                
            # Now calculate the upper and lower confidence limits
            ci = calc_HPDci(tmp_pz, zgrid, specz_sample)
            c50 = np.percentile(ci[ci <= 1.], 50)
            
            if iterations % 10 == 0:
            # Print to terminal
                print '\t{0:>4d}\t{1:>1.5f}'.format(iterations, c50)
            # If the confidence intervals are okay, break here
            #if (abs( in_pdf_1sigma - 0.6827) < 5e-4) or (in_pdf_1sigma - 0.6827 < 0.): break
        print '\t Final iteration: {2:>4d} {0:>4d}\t{1:>1.5f}'.format(iterations, c50, len(ci))
    """


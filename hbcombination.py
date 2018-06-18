
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
from scipy.stats import ks_2samp, norm

# Astropy
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from astropy.utils.console import ProgressBar
from astropy.visualization import LinearStretch, MinMaxInterval

import pickle
import emcee
import GPz

import priors
import pdf_calibration as pdf

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

import pickle

def save_gp(path, gp, bands, alpha_values, scale_band):
    out = {'gp': gp,
           'bands': bands,
           'alpha': alpha_values,
            'scale_band': scale_band}

    pickle.dump(file=open(path, "wb"), obj=out)

def load_gp(path):
    out = pickle.load(open(path, "rb"))

    gp = out['gp']
    bands = out['bands']
    alpha_values = out['alpha']
    try:
        scale_band = out['scale_band']
        return gp, bands, alpha_values, scale_band
    except:
        return gp, bands, alpha_values

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
    cut = np.logical_and(photoz >= 0, specz > 0.00)
    print('NGD: {0}'.format(cut.sum()))
    dz = photoz - specz

    ol1 = (np.abs(dz)/(1+specz) > 0.2)
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


def lnprior(mags, theta):
    intcp, slope = theta
    alphas = alphas_mag(mags, intcp, slope)

    if 0.5 < intcp < 50 and 0. < slope < 2. and (alphas > 0.).all():
        return 0.0
    return -np.inf

def lnlike(theta, mags, pz, zgrid, zspec):
    intcp, slope = theta
    ci, bins = pdf.calc_ci_dist(pz**(1/alphas_mag(mags, intcp, slope)[:, None]), zgrid, zspec)
    ci_dist = -1*np.log((ci[:80]-bins[:80])**2).sum()
    return ci_dist

def lnprob(theta, mags, pz, zgrid, zspec):
    lp = lnprior(mags, theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, mags, pz, zgrid, zspec)

def lnlikep(theta, mags, pz, pzprior, zgrid, zspec):
    intcp, slope = theta
    ci, bins = pdf.calc_ci_dist(pz**(1/alphas_mag(mags, intcp, slope)[:, None]) * pzprior,
                                zgrid, zspec)
    ci_dist = -1*np.log((ci[:80]-bins[:80])**2).sum()
    return ci_dist

def lnprobp(theta, mags, pz, pzprior, zgrid, zspec):
    lp = lnprior(mags, theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlikep(theta, mags, pz, pzprior, zgrid, zspec)

def alphas_mag(mags, intcp, slope, base = 16.):
    alt_mags = np.clip(mags, base, 35.)
    alphas = intcp + (alt_mags - base)*slope
    return alphas

def fitalphas(mags, pz, zgrid, zspec, alpha_start,
              nwalkers=10, nsamples=10, fburnin=0.1, nthreads = 10):
    """ Fit prior functional form to observed dataset with emcee

    """
    ndim = 2
    burnin = int(nsamples*fburnin)
    # Set up the sampler.
    pos = [np.array(alpha_start) + 0.5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(mags, pz, zgrid, zspec),
                                    threads=nthreads)


    # Clear and run the production chain.
    sampler.run_mcmc(pos, nsamples, rstate0=np.random.get_state())
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    i50, i16, i84 = np.percentile(samples[:,0], [50, 16, 84])
    s50, s16, s84 = np.percentile(samples[:,1], [50, 16, 84])
    return sampler, i50, s50

def fitalphasp(mags, pz, pzprior, zgrid, zspec, alpha_start,
              nwalkers=10, nsamples=10, fburnin=0.1, nthreads = 10):
    """ Fit prior functional form to observed dataset with emcee

    """
    ndim = 2
    burnin = int(nsamples*fburnin)
    # Set up the sampler.
    pos = [np.array(alpha_start) + 0.5*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobp,
                                    args=(mags, pz, pzprior, zgrid, zspec),
                                    threads=nthreads)


    f = open('achain.dat', 'w')
    f.close()

    with ProgressBar(nsamples) as bar:
        for result in sampler.sample(pos, iterations=nsamples, storechain=True):
            position = result[0]
            f = open('achain.dat', 'a')
            for k in range(position.shape[0]):
                f.write("{0:4d} {1}\n".format(k, " ".join(np.array(position[k]).astype('str'))))
            bar.update()

        #sampler.run_mcmc(pos, nsamples, rstate0=np.random.get_state())
        print("Done.")


    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    i50, i16, i84 = np.percentile(samples[:,0], [50, 16, 84])
    s50, s16, s84 = np.percentile(samples[:,1], [50, 16, 84])
    return sampler, i50, s50

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

def make_data(catalog, columns):
    X = []
    for col in columns:
        X.append(catalog['{0}Mag'.format(col)])

    for col in columns:
        X.append(catalog['{0}MagErr'.format(col)])

    X = np.array(X).T
    Y = np.array(catalog['z_spec'].data)
    to_keep = (((X > 0.) * np.isfinite(X)).sum(1) == (2*len(columns)))
    X = X[to_keep]

    n,d = X.shape
    filters = d/2

    Y = Y[to_keep].reshape(n, 1)
    # log the uncertainties of the magnitudes, any additional preprocessing should be placed here
    X[:, filters:] = np.log(X[:, filters:])

    return X, Y, filters, n, to_keep

def make_data(catalog, columns):
    X = []
    for col in columns:
        X.append(catalog['{0}_mag'.format(col)])

    for col in columns:
        X.append(catalog['{0}_magerr'.format(col)])

    X = np.array(X).T
    Y = np.array(catalog['z_spec'].data)
    to_keep = (((X > 0.) * np.isfinite(X)).sum(1) == (2*len(columns)))
    X = X[to_keep]

    n,d = X.shape
    filters = d/2

    Y = Y[to_keep].reshape(n, 1)
    # log the uncertainties of the magnitudes, any additional preprocessing should be placed here
    X[:, filters:] = np.log(X[:, filters:])

    return X, Y, filters, n, to_keep


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

    photom = photom.filled(fill_value=-99)

    AGN = (photom['AGN'] == 1)
    if pipe_params.gpz:
        try:
            IRAGN = (photom['IRClass'] >= 4)
            XR = (photom['XrayClass'] == 1)
            OPT = (photom['mqcAGN'] == 'True')
        except:
            pass
    GAL = np.invert(AGN)

    z_max_name = ['z_a', 'z_1', 'z_1']
    titles = ['EAZY', 'ATLAS (Brown+)', 'XMM-COSMOS (Salvato+)']
    filt = pipe_params.prior_fname

    for sbset in [AGN]:
        pzarr = []
        zouts = []
        alphas_fitted = []

        if (sbset == GAL).all():
            include_prior = pipe_params.include_prior_gal
            sbname = 'gal'
            alphas_init = [[1.5, 0.9], [2.5, 0.9], [5., 0.9]]
            fbad_max = 0.05
            fbad_min = 0.0
            lzc = 0.003
        else:
            include_prior = pipe_params.include_prior_agn
            sbname =  'agn'
            alphas_init = [[2., 0.9], [2., 0.9], [5., 0.9]]
            fbad_max = 0.2
            fbad_min = 0.
            lzc = 0.0

        pzarr_all = []
        catalog_all = []

        for itx, template in enumerate(pipe_params.templates[:]):
            print(template)

            """ Load Values/Arrays/Catalogs """
            folder = '{0}/testing/all_specz/{1}'.format(pipe_params.working_folder, template)
            basename='training_all_with_zp.{0}'.format(template)
            pz, zgrid = getPz('{0}/{1}'.format(folder, basename))
            catalog = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')

            check_id = np.array([i in catalog['id'] for i in photom['id']])
            photom = photom[check_id]
            GAL = GAL[check_id]
            AGN = AGN[check_id]

            pzarr_all.append(pz)
            catalog_all.append(catalog)

            if itx == 0:
                prior_params = np.load('{0}/{1}_{2}_prior_coeff.npz'.format(pipe_params.working_folder, filt, sbname))
                z0t = prior_params['z0t']
                kmt1 = prior_params['kmt1']
                kmt2 = prior_params['kmt2']
                alpha = prior_params['alpha']

                best = [z0t[0], kmt1[0], kmt2[0], alpha[0]]
                mags = np.array(photom[pipe_params.prior_colname])


                if include_prior:
                    pzprior = priors.pzl(zgrid, mags, *best, lzc=lzc)
                    pzprior_nomag = np.ones_like(zgrid)
                    pzprior_nomag /= np.trapz(pzprior_nomag, zgrid)
                    pzprior[mags < -90.] = pzprior_nomag
                else:
                    pzprior = np.ones_like(pz)


            mcut = (photom[pipe_params.prior_colname] > 0.) * (catalog_all[0]['nfilt'] > 0)
            pz = pz[sbset*mcut]
            catalog = catalog[sbset*mcut]

            magsc = photom[pipe_params.prior_colname][sbset*mcut]
            edges = [16, 17, 18, 19, 20, 21, 22]

            for ie in np.arange(len(edges)-1):
                mslice = np.logical_and(magsc > edges[ie], magsc < edges[ie+1])
                idslice = np.where(mslice)[0]


                cut = ShuffleSplit(len(idslice), 1, train_size=np.minimum(len(idslice), 3000)-1, test_size=1)
                for i, (sb, sbb) in enumerate(cut):
                    if ie == 0:
                        idsubset = idslice[sb]
                    else:
                        idsubset = np.append(idsubset, idslice[sb])




            """ Split and Do Training """
            # ShuffleSplit into training and test
            rs = ShuffleSplit(len(idsubset), 1, test_size = 0.33) #pipe_params.test_fraction)

            for i, (train_index, test_index) in enumerate(rs):
                zspec_train = catalog['z_spec'][idsubset][train_index]
                zspec_test = catalog['z_spec'][idsubset][test_index]

                pz_train = pz[idsubset][train_index]
                pz_test = pz[idsubset][test_index]

            pza_prior = zgrid[np.argmax(pz*pzprior[sbset*mcut], axis=1)]

            ol1, ol2, bias = calcStats(pza_prior[train_index], catalog['z_spec'][train_index])

            mag_col = pipe_params.alpha_colname #pipe_params.prior_colname
            mags = photom[mag_col][sbset*mcut][idsubset][train_index]

            # MCMC/Fit the best alpha to scale


            print('Doing fits...')
            sampler, imed, smed = fitalphasp(mags, pz_train,
                                            pzprior[sbset*mcut][idsubset][train_index],
                                            zgrid, zspec_train, alphas_init[itx],
                                            nwalkers=20, nsamples=1000, fburnin=0.2, nthreads = pipe_params.ncpus)


            #bestfit = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
            print('{0} {1}'.format(imed, smed))

            #alphas_best = bestfit
            best_idx = np.argmax(sampler.flatlnprobability)
            ibest, sbest = sampler.flatchain[best_idx]
            print('{0} {1}'.format(ibest, sbest))
            alphas_fitted.append([ibest, sbest])

            alphas_best = alphas_mag(photom[mag_col][sbset*mcut][idsubset][train_index],
                                     imed, smed)[:, None]
            pz_mod = pz_train**(1/(alphas_best)) * pzprior[sbset*mcut][idsubset][train_index]
            za_mod_train = zgrid[np.argmax(pz_mod, axis=1)]





            bright = (photom[mag_col][sbset*mcut][idsubset][train_index] >= 19.)*(photom[mag_col][sbset*mcut][idsubset][train_index] < 21.)
            faint = (photom[mag_col][sbset*mcut][idsubset][train_index] >= 21.)*(photom[mag_col][sbset*mcut][idsubset][train_index] < 23.)

            ci_train_all, bins = pdf.calc_ci_dist(pz_mod, zgrid, zspec_train)
            #ci_train_bright, bins = pdf.calc_ci_dist(pz_mod[bright], zgrid, zspec_train[bright])
            #ci_train_faint, bins = pdf.calc_ci_dist(pz_mod[faint], zgrid, zspec_train[faint])


            ci_train_orig, bins = pdf.calc_ci_dist(pz_train, zgrid, zspec_train)

            ci_test_orig, bins = pdf.calc_ci_dist(pz_test, zgrid, zspec_test)
            ci_test_mod, bins = pdf.calc_ci_dist(pz_test**(1./alphas_mag(photom[mag_col][sbset*mcut][idsubset][test_index], ibest, sbest)[:, None]) * pzprior[sbset*mcut][idsubset][test_index],
            zgrid, zspec_test)

            pz_mod_test = (pz_test**(1./alphas_mag(photom[mag_col][sbset*mcut][idsubset][test_index], ibest, sbest)[:, None]) * pzprior[sbset*mcut][idsubset][test_index])
            za_mod_test = zgrid[np.argmax(pz_mod_test, axis=1)]

            ol1, ol2, bias = calcStats(za_mod_test, zspec_test)
            ci_test_mod2, bins = pdf.calc_ci_dist(pz_mod_test, zgrid, zspec_test)


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

            magrange = np.arange(17., 23., 1.)
            magcols = plt.cm.viridis(MinMaxInterval()(magrange))
            for ic, magc in enumerate(magrange):
                sample = (photom[mag_col][sbset*mcut][train_index] >= magc-1.)*(photom[mag_col][sbset*mcut][train_index] < magc+1.)
                ci_train_mag, bins = pdf.calc_ci_dist(pz_train[sample], zgrid, zspec_train[sample])
                Ax[1].plot(bins, ci_train_mag, color=magcols[ic], ls='--', lw=2)

                ci_train_mag, bins = pdf.calc_ci_dist(pz_mod[sample], zgrid, zspec_train[sample])
                Ax[1].plot(bins, ci_train_mag, color=magcols[ic], lw=2)

            #Ax[1].plot(bins, ci_train_faint, color='firebrick', label='Faint', lw=2)
            #Ax[1].plot(bins, ci_train_bright, color='steelblue', label='Bright', lw=2)
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

            alphas_best = alphas_mag(photom[mag_col][sbset*mcut], ibest, sbest)[:, None]
            pz = pz**(1/alphas_best) * pzprior[sbset*mcut]
            pz /= np.trapz(pz, zgrid, axis=1)[:, None]

            pzarr.append(pz)
            zouts.append(catalog)

        zspec = photom['z_spec'][sbset*mcut]
        #pzarr = np.array(pzarr)

        pz_back = np.copy(pzarr)


        pzarr = list(pzarr)
        if pipe_params.gpz:

            sets = []
            bands = []
            alphas = []
            gpz = []
            scale_bands = []

            AGN = (photom['AGN'][sbset*mcut] == 1)

            if pipe_params.ir_gpz_path != None:
                IRAGN = (photom['IRClass'][sbset*mcut] >= 4)
                gp_ir, bands_ir, alpha_values_ir = load_gp('{0}'.format(pipe_params.ir_gpz_path))

                sets.append(IRAGN)
                bands.append(bands_ir)
                alphas.append(alpha_values_ir)
                gpz.append(gp_ir)

            if pipe_params.xray_gpz_path != None:
                XR = (photom['XrayClass'][sbset*mcut] == 1)
                gp_xray, bands_xray, alpha_values_xray = load_gp('{0}'.format(pipe_params.xray_gpz_path))

                sets.append(XR)
                bands.append(bands_xray)
                alphas.append(alpha_values_xray)
                gpz.append(gp_xray)

            if pipe_params.opt_gpz_path != None:
                try:
                    OPT = (photom['mqcAGN'][sbset*mcut] == 'True')
                except:
                    OPT = AGN

                gp_opt, bands_opt, alpha_values_opt, sc = load_gp('{0}'.format(pipe_params.opt_gpz_path))

                sets.append(OPT)
                bands.append(bands_opt)
                alphas.append(alpha_values_opt)
                gpz.append(gp_opt)
                scale_bands.append(sc)

            GAL = np.invert(AGN)


            if pipe_params.gal_gpz_paths != None:
                for path in pipe_params.gal_gpz_paths:
                    try:
                        g, b, a, sc = load_gp('{0}'.format(path))
                        scale_bands.append(sc)
                    except:
                        g, b, a = load_gp('{0}'.format(path))

                    sets.append(GAL)
                    bands.append(b)
                    alphas.append(a)
                    gpz.append(g)


            for ix, s in enumerate(sets):
                pz = np.ones((len(photom[sbset*mcut]), len(zgrid)))/zgrid.max()
                if s.sum() > 0:
                    X, Y, _, _, K = make_data(photom[sbset*mcut][s], bands[ix])

                    if K.sum() >= 1:
                        mu, sigma, modelV, noiseV, _ = gpz[ix].predict(X.copy())

                        mags = photom[sbset*mcut][scale_bands[ix]]
                        sigma *= alphas_mag(mags[s][K],
                                            *alphas[ix]).reshape(sigma.shape)
                        # if ix ==3:
                        #    sigma *=2

                        pz_gp = []

                        with ProgressBar(len(mu), ipython_widget=False) as bar:
                            for iz, z in enumerate(mu):
                                gaussian = norm(loc=mu[iz], scale=sigma[iz])
                                pz_gp.append(gaussian.pdf(zgrid))
                                bar.update()
                            pz[np.where(s)[0][K]] = pz_gp

                pzarr.append(pz)

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
            mags = np.array(photom[pipe_params.prior_colname])[sbset*mcut]

            pzbad = priors.pzl(zgrid, mags, *best, lzc=lzc)

            pzbad_nomag = np.ones_like(zgrid)
            pzbad_nomag /= np.trapz(pzbad_nomag, zgrid)
            pzbad[mags < -90.] = pzbad_nomag


        else:
            raise(ValueError('fbad parameter not recognised. Must be one of: flat/vol/mag'))

        pzbad_mag = np.copy(pzbad)

        #pzbad = pzbad[ca]

        rs = ShuffleSplit(len(catalog), 1, test_size = 0.9)
        for i, (train_index, test_index) in enumerate(rs):
            pz_train = pz[train_index]
            pz_test = pz[test_index]


        betas = np.linspace(1., len(pzarr[:]), 21)
        ci_dists = np.zeros_like(betas)

        for ia, beta in enumerate(betas):
            print(beta)
            if pipe_params.fbad_prior == 'mag':
                pzarr_hb = HBpz(pzarr[:,train_index], zgrid, pzbad[train_index], beta=beta, fbad_min=fbad_min, fbad_max=fbad_max)
            else:
                pzarr_hb = HBpz(pzarr[:,train_index], zgrid, pzbad, beta=beta, fbad_min=fbad_min, fbad_max=fbad_max)

            ci_hb, bins = pdf.calc_ci_dist(pzarr_hb, zgrid, zspec[train_index])
            ci_dists[ia] = -1*np.log(((ci_hb[:80]-bins[:80])**2).sum())

        fit = InterpolatedUnivariateSpline(betas, ci_dists, k=3)
        beta_fine = np.linspace(1., len(pzarr[:]), 1001)
        beta_best = beta_fine[np.argmax(fit(beta_fine))]
        print(beta_best)

        pzarr_hb = HBpz(pzarr[:], zgrid, pzbad, beta=beta_best, fbad_max = fbad_max, fbad_min=fbad_min)

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

        catpath = '{0}/HB_{1}_{2}_calibration.cat'.format(folder, sbname, pipe_params.prior_fname)
        hbcat.write(catpath, format='ascii.commented_header', overwrite=True)


        """
        Save Fitted Values to Dict
        """
        folder = '{0}/testing/all_specz'.format(pipe_params.working_folder)
        path = '{0}/HB_hyperparameters_{1}_{2}.npz'.format(folder, sbname, pipe_params.prior_fname)
        np.savez(path, alphas = alphas_fitted, beta = beta_best)

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


        bright = np.array((photom[mag_col] >= 12.)*(photom[mag_col] < 20.))[sbset*mcut]
        faint = np.array((photom[mag_col] >= 20.)*(photom[mag_col] < 24.))[sbset*mcut]

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
        plot_path = '{0}/pdf_calibration_HB_{1}_{2}.pdf'.format(folder, sbname,
                                                                pipe_params.prior_fname)

        Fig.savefig(plot_path, bbox_inches='tight', format='pdf')





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





        def plot_pzs(idn):
            Fig, Ax = plt.subplots(1)

            Ax.plot(zgrid, pzarr[0][idn], label='EAZY')
            Ax.plot(zgrid, pzarr[1][idn], label='ATLAS')
            Ax.plot(zgrid, pzarr[2][idn], label='XMM-COSMOS')

            Ax.plot(zgrid, pzarr[-3][idn], label='riz')
            Ax.plot(zgrid, pzarr[-2][idn], label='grizy')
            Ax.plot(zgrid, pzarr[-1][idn], label='grizy w1')

            Ax.plot(zgrid, pzarr_hb[idn], lw=2, color='k')
            leg = Ax.legend(loc='upper right')

            stack = pzarr[0][idn]+pzarr[1][idn]+pzarr[2][idn]
            stack /= np.trapz(stack, zgrid)
            #Ax.plot(zgrid, stack, color='0.5', lw=2)

            ymin, ymax = Ax.get_ylim()
            #Ax.plot(zgrid, pzbad[idn], color='0.5', lw=2)
            Ax.plot([zspec[idn], zspec[idn]], [0, ymax], '--', color='0.5')
            peakz, zlo, zhi, area = get_peak_z(pzarr_hb[idn], zgrid)

            #Ax.plot([peakz, peakz], [0, ymax], ':', color='olivedrab')

            pzc = find_ci_cut(pzarr_hb[idn], zgrid)
            pz_idx = (pzarr_hb[idn] > pzc)
            plt.fill_between(zgrid, pzarr_hb[idn], where=pz_idx,
                             color='0.3', alpha=0.3)

            Ax.set_xlim([0.0, 7.])
            #Ax.set_xscale('log')
            plt.show()

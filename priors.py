import numpy as np
import emcee

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def zmt(m0, z0t, kmt1, kmt2):
    return np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    
def pz(z, m0, z0t, kmt1, kmt2, alpha):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    vol = cosmo.differential_comoving_volume(z).value / 39467332216.008575
    #vol = z**alpha
    cutoff = np.exp(-1*((z[None, :] / zmt[:, None])**alpha))
    px = vol*cutoff
    
    zrange = np.append(0, np.logspace(-3, 1, 1000))
    zvol = cosmo.differential_comoving_volume(zrange).value / 39467332216.008575
    cutoff_all = np.exp(-1*((zrange[None,:] / zmt[:,None])**alpha))
    magnorm = zvol[None,:]*cutoff_all
    intgrl = np.trapz(magnorm, zrange, axis=1)
    
    return np.squeeze(px/intgrl[:, None])

def pzv(z, m0, z0t, kmt1, kmt2, alpha):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    vol = cosmo.differential_comoving_volume(z).value / 39467332216.008575
    #vol = z**alpha
    cutoff = np.exp(-1*((z / zmt)**alpha))
    px = vol*cutoff
    
    zrange = np.append(0, np.logspace(-3, 1, 1000))
    zvol = cosmo.differential_comoving_volume(zrange).value / 39467332216.008575
    cutoff_all = np.exp(-1*((zrange[None,:] / zmt[:,None])**alpha))
    magnorm = zvol[None,:]*cutoff_all
    intgrl = np.trapz(magnorm, zrange, axis=1)
    
    return px/intgrl


def lnprior(theta, m):
    z0t, kmt1, kmt2, alpha = theta

    if 0. < z0t < 3. and -2. < kmt1 < 2. and -2. < kmt2 < 2. and -3. < alpha < 3.0:
        return 0.0
    return -np.inf

def lnlike(theta, z, m):
    z0t, kmt1, kmt2, alpha = theta
    model = pz(z, m, z0t, kmt1, kmt2, alpha)
    return np.nansum(np.log(model))

def lnprob(theta, z, m):
    lp = lnprior(theta, m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, z, m)

def fitpriors(catalog_path, catalog_format, mag_col, z_col, 
              nwalkers=10, nsamples=10000, fburnin=0.1, nthreads = 6):
    """ Fit prior functional form to observed dataset with emcee
    
    """
    ndim = 4
    burnin = int(nsamples*fburnin)

    data = Table.read(catalog_path, format=catalog_format)

    cut = (data[z_col] > 0.)*(data[mag_col] > 0.) # Cut -99s etc
    data = data[cut][::2]

    zdata = data[z_col].data
    mdata = data[mag_col].data

    start = np.array([0.13, -0.09, 0.03, 1.126])
    
    # Set up the sampler.
    pos = [start + 0.001*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zdata, mdata), threads=nthreads)

    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsamples, rstate0=np.random.get_state())
    print("Done.")

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))


    # Compute the quantiles.
    z0t_mcmc, kmt1_mcmc, kmt2_mcmc, alpha_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                     zip(*np.percentile(samples, [16, 50, 84],
                                                                        axis=0)))

    quantiles = [z0t_mcmc, kmt1_mcmc, kmt2_mcmc, alpha_mcmc]
    best = [z0t_mcmc[0], kmt1_mcmc[0], kmt2_mcmc[0], alpha_mcmc[0]]
    return best, quantiles, samples, sampler


if __name__ == '__main__':
    """
    Input Variables
    
    """
    catalog_path = '/data2/ken/bootes/Bootes_merged_Icorr_2014a_all_ap4_mags.testsample.cat'
    catalog_format = 'ascii.commented_header'


    mags = ['ch1']
    #['I', 'Ks', 'ch1']
    
    for filt in mags:
        print('{0} Prior Fitting:'.format(filt))
        mag_col = '{0}_mag'.format(filt)
        z_col = 'z_spec'

        best, fits, samples, sampler = fitpriors(catalog_path, catalog_format, mag_col, z_col, nsamples=10000, nthreads=8)
        z0t, kmt1, kmt2, alpha = fits

        coeff_save_path = 'bootes_{0}_prior_coeff.npz'.format(filt)
        np.savez(coeff_save_path, z0t=z0t, kmt1=kmt1, kmt2=kmt2, alpha=alpha)



    """
    Useful Plots
    """
    do_plots = False
    
    if do_plots:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
        axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
        axes[0].yaxis.set_major_locator(MaxNLocator(5))

        axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
        axes[1].yaxis.set_major_locator(MaxNLocator(5))

        axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
        axes[2].yaxis.set_major_locator(MaxNLocator(5))

        axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
        axes[3].yaxis.set_major_locator(MaxNLocator(5))
        axes[3].set_xlabel("step number")

        fig.tight_layout(h_pad=0.0)


        zrange = np.linspace(0,4,1000)

        mrange = np.arange(21, 28, 1)
        data = Table.read(catalog_path, format='ascii.commented_header')

        cut = (data[z_col] > 0.)
        #data = data[cut][::10]
        data = data[data[z_col] >= 0.]

        zdata = data[z_col].data
        mdata = data[mag_col].data

        Fig, Ax = plt.subplots(1, figsize=(5,3.5))
        obs_peaks = []

        for m in mrange:
            pzm = pz(zrange, m, *best)
            pzm /= np.trapz(pzm, zrange)
            Ax.plot(zrange, pzm, label='{0}'.format(m), color='k')
            
            #cut = np.logical_and(mdata < m+0.5, mdata > m-0.5)
            #pk = np.percentile(zdata[cut], 50)
            #obs_peaks.append(pk)
            #print('{0} {1}'.format(m, pk))
            #Ax.hist(zdata[cut], normed=True, histtype='step', bins=51, range=(0,4))
            
        #Leg = Ax.legend(loc='upper right')
        plt.show()



    # NB Linear zmt evolution doesnt fit observed evolution in peak
    # Power-law fit looks a lot more in line. Re-cutting input sample down spec-z complete magnitudes to see if that
    # reduces biases
    # Linear best: [-0.1869, 0.0690, 1.126]




#a = pz(zdata, mdata, *[ 0.06891682, -0.05053156,  0.00889756,  1.17581103])


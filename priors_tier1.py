import numpy as np
import emcee
import os, re
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.utils.console import ProgressBar
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.visualization import MinMaxInterval

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--params", type=str,
                    help = "Parameter file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()
quiet = args.quiet

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def zmt(m0, z0t, kmt1, kmt2):
    return np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))

def pzf(z, m0, z0t, kmt1, kmt2, alpha1, lzc):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    vol = z**alpha1
    cutoff = np.exp(-1*((z[None, :] / zmt[:, None])**alpha1))
    px = (vol+lzc)*cutoff
    
    zrange = np.append(0, np.logspace(-3, 1, 1000))
    zvol = zrange**alpha1
    cutoff_all = np.exp(-1*((zrange[None,:] / zmt[:,None])**alpha1))
    magnorm = (zvol[None,:]+lzc)*cutoff_all
    intgrl = np.trapz(magnorm, zrange, axis=1)
    
    return np.squeeze(px/intgrl[:, None])    
    
def pz(z, m0, z0t, kmt1, kmt2, alpha):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    vol = cosmo.differential_comoving_volume(z).value / 39467332216.008575
    #vol = z**alpha
    cutoff = np.exp(-1*((z[None, :] / zmt[:, None])**alpha))
    px = (vol)*cutoff
    
    zrange = np.append(0, np.logspace(-3, 1, 1000))
    zvol = cosmo.differential_comoving_volume(zrange).value / 39467332216.008575
    cutoff_all = np.exp(-1*((zrange[None,:] / zmt[:,None])**alpha))
    magnorm = (zvol[None,:])*cutoff_all
    intgrl = np.trapz(magnorm, zrange, axis=1)
    
    return np.squeeze(px/intgrl[:, None])

def pzl(z, m0, z0t, kmt1, kmt2, alpha, lzc):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(1e-10, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
    vol = cosmo.differential_comoving_volume(z).value / 39467332216.008575
    #vol = z**alpha
    cutoff = np.exp(-1*((z[None, :] / zmt[:, None])**alpha))
    px = (vol+lzc)*cutoff
    
    zrange = np.append(0, np.logspace(-3, 1, 1000))
    zvol = cosmo.differential_comoving_volume(zrange).value / 39467332216.008575
    cutoff_all = np.exp(-1*((zrange[None,:] / zmt[:,None])**alpha))
    magnorm = (zvol[None,:]+lzc)*cutoff_all
    intgrl = np.trapz(magnorm, zrange, axis=1)
    
    return np.squeeze(px/intgrl[:, None])

def pzv(z, m0, z0t, kmt1, kmt2, alpha):
    m0 = np.array(m0, ndmin=1)
    zmt = np.maximum(0.03, z0t + kmt1*(m0-14.) + kmt2*((m0-14.)**2))
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
    z0t, kmt1, kmt2, alpha1 = theta

    if 0. < z0t < 1. and -0.5 < kmt1 < 0.5 and -0.5 < kmt2 < 0.5 and -3. < alpha1 < 3.0:
        return 0.0
    return -np.inf

def lnlike(theta, z, m):
    z0t, kmt1, kmt2, alpha1 = theta
    model = pzv(z, m, z0t, kmt1, kmt2, alpha1)
    like = np.sum(np.log(model))
    if not np.isfinite(like):
        return -np.inf
    
    return like

def lnprob(theta, z, m):
    lp = lnprior(theta, m)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, z, m)

def fitpriors(data, mag_col, z_col, start,
              nwalkers=20, nsamples=5000, fburnin=0.2, nthreads = 4, nskip=10):
    """ Fit prior functional form to observed dataset with emcee
    
    """
    ndim = 4
    burnin = int(nsamples*fburnin)

    #data = Table.read(catalog_path, format=catalog_format)

    cut = (data[z_col] > 0.001)*(data[mag_col] > 14.) # Cut -99s etc
    data = data[cut][::nskip]

    zdata = data[z_col].data
    mdata = data[mag_col].data

    
    
    # Set up the sampler.
    pos = [start + 0.1*np.random.randn(ndim)*np.abs(start) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zdata, mdata), threads=nthreads)

    # Clear and run the production chain.
    print("Running MCMC...")
    
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


    # Compute the quantiles.
    z0t_mcmc, kmt1_mcmc, kmt2_mcmc, alpha1_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                                                 zip(*np.percentile(samples, [16, 50, 84],
                                                                 axis=0)))

    quantiles = [z0t_mcmc, kmt1_mcmc, kmt2_mcmc, alpha1_mcmc]
    best = [z0t_mcmc[0], kmt1_mcmc[0], kmt2_mcmc[0], alpha1_mcmc[0]]
    return best, quantiles, samples, sampler


if __name__ == '__main__':
    """
    Input Variables
    
    """
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

    pzarr = []
    zouts = []

    #photometry = Table.read('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog), 
    #                        format = pipe_params.photometry_format)

    #photom = photometry[photometry['z_spec'] >= 0.]

    photom = Table.read('{0}/testing/{1}'.format(pipe_params.working_folder, 'training_all.cat'),
                        format='ascii.commented_header')    


    cut = (photom[pipe_params.prior_colname] > 0.)  * (photom['z_spec'] > 0.)
    photom = photom[cut]

    AGN = (photom['AGN'] == 1)
    GAL = np.invert(AGN)
    
    for ixs, sbset in enumerate([AGN]):
    
        if (sbset == GAL).all():
            sbname = 'gal'
            start = np.array([0.02, -0.007, 0.01, 2.])
            nskip = 20
            
        else:
            sbname =  'agn'
            start = np.array([0.54, -0.18, 0.028, 1.39])
            nskip = 1
            
        mags = ['r']
        #['I', 'Ks', 'ch1']
        
        
        for filt in mags:
            print('{0} Prior Fitting:'.format(filt))
            mag_col = pipe_params.prior_colname
            z_col = pipe_params.zspec_col

            best, fits, samples, sampler = fitpriors(photom[sbset], mag_col, z_col, start, 
                                                     nsamples=5000, nthreads=10, nskip=nskip)
            z0t, kmt1, kmt2, alpha1 = fits

            coeff_save_path = '{0}/{1}_{2}_prior_coeff.npz'.format(pipe_params.working_folder, filt, sbname)
            np.savez(coeff_save_path, z0t=z0t, kmt1=kmt1, kmt2=kmt2, alpha=alpha1)



        """
        Useful Plots
        """
        do_plots = True
        
        if do_plots:
            fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 9))
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


            zrange = np.linspace(0, 4, 1000)

            mrange = np.arange(17, 25, 1)
            data = photom[sbset]

            cut = (data[z_col] > 0.)
            #data = data[cut][::10]
            data = data[data[z_col] >= 0.]

            zdata = data[z_col].data
            mdata = data[mag_col].data

            Fig, Ax = plt.subplots(1, figsize=(5, 3.5))
            obs_peaks = []
            
            b1 = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
            b1[-1] = 0.

            intvl = MinMaxInterval()
            colors = plt.cm.viridis(intvl(mrange))

            for im, m in enumerate(mrange):
                pzm = pzl(zrange, m, *best, lzc=0.001)
                pzm /= np.trapz(pzm, zrange)
                P = Ax.plot(zrange, pzm, color=colors[im], 
                            lw=3, alpha=0.7, label='{0} = {1:.1f}'.format('$m_{I}$', m))
                
                
                cut = np.logical_and(mdata > m-0.5, mdata < m+0.5)
                Ax.hist(zdata[cut], normed=True, histtype='step', color=colors[im], linestyle='dashed',
                        bins=51, range=(0,4))
                
                Ax.set_ylabel(r'$p\left( z \mid m_{I} \right)$')
                Ax.set_xlabel('$z$')
                Leg = Ax.legend(loc='upper right', prop={'size':11})
                Leg.draw_frame(False)   
            
            Fig.tight_layout()
            fig_save_path = '{0}/{1}_{2}_prior_plot.pdf'.format(pipe_params.working_folder, filt, sbname)
            Fig.savefig(fig_save_path, format='pdf', bbox_inches='tight')
            #Leg = Ax.legend(loc='upper right')
    
    plt.show()
    
    
    """
    f = np.polyfit(mrange-14., obs_peaks, 2)
    fit = np.poly1d(f)
    mr = np.linspace(15, 25, 1000)
    
    plt.plot(mr, fit(mr-14.))
    plt.plot(mrange, obs_peaks, 'o')
    plt.show()
    """

    # NB Linear zmt evolution doesnt fit observed evolution in peak
    # Power-law fit looks a lot more in line. Re-cutting input sample down spec-z complete magnitudes to see if that
    # reduces biases
    # Linear best: [-0.1869, 0.0690, 1.126]




#a = pz(zdata, mdata, *[ 0.06891682, -0.05053156,  0.00889756,  1.17581103])


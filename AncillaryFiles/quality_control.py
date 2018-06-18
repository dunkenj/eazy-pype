import os, sys, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.stats import ks_2samp
from astropy.stats import bootstrap, bayesian_blocks
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

import astropy.units as u
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.modeling import models, fitting
from astropy.utils.console import ProgressBar
import emcee

g = Gaussian1DKernel(0.5)
i = 13

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--params", type=str,
                    help = "Parameter file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()
quiet = args.quiet




def calcStats(photoz, specz, verbose=False):
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
    sigma_ol1 = np.sqrt( np.sum((dz[np.invert(ol1)]/(1+specz[np.invert(ol1)]))**2) / 
                        float(len(dz[np.invert(ol1)])))
    
    ol2 = (np.abs(dz)/(1+specz) > 5*nmad )
    OLF2 = np.sum( ol2 ) / float(len(dz))
    sigma_ol2 = np.sqrt( np.sum((dz[np.invert(ol2)]/(1+specz[np.invert(ol2)]))**2) / 
                        float(len(dz[np.invert(ol2)])))
    KSscore = ks_2samp(specz, photoz)[0]
    
    if verbose:
        print('Sigma_all: {0:.3f}'.format(sigma_all))
        print('Sigma_NMAD: {0:.3f}'.format(nmad))
        print('Bias: {0:.3f}'.format(bias))
        
        print('OLF: Def1 = {0:.3f} Def2 = {1:0.3f}'.format(OLF1, OLF2))
        print('Sigma_OL: Def 1 = {0:.3f} Def2 = {1:0.3f}'.format(sigma_ol1, sigma_ol2))
        print('KS: {0:.3f}'.format(KSscore))
    
    return [sigma_all, nmad, bias, OLF1, sigma_ol1, OLF2, sigma_ol2, KSscore]

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

    folder = '{0}/full'.format(pipe_params.working_folder)
    zout = Table.read('{0}/photoz_all_merged.fits'.format(folder), format='fits')
    
    AGN = (photometry['AGN'] == 1)
    GAL = np.invert(AGN)
    
    """
    SPEC-Z ANALYSIS
    """
    all_chis = np.array([zout['chi_r_eazy'].data, zout['chi_r_cosmos'].data, zout['chi_r_atlas'].data])
    all_chis[all_chis < 0.] = np.nan
    
    all_zs = np.array([zout['zpeak_eazy'].data, zout['zpeak_cosmos'].data, zout['zpeak_atlas'].data])
    
    
    chi_best = np.nanmin(all_chis, axis=0)

    star = (chi_best > zout['chi_r_stellar']).astype('int')
    #qc = (zout['z1_max']-zout['z1_min']/(1+zout['z1_median']) < 0.5)
    #qc = (chi_best < 4.) #np.percentile(chi_best[AGN], 95))
    

    for sbset in [GAL]:

        if (sbset == GAL).all():
            #zsb_l = np.linspace(0, np.log10(1+zspec.max()), 7)
            #zsbins = (10**zsb_l) - 1
            #zsbins = np.linspace(0.0, 1.5, 7)
            #zsbins= np.append(zsbins, np.linspace(2., 3., 2.))
            sbset_name = 'gal'
        else:

            #zsbins = np.insert(np.linspace(0, 5, 6), 1, 0.5)
            sbset_name = 'agn'
            
        photometry_qc_cuts = np.invert(star) #based on star classification

        z_spec_all = photometry['z_spec']
        zs_sample_idx = np.logical_and(z_spec_all >= 0., photometry_qc_cuts)

        zsets = ['zpeak_eazy', 'zpeak_atlas', 'zpeak_cosmos', 'z1_median', 'za_hb']
        names = ['EAZY', 'Atlas', 'XMM-COSMOS', 'HB', 'HB Peak']
        cols = ['steelblue', 'olivedrab', 'firebrick', '0.3', '0.6']
        
        zsmin = z_spec_all[z_spec_all > 0.].min()
        zsmax = z_spec_all[z_spec_all > 0.].max()

        #zsbins = 10**(np.linspace(0.05, 0.90309, 15)) - 1


        cc = zs_sample_idx * (zout['z1_median'] >= 0.) * sbset

        stats = []
        
        zz = zout[cc]
        zspec = z_spec_all[cc]

        zsb_l = np.linspace(0, np.log10(1+zspec.max()), 1.5)
        zsbins = (10**zsb_l) - 1
        
        #zsbins = bayesian_blocks(zspec)
        #zsbins = (10**zsb_l) - 1


        
        zmids = 0.5*(zsbins[:-1]+zsbins[1:])
        
        for iz, zmin in enumerate(zsbins[:-1]):
            zmax = zsbins[iz+1]
            
            zcut = np.logical_and(zspec >= zmin, zspec < zmax)
            print(zcut.sum())
            
            
            st = []
            for zset in zsets:
                bs = []
                
                zphot = zz[zset][zcut]
                zsc = zspec[zcut]
                ix = np.arange(len(zsc)).astype('int')
                
                samples = bootstrap(ix, bootnum=50)
                
                for sample in samples:
                    bs.append(calcStats(zphot[sample.astype('int')], zsc[sample.astype('int')]))
                
                st.append(bs)
            
            
            stats.append(st)
            
        stats = np.array(stats)
        #stats = stats[:, :, :, 1:-1]
        
        smean = stats[:, :, :, 1:-3].mean(2)
        sstd = stats[:, :, :, 1:-3].std(2)
        
        
        Fig, Ax = plt.subplots(2, 2, figsize=(10.5, 5.), sharex=True)
        Ax = Ax.flatten()
        
        ymaxes = [1.5, 0.12, 1.5, 1.5]
        ymins = [0.005, -0.12, 0.005, 0.005]
        yscales = ['log', 'linear', 'log', 'log']
        
        ylabels = [r'$\sigma_{\rm{NMAD}}$', r'bias', r'Outlier Fraction', r'$\sigma_{\rm{OL}}$']
        
        for j in range(len(Ax)):
        
            for i, zset in enumerate(zsets):
                Ax[j].fill_between(zmids, smean[:, i, j]+sstd[:, i, j], 
                                    smean[:,i,j]-sstd[:,i,j],
                                    color=cols[i], alpha=0.5)      
                Ax[j].plot(zmids, smean[:, i, j], color=cols[i], label=names[i], lw=2)


            Ax[j].set_xscale('log')
            Ax[j].set_xlabel('z')
            Ax[j].set_ylabel(ylabels[j])
            Ax[j].set_yscale(yscales[j])
            Ax[j].set_ylim(ymins[j], ymaxes[j])
            Ax[j].set_xticks([0.1, 0.5, 1., 2., 3., 4., 5.])
            Ax[j].set_xticklabels(['0.1', '0.5', '1', '2', '3', '4', '5'])
            Ax[j].set_xlim([0.1, 4.5])
            #Ax[j].ticklabel_format(axis='y', style='plain')
            #Ax[0].set_xscale('log')
            #Ax[1].set_xscale('log')
            #Ax[j].set_yscale('log')
            #Ax[j].set_ylim([0.01, 1.])
        
        Leg = Ax[-1].legend(loc='upper left', ncol=2, prop={'size':9})
        Leg.draw_frame(False)
        Fig.subplots_adjust(right=0.95, top=0.95, wspace=0.22, hspace=0.)
        Fig.tight_layout()
        Fig.savefig('{0}/plots/{1}_photoz_stats.pdf'.format(pipe_params.working_folder, sbset_name), format='pdf',
                    bbox_inches='tight')
        plt.show()
        
  


    """
    
    PAIRS ANALYSIS
    
    """


    def find_pair_zdist(coords, z, Nbins, dist, mindist):
        """ Get distribution of redshift offsets for close pairs
        
            Returns:
                np.histogram results of dz with Nbins

        """

        kdt = cKDTree(coords)
        pairs = kdt.query_pairs(dist)

        dzp = np.zeros(len(pairs))
        z1 = np.zeros(len(pairs))
        z2 = np.zeros(len(pairs))
        dr = np.zeros(len(pairs))
        
        for i, pair in enumerate(pairs):
            # NB pairs is generator so can't vectorise properly
            z1[i] = z[pair[0]]
            z2[i] = z[pair[1]]
            dzp[i] = (z1[i]-z2[i]) / (1+(0.5*(z1[i]+z2[i])))
            dr[i] = np.sqrt((np.diff(coords[pair,:], axis=0)**2).sum())

        above_min = (dr > mindist)
        histp = np.histogram(dzp[above_min], range=(-0.3, 0.3), bins=Nbins)
        
        return histp, z1, z2, dr

    """
    zsets = ['zpeak_eazy', 'zpeak_atlas', 'zpeak_cosmos', 'z1_median', 'za_hb']
    names = ['EAZY', 'Atlas', 'XMM-COSMOS', 'HB Median', 'HB Peak']
    
    #zsets = ['zm_eazy']
    #names = ['EAZY']
    #zsets = ['z1_median']
    #names = ['HB']
    
    Hlims = [19., 20., 21., 22., 23., 24.] #, 22.5, 23., 23.5, 24.]
    
    mag_name = pipe_params.prior_fname
    
    pair_scatter = Table()

    pair_scatter['{0}_limit'.format(mag_name)] = Hlims

    diff_all = []

    for iz, zset in enumerate(zsets):
        sigma_fwhm = np.zeros(len(Hlims))
        sigma_gaussian = np.zeros(len(Hlims))
        sigma_lorentzian = np.zeros(len(Hlims))

        sigma_gaussian_err = np.zeros(len(Hlims))
        sigma_lorentzian_err = np.zeros(len(Hlims))

        outlier_fracs = np.zeros(len(Hlims))
        print(names[iz])
        
        diff_t = []
        
        for jmag, mag in enumerate(Hlims):
            print mag
        
            
            mag_cut = np.logical_and(photometry['{0}_mag'.format(mag_name)] < mag, photometry['{0}_mag'.format(mag_name)] > 0.)
            cut = mag_cut * (zout[zset] >= 0.)
            #olprob = np.percentile(zout['Outlier_Prob'][cut], 90)
            #cut = cut #* (zout['Outlier_Prob'] < olprob)
            

            coords_full = np.array([photometry['RA'], photometry['DEC']]).T
            coords = coords_full[cut]

            dist = (20*u.arcsec).to(u.deg).value
            mindist = (2*u.arcsec).to(u.deg).value
            
            #Steps:

            #1. Get zdist for real pairs
            #2. For N iterations:
            #    a) Randomise positions
            #    b) Get zdist for real pairs



            z = zout[zset][cut] + 0.01*(np.random.rand(len(coords)) - 0.5)

            histp, z1, z2, dr = find_pair_zdist(coords, zout[zset][cut], 201, dist, mindist)

            histr = []
            bar = ProgressBar(3)
            for i in range(3):
                #print i
                new_coords = np.copy(coords_full)
                ix = np.random.randint(len(new_coords), size=len(z))
                new_coords = new_coords[ix]
                
                or1 = (np.random.rand(len(coords))-0.5)*5*dist
                or2 = (np.random.rand(len(coords))-0.5)*5*dist

                new_coords += np.array([or1, or2]).T
                
                r1, z1, z2, dr = find_pair_zdist(new_coords, z, 201, dist, mindist)
                histr.append(r1[0])
                bar.update()

            histr = np.array(histr)
            hist_rand = histr.mean(0)

            weights = hist_rand/histr.std(0)
            weights[np.isnan(weights)] = 0.
            weights[np.isinf(weights)] = 1.
            weights[weights < 1] = 0.
            
            diff = histp[0]-hist_rand
            bins = histp[1][:-1]+(0.5*np.diff(histp[1]))
            pdz = diff / np.trapz(np.maximum(diff, 0.), bins)
            cumhist = np.cumsum(np.maximum(pdz,0)*np.diff(bins)[0])
            
            pmin = griddata(diff[:101], bins[:101], 0.5*np.max(diff))
            pmax = griddata(diff[100:], bins[100:], 0.5*np.max(diff))
            sigma_fwhm[jmag] = pmax-pmin
            
            fit_l = fitting.LevMarLSQFitter()
            fit_g = fitting.LevMarLSQFitter()
            
            g_init = models.Gaussian1D(np.max(diff), mean=0., stddev=0.1)
            l_init = models.Lorentz1D(amplitude=np.max(diff), x_0=0., fwhm=0.1)

            g = fit_g(g_init, bins[50:151], diff[50:151], weights=weights[50:151])
            l = fit_l(l_init, bins[50:151], diff[50:151], weights=weights[50:151])
            
            sigma_gaussian[jmag] = g.parameters[-1]/np.sqrt(2)
            sigma_lorentzian[jmag] = l.parameters[-1]/2.
            
            gerr = np.sqrt(fit_g.fit_info['param_cov'].diagonal())[-1]
            lerr = np.sqrt(fit_l.fit_info['param_cov'].diagonal())[-1]
        
            sigma_gaussian_err[jmag] = gerr
            sigma_lorentzian_err[jmag] = lerr
            
            wpeak = (np.abs(bins) < 3*g.parameters[-1])
            area_peak = np.trapz(bins[wpeak], diff[wpeak])
            area_totl = np.trapz(bins, diff)
            
            outlier_fracs[jmag] = (area_totl-area_peak)/area_totl
            diff_t.append(diff)
        
        diff_all.append(diff_t)
        
        pair_scatter['{0}_sigma_fwhm'.format(names[iz])] = sigma_fwhm
        pair_scatter['{0}_sigma_fwhm'.format(names[iz])].format = '%.4f'
        
        print('\n')
        pair_scatter['{0}_sigma_g'.format(names[iz])] = sigma_gaussian
        pair_scatter['{0}_sigma_gerr'.format(names[iz])] = sigma_gaussian_err
        
        pair_scatter['{0}_sigma_l'.format(names[iz])] = sigma_lorentzian
        pair_scatter['{0}_sigma_lerr'.format(names[iz])] = sigma_lorentzian_err

        pair_scatter['{0}_sigma_g'.format(names[iz])].format = '%.4f'
        pair_scatter['{0}_sigma_l'.format(names[iz])].format = '%.4f'

        pair_scatter['{0}_sigma_gerr'.format(names[iz])].format = '%.4f'
        pair_scatter['{0}_sigma_lerr'.format(names[iz])].format = '%.4f'

        pair_scatter['{0}_ol_frac'.format(names[iz])] = outlier_fracs

    #pair_scatter.write('pair_scatter_data_ol.cat',format='ascii.commented_header')


    Fig, Ax = plt.subplots(1, figsize=(5.5,5))
    #Ax.bar(histp[1][:-1], diff, np.diff(histp[1]), color='0.7',linewidth=0)
    Ax.plot(bins, diff, 'o--', color='0.5', mew=0)
    xx = np.linspace(-0.5, 0.5, 1000)
    Ax.plot(xx, g(xx), lw=2, label='Gaussian: {0} = {1:.3f}'.format('$\sigma/\sqrt{2}$', g.parameters[-1]/np.sqrt(2)))
    Ax.plot(xx, l(xx), '--', lw=2, label='Lorentzian: {0} = {1:.3f}'.format('$\gamma/2$', l.parameters[-1]/2))
    #Leg = plt.legend(loc='upper right', prop={'size':11})


    Ax.set_xlabel(r'$\Delta z_{\rm{phot}}/(1+z_{\rm{mean}})$')
    Ax.set_ylabel('N')
    #Ax.set_ylim([-50, 790])
    #Ax.set_xlim([-0.5, 0.5])
    #Leg.draw_frame(False)
    Fig.subplots_adjust(bottom=0.15, left=0.12, top=0.95, right=0.95)
    #Fig.savefig('pairs_error_example_HB_21.5.pdf', format='pdf', bbox_inches='tight')
    plt.show()


    Fig, Ax = plt.subplots(1, figsize=(6.5,4))
    
    #Ax.fill_between(pair_scatter['I_limit'], 
    #        pair_scatter['HB_sigma_g']+pair_scatter['HB_sigma_gerr'],
    #        pair_scatter['HB_sigma_g']-pair_scatter['HB_sigma_gerr'],
    #        color='0.8')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['EAZY_sigma_fwhm']/2.355,
            '--', lw=2, color='steelblue',
            label='EAZY')
    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['Atlas_sigma_fwhm']/2.355,
            '-.', lw=2, color='olivedrab',
            label='Atlas')
    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['XMM-COSMOS_sigma_fwhm']/2.355,
            ':', lw=2, color='firebrick',
            label='XMM-COSMOS')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['HB Median_sigma_fwhm']/2.355,
            lw=4, color='0.3',
            label='Hierarchical Bayesian')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['HB Peak_sigma_fwhm']/2.355,
            lw=4, color='0.6',
            label='Hierarchical Bayesian')

    Ax.plot(21.6, 0.035, 'o', ms=9, mew=1.5, color='steelblue')
    Ax.plot(21.55, 0.039, 'd', ms=9, mew=1.5, color='firebrick')
    Ax.plot(21.5, 0.035, 'D', ms=9, mew=1.5, color='olivedrab')

    Ax.plot(21.55, 0.028, 'v', ms=11, mew=1.5, color='0.6')
    Ax.plot(21.45, 0.033, '*', ms=14, mew=1.5, color='0.3')

    Leg = Ax.legend(loc='upper left', ncol=2, prop={'size': 10})
    Leg.draw_frame(False)

    Ax.set_xlabel('$I$-band Magnitude Limit')
    Ax.set_ylabel('Scatter from pairs - FWHM/2 \n [$\Delta z / (1 + z)$]')
    Ax.set_xlim([19.5, 24.5])
    Ax.set_ylim([0., 0.3])
    Fig.subplots_adjust(left = 0.2, bottom=0.17, right = 0.95, top=0.95)
    Fig.savefig('pair_errors_fwhm.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    Fig, Ax = plt.subplots(1, figsize=(6.5,4))

    Ax.fill_between(pair_scatter['{0}_limit'.format(mag_name)], 
            pair_scatter['HB_sigma_g']+pair_scatter['HB_sigma_gerr'],
            pair_scatter['HB_sigma_g']-pair_scatter['HB_sigma_gerr'],
            color='0.8')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['EAZY_sigma_g'],
            '--', lw=2, color='steelblue',
            label='EAZY')
    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['Atlas_sigma_g'],
            '-.', lw=2, color='olivedrab',
            label='Atlas')
    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['XMM-COSMOS_sigma_g'],
            ':', lw=2, color='firebrick',
            label='SWIRE')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['HB Median_sigma_g'],
            lw=4, color='0.3',
            label='Hierarchical Bayesian')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['HB Peak_sigma_g'],
            lw=4, color='0.3',
            label='Hierarchical Bayesian')


    Ax.plot(21.6, 0.035, 'o', ms=9, mew=1.5, color='steelblue')
    Ax.plot(21.55, 0.039, 'd', ms=9, mew=1.5, color='firebrick')
    Ax.plot(21.5, 0.035, 'D', ms=9, mew=1.5, color='olivedrab')

    Ax.plot(21.55, 0.028, 'v', ms=11, mew=1.5, color='0.6')
    Ax.plot(21.45, 0.033, '*', ms=14, mew=1.5, color='0.3')

    Leg = Ax.legend(loc='upper left', ncol=2, prop={'size': 10})
    Leg.draw_frame(False)

    Ax.set_xlabel('$I$-band Magnitude Limit')
    Ax.set_ylabel('Scatter from pairs - Gaussian \n [$\Delta z / (1 + z)$]')
    Ax.set_xlim([20.5, 24.5])
    Ax.set_ylim([0., 0.13])
    Fig.subplots_adjust(left = 0.2, bottom=0.17, right = 0.95, top=0.95)
    Fig.savefig('pair_errors_gaussian_ol.pdf', format='pdf', bbox_inches='tight')



    Fig, Ax = plt.subplots(1, figsize=(6.5,4))



    Ax.fill_between(pair_scatter['{0}_limit'.format(mag_name)], 
            pair_scatter['HB_sigma_l']+pair_scatter['HB_sigma_lerr'],
            pair_scatter['HB_sigma_l']-pair_scatter['HB_sigma_lerr'],
            color='0.8')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['EAZY_sigma_l'],
            '--', lw=2, color='steelblue',
            label='EAZY')
    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['Atlas_sigma_l'],
            '-.', lw=2, color='olivedrab',
            label='Atlas')
    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['XMM-COSMOS_sigma_l'],
            ':', lw=2, color='firebrick',
            label='SWIRE')

    Ax.plot(pair_scatter['{0}_limit'.format(mag_name)], pair_scatter['HB_sigma_l'],
            lw=4, color='0.3',
            label='Hierarchical Bayesian')

    Ax.plot(21.6, 0.035, 'o', ms=9, mew=1.5, color='steelblue')
    Ax.plot(21.55, 0.039, 'd', ms=9, mew=1.5, color='firebrick')
    Ax.plot(21.5, 0.035, 'D', ms=9, mew=1.5, color='olivedrab')

    Ax.plot(21.55, 0.028, 'v', ms=11, mew=1.5, color='0.6')
    Ax.plot(21.45, 0.033, '*', ms=14, mew=1.5, color='0.3')


    Leg = Ax.legend(loc='upper left', ncol=2, prop={'size': 10})
    Leg.draw_frame(False)

    Ax.set_xlabel('$I$-band Magnitude Limit')
    Ax.set_ylabel('Scatter from pairs - Lorentz \n [$\Delta z / (1 + z)$]')
    Ax.set_xlim([20.5, 24.5])
    Ax.set_ylim([0., 0.13])
    Fig.subplots_adjust(left = 0.17, bottom=0.17, right = 0.95, top=0.95)
    Fig.savefig('pair_errors_lorentzian_ol.pdf', format='pdf', bbox_inches='tight')

    plt.show()

    """






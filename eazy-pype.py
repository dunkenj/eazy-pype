import numpy as np
import array
import os, sys, shutil
import re
import time
import pickle
import multiprocessing as mp
from subprocess import call
import GPz
import smpy.smpy as S

import matplotlib.pyplot as plt

#from astropy.table import Table, Column
from astropy import units as u
from astropy.table import Table
from astropy.utils.console import ProgressBar
from sklearn.cross_validation import ShuffleSplit
from scipy.stats import norm

# Other Eazy-pype Functions
import validation
import zeropoints
import priors
import pdf_calibration
import hbcombination as hb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-p","--params", type=str,
                    help = "Parameter file")
parser.add_argument("-q", "--quiet", help = "Suppress extra outputs",
                    action = "store_true")
args = parser.parse_args()
quiet = args.quiet

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


# Classes from EazyPy - G.Brammer
class EazyParam():
    """
    Read an Eazy zphot.param file.

    Example:

    >>> params = EazyParam(PARAM_FILE='zphot.param')
    >>> params['Z_STEP']
    '0.010'

    """
    def __init__(self, PARAM_FILE='zphot.param', READ_FILTERS=False):
        self.filename = PARAM_FILE
        self.param_path = os.path.dirname(PARAM_FILE)

        f = open(PARAM_FILE,'r')
        self.lines = f.readlines()
        f.close()

        self._process_params()

        filters = []
        templates = []
        for line in self.lines:
            if line.startswith('#  Filter'):
                filters.append(ParamFilter(line))
            if line.startswith('#  Template'):
                templates.append(line.split()[3])

        self.NFILT = len(filters)
        self.filters = filters
        self.templates = templates

        if READ_FILTERS:
            RES = FilterFile(self.params['FILTERS_RES'])
            for i in range(self.NFILT):
                filters[i].wavelength = RES.filters[filters[i].fnumber-1].wavelength
                filters[i].transmission = RES.filters[filters[i].fnumber-1].transmission

    def _process_params(self):
        params = {}
        formats = {}
        self.param_names = []
        for line in self.lines:
            if line.startswith('#') is False:
                lsplit = line.split()
                if lsplit.__len__() >= 2:
                    params[lsplit[0]] = lsplit[1]
                    self.param_names.append(lsplit[0])
                    try:
                        flt = float(lsplit[1])
                        formats[lsplit[0]] = 'f'
                        params[lsplit[0]] = flt
                    except:
                        formats[lsplit[0]] = 's'

        self.params = params
        #self.param_names = params.keys()
        self.formats = formats

    def show_filters(self):
        for filter in self.filters:
            print ' F%d, %s, lc=%f' %(filter.fnumber, filter.name, filter.lambda_c)

    def write(self, file=None):
        if file == None:
            print 'No output file specified...'
        else:
            fp = open(file,'w')
            for param in self.param_names:
                if isinstance(self.params[param], np.str):
                    fp.write('%-25s %s\n' %(param, self.params[param]))
                else:
                    fp.write('%-25s %f\n' %(param, self.params[param]))
                    #str = '%-25s %'+self.formats[param]+'\n'
            #
            fp.close()

    #
    def __setitem__(self, param_name, value):
        self.params[param_name] = value

    def __getitem__(self, param_name):
        """
    __getitem__(param_name)

        >>> cat = mySexCat('drz.cat')
        >>> print cat['NUMBER']

        """
        if param_name not in self.param_names:
            print ('Column %s not found.  Check `column_names` attribute.'
                    %column_name)
            return None
        else:
            #str = 'out = self.%s*1' %column_name
            #exec(str)
            return self.params[param_name]

    def __setitem__(self, param_name, value):
        self.params[param_name] = value

def readEazyPDF(params, MAIN_OUTPUT_FILE='photz', OUTPUT_DIRECTORY='./OUTPUT', CACHE_FILE='Same'):
    """
tempfilt, coeffs, temp_sed, pz = readEazyBinary(MAIN_OUTPUT_FILE='photz', \
                                                OUTPUT_DIRECTORY='./OUTPUT', \
                                                CACHE_FILE = 'Same')

    Read Eazy BINARY_OUTPUTS files into structure data.

    If the BINARY_OUTPUTS files are not in './OUTPUT', provide either a relative or absolute path
    in the OUTPUT_DIRECTORY keyword.

    By default assumes that CACHE_FILE is MAIN_OUTPUT_FILE+'.tempfilt'.
    Specify the full filename if otherwise.
    """
    root = OUTPUT_DIRECTORY+'/'+MAIN_OUTPUT_FILE

    ###### .tempfilt
    if CACHE_FILE == 'Same':
        CACHE_FILE = root+'.tempfilt'

    if os.path.exists(CACHE_FILE) is False:
        print ('File, %s, not found.' %(CACHE_FILE))
        return -1,-1,-1,-1

    f = open(CACHE_FILE,'rb')

    s = np.fromfile(file=f,dtype=np.int32, count=4)
    NFILT=s[0]
    NTEMP=s[1]
    NZ=s[2]
    NOBJ=s[3]
    tempfilt = np.fromfile(file=f,dtype=np.double,count=NFILT*NTEMP*NZ).reshape((NZ,NTEMP,NFILT)).transpose()
    lc = np.fromfile(file=f,dtype=np.double,count=NFILT)
    zgrid = np.fromfile(file=f,dtype=np.double,count=NZ)
    fnu = np.fromfile(file=f,dtype=np.double,count=NFILT*NOBJ).reshape((NOBJ,NFILT)).transpose()
    efnu = np.fromfile(file=f,dtype=np.double,count=NFILT*NOBJ).reshape((NOBJ,NFILT)).transpose()
    f.close()

    ###### .pz
    if os.path.exists(root+'.pz'):
        PZ_FILE = root+'.pz'
        f = open(PZ_FILE,'rb')
        s = np.fromfile(file=f,dtype=np.int32, count=2)
        NZ=s[0]
        NOBJ=s[1]
        chi2fit = np.fromfile(file=f,dtype=np.double,count=NZ*NOBJ).reshape((NOBJ,NZ)).transpose()
        f.close()

        if params['APPLY_PRIOR'] == 'Y':
            ### This will break if APPLY_PRIOR No
            s = np.fromfile(file=f,dtype=np.int32, count=1)

            if len(s) > 0:
                NK = s[0]
                kbins = np.fromfile(file=f,dtype=np.double,count=NK)
                priorzk = np.fromfile(file=f, dtype=np.double, count=NZ*NK).reshape((NK,NZ)).transpose()
                kidx = np.fromfile(file=f,dtype=np.int32,count=NOBJ)
                pz = {'NZ':NZ,'NOBJ':NOBJ,'NK':NK, 'chi2fit':chi2fit,
                      'kbins':kbins, 'priorzk':priorzk,'kidx':kidx}
            else:
                priorzk = np.ones((1,NZ))
                kidx = np.zeros(NOBJ)
                pz = {'NZ':NZ,'NOBJ':NOBJ,'NK':0, 'chi2fit':chi2fit,
                      'kbins':[0], 'priorzk':priorzk,'kidx':kidx}
                #pz = None

            ###### Get p(z|m) from prior grid
            #print kidx, pz['priorzk'].shape
            if (kidx > 0) & (kidx < priorzk.shape[1]):
                prior = priorzk[:,kidx]
            else:
                prior = np.ones(NZ)

            ###### Convert Chi2 to p(z)
            pzi = np.exp(-0.5*(chi2fit-np.min(chi2fit)))*prior
            if np.sum(pzi) > 0:
                pzi/=np.trapz(pzi, zgrid, axis=0)
        else:
            # Convert to PDF and normalise
            pzi = np.exp(-0.5*chi2fit)
            if np.sum(pzi) > 0:
                pzi /= np.trapz(pzi, zgrid, axis=0)

    else:
        pzi, zgrid = None, None

    ###### Done.
    return [pzi, zgrid]

# Define functions
def runeazy(params='zphot.param', translate=None, zeropoints=None,
            eazypath = pipe_params.eazypath, verbose=True):
    """ Run EAZY for a given params file.

    Args:
        params: string - name of EAZY params file
        eazypath: string - path to run eazy (if not in eazy/inputs)
        verbose: boolean - If True, EAZY terminal output is printed
                           Else, EAZY terminal output is place in text
                           file corresponding to the input params file.
    """
    if verbose:
        stdout = None
        stderr = None
    else:
        stderr = open(params+'.stderr.txt','w')
        stdout = open(params+'.stdout.txt','w')

    if zeropoints == None:
        command = ('{eazypath} -p {params} -t {translate}'.format(eazypath = eazypath, params=params,
                                                                  translate=translate))
    else:
        command = ('{eazypath} -p {params} -t {translate} -z {zeropoints}'.format(eazypath = eazypath, params=params,
                                                                                  translate=translate, zeropoints = zeropoints))
    if verbose:
        print(command)

    output = call(command, stdout = stdout, stderr = stderr, shell=True)
    return output


def eazyfunction_worker(inputQueue, outputQueue, outLock, template,
                        translate, zeropoints, eazypath, verbose=False, clean=True):
    """ Run EAZY for a given catalog

    Args:
        gal: int - index of galaxy to fit within catalog (in shared memory)
        params: EazyParam class
        outdir: string - Path to output directory (if not current directory)
        clean: boolean - If True, tidy up and remove intermediate files.

    """
    for i in iter(inputQueue.get, 'STOP'):
        params_path = '{0}/full/{1}/{2}.{3}.param'.format(pipe_params.working_folder, i+1, template, i+1)

        # Run eazy with params file
        out = runeazy(params_path, translate=translate, zeropoints=zeropoints,
                      verbose=verbose, eazypath=eazypath)

        outLock.acquire()
        outputQueue.put(params_path)
        outLock.release()

def hb_worker(inputQueue, outputQueue, outLock):
    """ Do Hierarchical Bayesian combination for subset of catalog

    """
    for i in iter(inputQueue.get, 'STOP'):
        pzarr = []
        zouts = []

        photometry_path = '{0}/full/{1}/{2}.cat'.format(pipe_params.working_folder, i+1, i+1)
        photom = Table.read(photometry_path,
                            format='ascii.commented_header')

        AGN = (photom['AGN'] == 1)
        if pipe_params.gpz:
            if pipe_params.ir_gpz_path != None:
                IRAGN = (photom['IRClass'] >= 4)

            if pipe_params.xray_gpz_path != None:
                XR = (photom['XrayClass'] == 1)

            if pipe_params.opt_gpz_path != None:
                try:
                    OPT = (photom['mqcAGN'][sbset*mcut] == 'True')
                except:
                    OPT = AGN

        GAL = np.invert(AGN)

        z_max_name = ['z_a', 'z_1', 'z_1']
        titles = ['EAZY', 'ATLAS (Brown+)', 'XMM-COSMOS (Salvato+)']

        folder = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)

        for itx, template in enumerate(pipe_params.templates):
            #print(template)

            """ Load Values/Arrays/Catalogs """
            basename='{0}.{1}'.format(template, i+1)
            pz, zgrid = hb.getPz('{0}/{1}'.format(folder, basename))

            if pipe_params.include_prior:
                mags = np.array(photom[pipe_params.prior_colname])

                pzprior = np.ones_like(pz)
                pzprior /= np.trapz(pzprior, zgrid, axis=1)[:, None]

                if pipe_params.include_prior_gal:
                    pzprior[GAL] = priors.pzl(zgrid, mags[GAL], *best_prior_params_gal, lzc=0.003)

                if pipe_params.include_prior_agn:
                    pzprior[AGN] = priors.pzl(zgrid, mags[AGN], *best_prior_params_agn, lzc=0.000)

                pzprior_nomag = np.ones_like(zgrid)
                pzprior_nomag /= np.trapz(pzprior_nomag, zgrid)
                pzprior[mags < -90.] = pzprior_nomag


            catalog = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')

            alphas_best_gal = hb.alphas_mag(photom[pipe_params.alpha_colname][GAL], *alphas_gal[itx])[:, None]
            pz[GAL] = pz[GAL]**(1/alphas_best_gal)

            alphas_best_agn = hb.alphas_mag(photom[pipe_params.alpha_colname][AGN], *alphas_agn[itx])[:, None]
            pz[AGN] = pz[AGN]**(1/alphas_best_agn)

            if pipe_params.include_prior:
                pz *= pzprior
                pz /= np.trapz(pz, zgrid, axis=1)[:, None]

            pzarr.append(pz)
            zouts.append(catalog)

        if pipe_params.gpz:
            sets = []
            bands = []
            alphas = []
            gpz = []
            scale_bands = []

            AGN = (photom['AGN'] == 1)

            if pipe_params.ir_gpz_path != None:
                IRAGN = (photom['IRClass'] >= 4)
                gp_ir, bands_ir, alpha_values_ir = load_gp('{0}'.format(pipe_params.ir_gpz_path))

                sets.append(IRAGN)
                bands.append(bands_ir)
                alphas.append(alpha_values_ir)
                gpz.append(gp_ir)

            if pipe_params.xray_gpz_path != None:
                XR = (photom['XrayClass'] == 1)
                gp_xray, bands_xray, alpha_values_xray = load_gp('{0}'.format(pipe_params.xray_gpz_path))

                sets.append(XR)
                bands.append(bands_xray)
                alphas.append(alpha_values_xray)
                gpz.append(gp_xray)

            if pipe_params.opt_gpz_path != None:
                try:
                    OPT = (photom['mqcAGN'] == 'True')
                except:
                    OPT = AGN

                try:
                    gp_opt, bands_opt, alpha_values_opt, sc = load_gp('{0}'.format(pipe_params.opt_gpz_path))
                    scale_bands.append(sc)
                except:
                    gp_opt, bands_opt, alpha_values_opt = load_gp('{0}'.format(pipe_params.opt_gpz_path))

                sets.append(OPT)
                bands.append(bands_opt)
                alphas.append(alpha_values_opt)
                gpz.append(gp_opt)

            GAL = np.invert(AGN)

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
                pz = np.ones((len(photom), len(zgrid)))/7.

                if s.sum() >= 1:
                    X, Y, _, _, K = make_data(photom[s], bands[ix])
                    if K.sum() >= 1:
                        mu, sigma, modelV, noiseV, _ = gpz[ix].predict(X.copy())

                        sigma *= hb.alphas_mag(photom[pipe_params.prior_colname][s][K],
                                              *alphas[ix]).reshape(sigma.shape)

                        pz_gp = []

                        for iz, z in enumerate(mu):
                            gaussian = norm(loc=mu[iz], scale=sigma[iz])
                            pz_gp.append(gaussian.pdf(zgrid))
                        #bar.update()
                        pz[np.where(s)[0][K]] = pz_gp

                pzarr.append(pz)


        pzarr = np.array(pzarr)
        pzarr /= np.trapz(pzarr, zgrid, axis=-1)[:, :, None]

        if pipe_params.fbad_prior == 'flat':
            pzbad = np.ones_like(pzarr[0]) # Flat prior
            pzbad /= np.trapz(pzbad, zgrid, axis=1)[:,None]

        elif pipe_params.fbad_prior == 'vol':
            pzbad = hb.cosmo.differential_comoving_volume(zgrid)
            pzbad /= np.trapz(pzbad, zgrid) # Volume element prior (for our cosmology)
            pzbad = pzbad[:, None] * np.ones_like(pzarr[0])

        elif pipe_params.fbad_prior == 'mag':
            mags = np.array(photom[pipe_params.prior_colname])

            pzbad = np.zeros_like(pzarr[0])
            pzbad[GAL] = priors.pzl(zgrid, mags[GAL], *best_prior_params_gal, lzc=0.003)
            pzbad[AGN] = priors.pzl(zgrid, mags[AGN], *best_prior_params_agn, lzc=0.000)

            pzbad_nomag = np.ones_like(zgrid)
            pzbad_nomag /= np.trapz(pzbad_nomag, zgrid)
            pzbad[mags < -90.] = pzbad_nomag

        pzarr_hb = np.zeros_like(pzarr[0])
        pzarr_hb[GAL] = hb.HBpz(pzarr[:, GAL], zgrid, pzbad[GAL], beta_gal, fbad_max = 0.05, fbad_min = 0.0)
        pzarr_hb[AGN] = hb.HBpz(pzarr[:, AGN], zgrid, pzbad[AGN], beta_agn, fbad_max = 0.2, fbad_min = 0.)
        pzarr_hb /= np.trapz(pzarr_hb, zgrid, axis=1)[:, None]

        hbcat = hb.pz_to_catalog(pzarr_hb, zgrid, zouts[0], verbose=False)

        catpath = '{0}.{1}.cat'.format('HB', i+1)
        hbcat.write('{0}/{1}'.format(folder, catpath), format='ascii.commented_header')

        np.savez('{0}/{1}.npz'.format(folder, 'HB_pz'), zgrid=zgrid, pz=pzarr_hb)

        outLock.acquire()
        outputQueue.put(catpath)
        outLock.release()


def save_gp(path, gp, bands, alpha_values):
    out = {'gp': gp,
           'bands': bands,
           'alpha': alpha_values}

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


def make_data(catalog, columns):
    X = []
    for col in columns:
        X.append(catalog['{0}{1}'.format(col, pipe_params.mag_col)])
    for col in columns:
        X.append(catalog['{0}{1}'.format(col, pipe_params.magerr_col)])

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

def maybe_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == '__main__':
    """
    Section 1 - Initialise and Validate Inputs

    """

    # Check paths and filenames are valid
    if os.path.isdir(pipe_params.working_folder):
        pass
    else:
        os.mkdir(params.working_folder)

    # Read in catalog
    try:
        photometry = Table.read('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog),
                                format = pipe_params.photometry_format)
    except:
        raise

    if pipe_params.process_outliers:

        print('Filtering for photometry outlier/bad photometry')
        #ec = ['Total_flux', 'E_Total_flux']
        ec = None
        new_path, bad_frac = validation.process('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog),
                                                '{0}/{1}'.format(pipe_params.working_folder, pipe_params.translate_file),
                                                '{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file),
                                                cat_format=pipe_params.photometry_format,
                                                exclude_columns = ec,
                                                flux_col = pipe_params.flux_col,
                                                fluxerr_col = pipe_params.fluxerr_col,
                                                correct_extinction=pipe_params.correct_extinction)
        photometry = Table.read(new_path, format=pipe_params.photometry_format)


    try:
        translate_init = Table.read('{0}/{1}'.format(pipe_params.working_folder, pipe_params.translate_file),
                                    format='ascii.no_header')
        fnames = translate_init['col1']
    except:
        raise

    # Validate column names listed in translate file are in the photometry catalog
    for filt in fnames:
        found = filt in photometry.colnames
        if found:
            pass
        else:
            print('Filter name "{0}" not found in photometry catalog'.format(filt))
            sys.exit('Filter mis-match')


    # Parse filter file with smpy - get central wavelengths back
    filt_obj = S.LoadEAZYFilters('{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file))

    keep = np.ones(len(translate_init)).astype('bool')

    for il, line in enumerate(translate_init):
        filtnum = int(line['col2'][1:])-1
        if filt_obj.filters[filtnum].lambda_c > 3*u.micron:
            keep[il] = False

    translate_temp_name = '{0}/{1}'.format(pipe_params.working_folder,
                                               pipe_params.translate_file+'.opt')
    # Write out translate file with only optical/near-IR bands (for stellar fits)
    translate_init[keep].write(translate_temp_name, format='ascii.no_header')


    # Make folders
    test_folder = pipe_params.working_folder+'/testing'
    if not os.path.isdir(test_folder):
        os.mkdir(test_folder)
    # ...


    training_cut = (photometry[pipe_params.zspec_col] >= 0.)
    photometry_training = photometry[training_cut]
    fname = '{0}/training_all.cat'.format(test_folder)
    if os.path.isfile(fname):
        os.remove(fname)
    photometry_training.write(fname, format='ascii.commented_header', fill_values=[(photometry_training.masked, '-99.')])

    rs = ShuffleSplit(len(photometry_training), pipe_params.Ncrossval, test_size = pipe_params.test_fraction)

    for i, (train_index, test_index) in enumerate(rs):
        print('Writing test and training subset: {0}'.format(i+1))

        fname = '{0}/training_subset{1}.cat'.format(test_folder, i+1)
        if os.path.isfile(fname):
            os.remove(fname)
        training_subset = photometry_training[np.sort(train_index)]
        training_subset.write(fname, format='ascii.commented_header', fill_values=[(training_subset.masked, '-99.')])

        zbins = [0., 0.2, 0.4, 0.6, 0.8, 1., 1.5, 2.5] #
        zs_train = training_subset['z_spec']

        for iz, zl in enumerate(zbins[:-1]):
            zu = zbins[iz+1]

            subset = np.logical_and(zs_train >= zl, zs_train < zu)
            fname = '{0}/training_subset{1}_zs{2}.cat'.format(test_folder, i+1, iz)
            training_subset[subset].write(fname, format='ascii.commented_header', fill_values=[(training_subset.masked, '-99.')])



        fname = '{0}/test_subset{1}.cat'.format(test_folder, i+1)
        if os.path.isfile(fname):
            os.remove(fname)
        test_subset = photometry_training[np.sort(test_index)]
        test_subset.write(fname, format='ascii.commented_header', fill_values=[(test_subset.masked, '-99.')])


    """
    Section 2 - Zeropoint Offsets

    """

    if pipe_params.do_zp:

        for itx, template in enumerate(pipe_params.templates):

            # Write out new translate file for different wavelength cuts
            keep = np.ones(len(translate_init)).astype('bool')

            for il, line in enumerate(translate_init):
                filtnum = int(line['col2'][1:])-1
                if filt_obj.filters[filtnum].lambda_c > pipe_params.lambda_fit_max[itx]*u.micron:
                    keep[il] = False

            # Write out translate file with only optical/near-IR bands (for stellar fits)
            translate_temp_name = '{0}/{1}.{2}'.format(pipe_params.working_folder, pipe_params.translate_file, template)
            translate_init[keep].write(translate_temp_name, format='ascii.no_header')
            #print('{0} ---Calculating ZP--- {1}'.format('\n', '\n')
            print('Templates: {0}'.format(template))

            ##########################
            # Fixed throughout section
            ezparam = EazyParam(pipe_params.defaults[itx])
            ezparam['FILTERS_RES'] = '{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file)
            ezparam['TEMPLATE_COMBOS'] = pipe_params.fitting_mode[itx]
            ezparam['TEMP_ERR_A2'] = pipe_params.template_error_norm[itx]
            ezparam['SYS_ERR'] = pipe_params.additional_errors[itx]
            ezparam['LAF_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/LAFcoeff.txt')
            ezparam['DLA_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/DLAcoeff.txt')

            #########################


            """
            Section 2a - Calculating ZPs

            """
            print('{0} ---Calculating ZP--- {1}'.format('\n', '\n'))
            ezparam['CATALOG_FILE'] = '{0}/training_subset1.cat'.format(test_folder)
            ezparam['MAIN_OUTPUT_FILE'] = 'training_subset1_calc_zp.{0}'.format(template)

            outdir = '{0}/testing/{1}'.format(pipe_params.working_folder, template)
            maybe_mkdir(outdir)

            ezparam['DUMP_TEMPLATE_CACHE'] = '1'
            cache_name = 'training_subset1_calc_zp.{0}.tempfilt'.format(template)
            ezparam['CACHE_FILE'] = cache_name

            ezparam['OUTPUT_DIRECTORY'] = outdir
            ezparam['FIX_ZSPEC'] = '1'
            param_name = '{0}/training_subset1_calc_zp.{1}.param'.format(outdir, template)
            ezparam.write(param_name)


            runeazy(params = param_name, eazypath = pipe_params.eazypath,
                    translate = translate_temp_name, verbose=False)


            zp_in = '{0}/training_subset1_calc_zp.{1}'.format(outdir, template)
            zp_path, Fig, _, _ = zeropoints.calc_zeropoints(zp_in, verbose=True,
                                                            fend=pipe_params.flux_col)
            if pipe_params.do_zp_tests:
                zp_paths = []
                zp_with_z = []
                zp_scatter_with_z = []

                for iz, zl in enumerate(zbins[:-1]):
                    print(iz+1)
                    ezparam['CATALOG_FILE'] = '{0}/training_subset1_zs{1}.cat'.format(test_folder, iz)
                    ezparam['MAIN_OUTPUT_FILE'] = 'training_subset1_zs{1}_calc_zp.{0}'.format(template, iz)

                    outdir = '{0}/testing/{1}'.format(pipe_params.working_folder, template)
                    maybe_mkdir(outdir)

                    ezparam['DUMP_TEMPLATE_CACHE'] = '1'
                    cache_name = 'training_subset1_zs{1}_calc_zp.{0}.tempfilt'.format(template, iz)
                    ezparam['CACHE_FILE'] = cache_name

                    ezparam['OUTPUT_DIRECTORY'] = outdir
                    ezparam['FIX_ZSPEC'] = '1'
                    param_name = '{0}/training_subset1_zs{2}_calc_zp.{1}.param'.format(outdir, template, iz)
                    ezparam.write(param_name)


                    runeazy(params = param_name, eazypath = pipe_params.eazypath,
                            translate = translate_temp_name, verbose=False)

                    zp_in = '{0}/training_subset1_zs{2}_calc_zp.{1}'.format(outdir, template, iz)
                    zp_path_s, Fig, zpz, zpz_sd = zeropoints.calc_zeropoints(zp_in,
                                                              verbose=True,
                                                              fend=pipe_params.flux_col)
                    plt.close(Fig)
                    zp_paths.append(zp_path_s)
                    zp_with_z.append(zpz)
                    zp_scatter_with_z.append(zpz_sd)


                zp_with_z = np.array(zp_with_z).T
                zp_scatter_with_z = np.array(zp_scatter_with_z).T

                np.savez('{0}/zp_evolution_{1}.npz'.format(outdir, template),
                         zbins = zbins, zp = zp_with_z, zp_std = zp_scatter_with_z)


            """
            Section 2b - Testing ZPs

            - Run once on test sample without zeropoint offsets
            - Run again on test sample *with* zeropoint offsets

            """

            if pipe_params.do_zp_tests:
                ### Without ZP Offsets ###
                print('{0} ---No ZP--- {1}'.format('\n', '\n'))
                ezparam['CATALOG_FILE'] = '{0}/training_subset1_zs5.cat'.format(test_folder)
                ezparam['MAIN_OUTPUT_FILE'] = 'training_subset1_zs5_no_zp.{0}'.format(template)

                outdir = '{0}/testing/{1}/no_zp'.format(pipe_params.working_folder, template)
                maybe_mkdir(outdir)

                ezparam['DUMP_TEMPLATE_CACHE'] = '1'
                cache_name = 'training_subset1_zs5_no_zp.{0}.tempfilt'.format(template)
                ezparam['CACHE_FILE'] = cache_name

                ezparam['OUTPUT_DIRECTORY'] = outdir
                ezparam['FIX_ZSPEC'] = '0'
                ezparam['GET_ZP_OFFSETS'] = '0'
                param_name = '{0}/training_subset1_zs5_no_zp.{1}.param'.format(outdir, template)
                ezparam.write(param_name)


                runeazy(params = param_name, eazypath = pipe_params.eazypath,
                        translate = translate_temp_name, verbose=False)

                zout_no_zp = Table.read('{0}/training_subset1_zs5_no_zp.{1}.zout'.format(outdir, template),
                                        format='ascii.commented_header')

                ### With ZP Offsets ###
                print('{0} ---With ZP--- {1}'.format('\n', '\n'))
                ezparam['CATALOG_FILE'] = '{0}/training_subset1_zs5.cat'.format(test_folder)
                ezparam['MAIN_OUTPUT_FILE'] = 'training_subset1_zs5_with_zp.{0}'.format(template)

                outdir = '{0}/testing/{1}/with_zp'.format(pipe_params.working_folder, template)
                maybe_mkdir(outdir)

                ezparam['DUMP_TEMPLATE_CACHE'] = '1'
                cache_name = 'training_subset1_zs5_with_zp.{0}.tempfilt'.format(template)
                ezparam['CACHE_FILE'] = cache_name

                ezparam['OUTPUT_DIRECTORY'] = outdir
                ezparam['FIX_ZSPEC'] = '0'
                ezparam['GET_ZP_OFFSETS'] = '1'
                param_name = '{0}/training_subset1_zs5_with_zp.{1}.param'.format(outdir, template)
                ezparam.write(param_name)

                shutil.copy(zp_path, 'zphot.zeropoint')
                shutil.copy(zp_path, '{0}/{1}.zphot.zeropoint'.format(pipe_params.working_folder, template))

                runeazy(params = param_name, eazypath = pipe_params.eazypath,
                        translate = translate_temp_name, zeropoints = 'zphot.zeropoint', verbose=False)

                zout_zp = Table.read('{0}/training_subset1_zs5_with_zp.{1}.zout'.format(outdir, template),
                                     format='ascii.commented_header')

                stats_no_zp = pdf_calibration.calcStats(zout_no_zp['z_peak'], zout_no_zp['z_spec'], verbose=False)
                stats_zp = pdf_calibration.calcStats(zout_zp['z_peak'], zout_zp['z_spec'], verbose=False)

                stats_string = """
                {0[0]:>12s} {0[1]:>10s} {0[2]:>10s}
                {0[3]}
                {1[0]:>12s} {2[0]:>10.3f} {3[0]:>10.3f}
                {1[1]:>12s} {2[1]:>10.3f} {3[1]:>10.3f}
                {1[2]:>12s} {2[2]:>10.3f} {3[2]:>10.3f}
                {1[3]:>12s} {2[3]:>10.3f} {3[3]:>10.3f}
                {1[4]:>12s} {2[4]:>10.3f} {3[4]:>10.3f}
                {1[5]:>12s} {2[5]:>10.3f} {3[5]:>10.3f}
                {1[6]:>12s} {2[6]:>10.3f} {3[6]:>10.3f}
                {1[7]:>12s} {2[7]:>10.3f} {3[7]:>10.3f}
                """.format(['Param', 'No ZP', 'With ZP', '-'*35],
                           ['Sigma_all', 'Sigma_NMAD', 'Bias', 'OLF Def1', 'Sigma_OL1', 'OLF Def1', 'Sigma_OL1', 'KS'],
                           stats_no_zp,
                           stats_zp)
                print(stats_string)


            """
            Section 2c - Testing ZPs for Average offsets

            - Run once on test sample without zeropoint offsets
            - Run again on test sample *with* zeropoint offsets

            """

            ### Without ZP Offsets ###
            print('{0} ---No ZP--- {1}'.format('\n', '\n'))
            ezparam['CATALOG_FILE'] = '{0}/test_subset1.cat'.format(test_folder)
            ezparam['MAIN_OUTPUT_FILE'] = 'test_subset1_no_zp.{0}'.format(template)

            outdir = '{0}/testing/{1}/no_zp'.format(pipe_params.working_folder, template)
            maybe_mkdir(outdir)

            ezparam['DUMP_TEMPLATE_CACHE'] = '1'
            cache_name = 'test_subset1_no_zp.{0}.tempfilt'.format(template)
            ezparam['CACHE_FILE'] = cache_name

            ezparam['OUTPUT_DIRECTORY'] = outdir
            ezparam['FIX_ZSPEC'] = '0'
            ezparam['GET_ZP_OFFSETS'] = '0'
            param_name = '{0}/test_subset1_no_zp.{1}.param'.format(outdir, template)
            ezparam.write(param_name)


            runeazy(params = param_name, eazypath = pipe_params.eazypath,
                    translate = translate_temp_name, verbose=False)

            zout_no_zp = Table.read('{0}/test_subset1_no_zp.{1}.zout'.format(outdir, template),
                                    format='ascii.commented_header')

            ### With ZP Offsets ###
            print('{0} ---With ZP--- {1}'.format('\n', '\n'))
            ezparam['CATALOG_FILE'] = '{0}/test_subset1.cat'.format(test_folder)
            ezparam['MAIN_OUTPUT_FILE'] = 'test_subset1_with_zp.{0}'.format(template)

            outdir = '{0}/testing/{1}/with_zp'.format(pipe_params.working_folder, template)
            maybe_mkdir(outdir)

            ezparam['DUMP_TEMPLATE_CACHE'] = '1'
            cache_name = 'test_subset1_with_zp.{0}.tempfilt'.format(template)
            ezparam['CACHE_FILE'] = cache_name

            ezparam['OUTPUT_DIRECTORY'] = outdir
            ezparam['FIX_ZSPEC'] = '0'
            ezparam['GET_ZP_OFFSETS'] = '1'
            param_name = '{0}/test_subset1_with_zp.{1}.param'.format(outdir, template)
            ezparam.write(param_name)

            shutil.copy(zp_path, 'zphot.zeropoint')
            shutil.copy(zp_path, '{0}/{1}.zphot.zeropoint'.format(pipe_params.working_folder, template))

            runeazy(params = param_name, eazypath = pipe_params.eazypath,
                    translate = translate_temp_name, zeropoints = 'zphot.zeropoint', verbose=False)

            zout_zp = Table.read('{0}/test_subset1_with_zp.{1}.zout'.format(outdir, template),
                                 format='ascii.commented_header')

            stats_no_zp = pdf_calibration.calcStats(zout_no_zp['z_peak'], zout_no_zp['z_spec'], verbose=False)
            stats_zp = pdf_calibration.calcStats(zout_zp['z_peak'], zout_zp['z_spec'], verbose=False)

            stats_string = """
            {0[0]:>12s} {0[1]:>10s} {0[2]:>10s}
            {0[3]}
            {1[0]:>12s} {2[0]:>10.3f} {3[0]:>10.3f}
            {1[1]:>12s} {2[1]:>10.3f} {3[1]:>10.3f}
            {1[2]:>12s} {2[2]:>10.3f} {3[2]:>10.3f}
            {1[3]:>12s} {2[3]:>10.3f} {3[3]:>10.3f}
            {1[4]:>12s} {2[4]:>10.3f} {3[4]:>10.3f}
            {1[5]:>12s} {2[5]:>10.3f} {3[5]:>10.3f}
            {1[6]:>12s} {2[6]:>10.3f} {3[6]:>10.3f}
            {1[7]:>12s} {2[7]:>10.3f} {3[7]:>10.3f}
            """.format(['Param', 'No ZP', 'With ZP', '-'*35],
                       ['Sigma_all', 'Sigma_NMAD', 'Bias', 'OLF Def1', 'Sigma_OL1', 'OLF Def1', 'Sigma_OL1', 'KS'],
                       stats_no_zp,
                       stats_zp)
            print(stats_string)


        """
        Section 2d - Run ZP on full spectroscopic sample

        - Pass through to PDF/HB combination calibration

        """

        for itx, template in enumerate(pipe_params.templates):

            translate_temp_name = '{0}/{1}.{2}'.format(pipe_params.working_folder, pipe_params.translate_file, template)

            ##########################
            # Fixed throughout section
            ezparam = EazyParam(pipe_params.defaults[itx])
            ezparam['FILTERS_RES'] = '{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file)
            ezparam['TEMPLATE_COMBOS'] = pipe_params.fitting_mode[itx]
            ezparam['TEMP_ERR_A2'] = pipe_params.template_error_norm[itx]
            ezparam['SYS_ERR'] = pipe_params.additional_errors[itx]
            ezparam['LAF_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/LAFcoeff.txt')
            ezparam['DLA_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/DLAcoeff.txt')
            #########################


            ### With ZP Offsets ###
            print('{0} ---{1}: With ZP--- {2}'.format('\n', template, '\n'))
            ezparam['CATALOG_FILE'] = '{0}/training_all.cat'.format(test_folder)
            ezparam['MAIN_OUTPUT_FILE'] = 'training_all_with_zp.{0}'.format(template)

            maybe_mkdir('{0}/testing/all_specz/'.format(pipe_params.working_folder))
            outdir = '{0}/testing/all_specz/{1}'.format(pipe_params.working_folder, template)
            maybe_mkdir(outdir)

            ezparam['DUMP_TEMPLATE_CACHE'] = '1'
            cache_name = 'training_all_with_zp.{0}.tempfilt'.format(template)
            ezparam['CACHE_FILE'] = cache_name

            ezparam['OUTPUT_DIRECTORY'] = outdir
            ezparam['FIX_ZSPEC'] = '0'
            ezparam['GET_ZP_OFFSETS'] = '1'
            param_name = '{0}/training_all_with_zp.{1}.param'.format(outdir, template)
            ezparam.write(param_name)

            shutil.copy('{0}/{1}.zphot.zeropoint'.format(pipe_params.working_folder, template), 'zphot.zeropoint')

            runeazy(params = param_name, eazypath = pipe_params.eazypath,
                    translate = translate_temp_name, zeropoints = 'zphot.zeropoint', verbose=False)

            #zout_zp = Table.read('{0}/test_subset1_with_zp.{1}.zout'.format(outdir, template),
            #                     format='ascii.commented_header')



    if pipe_params.do_subcats:
        ### Make subset catalogs ###
        maybe_mkdir('{0}/full'.format(pipe_params.working_folder))
        # Build new catalog in EAZY format
        subsize = int(pipe_params.block_size)
        nsteps = int(len(photometry)/subsize)+1

        keep_cols = ['id', 'z_spec', 'AGN',
                     pipe_params.prior_colname]

        for col in ['IRClass', 'XrayClass', 'mqcAGN']:
            if col in photometry.colnames:
                keep_cols.append(col)

        for col in photometry.colnames:
            if np.logical_or(col.endswith(pipe_params.flux_col), col.endswith(pipe_params.fluxerr_col)):
                keep_cols.append(col)
            if col.endswith(pipe_params.mag_col):
                keep_cols.append(col)
            if col.endswith(pipe_params.magerr_col):
                keep_cols.append(col)

        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                ### Make folders
                subset_dir = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
                maybe_mkdir(subset_dir)

                ### Save catalog
                subset_photometry = Table(photometry[i*subsize:(i+1)*subsize])
                subset_photometry.keep_columns(keep_cols)
                subset_photometry.write('{0}/{1}.cat'.format(subset_dir, i+1),
                                        format='ascii.commented_header',
                                        overwrite=True)

                subset_photometry['z_spec'] = np.zeros(len(subset_photometry))
                subset_photometry.write('{0}/{1}.forstellar.cat'.format(subset_dir, i+1),
                                        format='ascii.commented_header',
                                        overwrite=True)

                bar.update()



    """
    Section 3 - Running Full Catalogs

    """
    if pipe_params.do_full:
        ### Make subset catalogs ###
        maybe_mkdir('{0}/full'.format(pipe_params.working_folder))
        # Build new catalog in EAZY format
        subsize = int(pipe_params.block_size)
        nsteps = int(len(photometry)/subsize)+1

        keep_cols = ['id', 'z_spec', 'AGN',
                     pipe_params.prior_colname]

        for col in ['IRClass', 'XrayClass', 'mqcAGN']:
            if col in photometry.colnames:
                keep_cols.append(col)

        for col in photometry.colnames:
            if np.logical_or(col.endswith(pipe_params.flux_col), col.endswith(pipe_params.fluxerr_col)):
                keep_cols.append(col)
            if col.endswith(pipe_params.mag_col):
                keep_cols.append(col)
            if col.endswith(pipe_params.magerr_col):
                keep_cols.append(col)

        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                ### Make folders
                subset_dir = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
                maybe_mkdir(subset_dir)

                ### Save catalog
                subset_photometry = Table(photometry[i*subsize:(i+1)*subsize])
                subset_photometry.keep_columns(keep_cols)
                subset_photometry.write('{0}/{1}.cat'.format(subset_dir, i+1),
                                        format='ascii.commented_header', overwrite=True)

                subset_photometry['z_spec'] = np.zeros(len(subset_photometry))
                subset_photometry.write('{0}/{1}.forstellar.cat'.format(subset_dir, i+1),
                                        format='ascii.commented_header', overwrite=True, fill_values=[(subset_photometry.masked, '-99.')])

                bar.update()

        # Begin EAZY fits for each template set
        for itx, template in enumerate(pipe_params.templates):
            print('\n\n{0}\n{1}{2:^16s}{3}\n{4}'.format('='*80, '='*32, template.upper(), '='*32, '='*80))
            translate_temp_name = '{0}/{1}.{2}'.format(pipe_params.working_folder, pipe_params.translate_file, template)

            ##########################
            # Fixed throughout section
            ezparam = EazyParam(pipe_params.defaults[itx])
            ezparam['FILTERS_RES'] = '{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file)
            ezparam['TEMPLATE_COMBOS'] = pipe_params.fitting_mode[itx]
            ezparam['TEMP_ERR_A2'] = pipe_params.template_error_norm[itx]
            ezparam['SYS_ERR'] = pipe_params.additional_errors[itx]
            ezparam['LAF_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/LAFcoeff.txt')
            ezparam['DLA_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/DLAcoeff.txt')
            #########################


            ### With ZP Offsets ###
            print('Writing param files:')
            with ProgressBar(nsteps) as bar:
                for i in range(nsteps):
                    #if np.logical_and(itx==0, np.logical_or(i == 50, i == 96)):
                    #    ezparam['N_MIN_COLORS'] = '6'
                    #    print('Modifying N_min_col')

                    ezparam['CATALOG_FILE'] = '{0}/full/{1}/{2}.cat'.format(pipe_params.working_folder, i+1, i+1)
                    ezparam['MAIN_OUTPUT_FILE'] = '{0}.{1}'.format(template, i+1)

                    maybe_mkdir('{0}/testing/all_specz/'.format(pipe_params.working_folder))
                    outdir = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
                    maybe_mkdir(outdir)

                    ezparam['DUMP_TEMPLATE_CACHE'] = '1'
                    cache_name = '{0}.{1}.tempfilt'.format(template, i+1)
                    ezparam['CACHE_FILE'] = cache_name

                    ezparam['OUTPUT_DIRECTORY'] = outdir
                    ezparam['FIX_ZSPEC'] = '0'
                    ezparam['GET_ZP_OFFSETS'] = '1'

                    param_name = '{0}/{1}.{2}.param'.format(outdir, template, i+1)
                    ezparam.write(param_name)
                    bar.update()

            shutil.copy('{0}/{1}.zphot.zeropoint'.format(pipe_params.working_folder, template), 'zphot.zeropoint')


            print()

            inputQueue = mp.Queue()
            outputQueue = mp.Queue()
            outLock = mp.Lock()

            verbose = False
            clean = True

            outpaths = []


            print('Beginning loop:')
            with ProgressBar(nsteps) as bar:
                for i in range(nsteps):
                    inputQueue.put(i)

                for i in range(pipe_params.ncpus):
                    mp.Process(target=eazyfunction_worker,
                               args=(inputQueue, outputQueue, outLock,
                                       template,
                                       translate_temp_name, 'zphot.zeropoint',
                                       pipe_params.eazypath,
                                       verbose)).start()

                for i in range(nsteps):
                    path = outputQueue.get()
                    outpaths.append(path)
                    bar.update(i+1)

                for i in range( pipe_params.ncpus ):
                    inputQueue.put( 'STOP' )

                inputQueue.close()
                outputQueue.close()

            print('{0}{1:^16s}{2}'.format('='*32,'Completed'.upper(), '='*32))

    if pipe_params.do_stellar:
        """
        Star fits

        """
        subsize = pipe_params.block_size
        nsteps = int(len(photometry)/subsize)+1

        print('\n\n{0}\n{1}{2:^16s}{3}\n{4}'.format('='*80, '='*32, 'Pickles'.upper(), '='*32, '='*80))

        translate_temp_name = '{0}/{1}'.format(pipe_params.working_folder,
                                                   pipe_params.translate_file+'.opt')


        ##########################
        # Fixed throughout section
        ezparam = EazyParam(pipe_params.stellar_params)
        ezparam['FILTERS_RES'] = '{0}/{1}'.format(pipe_params.working_folder, pipe_params.filter_file)
        ezparam['TEMPLATE_COMBOS'] = '1'
        ezparam['TEMP_ERR_A2'] = pipe_params.template_error_norm[0]
        ezparam['SYS_ERR'] = pipe_params.additional_errors[0]
        ezparam['LAF_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/LAFcoeff.txt')
        ezparam['DLA_FILE'] = '{0}/{1}'.format(pipe_params.eazypath[:-9], 'inputs/templates/DLAcoeff.txt')
        #########################


        ### With ZP Offsets ###
        print('Writing param files:')
        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                ezparam['CATALOG_FILE'] = '{0}/full/{1}/{2}.forstellar.cat'.format(pipe_params.working_folder,
                                                                                   i+1, i+1)
                ezparam['MAIN_OUTPUT_FILE'] = '{0}.{1}'.format('pickles', i+1)

                outdir = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
                maybe_mkdir(outdir)

                ezparam['DUMP_TEMPLATE_CACHE'] = '1'
                cache_name = '{0}.{1}.tempfilt'.format('pickles', i+1)
                ezparam['CACHE_FILE'] = cache_name

                ezparam['OUTPUT_DIRECTORY'] = outdir
                ezparam['FIX_ZSPEC'] = '1'
                ezparam['GET_ZP_OFFSETS'] = '0'

                param_name = '{0}/{1}.{2}.param'.format(outdir, 'pickles', i+1)
                ezparam.write(param_name)
                bar.update()

        #shutil.copy('{0}/{1}.zphot.zeropoint'.format(pipe_params.working_folder, template), 'zphot.zeropoint')


        print()

        inputQueue = mp.Queue()
        outputQueue = mp.Queue()
        outLock = mp.Lock()

        verbose = False
        clean = True

        outpaths = []


        print('Beginning loop:')
        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                inputQueue.put(i)

            for i in range(pipe_params.ncpus):
                mp.Process(target = eazyfunction_worker,
                           args = (inputQueue, outputQueue, outLock, 'pickles',
                                   translate_temp_name, None,
                                   pipe_params.eazypath, verbose)).start()

            for i in range(nsteps):
                path = outputQueue.get()
                outpaths.append(path)
                bar.update(i+1)

            for i in range( pipe_params.ncpus ):
                inputQueue.put( 'STOP' )

            inputQueue.close()
            outputQueue.close()

        print('{0}{1:^16s}{2}'.format('='*32,'Completed'.upper(), '='*32))



    """
    Section 4 - Post-processing

    """


    if pipe_params.do_hb:
        subsize = pipe_params.block_size
        nsteps = int(len(photometry)/subsize)+1

        folder = '{0}/testing/all_specz'.format(pipe_params.working_folder)
        try:
            path = '{0}/HB_hyperparameters_gal_{1}.npz'.format(folder, pipe_params.prior_fname)
            hyperparams_gal = np.load(path)

            path = '{0}/HB_hyperparameters_agn_{1}.npz'.format(folder, pipe_params.prior_fname)
            hyperparams_agn = np.load(path)

        except:
            print('Hyper-parameters not found. Assuming calibration not run. Beginning HB calibration...')
            command = ('python hbcombination.py -p {params}'.format(params = args.params))

            output = call(command, shell=True)

            path = '{0}/HB_hyperparameters_gal_{1}.npz'.format(folder, pipe_params.prior_fname)
            hyperparams_gal = np.load(path)

            path = '{0}/HB_hyperparameters_agn_{1}.npz'.format(folder, pipe_params.prior_fname)
            hyperparams_agn = np.load(path)


        alphas_gal = hyperparams_gal['alphas']
        beta_gal = hyperparams_gal['beta']

        alphas_agn = hyperparams_agn['alphas']
        beta_agn = hyperparams_agn['beta']

        if (pipe_params.fbad_prior == 'mag') or pipe_params.include_prior:
            filt = pipe_params.prior_fname
            folder = '{0}/testing/all_specz'.format(pipe_params.working_folder)
            prior_params = np.load('{0}/{1}_{2}_prior_coeff.npz'.format(pipe_params.working_folder, filt, 'gal'))
            z0t = prior_params['z0t']
            kmt1 = prior_params['kmt1']
            kmt2 = prior_params['kmt2']
            alpha = prior_params['alpha']

            best_prior_params_gal = [z0t[0], kmt1[0], kmt2[0], alpha[0]]

            prior_params = np.load('{0}/{1}_{2}_prior_coeff.npz'.format(pipe_params.working_folder, filt, 'agn'))
            z0t = prior_params['z0t']
            kmt1 = prior_params['kmt1']
            kmt2 = prior_params['kmt2']
            alpha = prior_params['alpha']
            best_prior_params_agn = [z0t[0], kmt1[0], kmt2[0], alpha[0]]


        print()

        inputQueue = mp.Queue()
        outputQueue = mp.Queue()
        outLock = mp.Lock()

        verbose = False
        clean = True

        outpaths = []


        print('Beginning loop:')
        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                inputQueue.put(i)

            for i in range(pipe_params.ncpus):
                mp.Process(target = hb_worker,
                           args = (inputQueue, outputQueue, outLock)).start()

            for i in range(nsteps):
                path = outputQueue.get()
                outpaths.append(path)
                bar.update(i+1)

            for i in range( pipe_params.ncpus ):
                inputQueue.put( 'STOP' )

            inputQueue.close()
            outputQueue.close()

        print('{0}{1:^16s}{2}'.format('='*32,'Completed'.upper(), '='*32))


    """
    Section 5 - Merging and formatting outputs

    """

    if pipe_params.do_merge:
        command = ('python merge.py -p {params}'.format(params = args.params))
        output = call(command, shell=True)

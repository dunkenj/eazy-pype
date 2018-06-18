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

    #root='COSMOS/OUTPUT/cat3.4_default_lines_zp33sspNoU'

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
            eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy', verbose=True):
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


    Main steps:
        1. Modify EazyParam class to corresponding catalog, filter,
           output paths etc.
        2. Run Eazy with generated params file ('runeazy')

    """
    for i in iter(inputQueue.get, 'STOP'):
        params_path = '{0}/full/{1}/{2}.{3}.param'.format(pipe_params.working_folder, i+1, template, i+1)

        # Run eazy with params file
        out = runeazy(params_path, translate=translate, zeropoints=zeropoints,
                      verbose=verbose, eazypath=eazypath)

        outLock.acquire()
        outputQueue.put(params_path)
        outLock.release()

"""
def eazyfunction_worker(inputQueue, outputQueue, outLock, paramLock,
                        outdir, translate, zeropoints, verbose=False, clean=True):
     Run EAZY for a given catalog

    Args:
        gal: int - index of galaxy to fit within catalog (in shared memory)
        params: EazyParam class
        outdir: string - Path to output directory (if not current directory)
        clean: boolean - If True, tidy up and remove intermediate files.


    Main steps:
        1. Modify EazyParam class to corresponding catalog, filter,
           output paths etc.
        2. Run Eazy with generated params file ('runeazy')


    for it in iter(inputQueue.get, 'STOP'):
        gal = it
        # (gal, params, outdir = '', clean = False, verbose=True)
    # Make catalog
        galaxy_fluxes = Table(eazycat[gal*subsize:(gal+1)*subsize])
        galaxy_fluxes.write(baseout+str(gal)+'.eazy',format='ascii.commented_header')


        # Modify and save params file
        with paramLock:
            params['CACHE_FILE'] = baseout+str(gal)+'.tempfilt'
            params['CATALOG_FILE'] = baseout+str(gal)+'.eazy'
            params['OUTPUT_DIRECTORY'] = outdir
            params['MAIN_OUTPUT_FILE'] = baseout+str(gal)
            params.write(baseout+str(gal)+'.param')

        # Run eazy with params file
        out = runeazy(baseout+str(gal)+'.param', translate=translate, zeropoints=zeropoints, verbose=verbose)

        with outLock:
            outputQueue.put(baseout+str(gal))
"""
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
    return gp, bands, alpha_values

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

def maybe_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def hb_worker(inputQueue, outputQueue, outLock):
    """ Run EAZY for a given catalog

    Args:
        gal: int - index of galaxy to fit within catalog (in shared memory)
        params: EazyParam class
        outdir: string - Path to output directory (if not current directory)
        clean: boolean - If True, tidy up and remove intermediate files.


    Main steps:
        1. Modify EazyParam class to corresponding catalog, filter,
           output paths etc.
        2. Run Eazy with generated params file ('runeazy')

    """
    for i in iter(inputQueue.get, 'STOP'):
        pzarr = []
        zouts = []

        photometry_path = '{0}/full/{1}/{2}.cat'.format(pipe_params.working_folder, i+1, i+1)
        photom = Table.read(photometry_path,
                            format='ascii.commented_header')

        AGN = (photom['AGN'] == 1)
        if pipe_params.gpz:
            IRAGN = (photom['IRClass'] >= 4)
            XR = (photom['XrayClass'] == 1)
            OPT = (photom['mqcAGN'] == 'True')
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

                pzprior = np.zeros_like(pz)
                pzprior[GAL] = priors.pzl(zgrid, mags[GAL], *best_prior_params_gal, lzc=0.003)
                pzprior[AGN] = priors.pzl(zgrid, mags[AGN], *best_prior_params_agn, lzc=0.000)

                pzprior_nomag = np.ones_like(zgrid)
                pzprior_nomag /= np.trapz(pzprior_nomag, zgrid)
                pzprior[mags < -90.] = pzprior_nomag

                pz *= pzprior
                pz /= np.trapz(pz, zgrid, axis=1)[:, None]

            catalog = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')

            alphas_best_gal = hb.alphas_mag(photom[pipe_params.alpha_colname][GAL], *alphas_gal[itx])[:, None]
            pz[GAL] = pz[GAL]**(1/alphas_best_gal)

            alphas_best_agn = hb.alphas_mag(photom[pipe_params.alpha_colname][AGN], *alphas_agn[itx])[:, None]
            pz[AGN] = pz[AGN]**(1/alphas_best_agn)

            pzarr.append(pz)
            zouts.append(catalog)

        if pipe_params.gpz:
            gp_ir, bands_ir, alpha_values_ir = load_gp('{0}'.format(pipe_params.ir_gpz_path))
            gp_xray, bands_xray, alpha_values_xray = load_gp('{0}'.format(pipe_params.xray_gpz_path))
            gp_opt, bands_opt, alpha_values_opt = load_gp('{0}'.format(pipe_params.opt_gpz_path))
            #gp_gal, bands_gal, alpha_values_gal = load_gp('{0}'.format(pipe_params.gal_gpz_path))

            sets = [IRAGN, XR, OPT]
            bands = [bands_ir, bands_xray, bands_opt]
            alphas = [alpha_values_ir, alpha_values_xray, alpha_values_opt]
            gpz = [gp_ir, gp_xray, gp_opt]
            
            for path in pipe_params.gal_gpz_paths:
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
            pzbad = np.ones_like(zgrid) # Flat prior
            pzbad /= np.trapz(pzbad, zgrid)

        elif pipe_params.fbad_prior == 'vol':
            pzbad = hb.cosmo.differential_comoving_volume(zgrid)
            pzbad /= np.trapz(pzbad, zgrid) # Volume element prior (for our cosmology)

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
        pzarr_hb[AGN] = hb.HBpz(pzarr[:, AGN], zgrid, pzbad[AGN], beta_agn, fbad_max = 0.3, fbad_min = 0.)
        pzarr_hb /= np.trapz(pzarr_hb, zgrid, axis=1)[:, None]

        hbcat = hb.pz_to_catalog(pzarr_hb, zgrid, zouts[0], verbose=False)

        catpath = '{0}.{1}.cat'.format('HB', i+1)
        hbcat.write('{0}/{1}'.format(folder, catpath), format='ascii.commented_header')

        np.savez('{0}/{1}.npz'.format(folder, 'HB_pz'), zgrid=zgrid, pz=pzarr_hb)

        outLock.acquire()
        outputQueue.put(catpath)
        outLock.release()



"""
Section 4 - Post-processing

"""

do_hb = True

    # Read in catalog
try:
    photometry = Table.read('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog),
                            format = pipe_params.photometry_format)
except:
    raise

if do_hb:
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

    if pipe_params.fbad_prior == 'mag':
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

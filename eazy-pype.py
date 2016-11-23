import numpy as np
import array
import os, sys, shutil
import re
import time
import multiprocessing as mp
from subprocess import call

import smpy.smpy as S
#from astropy.table import Table, Column
from astropy import units as u
from astropy.table import Table
from astropy.utils.console import ProgressBar
from sklearn.cross_validation import ShuffleSplit

import validation
import zeropoints
import pdf_calibration

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
                        translate, zeropoints, verbose=False, clean=True):
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
        out = runeazy(params_path, translate=translate, zeropoints=zeropoints, verbose=verbose)
                        
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
        ec = ['Total_flux', 'E_Total_flux']
        new_path, bad_frac = validation.process('{0}/{1}'.format(pipe_params.working_folder, pipe_params.photometry_catalog),
                                                cat_format=pipe_params.photometry_format,
                                                exclude_columns = ec)
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
    
        
    training_cut = (photometry[pipe_params.zspec_col] >= -1000.) 
    photometry_training = photometry[training_cut]
    fname = '{0}/training_all.cat'.format(test_folder)
    if os.path.isfile(fname):
        os.remove(fname)
    photometry_training.write(fname, format='ascii.commented_header')

    rs = ShuffleSplit(len(photometry_training), pipe_params.Ncrossval, test_size = pipe_params.test_fraction)

    for i, (train_index, test_index) in enumerate(rs):
        print('Writing test and training subset: {0}'.format(i+1))
        
        fname = '{0}/training_subset{1}.cat'.format(test_folder, i+1)
        if os.path.isfile(fname):
            os.remove(fname)
        training_subset = photometry_training[np.sort(train_index)]
        training_subset.write(fname, format='ascii.commented_header')

        fname = '{0}/test_subset{1}.cat'.format(test_folder, i+1)
        if os.path.isfile(fname):
            os.remove(fname)
        test_subset = photometry_training[np.sort(test_index)]
        test_subset.write(fname, format='ascii.commented_header')


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

            
            runeazy(params = param_name, 
                    translate = translate_temp_name, verbose=False)
            
        
            zp_path, Fig = zeropoints.calc_zeropoints('{0}/training_subset1_calc_zp.{1}'.format(outdir, template),
                                                      verbose=True)
        
        
        
            """
            Section 2b - Testing ZPs
            
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

            
            runeazy(params = param_name, 
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
            
            runeazy(params = param_name, 
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
        Section 2c - Run ZP on full spectroscopic sample
        
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

            runeazy(params = param_name, 
                    translate = translate_temp_name, zeropoints = 'zphot.zeropoint', verbose=False)
        
            #zout_zp = Table.read('{0}/test_subset1_with_zp.{1}.zout'.format(outdir, template),
            #                     format='ascii.commented_header')           
    
    """
    Section 3 - Running Full Catalogs
    
    """
    if pipe_params.do_full:
        ### Make subset catalogs ###
        maybe_mkdir('{0}/full'.format(pipe_params.working_folder))
        # Build new catalog in EAZY format
        subsize = pipe_params.block_size
        nsteps = int(len(photometry)/subsize)+1
        
        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                ### Make folders
                subset_dir = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
                maybe_mkdir(subset_dir)
                
                ### Save catalog
                subset_photometry = Table(photometry[i*subsize:(i+1)*subsize])
                subset_photometry.write('{0}/{1}.cat'.format(subset_dir, i+1),format='ascii.commented_header')
                
                subset_photometry['z_spec'] = np.zeros(len(subset_photometry))
                subset_photometry.write('{0}/{1}.forstellar.cat'.format(subset_dir, i+1),format='ascii.commented_header')
                
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
            #########################
            

            ### With ZP Offsets ###
            print('Writing param files:')
            with ProgressBar(nsteps) as bar:
                for i in range(nsteps):
                    ezparam['CATALOG_FILE'] = '{0}/full/{1}/{2}.cat'.format(pipe_params.working_folder, i+1, i+1)
                    ezparam['MAIN_OUTPUT_FILE'] = '{0}.{1}.cat'.format(template, i+1)
                    
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
                    mp.Process(target = eazyfunction_worker, 
                               args = (inputQueue, outputQueue, outLock, template,
                                       translate_temp_name, 'zphot.zeropoint', verbose)).start()
                
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
        #########################
        

        ### With ZP Offsets ###
        print('Writing param files:')
        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):
                ezparam['CATALOG_FILE'] = '{0}/full/{1}/{2}.forstellar.cat'.format(pipe_params.working_folder, 
                                                                                   i+1, i+1)
                ezparam['MAIN_OUTPUT_FILE'] = '{0}.{1}.cat'.format('pickles', i+1)
                
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
                                   translate_temp_name, None, verbose)).start()
            
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
        path = '{0}/HB_hyperparameters.npz'.format(folder)
        hyperparams = np.load(path)
        alphas = hyperparams['alphas']
        beta = hyperparams['beta']
    

        with ProgressBar(nsteps) as bar:
            for i in range(nsteps):               
                pzarr = []
                zouts = []
                photometry_path = '{0}/full/{1}/{2}.cat'.format(pipe_params.working_folder, i+1, i+1)
                photom = Table.read(photometry_path,
                                    format='ascii.commented_header')    
                
                z_max_name = ['z_a', 'z_1', 'z_1']
                titles = ['EAZY', 'XMM-COSMOS (Salvato+)', 'ATLAS (Brown+)']
                
                folder = '{0}/full/{1}'.format(pipe_params.working_folder, i+1)
                
                for itx, template in enumerate(pipe_params.templates):
                    print(template)
                    
                    """ Load Values/Arrays/Catalogs """
                    folder = '{0}/testing/all_specz/{1}'.format(pipe_params.working_folder, template)
                    basename='training_all_with_zp.{0}'.format(template)
                    pz, zgrid = getPz('{0}/{1}'.format(folder, basename))
                    catalog = Table.read('{0}/{1}.zout'.format(folder, basename), format='ascii.commented_header')
        
    
    """
    Section 5 - Merging and formatting outputs
    
    """
    
    
    
    
        

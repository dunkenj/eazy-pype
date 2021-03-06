"""
Main inputs:
(Change for all fields)

"""
eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy'
working_folder = '/data2/ken/photoz/bootes_3as_ext2'
photometry_catalog = 'Bootes_merged_Icorr_2014a_all_ap3_mags.zs.sample.fits'
photometry_format = 'fits'

filter_file = 'filter.bootes_mbrown_2014a.res'
translate_file = 'brown.zphot.2014.translate'

zspec_col = 'z_spec'

flux_col = 'flux'
fluxerr_col ='fluxerr'

do_zp = True
do_zp_tests = False
do_subcats = False

do_full = False
do_stellar = False
do_hb = False
do_merge = False

"""
Training parameters

"""
Ncrossval = 1
test_fraction = 0.05

process_outliers = False
correct_extinction = False

"""
Fitting Parameters
(Change only when needed)

"""

# Templates: Any combination of 'eazy', 'swire', 'atlas'
templates = ['eazy', 'atlas', 'cosmos']#, 'swire']#, 'cosmos', 'atlas'] #,'cosmos', 'atlas']  
fitting_mode = ['a', '1', '1']

defaults = ['defaults/zphot.eazy',
            'defaults/zphot.atlas_ext',
            'defaults/zphot.cosmos']
            #'defaults/zphot.eazy',
            #'defaults/zphot.atlas',
            #'defaults/zphot.swire']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0., 0.1, 0.1]
template_error_norm = [1., 0., 0.]
template_error_file = ''
lambda_fit_max = [5., 20., 20.]



"""
Combination Parameters

"""
include_prior = True
fbad_prior = 'mag' # 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_fname = 'ch2_mag'
prior_colname = 'ch2_mag'
alpha_colname = 'I_mag'

gpz = False
"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 10



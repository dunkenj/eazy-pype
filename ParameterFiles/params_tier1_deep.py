"""
Main inputs:
(Change for all fields)

"""
eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy'
working_folder = '/data2/ken/photoz/tier1_calibration'
photometry_catalog = 'tier1_ps1_wise_merged_subset.fits'
photometry_format = 'fits'

filter_file = 'tier1_filters.res'
translate_file = 'tier1.translate'

zspec_col = 'z_spec'

flux_col = 'Flux'
fluxerr_col = 'FluxErr'

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
test_fraction = 0.2

process_outliers = True
correct_extinction = True

"""
Fitting Parameters
(Change only when needed)

"""

# Templates: Any combination of 'eazy', 'swire', 'atlas'
templates = ['eazy', 'atlas', 'cosmos']#, 'swire']#, 'cosmos', 'atlas'] #,'cosmos', 'atlas']
fitting_mode = ['a', '1', '1']
defaults = ['defaults/zphot.eazy',
            'defaults/zphot.atlas',
            'defaults/zphot.cosmos']
            #'defaults/zphot.swire']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0.0, 0.0, 0.0]
template_error_norm = [0., 0., 0.]
template_error_file = ''
lambda_fit_max = [5., 15., 15.]



"""
Combination Parameters

"""
include_prior = True
fbad_prior = 'mag' # 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_fname = 'iFAp'
prior_colname = 'iFApMag'
alpha_colname = 'iFApMag'


"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 10

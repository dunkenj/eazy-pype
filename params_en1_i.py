"""
Main inputs:
(Change for all fields)

"""
eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy '
working_folder = '/data2/ken/EN1_pani'
photometry_catalog = 'en1_phot_with_zspec.fits'
photometry_format = 'fits'

filter_file = 'EN1_filters.res'
translate_file = 'EN1.translate'

zspec_col = 'z_spec'

flux_col = 'flux'
fluxerr_col ='fluxerr'

do_zp = False
do_zp_tests = False
do_subcats = False

do_full = False
do_stellar = False
do_hb = True
do_merge = True

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
            #'defaults/zphot.eazy',
            #'defaults/zphot.atlas',
            #'defaults/zphot.swire']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0.0, 0.0, 0.0]
template_error_norm = [1., 1., 1.]
template_error_file = ''
lambda_fit_max = [5., 30., 30.]



"""
Combination Parameters

"""
include_prior = True
fbad_prior = 'mag' # 'flat', 'vol' or 'mag'
#prior_parameter_path = 'en1_i_prior_coeff.npz'
prior_fname = 'pani_mag'
prior_colname = 'pani_mag'
alpha_colname = 'pani_mag'


"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 10

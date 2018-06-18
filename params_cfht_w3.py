"""
Main inputs:
(Change for all fields)

"""
eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy'
working_folder = '/data2/ken/CFHTW3'
photometry_catalog = 'cfhtls_dw3_phot_with_zspec.fits'
photometry_format = 'fits'

filter_file = 'CFHT_filters.res'
translate_file = 'CFHT.translate'

zspec_col = 'z_spec'

flux_col = 'flux'
fluxerr_col = 'fluxerr'

do_zp = True
do_zp_tests = False

do_subcats = False
do_full = True
do_stellar = True
do_hb = False
do_merge = False

"""
Training parameters

"""
Ncrossval = 1
test_fraction = 0.2

process_outliers = False
correct_extinction = True

"""
Fitting Parameters
(Change only when needed)

"""

# Templates: Any combination of 'eazy', 'cosmos', 'atlas'
templates = ['eazy', 'atlas', 'cosmos'] 
fitting_mode = ['a', '1', '1']
defaults = ['defaults/zphot.eazy',
            'defaults/zphot.atlas',
            'defaults/zphot.cosmos']
            #'defaults/zphot.swire']

stellar_params = 'defaults/zphot.pickles'

additional_errors = [0.0, 0.0, 0.0]
template_error_norm = [1., 1., 1.]
template_error_file = ''
lambda_fit_max = [5., 30., 30.]



"""
Combination Parameters

"""

fbad_prior = 'mag' # 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_fname = 'ks'
prior_colname = 'ks_mag'
alpha_colname = 'ip_mag'


"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 5



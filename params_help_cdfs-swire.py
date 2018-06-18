"""
Main inputs:
(Change for all fields)

"""
eazypath = '/data2/ken/photoz/eazy-photoz/src/eazy'
working_folder = '/data2/ken/HELP/CDFS-SWIRE'
photometry_catalog = 'master_catalogue_cdfs-swire_20171103_processed.fits.mod'

photometry_format = 'fits'

filter_file = 'cdfs-swire_filters.res'
translate_file = 'cdfs-swire.translate'

zspec_col = 'z_spec'

flux_col = 'flux'
fluxerr_col = 'fluxerr'
mag_col = '_mag'
magerr_col = '_magerr'

do_zp = False
do_zp_tests = False

do_subcats = True
do_full = False
do_stellar = False
do_hb = True
do_merge = True

"""
Training parameters

"""
Ncrossval = 1
test_fraction = 0.2

process_outliers = False
correct_extinction = False

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
include_prior = True
include_prior_gal = True
include_prior_agn = True
fbad_prior = 'flat' # 'flat', 'vol' or 'mag'
prior_parameter_path = 'bootes_I_prior_coeff.npz'
prior_fname = 'r_any_mag' 
prior_colname = 'r_any_mag' 
alpha_colname = 'r_any_mag' 

gpz = True

ir_gpz_path = None
xray_gpz_path = None
opt_gpz_path = None
gal_gpz_paths = ['/data2/ken/HELP/CDFS-SWIRE/gpz/ps1_griz_weight_save.pkl',
                 '/data2/ken/HELP/CDFS-SWIRE/gpz/ps1_griz_video_yjk_weight_save.pkl']


"""
System Parameters
(Specific system only - fixed after installation)

"""

block_size = 1e4
ncpus = 8


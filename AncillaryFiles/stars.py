import os, sys, re
import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from scipy.stats import ks_2samp
from scipy.stats import randint as sp_randint
from astropy.stats import bootstrap
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

import astropy.units as u
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar
import emcee

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
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


    cat_crd = SkyCoord(photometry['RA'], photometry['DEC'], unit='deg')

    sdss = Table.read('/data2/ken/proposals/af2_jun2016/sdss_bootes_v2.fits')
    sdss = sdss[sdss['type'] == 6]
    
    sdss_crd = SkyCoord(sdss['ra'], sdss['dec'], unit='deg')
    
    mag_cut = (photometry['I_mag'] < 20.) * (photometry['FLAG_DEEP'] == 1)

    idx, d2d, d3d = cat_crd.match_to_catalog_sky(sdss_crd)
    star = (d2d < 2*u.arcsec)*mag_cut
    gal = np.invert(star) * mag_cut

    forclass = zout.copy()
    forclass.keep_columns(['chi_r_eazy', 'chi_r_cosmos', 'chi_r_atlas', 'chi_r_stellar'])
    forclass.add_column(photometry['I_mag'].astype('f8'))
    forclass.add_column(photometry['CLASS_STAR'].astype('f8'))
    forclass['DZ'] = (zout['z1_max']-zout['z1_min'])/zout['z1_median']
    #forclass['STAR'] = np.array(star).astype('int')
    forclass = forclass[mag_cut]

    class_info = forclass.as_array()
    class_info = class_info.view(np.float64).reshape(class_info.shape + (-1,))

    stars = star[mag_cut].astype('int')

    shuff  = ShuffleSplit(len(class_info), 1, test_size = 2000, random_state = 0)
    for t1, t2 in shuff:
        train_index, test_index = t1, t2

    info_train = class_info[train_index]
    info_test  = class_info[test_index]
    OL_train = stars[train_index]
    OL_test = stars[test_index]

    clf = RandomForestClassifier(n_estimators=20)


    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")


    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 6),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(info_train, OL_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_, n_top=10)

    stop

    forest = RandomForestClassifier(n_estimators=100, **random_search.best_params_)
    scores = cross_val_score(forest, class_info, stars)
    print scores.mean()

    forest.fit(info_train, OL_train)
    #joblib.dump(forest, 'trained_forest_classifier.pkl')

    pred_forest = forest.predict(info_test)
    pred_forest_prob = forest.predict_proba(info_test)[:,1]
    
    print(classification_report(OL_test, pred_forest, target_names=['GAL', 'STAR']))
    stop

    pred_all = forest.predict_proba(class_info)    

    all_chis = np.array([zout['chi_r_eazy'].data, zout['chi_r_cosmos'].data, zout['chi_r_atlas'].data])
    all_chis[all_chis < 0.] = np.nan
    chi_best = np.nanmin(all_chis, axis=0)

    star = (chi_best > zout['chi_r_stellar']) * (zout['chi_r_stellar'] < 4.)

    Fig, Ax = plt.subplots(3, figsize=(5.5, 10.))
    
    Ax[0].semilogx(zout['chi_r_eazy'][gal]/zout['chi_r_stellar'][gal], photometry['CLASS_STAR'][gal], ',')
    Ax[0].semilogx(zout['chi_r_eazy'][star]/zout['chi_r_stellar'][star], photometry['CLASS_STAR'][star], 'o', alpha=0.1)
    #Ax[0].plot([1e-2, 100], [1e-2, 100], 'k--', lw=2)

    Ax[1].semilogx(zout['chi_r_cosmos'][gal]/zout['chi_r_stellar'][gal], photometry['CLASS_STAR'][gal], ',')
    Ax[1].semilogx(zout['chi_r_cosmos'][star]/zout['chi_r_stellar'][star], photometry['CLASS_STAR'][star], 'o', alpha=0.1)
    #Ax[1].plot([1e-2, 100], [1e-2, 100], 'k--', lw=2)

    Ax[2].semilogx(zout['chi_r_atlas'][gal]/zout['chi_r_stellar'][gal], photometry['CLASS_STAR'][gal], ',')
    Ax[2].semilogx(zout['chi_r_atlas'][star]/zout['chi_r_stellar'][star], photometry['CLASS_STAR'][star], 'o', alpha=0.1)
    #Ax[2].plot([1e-2, 100], [1e-2, 100], 'k--', lw=2)
        


    
    plt.show()

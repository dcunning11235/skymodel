import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA
from sklearn import preprocessing as skpp
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import ICAize

import fnmatch
import os
import os.path
import sys
import pickle as pk

from astropy.utils.compat import argparse

RANDOM_STATE = 456371


hide_plots=True #False
use_spca=False #True
use_pca=False #True
restrict_delta = True
restrict_color = 'blue'
include_knn = False #True
include_linear = False #True
linear_only = False #True
order_3 = False #True
order_4 = False #True #False
n_estimators = 200
save_out = False
scale = False #True
scale_std = False
proc_ecliptic_angles = False
proc_lunar_sep = False #True
proc_lunar_mag = False #True
reg_type = 'etr'
bootstrap = False
min_samples_split = 1
compute_chi_fit = True

def load_observation_metadata(path='.', file='annotated_metadata.csv'):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    if proc_lunar_mag:
        data['LUNAR_MAGNITUDE'] = np.power(2.512, -data['LUNAR_MAGNITUDE'])

    return data

def trim_observation_metadata(data, copy=False):
    if copy:
        data = data.copy()

    kept_columns = ['EXP_ID', 'RA', 'DEC',
                    'AZ',
                    'ALT', 'AIRMASS',
                    'LUNAR_MAGNITUDE', 'LUNAR_ELV', 'LUNAR_SEP', 'SOLAR_ELV',
                    'SOLAR_SEP', 'GALACTIC_CORE_SEP',
                    'GALACTIC_PLANE_SEP',
                    'SS_COUNT', 'SS_AREA',
                    'ECLIPTIC_PLANE_SEP',
                    'ECLIPTIC_PLANE_SOLAR_SEP']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Build and test models based on dim reductions and provided spectra'
    )
    parser.add_argument(
        '--metadata_path', type=str, default='.', metavar='PATH',
        help='Metadata path to work from, if not ''.'''
    )
    parser.add_argument(
        '--spectra_path', type=str, default='.', metavar='PATH',
        help='Spectra path to work from, if not ''.'''
    )
    parser.add_argument(
        '--method', type=str, default='ICA', metavar='METHOD',
        help='Dim reduction method to load data for'
    )

    parser.add_argument(
        '--test_inds', type=str, default=None, metavar='TEST_INDS',
        help='Indicies of loaded spectra to use in testing model; \'ALL\' is ' \
            'leave-one-out for the whole set, indvidual numbers or ranges do leave-one-out '\
            'for those specified spectra.  OVERRIDES --test_folds; IGNORED by --hyper_fit'
    )
    parser.add_argument(
        '--test_folds', type=int, default=None, metavar='TEST_FOLDS',
        help='Do k-fold cross validation with specified number of folds.  Defaults to 3.'
    )
    parser.add_argument(
        '--hyper_fit', action='store_true',
        help='Do hyperparamter estimation using CV (see --test_inds or --test_folds), and output'
    )
    parser.add_argument(
        '--hyper_fit_iters', type=int, default=50, metavar='HYPER_FIT_ITERS',
        help='Number of iterations when fitting hyper-params'
    )
    parser.add_argument(
        '--hyper_outputfbk', action='store_true',
        help='If set, outputs \'grid_scores_\' data from RandomizedSearchCV'
    )

    parser.add_argument(
        '--n_jobs', type=int, default=1, metavar='N_JOBS',
        help='N_JOBS'
    )

    parser.add_argument(
        '--model', type=str, choices=['ET', 'RF', 'GP', 'KNN'], default='ET',
        help='Which model type to use: ET (Extra Trees), RF (Random Forest), GP (Gaussian Process), or KNN'
    )
    parser.add_argument(
        '--save_model', action='store_true',
        help='Whether or not to save the (last/best) model built for e.g. --hyper_fit'
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help='Whether or not to load the model from --model_path'
    )
    parser.add_argument(
        '--model_path', type=str, default='model.pkl', metavar='MODEL_PATH',
        help='COMPLETE path from which to load a model'
    )

    parser.add_argument(
        '--metadata_flags', type=str, default=None, metavar='METADATA_FLAGS',
        help='Flags specifying observational metadata pre-processing, e.g. LUNAR_MAG which takes the '\
            'magnitude and linearizes it (ignoring that it is an area magnitude)'
    )
    args = parser.parse_args()

    obs_metadata = trim_observation_metadata(load_observation_metadata(args.metadata_path))
    sources, components, exposures, wavelengths = ICAize.deserialize_data(args.spectra_path, args.method)
    source_model = ICAize.unpickle_model(args.spectra_path, args.method)

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(exposures)

    reduced_obs_metadata.remove_column('EXP_ID')
    md_len = len(reduced_obs_metadata)
    var_count = len(reduced_obs_metadata.columns)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))
    Y_arr = sources[sorted_inds]

    if args.load_model:
        predictive_model = load_model(args.model_path)
    else:
        predictive_model = get_model(args.model)

    if args.hyper_fit:
        pdist = get_param_distribution_for_model(args.model)
        rcv = RandomizedSearchCV(predictive_model, param_distributions=pdist,
                            n_iter=args.hyper_fit_iters, random_state=RANDOM_STATE,
                            error_score=0, cv=args.test_folds, n_jobs=args.n_jobs)

        rcv.fit(X_arr, Y_arr)

        print(rcv.best_score_)
        print(rcv.best_params_)
        if args.hyper_outputfbk:
            print("=+"*10 + "=")
            print(rcv.grid_scores_)
            print("=+"*10 + "=")

        if args.save_model:
            save_model(rcv.best_estimator_, args.model_path)
    elif args.test_folds is not None:
        pass
    elif args.test_inds is not None:
        pass

def get_param_distribution_for_model(model_str):
    pdist = {}

    if model_str in ['ET', 'RF']:
        pdist['n_estimators'] = sp_randint(100, 500)
        pdist['max_features'] = sp_uniform(0.5, 1)
        pdist['min_samples_split'] = sp_randint(1, 15)
        pdist['min_samples_leaf'] = sp_randint(1, 15)
        pdist['bootstrap'] = [True, False]
    elif model_str == 'GP':
        pdist['corr'] = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']
        pdist['thetaL'] = sp_uniform(1e-5, 3e-1)
        pdist['thetaU'] = sp_uniform(7e-1, 1)
        pdist['random_start'] = sp_randint(2, 10)
    elif model_str == 'KNN':
        pdist['weights'] = ['uniform', 'distance']
        pdist['metric'] = ['euclidean', 'manhattan', 'chebyshev']
        pdist['n_neighbors'] = sp_randint(2, 50)
    return pdist

def get_model(model_str):
    if model_str == 'ET':
        return ensemble.ExtraTreesRegressor(random_state=RANDOM_STATE)
    elif model_str == 'RF':
        return ensemble.RandomForestRegressor(random_state=RANDOM_STATE)
    elif model_str == 'GP':
        return gaussian_process.GaussianProcess(random_state=RANDOM_STATE)
    elif model_str == 'KNN':
        return neighbors.KNeighborsRegressor()

def load_model(model_path):
    pickle_file = open(model_path, 'rb')
    model = pk.load(pickle_file)
    pickle_file.close()

    return model

def save_model(model, model_path):
    pickle_file = open(model_path, 'wb')
    model = pk.dump(model, pickle_file)
    pickle_file.close()

if __name__ == '__main__':
    main()

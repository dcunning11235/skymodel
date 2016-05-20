import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm
from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

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
import random

from astropy.utils.compat import argparse

RANDOM_STATE = 456371

def load_observation_metadata(path='.', file='annotated_metadata.csv', flags=""):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    if "proc_lunar_mag" in flags:
        data['LUNAR_MAGNITUDE'] = np.power(2.512, -data['LUNAR_MAGNITUDE'])

    if "wrap_angles" in flags:
        data['AZ'] = (data['AZ'] + 360.0) % 360.0
        data['RA'] = (data['RA'] + 360.0) % 360.0
        data['DEC'] = (data['DEC'] + 360.0) % 360.0
        data['LUNAR_SEP'] = (data['LUNAR_SEP'] + 360.0) % 360.0
        data['SOLAR_SEP'] = (data['LUNAR_SEP'] + 360.0) % 360.0
        data['GALACTIC_CORE_SEP'] = (data['GALACTIC_CORE_SEP'] + 360.0) % 360.0
        data['GALACTIC_PLANE_SEP'] = (data['GALACTIC_PLANE_SEP'] + 360.0) % 360.0
        #data['ECLIPTIC_PLANE_SEP'] = (data['ECLIPTIC_PLANE_SEP'] + 360.0) % 360.0
        #data['ECLIPTIC_PLANE_SOLAR_SEP'] = (data['ECLIPTIC_PLANE_SOLAR_SEP'] + 360.0) % 360.0

    if "cos_sep_angles" in flags:
        data['LUNAR_SEP'] = np.cos(data['LUNAR_SEP'] * np.pi/180.0)
        data['SOLAR_SEP'] = np.cos(data['LUNAR_SEP'] * np.pi/180.0)
        data['GALACTIC_CORE_SEP'] = np.cos(data['GALACTIC_CORE_SEP'] * np.pi/180.0)
        data['GALACTIC_PLANE_SEP'] = np.cos(data['GALACTIC_PLANE_SEP'] * np.pi/180.0)
        #data['ECLIPTIC_PLANE_SEP'] = np.cos(data['ECLIPTIC_PLANE_SEP'] * np.pi/180.0)
        #data['ECLIPTIC_PLANE_SOLAR_SEP'] = np.cos(data['ECLIPTIC_PLANE_SOLAR_SEP'] * np.pi/180.0)

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
                    'SS_COUNT', 'SS_AREA']
                    #,'ECLIPTIC_PLANE_SEP', 'ECLIPTIC_PLANE_SOLAR_SEP']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def action_build(args):
    pass

def action_compare(args):
    pass

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Build and test models based on dim reductions and provided spectra'
    )
    subparsers = parser.add_subparsers(dest='subparser_name')


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


    parser_compare = subparsers.add_parser('compare')
    parser_compare.set_defaults(func=action_compare)
    parser_compare.add_argument(
        '--folds', type=int, default=None, metavar='TEST_FOLDS',
        help='Do k-fold cross validation with specified number of folds.  Defaults to 3.'
    )
    parser_compare.add_argument(
        '--iters', type=int, default=50, metavar='HYPER_FIT_ITERS',
        help='Number of iterations when fitting hyper-params'
    )
    parser_compare.add_argument(
        '--outputfbk', action='store_true',
        help='If set, outputs \'grid_scores_\' data from RandomizedSearchCV'
    )
    parser_compare.add_argument(
        '--save_best', action='store_true',
        help='Whether or not to save the (last/best) model built for e.g. --hyper_fit'
    )
    parser_compare.add_argument(
        '--scorer', type=str, choices=['R2', 'MAE', 'MSE'], default='R2',
        help='Which scoring method to use to determine ranking of model instances.'
    )


    parser.add_argument(
        '--n_jobs', type=int, default=1, metavar='N_JOBS',
        help='N_JOBS'
    )


    parser.add_argument(
        '--model', type=str, choices=['ET', 'RF', 'GP', 'KNN', 'SVR'], default='ET',
        help='Which model type to use: ET (Extra Trees), RF (Random Forest), GP (Gaussian Process), KNN, or SVR (Support Vector Regression)'
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
        '--metadata_flags', type=str, default='', metavar='METADATA_FLAGS',
        help='Flags specifying observational metadata pre-processing, e.g. LUNAR_MAG which takes the '\
            'magnitude and linearizes it (ignoring that it is an area magnitude)'
    )

    args = parser.parse_args()

    obs_metadata = trim_observation_metadata(load_observation_metadata(args.metadata_path, flags=args.metadata_flags))
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

    if args.subparser_name == 'compare':
        pdist = get_param_distribution_for_model(args.model, args.iters)

        scorer = None
        if args.scorer == 'R2':
            scorer = make_scorer(r2_score, multioutput='uniform_average')
        elif args.scorer == 'MAE':
            scorer = make_scorer(MAE, greater_is_better=False)
        elif args.scorer == 'MSE':
            scorer = make_scorer(mean_squared_error, greater_is_better=False, multioutput='uniform_average')

        rcv = RandomizedSearchCV(predictive_model, param_distributions=pdist,
                            n_iter=args.iters, random_state=RANDOM_STATE,
                            error_score=0, cv=args.folds, n_jobs=args.n_jobs,
                            scoring=scorer)
        rcv.fit(X_arr, Y_arr)

        print(rcv.best_score_)
        print(rcv.best_params_)
        if args.outputfbk:
            print("=+"*10 + "=")
            for val in rcv.grid_scores_:
                print(val)
            print("=+"*10 + "=")

        if args.save_best:
            save_model(rcv.best_estimator_, args.model_path)

def MAE(Y, y):
    return np.mean(np.median(np.abs(Y - y), axis=1))

def get_param_distribution_for_model(model_str, iter_count):
    pdist = {}

    if model_str in ['ET', 'RF']:
        pdist['n_estimators'] = sp_randint(100, 500)
        pdist['max_features'] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]#sp_uniform(0.5, 1)
        pdist['min_samples_split'] = sp_randint(1, 15)
        pdist['min_samples_leaf'] = sp_randint(1, 15)
        pdist['bootstrap'] = [True, False]
    elif model_str == 'GP':
        #Fails because Gp will accpet either single values or array-like values, and it seems
        #RandomizedSearchCV etc. get confused (as they must, given no other information)
        corr_methods = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']
        theta0_range = sp_uniform(1e-3, 8e-1)
        thetaL_range = sp_uniform(1e-5, 3e-1)
        thetaU_range = sp_uniform(7e-1, 1)
        random_start_range = sp_randint(1, 3)

        pdist = []
        for i in range(iter_count):
            trial_dict = {}
            trial_dict['corr'] = random.choice(corr_methods)
            trial_dict['theta0'] = theta0_range.rvs()
            trial_dict['thetaL'] = thetaL_range.rvs()
            trial_dict['thetaU'] = thetaU_range.rvs()
            trial_dict['random_start'] = random_start_range.rvs()
            pdist.append(trial_dict)
    elif model_str == 'KNN':
        pdist['weights'] = ['uniform', 'distance']
        pdist['metric'] = ['euclidean', 'manhattan', 'chebyshev']
        pdist['n_neighbors'] = sp_randint(2, 50)
    elif model_str == 'SVR':
        pdist['kernel'] = ['rbf', 'sigmoid', 'poly']
        pdist['degree'] = sp_randint(2, 6)
        pdist['gamma'] = sp_uniform(1e-2, 1)
        pdist['coef0'] = sp_uniform(0, 1)
        pdist['epsilon'] = sp_uniform(1e-2, 3e-1)

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
    elif model_str == 'SVR':
        return svm.SVR()

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

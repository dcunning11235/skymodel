import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join
import astropy.coordinates as ascoord

from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import neighbors
from sklearn import svm
from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA

from sklearn.metrics import (make_scorer, mean_squared_error, r2_score, explained_variance_score)

from sklearn import preprocessing as skpp
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pp

from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import ICAize

import fnmatch
import os
import os.path
import sys
import pickle as pk
import random
from functools import partial
import pytz
from pytz import timezone

from astropy.utils.compat import argparse

import annotate_obs_metadata as aom

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
        data['ECLIPTIC_PLANE_SEP'] = (data['ECLIPTIC_PLANE_SEP'] + 360.0) % 360.0
        data['ECLIPTIC_PLANE_SOLAR_SEP'] = (data['ECLIPTIC_PLANE_SOLAR_SEP'] + 360.0) % 360.0

    if "cos_sep_angles" in flags:
        data['LUNAR_SEP'] = np.cos(data['LUNAR_SEP'] * np.pi/180.0)
        data['SOLAR_SEP'] = np.cos(data['LUNAR_SEP'] * np.pi/180.0)
        data['GALACTIC_CORE_SEP'] = np.cos(data['GALACTIC_CORE_SEP'] * np.pi/180.0)
        data['GALACTIC_PLANE_SEP'] = np.cos(data['GALACTIC_PLANE_SEP'] * np.pi/180.0)
        data['ECLIPTIC_PLANE_SEP'] = np.cos(data['ECLIPTIC_PLANE_SEP'] * np.pi/180.0)
        data['ECLIPTIC_PLANE_SOLAR_SEP'] = np.cos(data['ECLIPTIC_PLANE_SOLAR_SEP'] * np.pi/180.0)

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
                    'ECLIPTIC_PLANE_SEP', 'ECLIPTIC_PLANE_SOLAR_SEP']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

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
    parser.add_argument(
        '--compacted_path', type=str, default=None, metavar='COMPATED_PATH',
        help='Path to find compacted/arrayized data; setting this will cause --path, --pattern to be ignored'
    )

    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument(
        '--folds', type=int, default=3, metavar='TEST_FOLDS',
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
        '--scorer', type=str, choices=['R2', 'MAE', 'MSE', 'LL', 'EXP_VAR', 'MAPED', 'MSEMV'], default='R2',
        help='Which scoring method to use to determine ranking of model instances.'
    )
    parser_compare.add_argument(
        '--use_spectra', action='store_true',
        help='Whether scoring is done against the DM components or the predicted spectra'
    )
    parser_compare.add_argument(
        '--ivar_cutoff', type=float, default=0.001, metavar='IVAR_CUTOFF',
        help='data with inverse variace below cutoff is masked as if ivar==0'
    )
    parser_compare.add_argument(
        '--plot_final_errors', action='store_true',
        help='If set, will plot the errors from the final/best model, for the whole dataset, from ' + \
            'the best model re-trained on CV folds used for testing.' + \
            'Plots all errors on top of each other with low-ish alpha, to give a kind of visual ' + \
            'density map of errors.'
    )

    args = parser.parse_args()

    obs_metadata = trim_observation_metadata(load_observation_metadata(args.metadata_path, flags=args.metadata_flags))
    sources, components, exposures, wavelengths = ICAize.deserialize_data(args.spectra_path, args.method)
    source_model, ss, model_args = ICAize.unpickle_model(args.spectra_path, args.method)

    comb_flux_arr, comb_exposure_arr, comb_wavelengths = None, None, None
    if args.use_spectra:
        comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = ICAize.load_data(args)

        filter_arr = np.in1d(comb_exposure_arr, exposures)
        comb_flux_arr = comb_flux_arr[filter_arr]
        comb_exposure_arr = comb_exposure_arr[filter_arr]

        sorted_inds = np.argsort(comb_exposure_arr)
        comb_flux_arr = comb_flux_arr[sorted_inds]
        comb_exposure_arr = comb_exposure_arr[sorted_inds]

        del comb_ivar_arr
        del comb_masks

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
            scorer = make_scorer(R2)
        elif args.scorer == 'MAE':
            if args.use_spectra:
                p_MAE_ = partial(MAE, Y_full=Y_arr, flux_arr=comb_flux_arr,
                            source_model=source_model, ss=ss,
                            source_model_args=model_args, method=args.method)
                scorer = make_scorer(p_MAE_, greater_is_better=False)
            else:
                scorer = make_scorer(MAE, greater_is_better=False)
        elif args.scorer == 'MSE':
            if args.use_spectra:
                p_MSE_ = partial(MSE, Y_full=Y_arr, flux_arr=comb_flux_arr,
                            source_model=source_model, ss=ss,
                            source_model_args=model_args, method=args.method)
                scorer = make_scorer(p_MSE_, greater_is_better=False)
            else:
                scorer = make_scorer(MSE, greater_is_better=False)
        elif args.scorer == 'MSEMV':
            if args.use_spectra:
                p_MSEMV_ = partial(MSEMV, Y_full=Y_arr, flux_arr=comb_flux_arr,
                            source_model=source_model, ss=ss,
                            source_model_args=model_args, method=args.method)
                scorer = make_scorer(p_MSEMV_, greater_is_better=False)
            else:
                scorer = make_scorer(MSEMV, greater_is_better=False)
        elif args.scorer == 'EXP_VAR':
            if args.use_spectra:
                p_EXP_VAR_ = partial(EXP_VAR, Y_full=Y_arr, flux_arr=comb_flux_arr,
                            source_model=source_model, ss=ss,
                            source_model_args=model_args, method=args.method)
                scorer = make_scorer(p_EXP_VAR_)
            else:
                scorer = make_scorer(EXP_VAR)
        elif args.scorer == 'MAPED':
            if args.use_spectra:
                p_MAPED_ = partial(MAPED, Y_full=Y_arr, flux_arr=comb_flux_arr,
                            source_model=source_model, ss=ss,
                            source_model_args=model_args, method=args.method)
                scorer = make_scorer(p_MAPED_, greater_is_better=False)
            else:
                scorer = make_scorer(MAPED, greater_is_better=False)
        elif args.scorer == 'LL':
            scorer = None

        folder = ShuffleSplit(exposures.shape[0], n_iter=args.folds, test_size=1.0/args.folds,
                            random_state=12345)

        if args.model == 'GP':
            predictive_model.random_start = args.folds
            rcv = GridSearchCV(predictive_model, param_grid=pdist,
                            error_score=0, cv=3, n_jobs=args.n_jobs,
                            scoring=scorer)
                            #random_state=RANDOM_STATE,
                            #n_iter=args.iters,
        else:
            rcv = RandomizedSearchCV(predictive_model, param_distributions=pdist,
                            n_iter=args.iters, cv=folder, n_jobs=args.n_jobs,
                            scoring=scorer)

        # This is going to fit X (metdata) to Y (DM'ed sources).  But there are
        # really two tests here:  how well hyperparams fit/predict the sources
        # and how well they fit/predict the actual source spectra.  Until I know
        # better, I 'm going to need to build a way to test both.
        rcv.fit(X_arr, Y_arr)

        print(rcv.best_score_)
        print(rcv.best_params_)
        print(rcv.best_estimator_)
        if args.outputfbk:
            print("=+"*10 + "=")
            for val in rcv.grid_scores_:
                print(val)
            print("=+"*10 + "=")

        if args.save_best:
            save_model(rcv.best_estimator_, args.model_path)

        if args.plot_final_errors:
            for train_inds, test_inds in folder:
                rcv.best_estimator_.fit(X_arr[train_inds], Y_arr[train_inds])
                predicted = rcv.best_estimator_.predict(X_arr[test_inds])
                back_trans_flux = ICAize.inverse_transform(predicted, source_model, ss, args.method, model_args)
                diffs = np.abs(comb_flux_arr[test_inds] - back_trans_flux)
                #Is there not 'trick' to getting matplotlib to do this without a loop?
                for i in range(diffs.shape[0]):
                    plt.plot(comb_wavelengths, diffs[i, :], 'b-', alpha=0.01)
            plt.show()

def MAE(Y, y, multioutput='uniform_average', Y_full=None, flux_arr=None, source_model=None,
        ss=None, source_model_args=None, method=None):
    if Y_full is not None and flux_arr is not None and source_model is not None and ss is not None:
        inds = get_inds_(Y, Y_full)
        back_trans_flux = ICAize.inverse_transform(y, source_model, ss, method, source_model_args)
        return float(np.mean(np.median(np.abs(flux_arr[inds] - back_trans_flux), axis=1)))
    else:
        return float(np.mean(np.median(np.abs(Y - y), axis=1)))

def EXP_VAR(Y, y, multioutput='uniform_average', Y_full=None, flux_arr=None, source_model=None,
        ss=None, source_model_args=None, method=None):
    if Y_full is not None and flux_arr is not None and source_model is not None and ss is not None:
        inds = get_inds_(Y, Y_full)
        back_trans_flux = ICAize.inverse_transform(y, source_model, ss, method, source_model_args)
        try:
            return explained_variance_score(flux_arr[inds], back_trans_flux, multioutput=multioutput)
        except:
            return float(np.mean(np.var(flux_arr[inds] - back_trans_flux, axis=1) / np.var(flux_arr[inds], axis=1)))
    else:
        try:
            return explained_variance_score(Y, y, multioutput=multioutput)
        except:
            return float(np.mean(np.var(Y - y, axis=1) / np.var(Y, axis=1)))

def MAPED(Y, y, multioutput='uniform_average', power=4, cutoff=0.1, Y_full=None, flux_arr=None, source_model=None,
        ss=None, source_model_args=None, method=None):
    #Mean Absolute Power Error Difference;  take sum of (absolute) diffs, subtract MAPE from it
    if Y_full is not None and flux_arr is not None and source_model is not None and ss is not None:
        inds = get_inds_(Y, Y_full)
        back_trans_flux = ICAize.inverse_transform(y, source_model, ss, method, source_model_args)

        diffs = np.abs(flux_arr[inds] - back_trans_flux)
        diffs[diffs < cutoff] = 0

        sums = np.sum(diffs, axis=1)
        diffs = np.sum(np.power(diffs, power), axis=1)

        return float(np.mean(np.abs(sums - np.power(diffs, 1.0/power))) / flux_arr.shape[1])
    else:
        diffs = np.abs(Y - y)
        diffs[diff < cutoff] = 0

        sums = np.sum(diffs, axis=1)
        diffs = np.sum(np.power(diffs, power), axis=1)

        return float(np.mean(np.abs(sums - np.power(diffs, 1.0/power))) / Y.shape[1])

def get_inds_(Y, Y_full):
    # Figure out the right way to do this... don't want to rewrite 4/5 of the
    # GridSearch/cross_validation code.  And I don't know a *good* way do this
    # array comparison 'right'
    inds = []
    for i in range(Y.shape[0]):
        ind = np.where((Y_full == Y[i, :]).all(axis=1))
        if len(ind) > 0:
            inds.append(ind[0])
    return np.concatenate(inds)

def MSE(Y, y, multioutput='uniform_average', Y_full=None, flux_arr=None, source_model=None,
        ss=None, source_model_args=None, method=None):
    if Y_full is not None and flux_arr is not None and source_model is not None and ss is not None:
        inds = get_inds_(Y, Y_full)
        back_trans_flux = ICAize.inverse_transform(y, source_model, ss, method, source_model_args)

        try:
            return mean_squared_error(flux_arr[inds], back_trans_flux, multioutput=multioutput)
        except:
            return mean_squared_error(flux_arr[inds], back_trans_flux)
    else:
        try:
            yss = pp.MaxAbsScaler()
            Y = yss.fit_transform(Y)
            y = yss.transform(y)
        except:
            scalefactor = np.amax(np.abs(Y), axis=0)
            Y = Y / scalefactor
            y = y / scalefactor

        try:
            return mean_squared_error(Y, y, multioutput=multioutput)
        except:
            return mean_squared_error(Y, y)

def MSEMV(Y, y, multioutput='uniform_average', Y_full=None, flux_arr=None, source_model=None,
        ss=None, source_model_args=None, method=None):
    mse = MSE(Y, y, multioutput)
    var = 0

    if Y_full is not None and flux_arr is not None and source_model is not None and ss is not None:
        inds = get_inds_(Y, Y_full)
        #back_trans_flux = ICAize.inverse_transform(y, source_model, ss, method, source_model_args)

        var = np.mean(np.var(flux_arr[inds], axis=0))
    else:
        try:
            yss = pp.MaxAbsScaler()
            Y = yss.fit_transform(Y)
        except:
            scalefactor = np.amax(np.abs(Y), axis=0)
            Y = Y / scalefactor

        var = np.mean(np.var(Y, axis=0))

    return mse - var

def R2(Y, y, multioutput='uniform_average'):
    try:
        return r2_score(Y, y, multioutput=multioutput)
    except:
        return r2_score(Y, y)

def get_param_distribution_for_model(model_str, iter_count):
    pdist = {}

    if model_str in ['ET', 'RF']:
        pdist['n_estimators'] = sp_randint(100, 500)
        pdist['max_features'] = [0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1]#sp_uniform(0.5, 1)
        pdist['min_samples_split'] = sp_randint(1, 15)
        pdist['min_samples_leaf'] = sp_randint(1, 15)
        pdist['bootstrap'] = [True, False]
    elif model_str == 'GP':
        #Fails because Gp will accpet either single values or array-like values, and it seems
        #RandomizedSearchCV etc. get confused (as they must, given no other information)
        #corr_methods = ['absolute_exponential', 'squared_exponential', 'generalized_exponential', 'cubic', 'linear']
        pdist['corr'] = ['absolute_exponential', 'squared_exponential', 'cubic', 'linear']
        #theta0_range = sp_uniform(0.1, 0.9)
        #thetaL_range = sp_uniform(1e-5, 3e-1)
        #thetaU_range = sp_uniform(7e-1, 1)
        #random_start_range = sp_randint(1, 3)

        '''
        pdist = []
        for i in range(iter_count):
            trial_dict = {}
            trial_dict['corr'] = [random.choice(corr_methods)]
            #trial_dict['theta0'] = [theta0_range.rvs()]
            #trial_dict['thetaL'] = [[thetaL_range.rvs()]]
            #trial_dict['thetaU'] = [[thetaU_range.rvs()]]
            #trial_dict['random_start'] = [random_start_range.rvs()]
            pdist.append(trial_dict)
        '''
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
        return gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=8e-1, random_state=RANDOM_STATE)
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

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.base import clone as est_clone
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import NMF
from sklearn.decomposition import DictionaryLearning
from sklearn.manifold import Isomap

from sklearn import preprocessing as skpp
from sklearn.metrics import (explained_variance_score, mean_squared_error, median_absolute_error, r2_score, mean_absolute_error)
from sklearn.metrics import r2_score

from sklearn.covariance import MinCovDet
from sklearn.covariance import LedoitWolf

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold

from sklearn.utils.extmath import fast_logdet
from scipy import linalg

import fnmatch
import os
import os.path
import sys
import random
import pickle as pk
from multiprocessing.pool import ThreadPool
import itertools as it

from astropy.utils.compat import argparse

import arrayize

random_state=1234975
data_file = "{}_sources_and_mixing.npz"
pickle_file = "{}_pickle.pkl"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Compute PCA/ICA/NMF/etc. components over set of stacked spectra, save those out, and pickle model'
    )
    subparsers = parser.add_subparsers(dest='subparser_name')

    parser.add_argument(
        '--pattern', type=str, default='stacked*exp??????.*', metavar='PATTERN',
        help='File pattern for stacked sky fibers.'
    )
    parser.add_argument(
        '--path', type=str, default='.', metavar='PATH',
        help='Path to work from, if not ''.'''
    )
    parser.add_argument(
        '--compacted_path', type=str, default=None, metavar='COMPATED_PATH',
        help='Path to find compacted/arrayized data; setting this will cause --path, --pattern to be ignored'
    )
    parser.add_argument(
        '--method', type=str, default=['ICA'], metavar='METHOD',
        choices=['ICA', 'PCA', 'SPCA', 'NMF', 'ISO', 'KPCA', 'FA', 'DL'], nargs='+',
        help='Which dim. reduction method to use'
    )
    parser.add_argument(
        '--scale', action='store_true',
        help='Should inputs variance be scaled?  Defaults to mean subtract and value scale, but w/out this does not scale variance.'
    )
    parser.add_argument(
        '--ivar_cutoff', type=float, default=0.001, metavar='IVAR_CUTOFF',
        help='data with inverse variace below cutoff is masked as if ivar==0'
    )
    parser.add_argument(
        '--n_iter', type=int, default=1200, metavar='MAX_ITER',
        help='Maximum number of iterations to allow for convergence.  For SDSS data 1000 is a safe number of ICA, while SPCA requires larger values e.g. ~2000 to ~2500'
    )
    parser.add_argument(
        '--n_jobs', type=int, default=None, metavar='N_JOBS',
        help='N_JOBS'
    )

    parser_compare = subparsers.add_parser('compare')
    parser_compare.add_argument(
        '--max_components', type=int, default=50, metavar='COMP_MAX',
        help='Max number of components to use/test'
    )
    parser_compare.add_argument(
        '--min_components', type=int, default=0, metavar='COMP_MIN',
        help='Min number of compoenents to use/test'
    )
    parser_compare.add_argument(
        '--step_size', type=int, default=5, metavar='COMP_STEP',
        help='Step size from comp_min to comp_max'
    )
    parser_compare.add_argument(
        '--comparison', choices=['EXP_VAR', 'R2', 'MSE', 'MAE', 'mmAE', 'LL'], nargs='*', default=['EXP_VAR'],
        help='Comparison methods: Explained variance (score), R2 (score), mean sq. error (loss), MEDIAN absolute error (loss)'
    )
    parser_compare.add_argument(
        '--mle_if_avail', action='store_true',
        help='In additon to --comparison, include MLE if PCA or FA methods specified'
    )
    parser_compare.add_argument(
        '--plot_example_reconstruction', action='store_true',
        help='Pick a random spectrum, plot its actual and reconstructed versions'
    )

    parser_build = subparsers.add_parser('build')
    parser_build.add_argument(
        '--n_components', type=int, default=40, metavar='N_COMPONENTS',
        help='Number of ICA/PCA/etc. components'
    )
    parser_build.add_argument(
        '--n_neighbors', type=int, default=10, metavar='N_NEIGHBORS',
        help='Number of neighbots for e.g. IsoMap'
    )

    args = parser.parse_args()

    comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = load_data(args)

    if 'DL' in args.method:
        flux_arr = comb_flux_arr.astype(dtype=np.float64)
    else:
        flux_arr = comb_flux_arr

    ss = skpp.StandardScaler(with_std=False)
    if args.scale:
        ss = skpp.StandardScaler(with_std=True)

    if args.subparser_name == 'compare':
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for method in args.method:
            model = get_model(method, max_iter=args.n_iter, random_state=random_state, n_jobs=args.n_jobs)
            scores = {}
            mles_and_covs = args.mle_if_avail and (method == 'FA' or method == 'PCA')

            n_components = np.arange(args.min_components, args.max_components+1, args.step_size)
            for n in n_components:
                print("Cross validating for n=" + str(n) + " on method " + method)

                model.n_components = n

                comparisons = score_via_CV(args.comparison, flux_arr, model, ss,
                                    method, n_jobs=args.n_jobs, include_mle=mles_and_covs,
                                    plot_example_reconstruction = args.plot_example_reconstruction)
                for key, val in comparisons.items():
                    if key in scores:
                        scores[key].append(val)
                    else:
                        scores[key] = [val]

            '''
            if mles_and_covs:
                #ax2.axhline(cov_mcd_score(scaled_flux_arr, args.scale), color='violet', label='MCD Cov', linestyle='--')
                ax2.axhline(cov_lw_score(scaled_flux_arr, args.scale), color='orange', label='LW Cov', linestyle='--')
            '''
            
            for key, score_list in scores.items():
                if key != 'mle':
                    ax1.plot(n_components, score_list, label=method + ':' + key + ' scores')
                else:
                    ax2.plot(n_components, score_list, '-.', label=method + ' mle scores')

        ax1.set_xlabel('nb of components')
        ax1.set_ylabel('CV scores', figure=fig)

        ax1.legend(loc='lower left')
        ax2.legend(loc='lower right')

        plt.show()
    else:
        scaled_flux_arr = ss.fit_transform(flux_arr)

        for method in args.method:
            sources, components, model = dim_reduce(method, flux_arr if method == 'NMF' else scaled_flux_arr,
                                            args.n_components, args.n_neighbors,
                                            args.n_iter, random_state)
            serialize_data(sources, components, comb_exposure_arr, comb_wavelengths,
                            args.path, method)
            pickle_model((model, None if method == 'NMF' else ss), args.path, method)

def load_data(args):
    if args.compacted_path is not None:
        comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_wavelengths = arrayize.load_compacted_data(args.compacted_path)
        comb_masks = comb_ivar_arr <= args.ivar_cutoff
    else:
        comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = \
                load_all_in_dir(args.path, pattern=args.pattern, ivar_cutoff=args.ivar_cutoff)

    mask_summed = np.sum(comb_masks, axis=0)
    min_val_ind = np.min(np.where(mask_summed == 0))
    max_val_ind = np.max(np.where(mask_summed == 0))
    print "For data set, minimum and maximum valid indecies are:", (min_val_ind, max_val_ind)
    for i in range(comb_flux_arr.shape[0]):
        comb_flux_arr[i,:min_val_ind] = 0
        comb_flux_arr[i,max_val_ind+1:] = 0

    return comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths

def serialize_data(sources, components, exposures, wavelengths, path='.', method='ICA',
                    filename=None):
    if filename is None:
        filename = data_file.format(method)
    np.savez(os.path.join(path, filename), sources=sources, components=components,
            exposures=exposures, wavelengths=wavelengths)

def deserialize_data(path='.', method='ICA', filename=None):
    if filename is None:
        filename = data_file.format(method)
    npz = np.load(os.path.join(path, filename))

    sources = npz['sources']
    exposures = npz['exposures']
    wavelengths = npz['wavelengths']
    components = npz['components']

    npz.close()
    return sources, components, exposures, wavelengths

def pickle_model(model, path='.', method='ICA', filename=None):
    if filename is None:
        filename = pickle_file.format(method)
    file = open(os.path.join(path, filename), 'wb')
    pk.dump(model, file)
    file.close()

def unpickle_model(path='.', method='ICA', filename=None):
    if filename is None:
        filename = pickle_file.format(method)
    file = open(os.path.join(path, filename), 'rb')
    model, ss = pk.load(file)
    file.close()

    return model, ss

def get_model(method, n=None, n_neighbors=None, max_iter=None, random_state=None, n_jobs=None):
    model = None

    if method == 'ICA':
        model = FastICA(whiten=True)
    elif method == 'PCA':
        model = PCA()
    elif method == 'SPCA':
        model = SparsePCA()
        if n_jobs is not None:
            model.n_jobs = n_jobs
    elif method == 'NMF':
        model = NMF(solver='cd')
    elif method == 'ISO':
        model = Isomap()
    elif method == 'KPCA':
        model = KernelPCA(kernel='rbf', fit_inverse_transform=False, gamma=1, alpha=0.0001)
    elif method == 'FA':
        #model = FactorAnalysis(svd_method='lapack') #(tol=0.0001, iterated_power=4)
        model = FactorAnalysis(tol=0.0001, iterated_power=4)
    elif method == 'DL':
        model = DictionaryLearning(split_sign=True, fit_algorithm='cd', alpha=1)
        if n_jobs is not None:
            model.n_jobs = n_jobs

    if n is not None:
        model.n_components = n
    if max_iter is not None:
        model.max_iter = max_iter
    if random_state is not None:
        model.random_state = random_state
    if n_neighbors is not None:
        model.n_neighbors = n_neighbors


    return model

def get_components(method, model):
    if method == 'ISO':
        components = model.embedding_
    elif method == 'KPCA':
        components = model.alphas_
    else:
        components = model.components_

    return components

def dim_reduce(method, flux_arr, n=None, n_neighbors=None, max_iter=None, random_state=None):
    model = get_model(method, n, n_neighbors, max_iter, random_state)
    sources = model.fit_transform(flux_arr)
    components = get_components(method, model)

    return sources, components, model

def load_all_in_dir(path, pattern, ivar_cutoff=0):
    flux_list = []
    exp_list = []
    mask_list = []
    ivar_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            if file.endswith(".csv"):
                data = Table(Table.read(os.path.join(path, file), format="ascii.csv"), masked=True)
            elif file.endswith(".fits"):
                data = Table(Table.read(os.path.join(path, file), format="fits"), masked=True)

            mask = data['ivar'] <= ivar_cutoff
            ivar_list.append(np.array(data['ivar'], copy=False))

            exp = file.split("-")[2][3:]
            if exp.endswith("csv"):
                exp = int(exp[:-4])
            elif exp.endswith("fits"):
                exp = int(exp[:-5])
            else:
                exp = int(exp)

            if wavelengths is None:
                wavelengths = np.array(data['wavelength'], copy=False)

            flux_list.append(np.array(data['flux'], copy=False))
            mask_list.append(mask)
            exp_list.append(exp)

    flux_arr = np.array(flux_list)
    exp_arr = np.array(exp_list)
    mask_arr = np.array(mask_list)
    ivar_arr = np.array(ivar_list)

    return flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths

'''
def ___scorer(all_args):
    return _scorer(*all_args)
'''

def get_score_func(score_method):
    score_func = None
    if score_method == 'EXP_VAR':
        score_func = explained_variance_score
    elif score_method == 'R2':
        score_func = r2_score
    elif score_method == 'MSE':
        score_func = mean_squared_error
    elif score_method == 'MAE':
        score_func = median_absolute_error #Except can't use because doesn't support multioutput
    elif score_method == 'mmAE':
        score_func = mean_absolute_error
    elif score_method == 'LL':
        score_func = None

    return score_func

def _scorer(train_inds, test_inds, flux_arr, model, scaler, method, score_methods, include_mle):
    flux_test = flux_arr[test_inds]
    flux_conv_test = None
    if score_methods != ['LL']:
        flux_conv_test = transform_inverse_transform(flux_test, model, scaler, method)

    scores = {}

    for score_method in score_methods:
        #print("Calculating score:" + score_method)

        score_func = get_score_func(score_method)

        if score_func is not None:
            if score_method != 'MAE':
                scores[score_method] = score_func(flux_test, flux_conv_test, multioutput='uniform_average')
            else:
                scores[score_method] = np.mean(np.median(np.abs(flux_test - flux_conv_test), axis=1))

    if (include_mle or score_method == 'LL') and method in ['FA', 'PCA']:
        try:
            scores['mle'] = model.score(flux_test)
        except np.linalg.linalg.LinAlgError:
            scores['mle'] = -2**10 #float("-inf")
        except ValueError:
            scores['mle'] = -2**10 #float("-inf")

    #print("Scores: " + str(scores))
    return scores

'''
def ___modeler(all_args):
    return _modeler(*all_args)
'''

def _modeler(train_inds, test_inds, flux_arr, model, method):
    new_model = est_clone(model)
    flux_train = flux_arr[train_inds]
    #print("Training new model: " + str(new_model))

    if method == 'KPCA':
        new_model.fit_inverse_transform = False
        new_model.fit(flux_train)

        sqrt_lambdas = np.diag(np.sqrt(model.lambdas_))
        X_transformed = np.dot(model.alphas_, sqrt_lambdas)
        n_samples = X_transformed.shape[0]
        K = new_model._get_kernel(X_transformed)
        K.flat[::n_samples + 1] += model.alpha

        new_model.dual_coef_ = linalg.solve(K, flux_train)
        new_model.X_transformed_fit_ = X_transformed
        new_model.fit_inverse_transform = True
    else:
        if np.any(np.isinf(flux_train)):
            print("The given flux_train array contains inf's!")
        if np.any(np.isnan(flux_train)):
            print("The given flux_train array contains nan's!")
        new_model.fit(flux_train)

    #print("Returning new model: " + str(new_model))
    return new_model

def score_via_CV(score_methods, flux_arr, model, scaler, method, folds=3, n_jobs=1, include_mle=False,
                modeler=_modeler, scorer=_scorer, plot_example_reconstruction=False):
    scaled_flux_arr = flux_arr if method == 'NMF' else scaler.fit_transform(flux_arr)

    kf = KFold(len(scaled_flux_arr), n_folds=folds, shuffle=True)
    all_scores = []

    for train_inds, test_inds in kf:
        model = modeler(train_inds, test_inds, scaled_flux_arr, model, method)
        all_scores.append(scorer(train_inds, test_inds, scaled_flux_arr, model, scaler, method, score_methods, include_mle))

        if plot_example_reconstruction:
            sample_data = flux_arr[test_inds[100], :].reshape(1, -1)
            plt.plot(np.arange(flux_arr.shape[1]), sample_data[0, :], label="Sample Data")

            trans_data = transform_inverse_transform(sample_data, model, scaler, method)
            plt.plot(np.arange(flux_arr.shape[1]), trans_data[0, :], label="Trans Data")

            plt.legend()
            plt.show()

    collated_scores = {}
    for scores in all_scores:
        #print("Got scores obj of: " + str(scores))
        for key, val in scores.items():
            if key in collated_scores:
                collated_scores[key].append(val)
            else:
                collated_scores[key] = [val]

    final_scores = {}
    for key, vals in collated_scores.items():
        final_scores[key] = np.mean(vals)

    #print("Final_scores: " + str(final_scores))
    return final_scores

def transform_inverse_transform(flux_arr, model, ss, method):
    if ss is not None and method != 'NMF':
        trans_arr = ss.transform(flux_arr)
    else:
        trans_arr = flux_arr
    trans_arr = model.transform(trans_arr)
    return inverse_transform(trans_arr, model, ss, method)

def inverse_transform(dm_flux_arr, model, ss, method):
    if method in ['PCA', 'KPCA', 'ICA']:
        att_invtrans = model.inverse_transform(dm_flux_arr)
    else:
        #For FA, need to add back in mean, even if not taken out by scaler... right
        #now, have opted to not handle this, rather just assume/make mean-subtration
        #the norm/default scaler
        components = get_components(method, model)

        att_invtrans = np.zeros(shape=(dm_flux_arr.shape[0], components.shape[1]), dtype=float)
        for flux_n in xrange(dm_flux_arr.shape[0]):
            for comp_n in xrange(components.shape[0]):
                rec_comp = dm_flux_arr[flux_n, comp_n] * components[comp_n]
                att_invtrans[flux_n] += rec_comp

        #if method in ['FA']:
        #    att_invtrans += model_flux_mean

    if ss is not None and method != 'NMF':
        att_invtrans = ss.inverse_transform(att_invtrans)

    return att_invtrans

def cov_mcd_score(flux_arr, assume_centered):
    mcd = MinCovDet(assume_centered=assume_centered)
    return np.mean(cross_val_score(mcd, flux_arr))

def cov_lw_score(flux_arr, assume_centered):
    lw = LedoitWolf(assume_centered=assume_centered)
    return np.mean(cross_val_score(lw, flux_arr))


if __name__ == '__main__':
    main()

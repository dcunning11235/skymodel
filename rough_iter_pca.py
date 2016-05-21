import numpy as np
import matplotlib.pyplot as plt

from astropy.utils.compat import argparse

from sklearn.base import clone as est_clone

from scipy import linalg

import ICAize as iz

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
        help='Should inputs be scaled?  Will mean subtract and value scale, but does not scale variace.'
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
        '--comparison', choices=['EXP_VAR', 'R2', 'MSE', 'MAE'], nargs='*', default=['EXP_VAR'],
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

    comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = iz.load_data(args)

    if 'DL' in args.method:
        flux_arr = comb_flux_arr.astype(dtype=np.float64)
    else:
        flux_arr = comb_flux_arr
    scaled_flux_arr = None
    ss = None
    if args.scale:
        ss = skpp.StandardScaler(with_std=False)
        scaled_flux_arr = ss.fit_transform(flux_arr)
    else:
        scaled_flux_arr = flux_arr

    if args.subparser_name == 'compare':
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        for method in args.method:
            model = iz.get_model(method, max_iter=args.n_iter, random_state=iz.random_state, n_jobs=args.n_jobs)
            scores = {}
            mles_and_covs = args.mle_if_avail and (method == 'FA' or method == 'PCA')

            n_components = np.arange(args.min_components, args.max_components+1, args.step_size)
            for n in n_components:
                print("Cross validating for n=" + str(n) + " on method " + method)

                model.n_components = n

                comparisons = iz.score_via_CV(args.comparison,
                                    flux_arr if method == 'NMF' else scaled_flux_arr,
                                    model, method, n_jobs=args.n_jobs, include_mle=mles_and_covs,
                                    modeler=_iter_modeler, scorer=_iter_scorer)
                for key, val in comparisons.items():
                    if key in scores:
                        scores[key].append(val)
                    else:
                        scores[key] = [val]

            if mles_and_covs:
                #ax2.axhline(cov_mcd_score(scaled_flux_arr, args.scale), color='violet', label='MCD Cov', linestyle='--')
                ax2.axhline(cov_lw_score(scaled_flux_arr, args.scale), color='orange', label='LW Cov', linestyle='--')

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

def _iter_modeler(train_inds, test_inds, flux_arr, model, method):
    model_list = []
    flux_train = flux_arr[train_inds]
    flux_avg = np.mean(flux_train, axis=0)

    for i in range(2):
        new_model = est_clone(model)
        new_model.fit(flux_train)
        back_train = iz.transform_inverse_transform(flux_train, new_model, flux_avg, method)
        flux_train -= back_train

        model_list.append(new_model)

    return model_list, flux_avg

def _iter_scorer(train_inds, test_inds, flux_arr, model__and__model_flux_mean, method, score_methods, include_mle):
    model = model__and__model_flux_mean[0]
    model_flux_mean = model__and__model_flux_mean[1]

    flux_test = flux_arr[test_inds]
    flux_conv_test = None

    if score_methods != ['LL']:
        for pca_model in model:
            if flux_conv_test is None:
                flux_conv_test = iz.transform_inverse_transform(flux_test, pca_model, model_flux_mean, method)
                flux_test -= flux_conv_test
            else:
                residual = iz.transform_inverse_transform(flux_test, pca_model, model_flux_mean, method)
                flux_conv_test += residual
                flux_test -= residual

    scores = {}

    for score_method in score_methods:
        #print("Calculating score:" + score_method)

        score_func = iz.get_score_func(score_method)

        if score_func is not None:
            if score_method != 'MAE':
                scores[score_method] = score_func(flux_test, flux_conv_test, multioutput='uniform_average')
            else:
                scores[score_method] = np.mean(np.median(np.abs(flux_test - flux_conv_test), axis=1))

    if (include_mle or score_method == 'LL') and method in ['FA', 'PCA']:
        try:
            scores['mle'] = model.score(flux_test)
        except np.linalg.linalg.LinAlgError:
            scores['mle'] = 0 #-2**10 #float("-inf")
        except ValueError:
            scores['mle'] = 0 #-2**10 #float("-inf")

    #print("Scores: " + str(scores))
    return scores


if __name__ == '__main__':
    main()

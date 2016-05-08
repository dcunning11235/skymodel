import ICAize as ize
import stack

import matplotlib.pyplot as plt
import numpy as np

from astropy.utils.compat import argparse

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Build and test models based on dim reductions and provided spectra'
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
        '--file_path', type=str, default=None, metavar='FILE_PATH',
        help='COMPLETE path from which to load a dim reduction'
    )

    args = parser.parse_args()

    data_model = None
    scaler = None
    if args.file_path is not None:
        data_model, scaler = ize.unpickle_model(filename=args.file_path)
    else:
        data_model, scaler = ize.unpickle_model(path=args.spectra_path, method=args.method)
    components = ize.get_components(args.method, data_model)

    offset = 0
    for i, comp_i in enumerate(components):
        if i > 0:
            offset += np.max(np.abs(comp_i[comp_i < 0])) * 1.2
        plt.plot(stack.skyexp_wlen_out, comp_i + offset)
        offset += np.max(comp_i[comp_i > 0]) * 1.2
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

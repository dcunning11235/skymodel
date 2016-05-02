import pickle
import ICAize
import numpy as np

import os.path

def main():
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = \
        ICAize.load_all_in_dir('.', pattern="stacked*exp??????.csv")
    temp_flux_arr, temp_exp_arr, temp_ivar_arr, temp_mask_arr, temp_wavelengths = \
        ICAize.load_all_in_dir('.', pattern="stacked*exp??????.fits")

    if len(flux_arr) > 0:
        if len(temp_flux_arr) > 0:
            flux_arr = np.concatenate((flux_arr, temp_flux_arr))
            exp_arr = np.concatenate((exp_arr, temp_exp_arr))
            ivar_arr = np.concatenate((ivar_arr, temp_ivar_arr))
            wavelengths = np.concatenate((wavelengths, temp_wavelengths))
    elif len(temp_flux_arr) > 0:
        flux_arr = temp_flux_arr
        exp_arr = temp_exp_arr
        ivar_arr = temp_ivar_arr
        wavelengths = temp_wavelengths
    else:
        return

    np.savez("compacted_flux_data.npz", flux=flux_arr, exp=exp_arr, ivar=ivar_arr, wavelengths=wavelengths)

def load_compacted_data(path):
    data = np.load(os.path.join(path, "compacted_flux_data.npz"))
    return data['flux'], data['exp'], data['ivar'], data['wavelengths']

if __name__ == '__main__':
    main()

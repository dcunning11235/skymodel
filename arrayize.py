import pickle
import ICAize
import numpy as np

def main():
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = \
        ICAize.load_all_in_dir('.', use_con_flux=False, recombine_flux=False,
                        pattern="stacked*exp??????.csv")
    temp_flux_arr, temp_exp_arr, temp_ivar_arr, temp_mask_arr, temp_wavelengths = \
	ICAize.load_all_in_dir('.', use_con_flux=False, recombine_flux=False,
			pattern="stacked*exp??????.fits")
    flux_arr = np.concatenate((flux_arr, temp_flux_arr))
    exp_arr = np.concatenate((exp_arr, temp_exp_arr))
    ivar_arr = np.concatenate((ivar_arr, temp_ivar_arr))
    wavelengths = np.concatenate((wavelengths, temp_wavelengths))

    np.savez("compacted_flux_data.npz", flux=flux_arr, exp=exp_arr, ivar=ivar_arr, wavelengths=wavelengths)

if __name__ == '__main__':
    main()

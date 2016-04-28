import numpy as np

from astropy.table import Table

import sys
import datetime as dt
import posixpath

import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec
import bossdata.plate as bdplate
import bossdata.bits as bdbits

from speclite import accumulate
from speclite import resample

import matplotlib.pyplot as plt

from scipy.signal import find_peaks_cwt
from scipy.signal import general_gaussian
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema

from progressbar import ProgressBar, Percentage, Bar

from astropy.utils.compat import argparse

'''
This script takes in a list of plate-mjd-fiber combos as output by e.g. bossquery for
bossquery --what "PLATE,MJD,FIBER" \
--where "OBJTYPE='SKY' and THING_ID=-1 and ZWARNING in (1,5) and PLATE between 4501 and 6000" \
--sort "PLATE,MJD,FIBER" --full --max-rows 10000000 --save sky_fibers_4501_to_6000.dat

It groups entries in this list by plate-mjd, then stacks the spectra for all fibers
in each group, per exposure (so you end up with 7 sky spectra for the 79 fibers in the
plate-mjd 3586-55181.)

Since it works with exposures and not co-adds, it must process the data a bit in order to be
able to stack it:  namely, resample to a regular wavelength spacing (0.975 A) over a standard
range (3500.26 to 10422.76), combining the red and blue cameras.

Each of these combined sky specta is then saved out as a CSV file named:
    stacked_sky_{plate}-{mjd}-exp{exposure:02d}.csv (or .fits)
'''

finder = bdpath.Finder()
manager = bdremote.Manager()

peak_seek_end = 2820
skyexp_wlen_start = 3500.26
skyexp_wlen_delta = 0.975
skyexp_wlen_out = np.arange(skyexp_wlen_start, 10443, skyexp_wlen_delta)

skyexp_loglam_start = 3.5441
skyexp_loglam_delta = 0.00006675
skyexp_loglam_out = np.arange(skyexp_loglam_start, 4.01716, skyexp_loglam_delta)

allowed_bitmask = bdbits.bitmask_from_text(bdbits.SPPIXMASK, "BRIGHTSKY|BADSKYCHI") #|NOSKY")
pin_peaks = [4358.9, 5200.3, 5462.6, 5578.5]
pin_peaks_dtype = [('fiber', 'i4'), ('exposure', 'i4')]
for i in pin_peaks:
    pin_peaks_dtype.append( ("target_{:0.1f}".format(i), 'f4') )

#
####
####################################################################
####
#

def get_stacked_fiducial_wlen_pixel_offset(target):
    return np.rint((target - skyexp_wlen_start)/skyexp_wlen_delta)

def camera_id(camera, fiber):
    return camera[0] + ("1" if (fiber <= 500) else "2")

def stack_exposures(fiber_group, exposure=None, use_cframe=False, find_pins=False):
    exposure_list = []
    spec_sky_list = None

    filename = finder.get_spec_path(plate=fiber_group['PLATE'][0],
                mjd=fiber_group['MJD'][0], fiber=1,
                lite=False)
    spec = bdspec.SpecFile(manager.get(filename))

    if len(exposure_list) == 0:
        if exposure is None:
            exposure_list = spec.exposures.table['science'][0:(spec.num_exposures)]
        elif hasattr(exposure, '__iter__'):
            exposure_list.extend(exposure)
        else:
            exposure_list.append(exposure)
    if spec_sky_list is None:
        spec_sky_list = [None]*len(exposure_list)

    total_pin_peaks = []
    # These are really two separate methods.... should just separate
    if use_cframe:
        fiber_list_1 = fiber_group['FIBER'][fiber_group['FIBER'] <= 500]
        fiber_list_2 = fiber_group['FIBER'][fiber_group['FIBER'] > 500]

        def _prepend_frame_path(plate, frame_file):
            return posixpath.join(finder.redux_base, '{:04d}'.format(plate), frame_file)

        def _get_frame_data(fiber_list, camera, exp):
            use_calibrated = True

            exposure = '{0}-{1}-{2:08d}.{3}'.format('spCFrame', camera, exp, 'fits')
            frame = bdplate.FrameFile(manager.get(_prepend_frame_path(fiber_group['PLATE'][0], exposure)),
                                        1 if fiber_list[0] <= 500 else 2, use_calibrated)
            data = frame.get_valid_data(fiber_list, pixel_quality_mask=allowed_bitmask, include_sky=True, use_ivar=True)
            data['flux'] += data['sky']

            return data

        for i, exp in enumerate(exposure_list):
            r_1_data = _get_frame_data(fiber_list_1, 'r1', exp)
            b_1_data = _get_frame_data(fiber_list_1, 'b1', exp)
            spec_sky_list[i], pin_peaks_list = resample_regular(b_1_data, r_1_data,
                                spec_sky_list[i], exposure=exp,
                                use_loglam=False, fiber_list=fiber_list_1, find_pins=find_pins)
            total_pin_peaks.extend(pin_peaks_list)

            r_2_data = _get_frame_data(fiber_list_2, 'r2', exp)
            b_2_data = _get_frame_data(fiber_list_2, 'b2', exp)
            spec_sky_list[i], pin_peaks_list = resample_regular(b_2_data, r_2_data,
                                spec_sky_list[i], exposure=exp,
                                use_loglam=False, fiber_list=fiber_list_2, find_pins=find_pins)
            total_pin_peaks.extend(pin_peaks_list)
    else:
        for row in fiber_group:
            filename = finder.get_spec_path(plate=row['PLATE'], mjd=row['MJD'], fiber=row['FIBER'], lite=False)
            spec = bdspec.SpecFile(manager.get(filename))
            for exp in exposure_list:
                r_data = spec.get_valid_data(include_sky=True, exposure_index=exp, pixel_quality_mask=allowed_bitmask,
                                camera=camera_id('r', row['FIBER']), use_loglam=False, use_ivar=True,
                                include_wdisp=False)
                b_data = spec.get_valid_data(include_sky=True, exposure_index=exp, pixel_quality_mask=allowed_bitmask,
                                camera=camera_id('b', row['FIBER']), use_loglam=False, use_ivar=True,
                                include_wdisp=False)
                spec_sky_list[exp], pin_peaks_list = resample_regular(b_data, r_data, spec_sky_list[exp],
                                        use_loglam=False, use_skyexp_fid=False, find_pins=find_pins)
                total_pin_peaks.extend(pin_peaks_list)

    if len(total_pin_peaks) > 0:
        total_pin_peaks = np.vstack(total_pin_peaks)
    return exposure_list, spec_sky_list, total_pin_peaks

def resample_regular(b_data, r_data, accumulate_result, use_loglam=False, use_skyexp_fid=True,
                        exposure=None, fiber_list=None, find_pins=False):
    '''
    Fiber exposures are all on slightly different grids, with slightly different starting
    points; and these are not the co-add fiducial grid.  E.g. a 'red' spectra might have 3150
    pixels between something like 5800 and 10400 A, while the fid. grid has only roughly 2550
    pixels in this range.

    In order to build sky spectra that use both cameras we need to be able to combine multiple
    fibers; this requires lining up wlen's that are all slightly offset from each other.
    Additionally, need to be able to merge red and blue; this would be easiest if the two
    cameras' overlapping region used common bins.

    Seems reasonable to establish a new "exposure fiducial" grid; from inspecting data, the
    smallest wavelength delta is about 0.975 A, ranging up to about 1.120 A.  Using this
    smaller value, a nice-ish round number of 7100 spans from 3500.26 to 10422.76.

    In addition, late in the game change:  Going to pin resulting grid to 5200.26 (or,
    actually, 5200.66, since that is closest bin.)  Since this is on blue side, can't pin red
    spectrograph using this.  COULD choose another point on the red side, align there, and
    also pin, then sort out how to combine red and blue again.  But using CFrames, R and B
    should already be aligned; so resample, then pin, then accumulate.
    '''
    def _find_closest_peak(target_wavelength, grid, data):
        gt_ind = np.argmax(grid > target_wavelength)
        max_val_ind = np.argmax(data[gt_ind-2 : gt_ind+2])
        return grid[gt_ind-2+max_val_ind]

    def _find_grid_pins(resampled_row, fiber, exposure):
        data = resampled_row['flux']+resampled_row['sky']
        wavelength = resampled_row['wavelength']

        window = general_gaussian(21, p=0.5, sig=3)
        filtered = data[:peak_seek_end].copy()
        if np.any(filtered.mask) and np.any(~filtered.mask):
            filtered[filtered.mask] = np.interp(wavelength[filtered.mask], wavelength[~filtered.mask], filtered[~filtered.mask])
        filtered = fftconvolve(window, filtered)
        filtered = (np.ma.average(data[:peak_seek_end]) / np.ma.average(filtered)) * filtered
        filtered = np.roll(filtered, -10)[:peak_seek_end]
        filtered_peak_inds = np.array(argrelextrema(filtered, np.ma.greater))
        filtered_peak_wlens = (filtered_peak_inds*0.975)+3500.26

        peak_set = [fiber, exposure]
        for target in pin_peaks:
            peak_set.append(_find_closest_peak(target, wavelength, data))

        return tuple(peak_set)

    def _r_and_a(row, passthrough, accumulate_result, find_pins=True, fiber=None, exposure=None):
        grid = (skyexp_wlen_out if not use_loglam else skyexp_loglam_out) \
                    if use_skyexp_fid else (10**bdspec.fiducial_loglam if not use_loglam \
                    else bdspec.fiducial_loglam)
        resampled_data = resample(row, ('wavelength' if not use_loglam else 'loglam'), grid, passthrough)
        pins = None
        if find_pins:
            pins = _find_grid_pins(resampled_data, fiber, exposure)

        return accumulate(accumulate_result, resampled_data, data_out=accumulate_result,
                    join=('wavelength' if not use_loglam else 'loglam'),
                    add=('flux'), weight='ivar'), \
                pins

    pin_peaks_list = []
    passthrough = [name for name in b_data.dtype.names if name != ('wavelength' if not use_loglam else 'loglam')]
    if b_data.ndim > 1:
        if fiber_list is None:
            fiber_list = [None]*len(b_data)
        for b_row, r_row, fiber in zip(b_data, r_data, fiber_list):
            accumulate_result, peak_set = _r_and_a(b_row, passthrough, accumulate_result,
                                                find_pins=find_pins, fiber=fiber, exposure=exposure)
            if peak_set is not None:
                pin_peaks_list.append(peak_set) #Only for blue
            accumulate_result, peak_set = _r_and_a(r_row, passthrough, accumulate_result,
                                                find_pins=False)
    else:
        accumulate_result, peak_set = _r_and_a(b_data, passthrough, accumulate_result,
                                                find_pins=find_pins, fiber=fiber_list, exposure=exposure)
        if peak_set is not None:
            pin_peaks_list.append(peak_set) #Only for blue
        accumulate_result, peak_set = _r_and_a(r_data, passthrough, accumulate_result,
                                                find_pins=False)

    if len(pin_peaks_list) > 0:
        pin_peaks_list = np.vstack(pin_peaks_list)
    return accumulate_result, pin_peaks_list

def save_stacks(stacks, fiber_group, exposures, output_format):
    plate = fiber_group[0]['PLATE']
    mjd = fiber_group[0]['MJD']
    for stackedexp, exp in zip(stacks, exposures):
        '''
        flag_arr = (stackedexp['flux'] < -0.5)
        if np.any(flag_arr):
            print "GOT NEGATIVE SKY FLUX IN FINAL STACK!!!!"
            print plate, mjd, exp
            print stackedexp['wavelength'][flag_arr]
            print stackedexp['flux'][flag_arr]
            #Should put in ivar cutoff here.  Like any ivar < 0.001 is excluded, regardless of flux value
        '''
        exp_table = Table(data=stackedexp)
        if output_format == "FITS":
            exp_table.write("stacked_sky_{}-{}-exp{:02d}.fits".format(plate, mjd, exp), format="fits")
        elif output_format == "CSV":
            exp_table.write("stacked_sky_{}-{}-exp{:02d}.csv".format(plate, mjd, exp), format="ascii.csv")

def save_pins(pins, fiber_group):
    plate = fiber_group[0]['PLATE']
    mjd = fiber_group[0]['MJD']

    pins_arr = np.vstack(pins)

    names = []
    types = []
    for name in pin_peaks_dtype:
        names.append(name[0])
        types.append(name[1])
    exp_table = Table(data=pins_arr, names=names, dtype=types)

    exp_table_grouped = exp_table.group_by(['exposure'])
    for group in exp_table_grouped.groups:
        #Bother with fits?  Think this is something want to be able to easily visually/manually muck with
        group.write("stacked_sky_{}-{}-exp{:02d}_pins.csv".format(plate, mjd, group[0]['exposure']), format="ascii.csv")

    return exp_table

def main():
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Stack sky fibers from a PLATE/MJD into a single spectra.')
        parser.add_argument(
            '--fibers', type=str, default=None, metavar='FIBERS', required=True,
            help='File that contains list of PLATE, MJD, FIBER which are to be stacked (by PLATE/MJD).'
        )
        parser.add_argument(
            '--output', type=str, default='FITS', metavar='OUTPUT',
            help='Output format, either of FITS or CSV, defaults to FITS.'
        )
        parser.add_argument(
            '--pins', action='store_true',
            help='Whether or not to output "pins": file with wavelengths of prominent peaks between ~4400 and 5600'
        )
        args = parser.parse_args()

        sky_fibers_table = Table.read(args.fibers, format='ascii')
        sky_fibers_table = sky_fibers_table.group_by(["PLATE", "MJD"])

        progress_bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(sky_fibers_table)).start()
        counter = 0

        for group in sky_fibers_table.groups:
            exposures, stacks, pin_peaks = stack_exposures(group, use_cframe=True, find_pins=args.pins)

            save_stacks(stacks, group, exposures, args.output)
            if args.pins:
                save_pins(pin_peaks, group)

            counter += len(group)
            progress_bar.update(counter)
        progress_bar.finish()

if __name__ == '__main__':
    main()

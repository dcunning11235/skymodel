import numpy as np
from astropy.table import Table
from astropy.table import vstack
from astropy.time import Time
import astropy.coordinates as ascoord
import sys
import datetime as dt
import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec

from progressbar import ProgressBar, Percentage, Bar

from astropy.utils.compat import argparse

'''
This script takes in a list of plate-mjd combos (as output by e.g. bossquery for
    bossquery --what "PLATE,MJD" --where "PLATE<=4500 and FIBER=1" --save plate_mjd_to_4500.dat

It walks the list and outputs metadata about the exposures that make up those plates; it does
this by looking at metadata in (full) spec files for fibers 1 and 501.  Since there there
should (must) be a red and a blue exposure for each full exposure, corresponing color exposures
are output on one line.
'''

tai_base_date_time = dt.datetime(1858, 11, 17)
finder = bdpath.Finder()
manager = bdremote.Manager()

def file_deets(plate, mjd, gather=False):
    fiber = 1
    spec_path = finder.get_spec_path(plate, mjd, fiber, lite=False)
    spec_path = manager.get(spec_path)
    spec = bdspec.SpecFile(spec_path)
    exposure_list = spec.exposures.table['science'][0:(spec.num_exposures)]

    exposure_data = np.empty((spec.num_exposures, ),
                        dtype=[('PLATE', int), ('MJD', int), ('EXP_ID', "|S6"), ('RA', float), ('DEC', float),
                            ('AZ', float), ('ALT', float), ('AIRMASS', float), ('TAI-BEG', '|S25'), ('TAI-END', '|S25')])
    for i, exp in enumerate(spec.exposures.sequence):
        exp_header = spec.get_exposure_hdu(i, 'r1').read_header()
        exposure_data[i] = (int(plate), int(mjd), exp,
                        exp_header['RA'], exp_header['DEC'],
                        exp_header['AZ'], exp_header['ALT'], exp_header['AIRMASS'],
                        (tai_base_date_time + dt.timedelta(seconds=exp_header['TAI-BEG'])).isoformat(),
                        (tai_base_date_time + dt.timedelta(seconds=exp_header['TAI-END'])).isoformat())

    if not gather:
        for name in exposure_data.dtype.names:
            print name, "\t",
        print ""
        for row in exposure_data:
            for el in row:
                print el, "\t",
            print ""
        return None
    else:
        return exposure_data

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Pull metadata from FITS for PLATE/MJD combos, output.')

    parser.add_argument(
        '--plate', type=int, default=None, metavar='PLATE',
        help='Plate number to pull metadata for.')
    parser.add_argument(
        '--mjd', type=int, default=None, metavar='MJD',
        help='MJD of plate observation to use (can be omitted if only one value is possible)')
    parser.add_argument(
        '--fiber', type=int, default=1, metavar='FIBER',
        help='Fiber number identifying the spectrum of the requested PLATE-MJD to plot.')
    parser.add_argument(
        '--list_file', type=str, default=None, metavar='LIST_FILE',
        help='File that contains list of PLATE, MJD, FIBER records to output metadata for; e.g. the output from bossquery'
    )
    parser.add_argument(
        '--output', type=str, default='FITS', metavar='OUTPUT',
        help='Output format, either of ''FITS'' or ''CSV'', defaults to FITS.'
    )
    parser.add_argument(
        '--dont_gather', action='store_true',
        help='Flag to prevent storing output in file; just dumps metadata to stdout.'
    )
    args = parser.parse_args()

    if args.plate is not None and args.mjd is not None:
        file_deets(plate=args.plate, mjd=args.mjd, fiber=args.fiber)
    elif args.list_file is not None:
        plates_table = Table.read(args.list_file, format='ascii')

        exposure_table_list = []
        exposure_table = None

        if not args.dont_gather:
            progress_bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(plates_table)).start()
            counter = 0
        for row in plates_table:
            try:
                exposure_data = file_deets(row['PLATE'], row['MJD'], gather=not args.dont_gather)
                if exposure_data is not None:
                    exposure_table_list.append(Table(exposure_data))
            except RuntimeError as re:
                print "Caught runtime:"
                print re
            if not args.dont_gather:
                counter += 1
                progress_bar.update(counter)
        if not args.dont_gather:
            progress_bar.finish()

        if len(exposure_table_list):
            if len(exposure_table_list) > 1:
                exposure_table = vstack(exposure_table_list)
            else:
                exposure_table = exposure_table_list[0]

            if args.output.upper() == 'CSV':
                exposure_table.write("exposure_metadata.csv", format="ascii.csv")
            elif args.output.upper() == 'FITS':
                exposure_table.write("exposure_metadata.fits", format="fits")

if __name__ == '__main__':
    main()

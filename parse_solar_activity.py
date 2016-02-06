import astropy
from astropy.table import Table
from astropy.table import Column
import astropy.table as at
from astropy.utils.compat import argparse

import numpy as np

import datetime as dt
import sys

def date_converter(datecol):
    workcol = map(str.strip, datecol)
    workcol = map(str.split, workcol)

    def _date_it(tokens):
        new_date = dt.date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
        return new_date.strftime("%Y-%b-%d")

    workcol = map(_date_it, workcol)

    return workcol

def main():
    data_frames = []
    converters={
        'DATE': [(date_converter, astropy.io.ascii.core.StrType)]
    }
    for i in range(2, len(sys.argv)):
        activity_file = sys.argv[i]
        activity_table = Table.read(activity_file, format='ascii.fixed_width_two_line', converters=converters, guess=False)
        data_frames.append(activity_table["DATE", "SESC", "SSAREA"])
    total_data = at.vstack(data_frames)

    total_data.write("{}.csv".format(sys.argv[1]), format="ascii.csv")

if __name__ == '__main__':
    main()

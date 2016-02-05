import astropy
from astropy.table import Table
from astropy.table import Column
import astropy.table as at
from astropy.utils.compat import argparse

import numpy as np

import datetime as dt
import sys

'''
Solar and Lunar ephermeris data from JPL Horizons
Common options:
Ephemeris Type:  Observer
Observer Location:  Apache Point-Sloan Digital Sky Survey [645] ( 254 10' 45.9'' E, 32 46' 49.8'' N, 2791.2 m )
Start:  2009-12-01
End: 2014-07-01
Step: 15 m (minutes)

Lunar Options:
1-5,8-10,20,23,24,33

Solar Options:
1-5,9,20,23,24,33

Column widths (from inspection) for ephermeris files, as downloaded:
2-18:  Date/Time (UT)
20-21: Sun-Moon indicators
24-34: J2000 RA
36-46: J2000 Dec
48-58: Apparent RA
60-70: Apparent Dec
72-79: Change in (apparent) RA*Cosine(DEC)
82-89: dDEC / dt
91-98: Apparent Azimuth
101-107: Apparent Elevation
111-116: dAZ*Cos(ELE)
121-126: d(ELE)/dt
129-133: Airmass
136-140: Visual Magnitude Extinction
142-147: Apparent magnitude
151-154: Surface brightness (Mag/ArcSec)
157-162: Pct. Illumination
164-179: Delta (AU) (Distance)
182-191: DelDot (km/s) (Motion to/away from observer)
193-200: Sun-Observer-Target Angle
202-203: /T or /L (Leading or trailing Sun; evening or morning)
206-212: Sun-Target-Observer Angle
214-220: Target-Observer-Moon Angle
222-225: Pct Illumination
227-236: Galactic Longitute
238-247: Galactic Latitude

SOLAR PRESENCE (OBSERVING SITE)
  Time tag is followed by a blank, then a solar-presence symbol:

        '*'  Daylight (refracted solar upper-limb on or above apparent horizon)
        'C'  Civil twilight/dawn
        'N'  Nautical twilight/dawn
        'A'  Astronomical twilight/dawn
        ' '  Night OR geocentric ephemeris

LUNAR PRESENCE WITH TARGET RISE/TRANSIT/SET MARKER (OBSERVING SITE)
  The solar-presence symbol is immediately followed by another marker symbol:

        'm'  Refracted upper-limb of Moon on or above apparent horizon
        ' '  Refracted upper-limb of Moon below apparent horizon OR geocentric
        'r'  Rise    (target body on or above cut-off RTS elevation)
        't'  Transit (target body at or past local maximum RTS elevation)
        's'  Set     (target body on or below cut-off RTS elevation)
'''

def ra_converter(radcol):
    workcol = map(str.strip, radcol)
    workcol = map(str.split, workcol)

    def _float_it(tokens):
        return float(tokens[0])*15 + float(tokens[1])/4 + float(tokens[2])/240

    workcol = map(_float_it, workcol)

    return workcol

def dec_converter(deccol):
    workcol = map(str.strip, deccol)
    workcol = map(str.split, workcol)

    def _float_it(tokens):
        return float(tokens[0]) + float(tokens[1])/60 + float(tokens[2])/3600

    workcol = map(_float_it, workcol)

    return workcol

def sun_flag_converter(sunflagcol):
    def _swap_it(val):
        return "TRAIL" if val == '/T' else "LEAD"

    return map(_swap_it, sunflagcol)

def main():
    data_frames = []
    converters={
        'RA_ABS': [(ra_converter, astropy.io.ascii.core.FloatType)],
        'RA_APP': [(ra_converter, astropy.io.ascii.core.FloatType)],
        'DEC_ABS': [(dec_converter, astropy.io.ascii.core.FloatType)],
        'DEC_APP': [(dec_converter, astropy.io.ascii.core.FloatType)],
        'TL': [(sun_flag_converter, astropy.io.ascii.core.StrType)]
    }
    for i in range(2, len(sys.argv)):
        ephemeris_file = sys.argv[i]
        ephem_table = Table.read(ephemeris_file, format='ascii.fixed_width_two_line', converters=converters, guess=False)
        data_frames.append(ephem_table)
    total_data = at.vstack(data_frames)

    total_data.write("{}.csv".format(sys.argv[1]), format="ascii.csv")

if __name__ == '__main__':
    main()

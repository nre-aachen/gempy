import pandas as pn
import glob
import os
import numpy as np
path = '/home/bl3/Documents/CSIRO/MBS_split_by_holeID/MBS_split_by_holeID'
import theano.tensor as T
import theano
import sys, os
sys.path.append("/home/bl3/PycharmProjects/GeMpy/GeMpy")

# Importing GeMpy modules
import GeMpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

# Setting the extent
geo_data = GeMpy.import_data([3.377617e+05, 5.075333e+05,
                              6.743846e+06,1.011812e+07,
                              2.056000e+01, 4.884000e+02], [50,50,50])

# Reading the interfaces data, created in the previous section
inter = pn.read_pickle('/home/bl3/Documents/CSIRO/interfaces')

# Choosing the formations
inter = inter[inter['formation'].isin(['harz', 'dun', 'gabb', 'opx', 'base'])]

# Creating one arbitrary foliation (the algorithm needs at least one)
foli = inter[['X', 'Y', 'Z']].iloc[0]
foli = foli.to_frame().T
foli['azimuth'] = 0
foli['dip'] = 0
foli['polarity'] = 1
foli['formation'] = 'harz'

# Setting interfaces and foliations in GeMpy
GeMpy.set_interfaces(geo_data, inter)
GeMpy.set_foliations(geo_data, foli)


# Preparing data to interpolate

data_interp = GeMpy.set_interpolator(geo_data, u_grade = 0, compute_potential_field= False, compute_block_model = False,
                      verbose = 0)

input_data_T = data_interp.interpolator.tg.input_parameters_list()
input_data_P = data_interp.interpolator.data_prep()
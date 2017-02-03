# Importing
import theano.tensor as T
import sys, os
sys.path.append("../GeMpy")

# Importing GeMpy modules
import GeMpy

# Reloading (only for development purposes)
import importlib
importlib.reload(GeMpy)
# Usuful packages
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt

# This was to choose the gpu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Default options of printin
np.set_printoptions(precision = 6, linewidth= 130, suppress =  True)



geo_data = GeMpy.import_data([696000,747000,6863000,6950000,-20000, 2000],[ 40, 40, 80],
                         path_f = os.pardir+"/input_data/a_Foliations.csv",
                         path_i = os.pardir+"/input_data/a_Points.csv")

GeMpy.set_data_series(geo_data, {"EarlyGranite_Series":geo_data.formations[-1],
                      "BIF_Series":(geo_data.formations[0], geo_data.formations[1]),
                      "SimpleMafic_Series":geo_data.formations[2]},
                       order_series = ["EarlyGranite_Series",
                              "BIF_Series",
                              "SimpleMafic_Series"], verbose=0)

#GeMpy.set_interpolator(geo_data, u_grade = 0, compute_potential_field=True, verbose = 4)
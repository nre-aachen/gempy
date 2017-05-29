import pandas as pn
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import pymc3 as pm
import sys
import gempy
import coKriging as ck
import importlib
importlib.reload(ck)


# Read data
data = pn.read_pickle('../../CSIRO/Domained_data.pkl')
geomodel = np.load("../../CSIRO/3Dmodel.npy")
exp_variogram = pn.read_pickle("../../CSIRO/experimental_variogram.p")

# In the future the best would be to save a file with all the 3D model info
grid = gempy.InputData([422050, 423090, 8429400, 8432100, -500, 332], [50, 50, 50]).grid.grid

sgs = ck.SGS(exp_var=exp_variogram, properties=['Al_ppm-Al_ppm', 'Al_ppm-Ca_ppm', 'Ca_ppm-Al_ppm', 'Ca_ppm-Ca_ppm' ], n_exp = 5)
sgs.set_data(data)
sgs.set_lithology('opx')
sgs.set_geomodel(geomodel)

sgs.choose_lithology_elements(elem = ['Al_ppm', 'Ca_ppm'])
sgs.select_segmented_grid(grid)
model = sgs.fit_cross_cov(n_exp=5)
with model:
    start = pm.find_MAP() # Find starting value by optimization
    step = pm.Metropolis()
    #db = pm.backends.SQLite('SQtry.sqlite')
    trace = pm.sample(2000, step, init=start, progressbar=True, njobs=1);
sgs.set_trace(trace)

SED_f = ck.theano_sed()

df = sgs.data_to_inter[['X', 'Y', 'Z']]
selected_cluster_grid = sgs.grid_to_inter[5000:5002]
dist = SED_f(df, selected_cluster_grid)

# Checking the radio of the simulation
for r in range(100, 1000, 100):
    select = (dist < r).any(axis=1)
    if select.sum() > 50:
        break

j = np.zeros(sgs.grid_to_inter.shape[0], dtype=bool)
j[5000:5002] = True

sgs.solve_kriging(select, j)
import pandas as pn
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import pymc3 as pm
import sys
import gempy as gp
from GeoPhysics import GeoPhysicsPreprocessing


geo_data = gp.read_pickle('../Prototype Notebook/geo_data.pickle')
inter_data = gp.InterpolatorInput(geo_data, compile_theano=False)
gpp = GeoPhysicsPreprocessing(inter_data,300,  [696000,747000,6863000,6950000,-20000, 200], res_grav = [10, 10])
print(gpp)
gpp.looping_z_decomp(20)

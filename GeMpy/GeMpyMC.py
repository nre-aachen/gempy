"""
Module with classes and methods to perform bayesian statistics directly to the potential field method models using
pymc3
Tested on Ubuntu 14

Created on 10/01/2017

@author: Miguel de la Varga

"""

from .DataManagement import DataManagement
import pymc3 as pm

class GeMpyMC(DataManagement.InterpolatorClass):
    def __init__(self, geo_data):


"""
@author: Alexander Schaaf
"""
import pymc
import numpy as np
import math


def create_priors(data):
    """
    data = interp_data.interpolator.pandas_rest_layer_points
    # data_ref = interp_data.interpolator.ref_layer_points
    """
    # index = data.index

    X_priors = np.array([])
    Y_priors = np.array([])
    Z_priors = np.array([])

    # iterate over dataframe
    for i, row in data.T.iteritems():
        if math.isnan(row["X_std"]):
            X_priors = np.append(X_priors, row["X"])
        else:
            dist = pymc.Normal("X_" + str(i), row["X"], 1 / row["X_std"] ** 2)
            X_priors = np.append(X_priors, dist)

        if math.isnan(row["Y_std"]):
            Y_priors = np.append(Y_priors, row["Y"])
        else:
            dist = pymc.Normal("Y_" + str(i), row["Y"], 1 / row["Y_std"] ** 2)
            Y_priors = np.append(Y_priors, dist)

        if math.isnan(row["Z_std"]):
            Z_priors = np.append(Z_priors, row["Z"])
        else:
            dist = pymc.Normal("Z_" + str(i), row["Z"], 1 / row["Z_std"] ** 2)
            Z_priors = np.append(Z_priors, dist)

    return X_priors, Y_priors, Z_priors  #, index
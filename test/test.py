import pytest

import theano
import numpy as np
import sys
sys.path.append("../GeMpy")
import GeMpy


class TestNoFaults:

    def test_a(self):
        """
        2 Horizontal layers with drift oen
        """
        # Importing the data from csv files and settign extent and resolution
        geo_data = GeMpy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        data_interp = GeMpy.set_interpolator(geo_data,
                                             dtype="float64",
                                             verbose=['solve_kriging'])
        # This cell will go to the backend

        # Set all the theano shared parameters and return the symbolic variables (the input of the theano function)
        input_data_T = data_interp.interpolator.tg.input_parameters_list()

        # Prepare the input data (interfaces, foliations data) to call the theano function.
        # Also set a few theano shared variables with the len of formations series and so on
        input_data_P = data_interp.interpolator.data_prep(u_grade=[3])

        # Compile the theano function.
        self.compiled_f = theano.function(input_data_T, data_interp.interpolator.tg.whole_block_model(),
                                     allow_input_downcast=True, profile=True)
        sol = self.compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
                         input_data_P[5])

        real_sol = np.load('test_a_sol.npy')

        np.testing.assert_array_almost_equal(sol, real_sol)

        GeMpy.plot_section(geo_data, 25, block=sol[0, 0, :], direction='y', plot_data=True)
        GeMpy.plot_potential_field(geo_data, sol[0, 1, :], 25)



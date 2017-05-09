import pytest

import theano
import numpy as np
import sys
sys.path.append("../GeMpy")
import GeMpy


class TestNoFaults:
    # Init interpolator
    @pytest.fixture(scope='class')
    def interpolator(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = GeMpy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        data_interp = GeMpy.set_interpolator(geo_data,
                                             dtype="float64",
                                             verbose=['solve_kriging'])
    @pytest.fixture(scope='class')
    def theano_f(self):
        # Importing the data from csv files and settign extent and resolution
        geo_data = GeMpy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        data_interp = GeMpy.set_interpolator(geo_data,
                                             dtype="float64",
                                             verbose=['solve_kriging'])

        # Set all the theano shared parameters and return the symbolic variables (the input of the theano function)
        input_data_T =   data_interp.interpolator.tg.input_parameters_list()

        # Prepare the input data (interfaces, foliations data) to call the theano function.
        # Also set a few theano shared variables with the len of formations series and so on
        input_data_P =   data_interp.interpolator.data_prep(u_grade=[3])

        # Compile the theano function.
        compiled_f = theano.function(input_data_T,   data_interp.interpolator.tg.whole_block_model(),
                                     allow_input_downcast=True, profile=True)

        return data_interp, compiled_f

    def test_a(self, theano_f):
        """
        2 Horizontal layers with drift one
        """

        data_interp = theano_f[0]
        compiled_f = theano_f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = GeMpy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_a/test_a_Foliations.csv",
                                     path_i="./GeoModeller/test_a/test_a_Points.csv")

        rescaled_data = GeMpy.rescale_data(geo_data)

        data_interp.interpolator._data_scaled = rescaled_data
        data_interp.interpolator.order_table()
        data_interp.interpolator.set_theano_shared_parameteres()

        # Prepare the input data (interfaces, foliations data) to call the theano function.
        # Also set a few theano shared variables with the len of formations series and so on
        input_data_P = data_interp.interpolator.data_prep(u_grade=[3])
        # Compile the theano function.

        sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
                         input_data_P[5])

        real_sol = np.load('test_a_sol.npy')
        np.testing.assert_array_almost_equal(sol, real_sol, decimal=3)

        GeMpy.plot_section(geo_data, 25, block=sol[0, 0, :], direction='y', plot_data=True)
        GeMpy.plot_potential_field(geo_data, sol[0, 1, :], 25)

    def test_b(self, theano_f):
        """
        Two layers a bit curvy, drift degree 1
        """
        data_interp = theano_f[0]
        compiled_f = theano_f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = GeMpy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_b/test_b_Foliations.csv",
                                     path_i="./GeoModeller/test_b/test_b_Points.csv")

        rescaled_data = GeMpy.rescale_data(geo_data)

        data_interp.interpolator._data_scaled = rescaled_data
        data_interp.interpolator.order_table()
        data_interp.interpolator.set_theano_shared_parameteres()

        # Prepare the input data (interfaces, foliations data) to call the theano function.
        # Also set a few theano shared variables with the len of formations series and so on
        input_data_P = data_interp.interpolator.data_prep(u_grade=[3])

        sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
                         input_data_P[5])

        real_sol = np.load('test_b_sol.npy')
        np.testing.assert_array_almost_equal(sol, real_sol, decimal=3)

    def test_c(self, theano_f):
        """
        Two layers a bit curvy, drift degree 2
        """
        data_interp = theano_f[0]
        compiled_f = theano_f[1]

        # Importing the data from csv files and settign extent and resolution
        geo_data = GeMpy.import_data([0, 10, 0, 10, -10, 0], [50, 50, 50],
                                     path_f="./GeoModeller/test_c/test_c_Foliations.csv",
                                     path_i="./GeoModeller/test_c/test_c_Points.csv")

        rescaled_data = GeMpy.rescale_data(geo_data)

        data_interp.interpolator._data_scaled = rescaled_data
        data_interp.interpolator.order_table()
        data_interp.interpolator.set_theano_shared_parameteres()

        # Prepare the input data (interfaces, foliations data) to call the theano function.
        # Also set a few theano shared variables with the len of formations series and so on
        input_data_P = data_interp.interpolator.data_prep(u_grade=[0])

        sol = compiled_f(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3], input_data_P[4],
                         input_data_P[5])

        real_sol = np.load('test_c_sol.npy')
        np.testing.assert_array_almost_equal(sol, real_sol, decimal=3)
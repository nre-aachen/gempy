from __future__ import division

import theano
import theano.tensor as T
import numpy as np
import sys, os
import pandas as pn
import theanograf
from Visualization import PlotData


class DataManagement(object):
    """
    Class to import the raw data of the model and set data classifications into formations and series

    Args:
        extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        Resolution ((Optional[list])): [nx, ny, nz]. Defaults to 50
        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_f: Path to the data bases of foliations. Default os.getcwd()

    Attributes:
        extent(list):  [x_min, x_max, y_min, y_max, z_min, z_max]
        resolution ((Optional[list])): [nx, ny, nz]
        Foliations(pandas.core.frame.DataFrame): Pandas data frame with the foliations data
        Interfaces(pandas.core.frame.DataFrame): Pandas data frame with the interfaces data
        formations(numpy.ndarray): Dictionary that contains the name of the formations
        series(pandas.core.frame.DataFrame): Pandas data frame which contains every formation within each series
    """

    # TODO: Data management using pandas, find an easy way to add values
    # TODO: Probably at some point I will have to make an static and dynamic data classes
    def __init__(self, extent,
                 resolution=[50, 50, 50],
                 path_i=None, path_f=None,
                 **kwargs):

        # Deprecated
        self.xmin = extent[0]
        self.xmax = extent[1]
        self.ymin = extent[2]
        self.ymax = extent[3]
        self.zmin = extent[4]
        self.zmax = extent[5]
        self.nx = resolution[0]
        self.ny = resolution[1]
        self.nz = resolution[2]
        # -------------------

        self.extent = extent
        self.resolution = resolution

        # TODO choose the default source of data. So far only
        if path_f:
            self.foliations = self.load_data_csv(data_type="foliations", path=path_f, **kwargs)
            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(self.foliations.columns), \
                "One or more columns do not match with the expected values " + str(self.foliations.columns)
        else:
            self.foliations = pn.DataFrame(columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series'])
        if path_i:
            self.interfaces = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(self.interfaces.columns), \
                "One or more columns do not match with the expected values " + str(self.interfaces.columns)
        else:
            self.interfaces = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation'])

        self._set_formations()
        self.series = self.set_series()
        self.calculate_gradient()

        # Create default grid object. (Is this necessary now?)
        self.grid = self.create_grid(extent=None, resolution=None, grid_type="regular_3D", **kwargs)

    def _set_formations(self):
        """
        Function to import the formations that will be used later on. By default all the formations in the tables are
        chosen.

        Returns:
             pandas.core.frame.DataFrame: Data frame with the raw data

        """

        try:
            # foliations may or may not be in all formations so we need to use interfaces
            self.formations = self.interfaces["formation"].unique()

            # TODO: Trying to make this more elegant?
            for el in self.formations:
                for check in self.formations:
                    assert (el not in check or el == check), "One of the formations name contains other" \
                                                             " string. Please rename." + str(el) + " in " + str(
                        check)

                    # TODO: Add the possibility to change the name in pandas directly
                    # (adding just a 1 in the contained string)
        except AttributeError:
            pass

    def calculate_gradient(self):
        """
        Calculate the gradient vector of module 1 given dip and azimuth to be able to plot the foliations

        Returns:
            self.foliations: extra columns with components xyz of the unity vector.
        """

        self.foliations['G_x'] = np.sin(np.deg2rad(self.foliations["dip"])) * \
                                 np.sin(np.deg2rad(self.foliations["azimuth"])) * self.foliations["polarity"]
        self.foliations['G_y'] = np.sin(np.deg2rad(self.foliations["dip"])) * \
                                 np.cos(np.deg2rad(self.foliations["azimuth"])) * self.foliations["polarity"]
        self.foliations['G_z'] = np.cos(np.deg2rad(self.foliations["dip"])) * self.foliations["polarity"]

    # TODO set new interface/set

    def create_grid(self, extent=None, resolution=None, grid_type="regular_3D", **kwargs):
        """
        Method to initialize the class grid. So far is really simple and only has the regular grid type

        Args:
            grid_type (str): regular_3D or regular_2D (I am not even sure if regular 2D still working)
            **kwargs: Arbitrary keyword arguments.

        Returns:
            self.grid(GeMpy_core.grid): Object that contain different grids
        """
        if not extent:
            extent = self.extent
        if not resolution:
            resolution = self.resolution

        return self.GridClass(extent, resolution, grid_type=grid_type, **kwargs)

    def get_raw_data(self, dtype='all'):

        import pandas as _pn
        if dtype == 'foliations':
            raw_data = self.foliations
        elif dtype == 'interfaces':
            raw_data = self.interfaces
        elif dtype == 'all':
            raw_data = _pn.concat([self.interfaces, self.foliations], keys=['interfaces', 'foliations'])
        return raw_data

    def i_set_data(self, dtype="foliations"):
        import qgrid
        qgrid.nbinstall(overwrite=True)
        qgrid.set_defaults(show_toolbar=True)
        assert dtype is 'foliations' or dtype is 'interfaces', 'dtype must be either foliations or interfaces'
        qgrid.show_grid(self.get_raw_data(dtype=dtype))

    @staticmethod
    def load_data_csv(data_type, path=os.getcwd(), **kwargs):
        """
        Method to load either interface or foliations data csv files. Normally this is in which GeoModeller exports it

        Args:
            data_type (str): 'interfaces' or 'foliations'
            path (str): path to the files. Default os.getcwd()
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        # TODO: in case that the columns have a different name specify in pandas which columns are interfaces /
        #  coordinates, dips and so on.
        # TODO: use pandas to read any format file not only csv

        if data_type == "foliations":
            return pn.read_csv(path, **kwargs)
        elif data_type == 'interfaces':
            return pn.read_csv(path, **kwargs)
        else:
            raise NameError('Data type not understood. Try interfaces or foliations')

        # TODO if we load different data the Interpolator parameters must be also updated:  Research how and implement

    def set_interfaces(self, interf_Dataframe, append=False):

        assert set(['X', 'Y', 'Z', 'formation']).issubset(interf_Dataframe.columns), \
            "One or more columns do not match with the expected values " + str(interf_Dataframe.columns)

        if append:
            self.interfaces = self.interfaces.append(interf_Dataframe)
        else:
            self.interfaces = interf_Dataframe

        self._set_formations()
        self.set_series()

    def set_foliations(self, foliat_Dataframe, append=False):

        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(foliat_Dataframe.columns)
        if append:
            self.foliations = self.foliations.append(foliat_Dataframe)
        else:
            self.foliations = foliat_Dataframe

        self._set_formations()
        self.set_series()
        self.calculate_gradient()

    def set_series(self, series_distribution=None, order=None):
        """
        Method to define the different series of the project

        Args:
            series_distribution (dict): with the name of the serie as key and the name of the formations as values.
            order(Optional[list]): order of the series by default takes the dictionary keys which until python 3.6 are
                random. This is important to set the erosion relations between the different series

        Returns:
            self.series: A pandas DataFrame with the series and formations relations
            self.interfaces: one extra column with the given series
            self.foliations: one extra column with the given series
        """

        if series_distribution is None:
            # TODO: Possibly we have to debug this function
            _series = {"Default serie": self.formations}

        else:
            assert type(series_distribution) is dict, "series_distribution must be a dictionary, " \
                                                      "see Docstring for more information"
            _series = series_distribution
        if not order:
            order = _series.keys()
        _series = pn.DataFrame(data=_series, columns=order)
        assert np.count_nonzero(np.unique(_series.values)) is len(self.formations), \
            "series_distribution must have the same number of values as number of formations %s." \
            % self.formations

        self.interfaces["series"] = [(i == _series).sum().argmax() for i in self.interfaces["formation"]]
        self.foliations["series"] = [(i == _series).sum().argmax() for i in self.foliations["formation"]]

        self.series = _series
        return _series

    class GridClass(object):
        """
        Class with set of functions to generate grids

        Args:
            extent (list):  [x_min, x_max, y_min, y_max, z_min, z_max]
            resolution (list): [nx, ny, nz].
            grid_type(str): Type of grid. So far only regular 3D is implemented
        """

        def __init__(self, extent, resolution, grid_type="regular_3D"):
            self._grid_ext = extent
            self._grid_res = resolution

            if grid_type == "regular_3D":
                self.grid = self.create_regular_grid_3d()
            elif grid_type == "regular_2D":
                self.grid = self.create_regular_grid_2d()
            else:
                print("Wrong type")

        def create_regular_grid_3d(self):
            """
            Method to create a 3D regular grid where is interpolated

            Returns:
                numpy.ndarray: Unraveled 3D numpy array where every row correspond to the xyz coordinates of a regular grid
            """

            g = np.meshgrid(
                np.linspace(self._grid_ext[0], self._grid_ext[1], self._grid_res[0], dtype="float32"),
                np.linspace(self._grid_ext[2], self._grid_ext[3], self._grid_res[1], dtype="float32"),
                np.linspace(self._grid_ext[4], self._grid_ext[5], self._grid_res[2], dtype="float32"), indexing="ij"
            )

            return np.vstack(map(np.ravel, g)).T.astype("float32")

    class InterpolatorClass(object):
        """
        Class which contain all needed methods to perform potential field implicit modelling in theano

        Args:
            _data(GeMpy_core.DataManagement): All values of a DataManagement object
            _grid(GeMpy_core.grid): A grid object
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
        """

        def __init__(self, _data, _grid=None, compute_block_model=True,
                     compute_potential_field=False, *args, **kwargs):

            verbose = kwargs.get('verbose', 0)

            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'
            theano.config.compute_test_value = 'ignore'
            u_grade = kwargs.get('u_grade', 2)

            self._data = _data

            if not _grid:
                self._grid = _data.grid
            else:
                self._grid = _grid

            self.tg = theanograf.TheanoGraph(u_grade)
            self.set_theano_shared_parameteres(_data, _grid, **kwargs)

            if compute_potential_field:

                self.potential_fields = []
                self._interpolate = self.compile_potential_field_function()
                self.potential_fields = [self.compute_potential_fields(i, verbose=verbose)
                                         for i in np.arange(len(self._data.series.columns))]

            if compute_block_model:

                self._block_export = self.compile_block_model_function()
                self.block = self.compute_block_model()

        def _aux_computations_block_model(self, for_in_ser, n_formation, verbose=0):
            """
            Private function with the bridge steps from the selection of serie to the input in theano

            Args:
                for_in_ser: array with the formation for the series to interpolate
                n_formation: number of formation in the series
                verbose: verbosity

            Returns:
                self.block(theano shared[numpy.ndarray]): 3D block with the corresponding formations
            """
            # TODO Probably here I should add some asserts for sanity check
            try:
                yet_simulated = (self.block.get_value() == 0) * 1
                if verbose > 0:
                    print(yet_simulated, (yet_simulated == 0).sum())
            except AttributeError:
                yet_simulated = np.ones_like(self._grid.grid[:, 0], dtype="int8")
                print("I am in the except")

            dips_position = self._data.foliations[
                self._data.foliations["formation"].str.contains(for_in_ser)] \
                [['X', 'Y', 'Z']].as_matrix()
            dip_angles = self._data.foliations[
                self._data.foliations["formation"].str.contains(for_in_ser)]["dip"].as_matrix()
            azimuth = self._data.foliations[
                self._data.foliations["formation"].str.contains(for_in_ser)]["azimuth"].as_matrix()
            polarity = self._data.foliations[
                self._data.foliations["formation"].str.contains(for_in_ser)]["polarity"].as_matrix()

            if for_in_ser.count("|") == 0:

                layers = self._data.interfaces[self._data.interfaces["formation"] == for_in_ser] \
                    [['X', 'Y', 'Z']].as_matrix()
                rest_layer_points = layers[1:]
                # TODO self.n_formation probably should not be self
                self.number_of_points_per_formation_T.set_value(np.array(rest_layer_points.shape[0], ndmin=1))
                ref_layer_points = np.tile(layers[0], (np.shape(layers)[0] - 1, 1))
            else:
                # TODO: This is ugly
                layers_list = []
                for formation in for_in_ser.split("|"):
                    layers_list.append(
                        self._data.interfaces[self._data.interfaces["formation"] == formation]
                        [['X', 'Y', 'Z']].as_matrix())
                layers = np.asarray(layers_list)

                rest_layer_points = layers[0][1:]
                rest_dim = np.array(layers[0][1:].shape[0], ndmin=1)
                for i in layers[1:]:
                    rest_layer_points = np.vstack((rest_layer_points, i[1:]))
                    rest_dim = np.append(rest_dim, rest_dim[-1] + i[1:].shape[0])
                self.tg.number_of_points_per_formation_T.set_value(rest_dim)
                ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))

            if verbose > 0:
                print("The serie formations are %s" % for_in_ser)
                if verbose > 1:
                    print("The formations are: \n"
                          "Layers ", self._data.interfaces[self._data.interfaces["formation"].str.contains(for_in_ser)],
                          " \n "
                          "foliations ",
                          self._data.foliations[self._data.foliations["formation"].str.contains(for_in_ser)])

            # self.grad is none so far. I have it for further research in the calculation of the Jacobian matrix

                if verbose > 2:
                    print('number_formations', n_formation)
                    print('rest_layer_points', rest_layer_points)

            return self._block_export(dips_position, dip_angles, azimuth, polarity,
                                      rest_layer_points, ref_layer_points,
                                      n_formation, yet_simulated)

        def _aux_computations_potential_field(self, for_in_ser, verbose=0):
            """
            Private function with the bridge steps from the selection of serie to the input in theano

            Args:
                for_in_ser: array with the formation for the series to interpolate
                verbose: verbosity

            Returns:
                numpy.ndarray: 3D array with the potential field
            """

            # TODO: change [:,:3] that is positional based for XYZ so is more consistent
            dips_position = self._data.foliations[
                self._data.foliations["formation"].str.contains(for_in_ser)] \
                [['X', 'Y', 'Z']].as_matrix()
            dip_angles = self._data.foliations[self._data.foliations["formation"].str.contains(for_in_ser)][
                "dip"].as_matrix()
            azimuth = self._data.foliations[self._data.foliations["formation"].str.contains(for_in_ser)][
                "azimuth"].as_matrix()
            polarity = self._data.foliations[self._data.foliations["formation"].str.contains(for_in_ser)][
                "polarity"].as_matrix()

            if for_in_ser.count("|") == 0:
                # layers = self._data.interfaces[self._data.interfaces["formation"].str.contains(for_in_ser)].as_matrix()[
                #         :, :3]
                layers = self._data.interfaces[self._data.interfaces["formation"] == for_in_ser] \
                    [['X', 'Y', 'Z']].as_matrix()
                rest_layer_points = layers[1:]
                ref_layer_points = np.tile(layers[0], (np.shape(layers)[0] - 1, 1))
            else:
                layers_list = []
                for formation in for_in_ser.split("|"):
                    layers_list.append(
                        self._data.interfaces[self._data.interfaces["formation"] == formation]
                        [['X', 'Y', 'Z']].as_matrix())
                layers = np.asarray(layers_list)
                rest_layer_points = np.vstack((i[1:] for i in layers))
                ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))

            if verbose > 0:
                print("The serie formations are %s" % for_in_ser)
                if verbose > 1:
                    print("The formations are: \n"
                          "Layers \n",
                          self._data.interfaces[self._data.interfaces["formation"].str.contains(for_in_ser)],
                          "\n foliations \n",
                          self._data.foliations[self._data.foliations["formation"].str.contains(for_in_ser)])

            self.Z_x, self.potential_interfaces, self.C, self.DK = self._interpolate(
                dips_position, dip_angles, azimuth, polarity,
                rest_layer_points, ref_layer_points)[:]

            potential_field = self.Z_x.reshape(self._data.nx, self._data.ny, self._data.nz)

            if verbose > 2:
                print("Dual Kriging weights: ", self.DK)
            if verbose > 3:
                print("C_matrix: ", self.C)

            return potential_field

        def _select_serie(self, series_name=0):
            """
            Return the formations of a given serie in string
            :param series_name: name or argument of the serie. Default first of the list
            :return: formations of a given serie in string separeted by |
            """
            if type(series_name) == int or type(series_name) == np.int64:
                _formations_in_serie = "|".join(self._data.series.ix[:, series_name].drop_duplicates())
            elif type(series_name) == str:
                _formations_in_serie = "|".join(self._data.series[series_name].drop_duplicates())
            return _formations_in_serie

        def set_theano_shared_parameteres(self, _data, _grid, **kwargs):
            """
            Basic interpolator parameters. Also here it is possible to change some flags of theano
            :param range_var: Range of the variogram, it is recommended the distance of the longest diagonal
            :param c_o: Sill of the variogram
            """
            # TODO: update Docstrig

            u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect = kwargs.get('nugget_effect', 0.01)
            rescaling_factor = kwargs.get('rescaling_factor', None)

            if not range_var:
                range_var = np.sqrt((_data.xmax - _data.xmin) ** 2 +
                                    (_data.ymax - _data.ymin) ** 2 +
                                    (_data.zmax - _data.zmin) ** 2)
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # Creation of shared variables

            self.tg.a_T.set_value(range_var)
            self.tg.c_o_T.set_value(c_o)
            self.tg.nugget_effect_grad_T.set_value(nugget_effect)

            assert (0 <= u_grade <= 2)

            if u_grade == 0:
                print('I am here')
                self.tg.u_grade_T.set_value(u_grade)
            else:
                self.tg.u_grade_T.set_value(3**u_grade)
            # TODO: To be sure what is the mathematical meaning of this

            if not rescaling_factor:
                max_coord = pn.concat([_data.foliations, _data.interfaces]).max()[['X', 'Y', 'Z']]
                min_coord = pn.concat([_data.foliations, _data.interfaces]).min()[['X', 'Y', 'Z']]
                rescaling_factor = np.max(max_coord - min_coord)

            self.tg.rescaling_factor_T.set_value(rescaling_factor)

            _universal_matrix = np.vstack((_grid.grid.T,
                                           (_grid.grid ** 2).T,
                                           _grid.grid[:, 0] * _grid.grid[:, 1],
                                           _grid.grid[:, 0] * _grid.grid[:, 2],
                                           _grid.grid[:, 1] * _grid.grid[:, 2]))

            self.tg.universal_matrix_T.set_value(_universal_matrix + 1e-10)
            self.tg.final_block.set_value(np.zeros_like(_grid.grid[:, 0]))
            self.tg.grid_val_T.set_value(_grid.grid + 10e-6)

     #    def _get_constant_parameters(self):
     #        """
     #        Deprecated?
     #
     #        Returns:
     #
     #        """
     #        return self.a_T, self.c_o_T, self.nugget_effect_grad_T

        def compile_potential_field_function(self):
            self._interpolate = theano.function(
                [self.tg.dips_position, self.tg.dip_angles, self.tg.azimuth, self.tg.polarity, self.tg.rest_layer_points,
                 self.tg.ref_layer_points, theano.In(self.tg.yet_simulated, value=np.ones_like(self._grid.grid[:, 0]))],
                [self.tg.Z_x, self.tg.potential_field_interfaces,
                 self.tg.C_matrix, self.tg.DK_parameters],
                on_unused_input="warn", profile=True, allow_input_downcast=True)
            return self._interpolate

        def compile_block_model_function(self):

            self._block_export = theano.function([self.tg.dips_position, self.tg.dip_angles, self.tg.azimuth,
                                                  self.tg.polarity, self.tg.rest_layer_points,
                                                  self.tg.ref_layer_points, self.tg.n_formation,
                                                  self.tg.yet_simulated],
                                                 None,
                                                 updates=[(self.tg.final_block, self.tg.potential_field_contribution)],
                                                 on_unused_input="warn", profile=True, allow_input_downcast=True,)
            #   mode=theano.compile.MonitorMode(
            #   post_func=detect_nan))
            #    mode=NanGuardMode(nan_is_error=True, inf_is_error=True,
            #                     big_is_error=True))

            return self._block_export

      #  def update_potential_fields(self, verbose=0):
      #      self.potential_fields = [self.compute_potential_fields(i, verbose=verbose)
      #                               for i in np.arange(len(self._data.series.columns))]

        def compute_block_model(self, series_number="all", verbose=0):
            """
            Method to compute the block model for the given series using data provided in the DataManagement object

            Args:
                series_number(str or int): series to interpolate
                verbose(int): level of verbosity during the computation

            Returns:
                self.block(theano shared[numpy.ndarray]): 3D block with the corresponding formations

            """

            if series_number == "all":
                series_number = np.arange(len(self._data.series.columns))
            for i in series_number:
                formations_in_serie = self._select_serie(i)
                # Number assigned to each formation
                n_formation = np.squeeze(np.where(np.in1d(self._data.formations, self._data.series.ix[:, i]))) + 1
                if verbose > 0:
                    print(n_formation)
                self.grad = self._aux_computations_block_model(formations_in_serie, np.array(n_formation, ndmin=1),
                                                               verbose=verbose)

                return self.tg.final_block

        def compute_potential_fields(self, series_name="all", verbose=0):
            """
            Compute an individual potential field.

            Args:
                series_name (str or int): name or number of series to interpolate
                verbose(int): int level of verbosity during the computation

            Returns:
                numpy.ndarray: 3D array with the potential field

            """

            #assert series_name is not "all", "Compute potential field only returns one potential field at the time"
            if series_name is 'all':
                for i in np.arange(len(self._data.series.columns)):
                    formations_in_serie = self._select_serie(i)
                    self.potential_fields.append(self._aux_computations_potential_field(formations_in_serie,
                                                                                        verbose=verbose))
            else:
                formations_in_serie = self._select_serie(series_name)
                self.potential_fields = self._aux_computations_potential_field(formations_in_serie, verbose=verbose)

            return self.potential_fields



        # def theano_compilation_3D(self):
        #     """
        #     Function that generates the symbolic code to perform the interpolation. Calling this function creates
        #      both the theano functions for the potential field and the block.
        #
        #     Returns:
        #         theano function for the potential field
        #         theano function for the block
        #     """
        #
        #     # Creation of symbolic variables
        #     dips_position = T.matrix("Position of the dips")
        #     dip_angles = T.vector("Angle of every dip")
        #     azimuth = T.vector("Azimuth")
        #     polarity = T.vector("Polarity")
        #     ref_layer_points = T.matrix("Reference points for every layer")
        #     rest_layer_points = T.matrix("Rest of the points of the layers")
        #
        #     # Init values
        #     n_dimensions = 3
        #     grade_universal = self.u_grade_T
        #
        #     # Calculating the dimensions of the
        #     length_of_CG = dips_position.shape[0] * n_dimensions
        #     length_of_CGI = rest_layer_points.shape[0]
        #     length_of_U_I = grade_universal
        #     length_of_C = length_of_CG + length_of_CGI + length_of_U_I
        #
        #     # Extra parameters
        #     i_reescale = 1 / (self.rescaling_factor_T ** 2)
        #     gi_reescale = 1 / self.rescaling_factor_T
        #
        #     # TODO: Check that the distances does not go nuts when I use too large numbers
        #
        #     # Here we create the array with the points to simulate:
        #     #   grid points except those who have been simulated in a younger serie
        #     #   interfaces points to segment the lithologies
        #     yet_simulated = T.vector("boolean function that avoid to simulate twice a point of a different serie")
        #     grid_val = T.vertical_stack((self.grid_val_T * yet_simulated.reshape(
        #                                 (yet_simulated.shape[0], 1))).nonzero_values().reshape((-1, 3)),
        #                                 rest_layer_points)
        #
        #     # ==========================================
        #     # Calculation of Cartesian and Euclidian distances
        #     # ===========================================
        #     # Auxiliary tile for dips and transformation to float 64 of variables in order to calculate
        #     #  precise euclidian
        #     # distances
        #     _aux_dips_pos = T.tile(dips_position, (n_dimensions, 1)).astype("float64")
        #     _aux_rest_layer_points = rest_layer_points.astype("float64")
        #     _aux_ref_layer_points = ref_layer_points.astype("float64")
        #     _aux_grid_val = grid_val.astype("float64")
        #
        #     # Calculation of euclidian distances giving back float32
        #     SED_rest_rest = (T.sqrt(
        #         (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
        #         (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
        #         2 * _aux_rest_layer_points.dot(_aux_rest_layer_points.T))).astype("float32")
        #
        #     SED_ref_rest = (T.sqrt(
        #         (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
        #         (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
        #         2 * _aux_ref_layer_points.dot(_aux_rest_layer_points.T))).astype("float32")
        #
        #     SED_rest_ref = (T.sqrt(
        #         (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
        #         (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
        #         2 * _aux_rest_layer_points.dot(_aux_ref_layer_points.T))).astype("float32")
        #
        #     SED_ref_ref = (T.sqrt(
        #         (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
        #         (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
        #         2 * _aux_ref_layer_points.dot(_aux_ref_layer_points.T))).astype("float32")
        #
        #     SED_dips_dips = (T.sqrt(
        #         (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
        #         (_aux_dips_pos ** 2).sum(1).reshape((1, _aux_dips_pos.shape[0])) -
        #         2 * _aux_dips_pos.dot(_aux_dips_pos.T))).astype("float32")
        #
        #     SED_dips_rest = (T.sqrt(
        #         (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
        #         (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
        #         2 * _aux_dips_pos.dot(_aux_rest_layer_points.T))).astype("float32")
        #
        #     SED_dips_ref = (T.sqrt(
        #         (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
        #         (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
        #         2 * _aux_dips_pos.dot(_aux_ref_layer_points.T))).astype("float32")
        #
        #     # Calculating euclidian distances between the point to simulate and the avalible data
        #     SED_dips_SimPoint = (T.sqrt(
        #         (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
        #         (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
        #         2 * _aux_dips_pos.dot(_aux_grid_val.T))).astype("float32")
        #
        #     SED_rest_SimPoint = (T.sqrt(
        #         (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
        #         (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
        #         2 * _aux_rest_layer_points.dot(_aux_grid_val.T))).astype("float32")
        #
        #     SED_ref_SimPoint = (T.sqrt(
        #         (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
        #         (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
        #         2 * _aux_ref_layer_points.dot(_aux_grid_val.T))).astype("float32")
        #
        #     # Cartesian distances between dips positions
        #     h_u = T.vertical_stack(
        #         T.tile(dips_position[:, 0] - dips_position[:, 0].reshape((dips_position[:, 0].shape[0], 1)),
        #                n_dimensions),
        #         T.tile(dips_position[:, 1] - dips_position[:, 1].reshape((dips_position[:, 1].shape[0], 1)),
        #                n_dimensions),
        #         T.tile(dips_position[:, 2] - dips_position[:, 2].reshape((dips_position[:, 2].shape[0], 1)),
        #                n_dimensions))
        #
        #     h_v = h_u.T
        #
        #     # Cartesian distances between dips and interface points
        #     # Rest
        #     hu_rest = T.vertical_stack(
        #         (dips_position[:, 0] - rest_layer_points[:, 0].reshape((rest_layer_points[:, 0].shape[0], 1))).T,
        #         (dips_position[:, 1] - rest_layer_points[:, 1].reshape((rest_layer_points[:, 1].shape[0], 1))).T,
        #         (dips_position[:, 2] - rest_layer_points[:, 2].reshape((rest_layer_points[:, 2].shape[0], 1))).T
        #     )
        #
        #     # Reference point
        #     hu_ref = T.vertical_stack(
        #         (dips_position[:, 0] - ref_layer_points[:, 0].reshape((ref_layer_points[:, 0].shape[0], 1))).T,
        #         (dips_position[:, 1] - ref_layer_points[:, 1].reshape((ref_layer_points[:, 1].shape[0], 1))).T,
        #         (dips_position[:, 2] - ref_layer_points[:, 2].reshape((ref_layer_points[:, 2].shape[0], 1))).T
        #     )
        #
        #     # Cartesian distances between reference points and rest
        #     hx = T.stack(
        #         (rest_layer_points[:, 0] - ref_layer_points[:, 0]),
        #         (rest_layer_points[:, 1] - ref_layer_points[:, 1]),
        #         (rest_layer_points[:, 2] - ref_layer_points[:, 2])
        #     ).T
        #
        #     # Cartesian distances between the point to simulate and the dips
        #     hu_SimPoint = T.vertical_stack(
        #         (dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
        #         (dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
        #         (dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        #     )
        #
        #     # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        #     # every gradient direction covariance
        #     perpendicularity_matrix = T.zeros_like(SED_dips_dips)
        #
        #     # Cross-covariances of x
        #     perpendicularity_matrix = T.set_subtensor(
        #         perpendicularity_matrix[0:dips_position.shape[0], 0:dips_position.shape[0]], 1)
        #
        #     # Cross-covariances of y
        #     perpendicularity_matrix = T.set_subtensor(
        #         perpendicularity_matrix[dips_position.shape[0]:dips_position.shape[0] * 2,
        #         dips_position.shape[0]:dips_position.shape[0] * 2], 1)
        #
        #     # Cross-covariances of y
        #     perpendicularity_matrix = T.set_subtensor(
        #         perpendicularity_matrix[dips_position.shape[0] * 2:dips_position.shape[0] * 3,
        #         dips_position.shape[0] * 2:dips_position.shape[0] * 3], 1)
        #
        #
        #    # printing = (self.c_o_T * i_reescale * (
        #    #     (SED_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
        #     #printing = (1 - 7 * (SED_rest_rest / self.a_T) ** 2 +
        #     #     35 / 4 * (SED_rest_rest / self.a_T) ** 3 -
        #     #     7 / 2 * (SED_rest_rest / self.a_T) ** 5 +
        #     #     3 / 4 * (SED_rest_rest / self.a_T) ** 7)
        #     printing = SED_rest_rest*self.a_T
        #     # ==========================
        #     # Creating covariance Matrix
        #     # ==========================
        #     # Covariance matrix for interfaces
        #     C_I = (self.c_o_T * i_reescale * (
        #         (SED_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
        #         (1 - 7 * (SED_rest_rest / self.a_T) ** 2 +
        #          35 / 4 * (SED_rest_rest / self.a_T) ** 3 -
        #          7 / 2 * (SED_rest_rest / self.a_T) ** 5 +
        #          3 / 4 * (SED_rest_rest / self.a_T) ** 7) -
        #         ((SED_ref_rest < self.a_T) *  # Reference - Rest
        #          (1 - 7 * (SED_ref_rest / self.a_T) ** 2 +
        #           35 / 4 * (SED_ref_rest / self.a_T) ** 3 -
        #           7 / 2 * (SED_ref_rest / self.a_T) ** 5 +
        #           3 / 4 * (SED_ref_rest / self.a_T) ** 7)) -
        #         ((SED_rest_ref < self.a_T) *  # Rest - Reference
        #          (1 - 7 * (SED_rest_ref / self.a_T) ** 2 +
        #           35 / 4 * (SED_rest_ref / self.a_T) ** 3 -
        #           7 / 2 * (SED_rest_ref / self.a_T) ** 5 +
        #           3 / 4 * (SED_rest_ref / self.a_T) ** 7)) +
        #         ((SED_ref_ref < self.a_T) *  # Reference - References
        #          (1 - 7 * (SED_ref_ref / self.a_T) ** 2 +
        #           35 / 4 * (SED_ref_ref / self.a_T) ** 3 -
        #           7 / 2 * (SED_ref_ref / self.a_T) ** 5 +
        #           3 / 4 * (SED_ref_ref / self.a_T) ** 7)))) #'+ 10e-6
        #
        #     SED_dips_dips = T.switch(T.eq(SED_dips_dips, 0), 1, SED_dips_dips)
        #
        #     # Covariance matrix for gradients at every xyz direction and their cross-covariances
        #     C_G = T.switch(
        #         T.eq(SED_dips_dips, 0),  # This is the condition
        #         0,  # If true it is equal to 0. This is how a direction affect another
        #         (  # else, following Chiles book
        #             (h_u * h_v / SED_dips_dips ** 2) *
        #             ((
        #                  (SED_dips_dips < self.a_T) *  # first derivative
        #                  (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_dips / self.a_T ** 3 -
        #                                  35 / 2 * SED_dips_dips ** 3 / self.a_T ** 5 +
        #                                  21 / 4 * SED_dips_dips ** 5 / self.a_T ** 7))) +
        #              (SED_dips_dips < self.a_T) *  # Second derivative
        #              self.c_o_T * 7 * (9 * SED_dips_dips ** 5 - 20 * self.a_T ** 2 * SED_dips_dips ** 3 +
        #                                15 * self.a_T ** 4 * SED_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
        #             (perpendicularity_matrix *
        #              (SED_dips_dips < self.a_T) *  # first derivative
        #              self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_dips / self.a_T ** 3 -
        #                            35 / 2 * SED_dips_dips ** 3 / self.a_T ** 5 +
        #                            21 / 4 * SED_dips_dips ** 5 / self.a_T ** 7)))
        #     )
        #
        #     # Setting nugget effect of the gradients
        #     # TODO: This function can be substitued by simply adding the nugget effect to the diag
        #     C_G = T.fill_diagonal(C_G, -self.c_o_T * (-14 / self.a_T ** 2) + self.nugget_effect_grad_T)
        #
        #     # Cross-Covariance gradients-interfaces
        #     C_GI = gi_reescale * (
        #         (hu_rest *
        #          (SED_dips_rest < self.a_T) *  # first derivative
        #          (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_rest / self.a_T ** 3 -
        #                           35 / 2 * SED_dips_rest ** 3 / self.a_T ** 5 +
        #                           21 / 4 * SED_dips_rest ** 5 / self.a_T ** 7))) -
        #         (hu_ref *
        #          (SED_dips_ref < self.a_T) *  # first derivative
        #          (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_ref / self.a_T ** 3 -
        #                           35 / 2 * SED_dips_ref ** 3 / self.a_T ** 5 +
        #                           21 / 4 * SED_dips_ref ** 5 / self.a_T ** 7)))
        #     ).T
        #
        #     if self.u_grade_T.get_value() == 3:
        #         # ==========================
        #         # Condition of universality 1 degree
        #
        #         # Gradients
        #         n = dips_position.shape[0]
        #         U_G = T.zeros((n * n_dimensions, n_dimensions))
        #         # x
        #         U_G = T.set_subtensor(
        #             U_G[:n, 0], 1)
        #         # y
        #         U_G = T.set_subtensor(
        #             U_G[n:n * 2, 1], 1
        #         )
        #         # z
        #         U_G = T.set_subtensor(
        #             U_G[n * 2: n * 3, 2], 1
        #         )
        #
        #     #    U_G = T.set_subtensor(U_G[:, -1], [0, 0, 0, 0, 0, 0])
        #
        #         # Interface
        #         U_I = -hx * gi_reescale
        #
        #     #    hxf = (T.lt(rest_layer_points[:, 0], 5) - T.lt(ref_layer_points[:, 0], 5))*2 + 1
        #
        #     #    U_I = T.horizontal_stack(U_I, T.stack(hxf).T)
        #
        #     elif self.u_grade_T.get_value() == 9:
        #         # ==========================
        #         # Condition of universality 2 degree
        #         # Gradients
        #
        #         n = dips_position.shape[0]
        #         U_G = T.zeros((n * n_dimensions, 3 * n_dimensions))
        #         # x
        #         U_G = T.set_subtensor(U_G[:n, 0], 1)
        #         # y
        #         U_G = T.set_subtensor(U_G[n * 1:n * 2, 1], 1)
        #         # z
        #         U_G = T.set_subtensor(U_G[n * 2: n * 3, 2], 1)
        #         # x**2
        #         U_G = T.set_subtensor(U_G[:n, 3], 2 * gi_reescale * dips_position[:, 0])
        #         # y**2
        #         U_G = T.set_subtensor(U_G[n * 1:n * 2, 4], 2 * gi_reescale * dips_position[:, 1])
        #         # z**2
        #         U_G = T.set_subtensor(U_G[n * 2: n * 3, 5], 2 * gi_reescale * dips_position[:, 2])
        #         # xy
        #         U_G = T.set_subtensor(U_G[:n, 6], gi_reescale * dips_position[:, 1])  # This is y
        #         U_G = T.set_subtensor(U_G[n * 1:n * 2, 6], gi_reescale * dips_position[:, 0])  # This is x
        #         # xz
        #         U_G = T.set_subtensor(U_G[:n, 7], gi_reescale * dips_position[:, 2])  # This is z
        #         U_G = T.set_subtensor(U_G[n * 2: n * 3, 7], gi_reescale * dips_position[:, 0])  # This is x
        #         # yz
        #         U_G = T.set_subtensor(U_G[n * 1:n * 2, 8], gi_reescale * dips_position[:, 2])  # This is z
        #         U_G = T.set_subtensor(U_G[n * 2:n * 3, 8], gi_reescale * dips_position[:, 1])  # This is y
        #
        #         # Interface
        #         U_I = - T.stack(
        #             gi_reescale * (rest_layer_points[:, 0] - ref_layer_points[:, 0]), # x
        #             gi_reescale * (rest_layer_points[:, 1] - ref_layer_points[:, 1]), # y
        #             gi_reescale * (rest_layer_points[:, 2] - ref_layer_points[:, 2]), # z
        #             gi_reescale ** 2 * (rest_layer_points[:, 0] ** 2 - ref_layer_points[:, 0] ** 2), # xx
        #             gi_reescale ** 2 * (rest_layer_points[:, 1] ** 2 - ref_layer_points[:, 1] ** 2), # yy
        #             gi_reescale ** 2 * (rest_layer_points[:, 2] ** 2 - ref_layer_points[:, 2] ** 2), # zz
        #             gi_reescale ** 2 * (rest_layer_points[:, 0] * rest_layer_points[:, 1] - ref_layer_points[:, 0] *
        #                                 ref_layer_points[:, 1]),
        #             gi_reescale ** 2 * (rest_layer_points[:, 0] * rest_layer_points[:, 2] - ref_layer_points[:, 0] *
        #                                 ref_layer_points[:, 2]),
        #             gi_reescale ** 2 * (rest_layer_points[:, 1] * rest_layer_points[:, 2] - ref_layer_points[:, 1] *
        #                                 ref_layer_points[:, 2]),
        #         ).T
        #
        #
        #
        #     # =================================
        #     # Creation of the Covariance Matrix
        #     # =================================
        #     C_matrix = T.zeros((length_of_C, length_of_C ))
        #
        #     # First row of matrices
        #     C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, 0:length_of_CG], C_G)
        #
        #     C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)
        #
        #     if not self.u_grade_T.get_value() == 0:
        #         C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, -length_of_U_I:], U_G)
        #
        #     # Second row of matrices
        #     C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        #     C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
        #                                length_of_CG:length_of_CG + length_of_CGI], C_I)
        #
        #     if not self.u_grade_T.get_value() == 0:
        #         C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, -length_of_U_I:], U_I)
        #
        #         # Third row of matrices
        #         C_matrix = T.set_subtensor(C_matrix[-length_of_U_I:, 0:length_of_CG], U_G.T)
        #         C_matrix = T.set_subtensor(C_matrix[-length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], U_I.T)
        #
        #     # =====================
        #     # Creation of the gradients G vector
        #     # Calculation of the cartesian components of the dips assuming the unit module
        #     G_x = T.sin(T.deg2rad(dip_angles)) * T.sin(T.deg2rad(azimuth)) * polarity
        #     G_y = T.sin(T.deg2rad(dip_angles)) * T.cos(T.deg2rad(azimuth)) * polarity
        #     G_z = T.cos(T.deg2rad(dip_angles)) * polarity
        #
        #     G = T.concatenate((G_x, G_y, G_z))
        #
        #     # Creation of the Dual Kriging vector
        #     b = T.zeros_like(C_matrix[:, 0])
        #     b = T.set_subtensor(b[0:G.shape[0]], G)
        #
        #     # Solving the kriging system
        #     # TODO: look for an eficient way to substitute nlianlg by a theano operation
        #     DK_parameters = T.dot(T.nlinalg.matrix_inverse(C_matrix), b)
        #
        #     # ==============
        #     # Interpolator
        #     # ==============
        #
        #     # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        #     # ravel form)
        #     weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T
        #
        #     # Gradient contribution
        #     sigma_0_grad = T.sum(
        #         (weights[:length_of_CG, :] *
        #          gi_reescale *
        #          (-hu_SimPoint *
        #           (SED_dips_SimPoint < self.a_T) *  # first derivative
        #           (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_SimPoint / self.a_T ** 3 -
        #                            35 / 2 * SED_dips_SimPoint ** 3 / self.a_T ** 5 +
        #                            21 / 4 * SED_dips_SimPoint ** 5 / self.a_T ** 7)))),
        #         axis=0)
        #
        #     # Interface contribution
        #     sigma_0_interf = (T.sum(
        #         -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
        #         (self.c_o_T * i_reescale * (
        #             (SED_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
        #             (1 - 7 * (SED_rest_SimPoint / self.a_T) ** 2 +
        #              35 / 4 * (SED_rest_SimPoint / self.a_T) ** 3 -
        #              7 / 2 * (SED_rest_SimPoint / self.a_T) ** 5 +
        #              3 / 4 * (SED_rest_SimPoint / self.a_T) ** 7) -
        #             ((SED_ref_SimPoint < self.a_T) *  # SimPoint- Ref
        #              (1 - 7 * (SED_ref_SimPoint / self.a_T) ** 2 +
        #               35 / 4 * (SED_ref_SimPoint / self.a_T) ** 3 -
        #               7 / 2 * (SED_ref_SimPoint / self.a_T) ** 5 +
        #               3 / 4 * (SED_ref_SimPoint / self.a_T) ** 7)))), axis=0))
        #
        #     # Universal drift contribution
        #     # Universal terms used to calculate f0
        #     _universal_terms_layers = T.horizontal_stack(
        #         rest_layer_points,
        #         (rest_layer_points ** 2),
        #         T.stack((rest_layer_points[:, 0] * rest_layer_points[:, 1],
        #                  rest_layer_points[:, 0] * rest_layer_points[:, 2],
        #                  rest_layer_points[:, 1] * rest_layer_points[:, 2]), axis=1)).T
        #
        #     universal_matrix = T.horizontal_stack(
        #         (self.universal_matrix_T * yet_simulated).nonzero_values().reshape((9, -1)),
        #         _universal_terms_layers)
        #
        #     if self.u_grade_T.get_value() == 0:
        #         f_0 = 0
        #     else:
        #         gi_rescale_aux = T.repeat(gi_reescale, 9)
        #         gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
        #         _aux_magic_term = T.tile(gi_rescale_aux[:grade_universal], (grid_val.shape[0], 1)).T
        #         f_0 = (T.sum(
        #             weights[-length_of_U_I:, :] * gi_reescale * _aux_magic_term *
        #             universal_matrix[:grade_universal]
        #             , axis=0))
        #
        #     # Contribution faults
        #    # f_1 = weights[-1, :] * T.lt(universal_matrix[0, :], 5) * 2 - 1
        #  #   f_1 = 0
        #     # Potential field
        #     # Value of the potential field
        #
        #     Z_x = (sigma_0_grad + sigma_0_interf + f_0)[:-rest_layer_points.shape[0]]
        #     potential_field_interfaces = (sigma_0_grad + sigma_0_interf + f_0)[-rest_layer_points.shape[0]:]
        #
        #     # Theano function to calculate a potential field
        #     self._interpolate = theano.function(
        #         [dips_position, dip_angles, azimuth, polarity, rest_layer_points, ref_layer_points,
        #          theano.In(yet_simulated, value=np.ones_like(self._grid.grid[:, 0]))],
        #         [Z_x, G_x, G_y, G_z, potential_field_interfaces, C_matrix, printing],
        #         on_unused_input="warn", profile=True, allow_input_downcast=True)
        #
        #     # =======================================================================
        #     #               CODE TO EXPORT THE BLOCK DIRECTLY
        #     # ========================================================================
        #
        #     # Aux shared parameters
        #    # infinite_pos = theano.shared(np.float32(np.inf))
        #    # infinite_neg = theano.shared(np.float32(-np.inf))
        #
        #     # Value of the lithology-segment
        #     n_formation = T.vector("The assigned number of the lithologies in this serie")
        #
        #     # Loop to obtain the average Zx for every intertace
        #     def average_potential(dim_a, dim_b, pfi):
        #         """
        #
        #         :param dim: size of the rest values vector per formation
        #         :param pfi: the values of all the rest values potentials
        #         :return: average of the potential per formation
        #         """
        #         average = pfi[T.cast(dim_a, "int32"): T.cast(dim_b, "int32")].sum() / (dim_b - dim_a)
        #         return average
        #
        #     potential_field_unique, updates1 = theano.scan(fn=average_potential,
        #                                                    outputs_info=None,
        #                                                    sequences=dict(
        #                                                        input=T.concatenate(
        #                                                            (T.stack(0),
        #                                                             self.number_of_points_per_formation_T)),
        #                                                        taps=[0, 1]),
        #                                                    non_sequences=potential_field_interfaces)
        #
        #     infinite_pos = T.max(potential_field_unique) + 10
        #     infinite_neg = T.min(potential_field_unique) - 10
        #
        #     # Loop to segment the distinct lithologies
        #     potential_field_iter = T.concatenate((T.stack(infinite_pos),
        #                                           potential_field_unique,
        #                                           T.stack(infinite_neg)))
        #
        #     def compare(a, b, n_formation, Zx):
        #         return T.le(Zx, a) * T.ge(Zx, b) * n_formation
        #
        #     block, updates2 = theano.scan(fn=compare,
        #                                   outputs_info=None,
        #                                   sequences=[dict(input=potential_field_iter, taps=[0, 1]),
        #                                              n_formation],
        #                                   non_sequences=Z_x)
        #
        #     # Adding to the block the contribution of the potential field
        #     potential_field_contribution = T.set_subtensor(
        #         self.block[T.nonzero(T.cast(yet_simulated, "int8"))[0]],
        #         block.sum(axis=0))
        #
        #     # Some gradient testing
        #     # grad = T.jacobian(T.flatten(printing), rest_layer_points)
        #     # grad = T.grad(T.sum(Z_x), self.a_T)
        #     from theano.compile.nanguardmode import NanGuardMode
        #
        #     def detect_nan(i, node, fn):
        #         for output in fn.outputs:
        #             if (not isinstance(output[0], np.random.RandomState) and
        #                     np.isnan(output[0]).any()):
        #                 print('*** NaN detected ***')
        #                 theano.printing.debugprint(node)
        #                 print('Inputs : %s' % [input[0] for input in fn.inputs])
        #                 print('Outputs: %s' % [output[0] for output in fn.outputs])
        #                 break
        #
        #     # Theano function to update the block
        #     self._block_export = theano.function([dips_position, dip_angles, azimuth, polarity, rest_layer_points,
        #                                           ref_layer_points, n_formation, yet_simulated], None,
        #                                          updates=[(self.block, potential_field_contribution)],
        #                                          on_unused_input="warn", profile=True, allow_input_downcast=True,)
        #                                      #   mode=theano.compile.MonitorMode(
        #                                      #       post_func=detect_nan))
        #                                       #    mode=NanGuardMode(nan_is_error=True, inf_is_error=True,
        #                                       #                     big_is_error=True))

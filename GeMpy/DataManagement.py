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
    def __init__(self,
                 extent,
                 resolution=[50, 50, 50],
                 path_i=None, path_f=None,
                 **kwargs):

        self.extent = np.array(extent)
        self.resolution = np.array(resolution)

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
        self.set_formation_number()
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
            # for el in self.formations:
            #     for check in self.formations:
            #         assert (el not in check or el == check), "One of the formations name contains other" \
            #                                                  " string. Please rename." + str(el) + " in " + str(
            #             check)

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

       # self.interfaces.reset_index(drop=False, inplace=True)
        self._set_formations()
        self.set_series()
        self.set_formation_number()

    def set_foliations(self, foliat_Dataframe, append=False):

        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(foliat_Dataframe.columns)
        if append:
            self.foliations = self.foliations.append(foliat_Dataframe)
        else:
            self.foliations = foliat_Dataframe

      #  self.foliations.reset_index(inplace=True, drop=True)
        self._set_formations()
        self.set_series()
        self.set_formation_number()
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
        self.interfaces["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.interfaces["formation"]]
        self.interfaces.sort_values(by='order_series', inplace=True)

        self.foliations["series"] = [(i == _series).sum().argmax() for i in self.foliations["formation"]]
        self.foliations["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.foliations["formation"]]
        self.foliations.sort_values(by='order_series', inplace=True)

        self.series = _series
        return _series

    def set_formation_number(self):
        try:
            ip_addresses = self.interfaces["formation"].unique()
            ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses)+1)))
            self.interfaces['formation number'] = self.interfaces['formation'].replace(ip_dict)
            self.foliations['formation number'] = self.foliations['formation'].replace(ip_dict)
        except ValueError:
            pass

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

          #  self.grid = np.vstack(map(np.ravel, g)).T.astype("float32")
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

        def __init__(self, _data_scaled, _grid_scaled=None, compute_block_model=False,
                     compute_potential_field=False, dtype = 'float32', *args, **kwargs):

            verbose = kwargs.get('verbose', 0)
            rescaling_factor = kwargs.get('rescaling_factor', None)

            # theano.config.optimizer = 'None'
            # theano.config.exception_verbosity = 'high'
            # theano.config.compute_test_value = 'ignore'

            self.dtype = dtype

            u_grade = kwargs.get('u_grade', 2)

            self._data_scaled = _data_scaled

            # In case someone wants to provide a grid
            if not _grid_scaled:
                self._grid_scaled = _data_scaled.grid
            else:
                self._grid_scaled = _grid_scaled

            # Importing the theano graph
            self.tg = theanograf.TheanoGraph_pro(u_grade)

            # Setting theano parameters
            self.set_theano_shared_parameteres(self._data_scaled, self._grid_scaled, **kwargs)
            self.data_prep()


            # Choosing if compute something directly
            if compute_potential_field:

                self.potential_fields = []
                self._interpolate = self.compile_potential_field_function()
                self.potential_fields = [self.compute_potential_fields(i, verbose=verbose)
                                         for i in np.arange(len(self._data_scaled.series.columns))]

            if compute_block_model:

                self._block_export = self.compile_block_model_function()
                self.block = self.compute_block_model()

        def data_prep(self):

            # We order the pandas table
            self._data_scaled.interfaces.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Drop works with the pandas indices so I DO need this thing working
            self._data_scaled.interfaces.reset_index(drop=True, inplace=True)

            # Size of every formation, SHARED
            len_interfaces = np.asarray(
                [np.sum(self._data_scaled.interfaces['formation number'] == i)
                 for i in self._data_scaled.interfaces['formation number'].unique()])

            # Position of the first term of every layer PYTHON
            ref_position = np.insert(len_interfaces[:-1], 0, 0).cumsum()

            # Drop using pandas indeces
            pandas_rest_layer_points = self._data_scaled.interfaces.drop(ref_position)

            # Size of every layer in rests
            len_rest_form = (len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form)

            # TODO: do I need this? PYTHON
            len_foliations = np.asarray(
                [np.sum(self._data_scaled.foliations['formation number'] == i)
                 for i in self._data_scaled.foliations['formation number'].unique()])

            self.pandas_rest = pandas_rest_layer_points
            # Size of every series SHARED
            len_series_i = np.asarray(
                [np.sum(pandas_rest_layer_points['order_series'] == i)
                 for i in pandas_rest_layer_points['order_series'].unique()])

            # We add the 0 at the beginning and set the shared value
            self.tg.len_series_i.set_value(np.insert(len_series_i, 0, 0).cumsum())

            len_series_f = np.asarray(
                [np.sum(self._data_scaled.foliations['order_series'] == i)
                 for i in self._data_scaled.foliations['order_series'].unique()])

            self.tg.len_series_f.set_value(np.insert(len_series_f, 0, 0).cumsum())

            # Rest layers matrix # VAR
            rest_layer_points = pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            # TODO delete
            self.rest_layer_points = rest_layer_points

            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            aux_1 = self._data_scaled.interfaces.iloc[ref_position][['X', 'Y', 'Z']].as_matrix()
            ref_layer_points = np.zeros((0, 3))

            for e, i in enumerate(len_interfaces):
                ref_layer_points = np.vstack((ref_layer_points,
                                              np.tile(aux_1[e], (i - 1, 1))))

            self.ref_layer_points = ref_layer_points

            # Check no reference points in rest points (at least in coor x)
            assert not any(aux_1[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

            # Foliations, VAR
            dips_position = self._data_scaled.foliations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self._data_scaled.foliations["dip"].as_matrix()
            azimuth = self._data_scaled.foliations["azimuth"].as_matrix()
            polarity = self._data_scaled.foliations["polarity"].as_matrix()

            idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                   ref_layer_points, rest_layer_points)]

            return idl

        def set_theano_shared_parameteres(self, _data_rescaled, _grid_rescaled, **kwargs):

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
                range_var = np.sqrt((_data_rescaled.extent[0] - _data_rescaled.extent[1]) ** 2 +
                                    (_data_rescaled.extent[2] - _data_rescaled.extent[3]) ** 2 +
                                    (_data_rescaled.extent[4] - _data_rescaled.extent[5]) ** 2)

            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            from IPython.core.debugger import Tracer

            # Creation of shared variables
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect))

            assert (0 <= u_grade <= 2)

            if u_grade == 0:
                self.tg.u_grade_T.set_value(u_grade)
            else:
                self.tg.u_grade_T.set_value(3**u_grade)
            # TODO: To be sure what is the mathematical meaning of this
            # TODO Deprecated
            self.tg.c_resc.set_value(1)

            _universal_matrix = np.vstack((_grid_rescaled.grid.T,
                                           (_grid_rescaled.grid ** 2).T,
                                           _grid_rescaled.grid[:, 0] * _grid_rescaled.grid[:, 1],
                                           _grid_rescaled.grid[:, 0] * _grid_rescaled.grid[:, 2],
                                           _grid_rescaled.grid[:, 1] * _grid_rescaled.grid[:, 2]))

            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](_universal_matrix + 1e-10))
            #self.tg.final_block.set_value(np.zeros_like(_grid_rescaled.grid[:, 0]))
            self.tg.final_block.set_value(np.zeros((_grid_rescaled.grid.shape[0])))
          #  self.tg.final_block.set_value(np.random.randint(0, 2, _grid_rescaled.grid.shape[0]))
            self.tg.grid_val_T.set_value(np.cast[self.dtype](_grid_rescaled.grid + 10e-6))
            self.tg.n_formation.set_value(np.insert(_data_rescaled.interfaces['formation number'].unique(),
                                                    0, 0)[::-1])

            self.tg.n_formation.set_value(_data_rescaled.interfaces['formation number'].unique())

            self.tg.n_formations_per_serie.set_value(
                np.insert(_data_rescaled.interfaces.groupby('order_series').formation.nunique().values.cumsum(),
                          0, 0))


















    #     """
    #     def _aux_computations_block_model(self, for_in_ser, n_formation, verbose=0):
    #         """
    #         Private function with the bridge steps from the selection of serie to the input in theano
    #
    #         Args:
    #             for_in_ser: array with the formation for the series to interpolate
    #             n_formation: number of formation in the series
    #             verbose: verbosity
    #
    #         Returns:
    #             self.block(theano shared[numpy.ndarray]): 3D block with the corresponding formations
    #         """
    #         # TODO Probably here I should add some asserts for sanity check
    #         try:
    #             yet_simulated = (self.tg.final_block.get_value() == 0) * 1
    #             if verbose > 0:
    #                 print(yet_simulated, (yet_simulated == 0).sum())
    #         except AttributeError:
    #             yet_simulated = np.ones_like(self._grid.grid[:, 0], dtype="int8")
    #            # print("I am in the except")
    #
    #         dips_position_tiled = self._data_scaled.foliations[
    #             self._data_scaled.foliations["formation"].str.contains(for_in_ser)] \
    #             [['X', 'Y', 'Z']].as_matrix()
    #         dip_angles = self._data_scaled.foliations[
    #             self._data_scaled.foliations["formation"].str.contains(for_in_ser)]["dip"].as_matrix()
    #         azimuth = self._data_scaled.foliations[
    #             self._data_scaled.foliations["formation"].str.contains(for_in_ser)]["azimuth"].as_matrix()
    #         polarity = self._data_scaled.foliations[
    #             self._data_scaled.foliations["formation"].str.contains(for_in_ser)]["polarity"].as_matrix()
    #
    #         if for_in_ser.count("|") == 0:
    #
    #             layers = self._data_scaled.interfaces[self._data_scaled.interfaces["formation"] == for_in_ser] \
    #                 [['X', 'Y', 'Z']].as_matrix()
    #             rest_layer_points = layers[1:]
    #             # TODO self.n_formation probably should not be self
    #             self.tg.number_of_points_per_formation_T.set_value(np.array(rest_layer_points.shape[0], ndmin=1))
    #             ref_layer_points = np.tile(layers[0], (np.shape(layers)[0] - 1, 1))
    #         else:
    #             # TODO: This is ugly
    #             layers_list = []
    #             for formation in for_in_ser.split("|"):
    #                 layers_list.append(
    #                     self._data_scaled.interfaces[self._data_scaled.interfaces["formation"] == formation]
    #                     [['X', 'Y', 'Z']].as_matrix())
    #             layers = np.asarray(layers_list)
    #
    #             rest_layer_points = layers[0][1:]
    #             rest_dim = np.array(layers[0][1:].shape[0], ndmin=1)
    #             for i in layers[1:]:
    #                 rest_layer_points = np.vstack((rest_layer_points, i[1:]))
    #                 rest_dim = np.append(rest_dim, rest_dim[-1] + i[1:].shape[0])
    #             self.tg.number_of_points_per_formation_T.set_value(rest_dim)
    #             ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))
    #
    #         if verbose > 0:
    #             print("The serie formations are %s" % for_in_ser)
    #             if verbose > 1:
    #                 print("The formations are: \n"
    #                       "Layers ", self._data_scaled.interfaces[self._data_scaled.interfaces["formation"].str.contains(for_in_ser)],
    #                       " \n "
    #                       "foliations ",
    #                       self._data_scaled.foliations[self._data_scaled.foliations["formation"].str.contains(for_in_ser)])
    #
    #         # self.grad is none so far. I have it for further research in the calculation of the Jacobian matrix
    #
    #             if verbose > 2:
    #                 print('number_formations', n_formation)
    #                 print('rest_layer_points', rest_layer_points)
    #
    #         if not getattr(self, '_block_export', None):
    #             self.compile_block_model_function()
    #
    #         res = self._block_export(dips_position_tiled, dip_angles, azimuth, polarity,
    #                                  rest_layer_points, ref_layer_points,
    #                                  n_formation, yet_simulated)
    #
    #         if verbose > 2:
    #             print('number of unique lithologies in the final block model',
    #                   np.unique(self.tg.final_block.get_value()))
    #
    #         self.input_parameters = [dips_position_tiled, dip_angles, azimuth, polarity,
    #                                  rest_layer_points, ref_layer_points,
    #                                  n_formation, yet_simulated]
    #         return res
    #
    #
    #
    #     def _aux_computations_potential_field(self, for_in_ser, verbose=0):
    #         """
    #         Private function with the bridge steps from the selection of serie to the input in theano
    #
    #         Args:
    #             for_in_ser: array with the formation for the series to interpolate
    #             verbose: verbosity
    #
    #         Returns:
    #             numpy.ndarray: 3D array with the potential field
    #         """
    #
    #         # TODO: change [:,:3] that is positional based for XYZ so is more consistent
    #         dips_position_tiled = self._data_scaled.foliations[
    #             self._data_scaled.foliations["formation"].str.contains(for_in_ser)] \
    #             [['X', 'Y', 'Z']].as_matrix()
    #         dip_angles = self._data_scaled.foliations[self._data_scaled.foliations["formation"].str.contains(for_in_ser)][
    #             "dip"].as_matrix()
    #         azimuth = self._data_scaled.foliations[self._data_scaled.foliations["formation"].str.contains(for_in_ser)][
    #             "azimuth"].as_matrix()
    #         polarity = self._data_scaled.foliations[self._data_scaled.foliations["formation"].str.contains(for_in_ser)][
    #             "polarity"].as_matrix()
    #
    #         if for_in_ser.count("|") == 0:
    #             # layers = self._data_scaled.interfaces[self._data_scaled.interfaces["formation"].str.contains(for_in_ser)].as_matrix()[
    #             #         :, :3]
    #             layers = self._data_scaled.interfaces[self._data_scaled.interfaces["formation"] == for_in_ser] \
    #                 [['X', 'Y', 'Z']].as_matrix()
    #             rest_layer_points = layers[1:]
    #             ref_layer_points = np.tile(layers[0], (np.shape(layers)[0] - 1, 1))
    #         else:
    #             layers_list = []
    #             for formation in for_in_ser.split("|"):
    #                 layers_list.append(
    #                     self._data_scaled.interfaces[self._data_scaled.interfaces["formation"] == formation]
    #                     [['X', 'Y', 'Z']].as_matrix())
    #             layers = np.asarray(layers_list)
    #             rest_layer_points = np.vstack((i[1:] for i in layers))
    #             ref_layer_points = np.vstack((np.tile(i[0], (np.shape(i)[0] - 1, 1)) for i in layers))
    #
    #         if verbose > 0:
    #             print("The serie formations are %s" % for_in_ser)
    #             if verbose > 1:
    #                 print("The formations are: \n"
    #                       "Layers \n",
    #                       self._data_scaled.interfaces[self._data_scaled.interfaces["formation"].str.contains(for_in_ser)],
    #                       "\n foliations \n",
    #                       self._data_scaled.foliations[self._data_scaled.foliations["formation"].str.contains(for_in_ser)])
    #
    #         self.tg.C_matrix.eval({self.tg.dips_position_tiled: dips_position_tiled,
    #                                #    'self.dip_angles': dip_angles,
    #                                #   'self.azimuth': azimuth,
    #                                #  'self.polarity': polarity,
    #                                self.tg.rest_layer_points: rest_layer_points,
    #                                self.tg.ref_layer_points: ref_layer_points})
    #
    #
    #
    #         potential_field_results = self._interpolate(
    #             dips_position_tiled, dip_angles, azimuth, polarity,
    #             rest_layer_points, ref_layer_points)[:]
    #
    #         self.Z_x, self.results = potential_field_results[0], potential_field_results[1:]
    #
    #         potential_field = self.Z_x.reshape(self._data_scaled.resolution[0],
    #                                            self._data_scaled.resolution[1],
    #                                            self._data_scaled.resolution[2])
    #
    #         if verbose > 2:
    #             print("Dual Kriging weights: ", self.results[2])
    #         if verbose > 3:
    #             print("C_matrix: ", self.results[1])
    #
    #         return potential_field
    #
    #     def _select_serie(self, series_name=0):
    #         """
    #         Return the formations of a given serie in string
    #         :param series_name: name or argument of the serie. Default first of the list
    #         :return: formations of a given serie in string separeted by |
    #         """
    #         if type(series_name) == int or type(series_name) == np.int64:
    #             _formations_in_serie = "|".join(self._data_scaled.series.ix[:, series_name].drop_duplicates())
    #         elif type(series_name) == str:
    #             _formations_in_serie = "|".join(self._data_scaled.series[series_name].drop_duplicates())
    #         return _formations_in_serie
    #
    #     def set_theano_shared_parameteres(self, _data_rescaled, _grid_rescaled, **kwargs):
    #         """
    #         Basic interpolator parameters. Also here it is possible to change some flags of theano
    #         :param range_var: Range of the variogram, it is recommended the distance of the longest diagonal
    #         :param c_o: Sill of the variogram
    #         """
    #         # TODO: update Docstrig
    #
    #         u_grade = kwargs.get('u_grade', 2)
    #         range_var = kwargs.get('range_var', None)
    #         c_o = kwargs.get('c_o', None)
    #         nugget_effect = kwargs.get('nugget_effect', 0.01)
    #         rescaling_factor = kwargs.get('rescaling_factor', None)
    #
    #       #  print("I am in the set theano shared", _data_rescaled, _data_rescaled.interfaces.head())
    #
    #         if not range_var:
    #             range_var = np.sqrt((_data_rescaled.extent[0] - _data_rescaled.extent[1]) ** 2 +
    #                                 (_data_rescaled.extent[2] - _data_rescaled.extent[3]) ** 2 +
    #                                 (_data_rescaled.extent[4] - _data_rescaled.extent[5]) ** 2)
    #         if not c_o:
    #             c_o = range_var ** 2 / 14 / 3
    #
    #         from IPython.core.debugger import Tracer
    #
    #         # Creation of shared variables
    #      #   print('range_var, c_o', range_var, c_o)
    #         self.tg.a_T.set_value(range_var)
    #         self.tg.c_o_T.set_value(c_o)
    #         self.tg.nugget_effect_grad_T.set_value(nugget_effect)
    #
    #         assert (0 <= u_grade <= 2)
    #
    #         if u_grade == 0:
    #             self.tg.u_grade_T.set_value(u_grade)
    #         else:
    #             self.tg.u_grade_T.set_value(3**u_grade)
    #         # TODO: To be sure what is the mathematical meaning of this
    #
    #         self.tg.c_resc.set_value(1)
    #
    #         _universal_matrix = np.vstack((_grid_rescaled.grid.T,
    #                                        (_grid_rescaled.grid ** 2).T,
    #                                        _grid_rescaled.grid[:, 0] * _grid_rescaled.grid[:, 1],
    #                                        _grid_rescaled.grid[:, 0] * _grid_rescaled.grid[:, 2],
    #                                        _grid_rescaled.grid[:, 1] * _grid_rescaled.grid[:, 2]))
    #
    #         self.tg.universal_grid_matrix_T.set_value(_universal_matrix + 1e-10)
    #         self.tg.final_block.set_value(np.zeros_like(_grid_rescaled.grid[:, 0]))
    #         self.tg.grid_val_T.set_value(_grid_rescaled.grid + 10e-6)
    #
    #  #    def _get_constant_parameters(self):
    #  #        """
    #  #        Deprecated?
    #  #
    #  #        Returns:
    #  #
    #  #        """
    #  #        return self.a_T, self.c_o_T, self.nugget_effect_grad_T
    #
    #     def compile_potential_field_function(self):
    #         self._interpolate = theano.function(
    #             [self.tg.dips_position_tiled, self.tg.dip_angles, self.tg.azimuth, self.tg.polarity,
    #              self.tg.rest_layer_points, self.tg.ref_layer_points,
    #              theano.In(self.tg.yet_simulated, value=np.ones_like(self._grid_scaled.grid[:, 0]))],
    #             [self.tg.Z_x, self.tg.potential_field_interfaces,
    #              self.tg.C_matrix, self.tg.DK_parameters, self.tg.printing],
    #             on_unused_input="warn", profile=True, allow_input_downcast=True)
    #         return self._interpolate
    #
    #     def compile_block_model_function(self):
    #
    #         self._block_export = theano.function([self.tg.dips_position_tiled, self.tg.dip_angles, self.tg.azimuth,
    #                                               self.tg.polarity, self.tg.rest_layer_points,
    #                                               self.tg.ref_layer_points, self.tg.n_formation,
    #                                               self.tg.yet_simulated],
    #                                              None,
    #                                              updates=[(self.tg.final_block, self.tg.potential_field_contribution)],
    #                                              on_unused_input="warn", profile=True, allow_input_downcast=True,)
    #         #   mode=theano.compile.MonitorMode(
    #         #   post_func=detect_nan))
    #         #    mode=NanGuardMode(nan_is_error=True, inf_is_error=True,
    #         #                     big_is_error=True))
    #
    #         return self._block_export
    #
    #   #  def update_potential_fields(self, verbose=0):
    #   #      self.potential_fields = [self.compute_potential_fields(i, verbose=verbose)
    #   #                               for i in np.arange(len(self._data_scaled.series.columns))]
    #
    #     def compute_block_model(self, series_number="all", verbose=0):
    #         """
    #         Method to compute the block model for the given series using data provided in the DataManagement object
    #
    #         Args:
    #             series_number(str or int): series to interpolate
    #             verbose(int): level of verbosity during the computation
    #
    #         Returns:
    #             self.block(theano shared[numpy.ndarray]): 3D block with the corresponding formations
    #
    #         """
    #
    #         if series_number == "all":
    #             series_number = np.arange(len(self._data_scaled.series.columns))
    #         for i in series_number:
    #             formations_in_serie = self._select_serie(i)
    #             # Number assigned to each formation
    #             n_formation = np.squeeze(np.where(np.in1d(self._data_scaled.formations, self._data_scaled.series.ix[:, i]))) + 1
    #             if verbose > 0:
    #                 print(n_formation)
    #             self.grad = self._aux_computations_block_model(formations_in_serie, np.array(n_formation, ndmin=1),
    #                                                            verbose=verbose)
    #       #  self.block = self.tg.final_block
    #         return self.tg.final_block
    #
    #     def compute_potential_fields(self, series_name="all", verbose=0):
    #         """
    #         Compute an individual potential field.
    #
    #         Args:
    #             series_name (str or int): name or number of series to interpolate
    #             verbose(int): int level of verbosity during the computation
    #
    #         Returns:
    #             numpy.ndarray: 3D array with the potential field
    #
    #         """
    #
    #         #assert series_name is not "all", "Compute potential field only returns one potential field at the time"
    #         if series_name is 'all':
    #             self.potential_fields = []
    #             for i in np.arange(len(self._data_scaled.series.columns)):
    #                 formations_in_serie = self._select_serie(i)
    #                 self.potential_fields.append(self._aux_computations_potential_field(formations_in_serie,
    #                                                                                     verbose=verbose))
    #         else:
    #             formations_in_serie = self._select_serie(series_name)
    #             self.potential_fields = self._aux_computations_potential_field(formations_in_serie, verbose=verbose)
    #
    #         return self.potential_fields
    #

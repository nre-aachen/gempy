from __future__ import division

import os
import sys

import numpy as np
import pandas as pn
import theanograf


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
            self.foliations = pn.DataFrame(columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity',
                                                    'formation', 'series'])
        if path_i:
            self.interfaces = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(self.interfaces.columns), \
                "One or more columns do not match with the expected values " + str(self.interfaces.columns)
        else:
            self.interfaces = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series'])

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

    def i_open_set_data(self, itype="foliations"):

        if self.foliations.empty:
            self.foliations = pn.DataFrame(
                np.array([0., 0., 0., 0., 0., 1., 'Default Formation', 'Default series']).reshape(1, 8),
                columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series']).\
                convert_objects(convert_numeric=True)

        if self.interfaces.empty:
            self.interfaces = pn.DataFrame(
                np.array([0, 0, 0, 'Default Formation', 'Default series']).reshape(1, 5),
                columns=['X', 'Y', 'Z', 'formation', 'series']).convert_objects(convert_numeric=True)

        import qgrid
        from ipywidgets import widgets
        from IPython.display import display
        qgrid.nbinstall(overwrite=True)
        qgrid.set_defaults(show_toolbar=True)
        assert itype is 'foliations' or itype is 'interfaces', 'itype must be either foliations or interfaces'

        self.pandas_frame = qgrid.show_grid(self.get_raw_data(dtype=itype))

    def i_close_set_data(self):
        self.pandas_frame.close()
        #self._set_formations()
        #self.series = self.set_series()
        self.set_formation_number()
        self.calculate_gradient()



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
        # assert np.count_nonzero(np.unique(_series.values)) is len(self.formations), \
        #     "series_distribution must have the same number of values as number of formations %s." \
        #     % self.formations

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

        def __init__(self, _data_scaled, _grid_scaled=None, *args, **kwargs):

            verbose = kwargs.get('verbose', [0])
            rescaling_factor = kwargs.get('rescaling_factor', None)
            dtype = kwargs.get('dtype', 'float32')
            self.dtype = dtype

            u_grade = kwargs.get('u_grade', 2)

            self._data_scaled = _data_scaled

            # In case someone wants to provide a grid
            if not _grid_scaled:
                self._grid_scaled = _data_scaled.grid
            else:
                self._grid_scaled = _grid_scaled

            # Importing the theano graph
            self.tg = theanograf.TheanoGraph_pro(u_grade, dtype=dtype, verbose=verbose,)

            # Setting theano parameters
            self.set_theano_shared_parameteres(self._data_scaled, self._grid_scaled, **kwargs)
            self.data_prep()

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

            # Creation of shared variables
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect))

            assert (0 <= u_grade <= 2)

            if u_grade == 0:
                self.tg.u_grade_T.set_value(u_grade)
            else:
                self.tg.u_grade_T.set_value(u_grade)
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
            self.tg.final_block.set_value(np.zeros((_grid_rescaled.grid.shape[0]), dtype='int'))
            self.tg.yet_simulated.set_value(np.ones((_grid_rescaled.grid.shape[0]), dtype='int'))
          #  self.tg.final_block.set_value(np.random.randint(0, 2, _grid_rescaled.grid.shape[0]))
            self.tg.grid_val_T.set_value(np.cast[self.dtype](_grid_rescaled.grid + 10e-6))
            self.tg.n_formation.set_value(np.insert(_data_rescaled.interfaces['formation number'].unique(),
                                                    0, 0)[::-1])

            self.tg.n_formation.set_value(_data_rescaled.interfaces['formation number'].unique())

            self.tg.n_formations_per_serie.set_value(
                np.insert(_data_rescaled.interfaces.groupby('order_series').formation.nunique().values.cumsum(),
                          0, 0))

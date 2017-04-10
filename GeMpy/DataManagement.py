from __future__ import division

import os
import sys

import numpy as np
import pandas as pn
import theanograf


class DataManagement(object):
    """
    -DOCS NOT UPDATED- Class to import the raw data of the model and set data classifications into formations and series

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

        # Set extent and resolution
        self.extent = np.array(extent)
        self.resolution = np.array(resolution)

        # TODO choose the default source of data. So far only csv
        # Create the pandas dataframes
        if path_f:
            self.foliations = self.load_data_csv(data_type="foliations", path=path_f, **kwargs)
            assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(self.foliations.columns), \
                "One or more columns do not match with the expected values " + str(self.foliations.columns)
        else:
            # if we dont read a csv we create an empty dataframe with the columns that have to be filled
            self.foliations = pn.DataFrame(columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity',
                                                    'formation', 'series'])
        if path_i:
            self.interfaces = self.load_data_csv(data_type="interfaces", path=path_i, **kwargs)
            assert set(['X', 'Y', 'Z', 'formation']).issubset(self.interfaces.columns), \
                "One or more columns do not match with the expected values " + str(self.interfaces.columns)
        else:
            self.interfaces = pn.DataFrame(columns=['X', 'Y', 'Z', 'formation', 'series'])


        # DEP-
        self._set_formations()

        # If not provided set default series
        self.series = self.set_series()
        # DEP- self.set_formation_number()

        # Compute gradients given azimuth and dips to plot data
        self.calculate_gradient()

        # Create default grid object. TODO: (Is this necessary now?)
        self.grid = self.create_grid(extent=None, resolution=None, grid_type="regular_3D", **kwargs)

    def _set_formations(self):
        """
        -DEPRECATED- Function to import the formations that will be used later on. By default all the formations in the tables are
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

    def get_raw_data(self, itype='all'):
        """
        Method that returns the interfaces and foliations pandas Dataframes. Can return both at the same time or only
        one of the two
        Args:
            itype: input data type, either 'foliations', 'interfaces' or 'all' for both.

        Returns:
            pandas.core.frame.DataFrame: Data frame with the raw data

        """
        import pandas as pn
        if itype == 'foliations':
            raw_data = self.foliations
        elif itype == 'interfaces':
            raw_data = self.interfaces
        elif itype == 'all':
            raw_data = pn.concat([self.interfaces, self.foliations], keys=['interfaces', 'foliations'])
        return raw_data

    def i_open_set_data(self, itype="foliations"):
        """
        Method to have interactive pandas tables in jupyter notebooks. The idea is to use this method to interact with
         the table and i_close_set_data to recompute the parameters that depend on the changes made. I did not find a
         easier solution than calling two different methods.
        Args:
            itype: input data type, either 'foliations' or 'interfaces'

        Returns:
            pandas.core.frame.DataFrame: Data frame with the changed data on real time
        """

        # if the data frame is empty the interactive table is bugged. Therefore I create a default raw when the method
        # is called
        if self.foliations.empty:
            self.foliations = pn.DataFrame(
                np.array([0., 0., 0., 0., 0., 1., 'Default Formation', 'Default series']).reshape(1, 8),
                columns=['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation', 'series']).\
                convert_objects(convert_numeric=True)

        if self.interfaces.empty:
            self.interfaces = pn.DataFrame(
                np.array([0, 0, 0, 'Default Formation', 'Default series']).reshape(1, 5),
                columns=['X', 'Y', 'Z', 'formation', 'series']).convert_objects(convert_numeric=True)

        # TODO leave qgrid as a dependency since in the end I did not change the code of the package
        import qgrid

        # Setting some options
        qgrid.nbinstall(overwrite=True)
        qgrid.set_defaults(show_toolbar=True)
        assert itype is 'foliations' or itype is 'interfaces', 'itype must be either foliations or interfaces'

        import warnings
        warnings.warn('Remember to call i_close_set_data after the editing.')

        # We kind of set the show grid to a variable so we can close it afterwards
        self.pandas_frame = qgrid.show_grid(self.get_raw_data(itype=itype))

        # TODO set

    def i_close_set_data(self):

        """
        Method to have interactive pandas tables in jupyter notebooks. The idea is to use this method to interact with
         the table and i_close_set_data to recompute the parameters that depend on the changes made. I did not find a
         easier solution than calling two different methods.
        Args:
            itype: input data type, either 'foliations' or 'interfaces'

        Returns:
            pandas.core.frame.DataFrame: Data frame with the changed data on real time
        """
        # We close it to guarantee that after this method it is not possible further modifications
        self.pandas_frame.close()
        # -DEP- self._set_formations()
        # -DEP- self.set_formation_number()
        # Set parameters
        self.series = self.set_series()
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

        # TODO if we load different data the Interpolator parameters must be also updated. Prob call gradients and
        # series

    def set_interfaces(self, interf_Dataframe, append=False):
        """
        Method to change or append a Dataframe to interfaces in place.
        Args:
            interf_Dataframe: pandas.core.frame.DataFrame with the data
            append: Bool: if you want to append the new data frame or substitute it
        """
        assert set(['X', 'Y', 'Z', 'formation']).issubset(interf_Dataframe.columns), \
            "One or more columns do not match with the expected values " + str(interf_Dataframe.columns)

        if append:
            self.interfaces = self.interfaces.append(interf_Dataframe)
        else:
            self.interfaces = interf_Dataframe

       #-DEP- self.interfaces.reset_index(drop=False, inplace=True)
       #-DEP- self._set_formations()
        self.set_series()
       #-DEP- self.set_formation_number()

    def set_foliations(self, foliat_Dataframe, append=False):
        """
          Method to change or append a Dataframe to foliations in place.
          Args:
              interf_Dataframe: pandas.core.frame.DataFrame with the data
              append: Bool: if you want to append the new data frame or substitute it
          """
        assert set(['X', 'Y', 'Z', 'dip', 'azimuth', 'polarity', 'formation']).issubset(
            foliat_Dataframe.columns), "One or more columns do not match with the expected values " +\
                                       str(foliat_Dataframe.columns)
        if append:
            self.foliations = self.foliations.append(foliat_Dataframe)
        else:
            self.foliations = foliat_Dataframe

        #-DEP- self.interfaces.reset_index(drop=False, inplace=True)
        #-DEP- self._set_formations()
        self.set_series()
        #-DEP- self.set_formation_number()
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
            # set to default series
            # TODO see if some of the formations have already a series and not overwrite
            _series = {"Default serie": self.interfaces["formation"].unique()}

        else:
            assert type(series_distribution) is dict, "series_distribution must be a dictionary, " \
                                                      "see Docstring for more information"

            # TODO if self.series exist already maybe we should append instead of overwrite
            _series = series_distribution

        # The order of the series is very important since it dictates which one is on top of the stratigraphic pile
        # If it is not given we take the dictionaries keys. NOTICE that until python 3.6 these keys are pretty much
        # random
        if not order:
            order = _series.keys()

        # TODO assert len order is equal to len of the dictionary

        # We create a dataframe with the links
        _series = pn.DataFrame(data=_series, columns=order)

        # Now we fill the column series in the interfaces and foliations tables with the correspondant series and
        # assigned number to the series
        self.interfaces["series"] = [(i == _series).sum().argmax() for i in self.interfaces["formation"]]
        self.interfaces["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.interfaces["formation"]]
        self.foliations["series"] = [(i == _series).sum().argmax() for i in self.foliations["formation"]]
        self.foliations["order_series"] = [(i == _series).sum().as_matrix().argmax() + 1
                                           for i in self.foliations["formation"]]

        # We sort the series altough is only important for the computation (we will do it again just before computing)
        self.interfaces.sort_values(by='order_series', inplace=True)
        self.foliations.sort_values(by='order_series', inplace=True)

        # Save the dataframe in a property
        self.series = _series

        return _series

    def set_formation_number(self):
        """
        Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
        to know it and also now the numbers must be set in the order of the series as well. Therefore this method
        has been moved to the interpolator class as preprocessing

        Returns: Column in the interfaces and foliations dataframes
        """
        try:
            ip_addresses = self.interfaces["formation"].unique()
            ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses)+1)))
            self.interfaces['formation number'] = self.interfaces['formation'].replace(ip_dict)
            self.foliations['formation number'] = self.foliations['formation'].replace(ip_dict)
        except ValueError:
            pass

    class GridClass(object):
        """
        -DOCS NOT UPDATED- Class with set of functions to generate grids

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
        -DOCS NOT UPDATED- Class which contain all needed methods to perform potential field implicit modelling in theano

        Args:
            _data(GeMpy_core.DataManagement): All values of a DataManagement object
            _grid(GeMpy_core.grid): A grid object
            **kwargs: Arbitrary keyword arguments.

        Keyword Args:
            verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
        """

        def __init__(self, _data_scaled, _grid_scaled=None, *args, **kwargs):

            # verbose is a list of strings. See theanograph
            verbose = kwargs.get('verbose', [0])
            # -DEP-rescaling_factor = kwargs.get('rescaling_factor', None)

            # Here we can change the dtype for stability and GPU vs CPU
            dtype = kwargs.get('dtype', 'float32')
            self.dtype = dtype

            # Drift grade
            u_grade = kwargs.get('u_grade', [2])

            # We hide the scaled copy of DataManagement object from the user. The scaling happens in GeMpy what is a
            # bit weird. Maybe at some point I should bring the function to this module
            self._data_scaled = _data_scaled

            # In case someone wants to provide a grid otherwise we extract it from the DataManagement object.
            if not _grid_scaled:
                self._grid_scaled = _data_scaled.grid
            else:
                self._grid_scaled = _grid_scaled

            # Importing the theano graph. The methods of this object generate different parts of graph.
            # See theanograf doc
            self.tg = theanograf.TheanoGraph_pro(dtype=dtype, verbose=verbose,)

            # Sorting data in case the user provides it unordered
            self.order_table()

            # Setting theano parameters
            self.set_theano_shared_parameteres(self._data_scaled, self._grid_scaled, **kwargs)

            # Extracting data from the pandas dataframe to numpy array in the required form for the theano function
            self.data_prep(u_grade=u_grade)

            # Avoid crashing my pc
            import theano
            if theano.config.optimizer != 'fast_run':
                assert self.tg.grid_val_T.get_value().shape[0] * \
                       np.math.factorial(len(self.tg.len_series_i.get_value())) < 2e6, \
                       'The grid is too big for the number of potential fields. Reduce the grid or change the' \
                       'optimization flag to fast run'

        def set_formation_number(self):
            """
                    Set a unique number to each formation. NOTE: this method is getting deprecated since the user does not need
                    to know it and also now the numbers must be set in the order of the series as well. Therefore this method
                    has been moved to the interpolator class as preprocessing

            Returns: Column in the interfaces and foliations dataframes
            """
            try:
                ip_addresses = self._data_scaled.interfaces["formation"].unique()
                ip_dict = dict(zip(ip_addresses, range(1, len(ip_addresses) + 1)))
                self._data_scaled.interfaces['formation number'] = self._data_scaled.interfaces['formation'].replace(ip_dict)
                self._data_scaled.foliations['formation number'] = self._data_scaled.foliations['formation'].replace(ip_dict)
            except ValueError:
                pass

        def order_table(self):
            """
            First we sort the dataframes by the series age. Then we set a unique number for every formation and resort
            the formations. All inplace
            """

            # We order the pandas table by series
            self._data_scaled.interfaces.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series'],  # , 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Give formation number
            self.set_formation_number()

            # We order the pandas table by formation (also by series in case something weird happened)
            self._data_scaled.interfaces.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            self._data_scaled.foliations.sort_values(by=['order_series', 'formation number'],
                                                     ascending=True, kind='mergesort',
                                                     inplace=True)

            # Pandas dataframe set an index to every row when the dataframe is created. Sorting the table does not reset
            # the index. For some of the methods (pn.drop) we have to apply afterwards we need to reset these indeces
            self._data_scaled.interfaces.reset_index(drop=True, inplace=True)

        def data_prep(self, **kwargs):
            """
            Ideally this method will extract the data from the pandas dataframes to individual numpy arrays to be input
            of the theano function. However since some of the shared parameters are function of these arrays shape I also
            set them here
            Returns:
                idl (list): List of arrays which are the input for the theano function:
                    - numpy.array: dips_position
                    - numpy.array: dip_angles
                    - numpy.array: azimuth
                    - numpy.array: polarity
                    - numpy.array: ref_layer_points
                    - numpy.array: rest_layer_points
            """

            u_grade = kwargs.get('u_grade', None)
            print(u_grade)
            # ==================
            # Extracting lengths
            # ==================
            # Array containing the size of every formation. Interfaces
            len_interfaces = np.asarray(
                [np.sum(self._data_scaled.interfaces['formation number'] == i)
                 for i in self._data_scaled.interfaces['formation number'].unique()])

            # Size of every layer in rests. SHARED (for theano)
            len_rest_form = (len_interfaces - 1)
            self.tg.number_of_points_per_formation_T.set_value(len_rest_form)

            # Position of the first point of every layer
            ref_position = np.insert(len_interfaces[:-1], 0, 0).cumsum()

            # Drop the reference points using pandas indeces to get just the rest_layers array
            pandas_rest_layer_points = self._data_scaled.interfaces.drop(ref_position)

            # TODO: do I need this? PYTHON
            # DEP- because per series the foliations do not belong to a formation but to the whole series
            # len_foliations = np.asarray(
            #     [np.sum(self._data_scaled.foliations['formation number'] == i)
            #      for i in self._data_scaled.foliations['formation number'].unique()])

            # -DEP- I think this was just a kind of print to know what was going on
            #self.pandas_rest = pandas_rest_layer_points

            # Array containing the size of every series. Interfaces.
            len_series_i = np.asarray(
                [np.sum(pandas_rest_layer_points['order_series'] == i)
                 for i in pandas_rest_layer_points['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_i.set_value(np.insert(len_series_i, 0, 0).cumsum())

            # Array containing the size of every series. Foliations.
            len_series_f = np.asarray(
                [np.sum(self._data_scaled.foliations['order_series'] == i)
                 for i in self._data_scaled.foliations['order_series'].unique()])

            # Cumulative length of the series. We add the 0 at the beginning and set the shared value. SHARED
            self.tg.len_series_f.set_value(np.insert(len_series_f, 0, 0).cumsum())

            # =========================
            # Choosing Universal drifts
            # =========================

            if u_grade is None:
                u_grade = np.zeros_like(len_series_i)
                u_grade[len_series_i > 12] = 9
                u_grade[(len_series_i > 6) & (len_series_i < 12)] = 3

            self.tg.u_grade_T.set_value(u_grade)

            # ================
            # Prepare Matrices
            # ================
            # Rest layers matrix # PYTHON VAR
            rest_layer_points = pandas_rest_layer_points[['X', 'Y', 'Z']].as_matrix()

            # TODO delete
            # -DEP- Again i was just a check point
            # self.rest_layer_points = rest_layer_points

            # Ref layers matrix #VAR
            # Calculation of the ref matrix and tile. Iloc works with the row number
            # Here we extract the reference points
            aux_1 = self._data_scaled.interfaces.iloc[ref_position][['X', 'Y', 'Z']].as_matrix()

            # We initialize the matrix
            ref_layer_points = np.zeros((0, 3))

            # TODO I hate loop it has to be a better way
            # Tiling very reference points as many times as rest of the points we have
            for e, i in enumerate(len_interfaces):
                ref_layer_points = np.vstack((ref_layer_points, np.tile(aux_1[e], (i - 1, 1))))

            # -DEP- was just a check point
            #self.ref_layer_points = ref_layer_points

            # Check no reference points in rest points (at least in coor x)
            assert not any(aux_1[:, 0]) in rest_layer_points[:, 0], \
                'A reference point is in the rest list point. Check you do ' \
                'not have duplicated values in your dataframes'

            # Foliations, this ones I tile them inside theano. PYTHON VAR
            dips_position = self._data_scaled.foliations[['X', 'Y', 'Z']].as_matrix()
            dip_angles = self._data_scaled.foliations["dip"].as_matrix()
            azimuth = self._data_scaled.foliations["azimuth"].as_matrix()
            polarity = self._data_scaled.foliations["polarity"].as_matrix()

            # Set all in a list casting them in the chosen dtype
            idl = [np.cast[self.dtype](xs) for xs in (dips_position, dip_angles, azimuth, polarity,
                   ref_layer_points, rest_layer_points)]

            return idl

        def set_theano_shared_parameteres(self, _data_rescaled, _grid_rescaled, **kwargs):
            """
            Here we create most of the kriging parameters. The user can pass them as kwargs otherwise we pick the
            default values from the DataManagement info. The share variables are set in place. All the parameters here
            are independent of the input data so this function only has to be called if you change the extent or grid or
            if you want to change one the kriging parameters.
            Args:
                _data_rescaled: DataManagement object
                _grid_rescaled: Grid object
            Keyword Args:
                u_grade (int): Drift grade. Default to 2.
                range_var (float): Range of the variogram. Default 3D diagonal of the extent
                c_o (float): Covariance at lag 0. Default range_var ** 2 / 14 / 3. See my paper when I write it
                nugget_effect (flaot): Nugget effect of foliations. Default to 0.01
            """

            # Kwargs
            u_grade = kwargs.get('u_grade', 2)
            range_var = kwargs.get('range_var', None)
            c_o = kwargs.get('c_o', None)
            nugget_effect = kwargs.get('nugget_effect', 0.01)

            # -DEP- Now I rescale the data so we do not need this
            # rescaling_factor = kwargs.get('rescaling_factor', None)

            # Default range
            if not range_var:
                range_var = np.sqrt((_data_rescaled.extent[0] - _data_rescaled.extent[1]) ** 2 +
                                    (_data_rescaled.extent[2] - _data_rescaled.extent[3]) ** 2 +
                                    (_data_rescaled.extent[4] - _data_rescaled.extent[5]) ** 2)

            # Default covariance at 0
            if not c_o:
                c_o = range_var ** 2 / 14 / 3

            # Asserting that the drift grade is in this range
           # assert (0 <= all(u_grade) <= 2)

            # Creating the drift matrix. TODO find the official name of this matrix?
            _universal_matrix = np.vstack((_grid_rescaled.grid.T,
                                           (_grid_rescaled.grid ** 2).T,
                                           _grid_rescaled.grid[:, 0] * _grid_rescaled.grid[:, 1],
                                           _grid_rescaled.grid[:, 0] * _grid_rescaled.grid[:, 2],
                                           _grid_rescaled.grid[:, 1] * _grid_rescaled.grid[:, 2]))

            # Setting shared variables
            # Range
            self.tg.a_T.set_value(np.cast[self.dtype](range_var))
            # Covariance at 0
            self.tg.c_o_T.set_value(np.cast[self.dtype](c_o))
            # Foliations nugget effect
            self.tg.nugget_effect_grad_T.set_value(np.cast[self.dtype](nugget_effect))

            # TODO change the drift to the same style I have the faults so I do not need to do this
            # # Drift grade
            # if u_grade == 0:
            #     self.tg.u_grade_T.set_value(u_grade)
            # else:
            #     self.tg.u_grade_T.set_value(u_grade)
                # TODO: To be sure what is the mathematical meaning of this -> It seems that nothing
                # TODO Deprecated
                # self.tg.c_resc.set_value(1)

            # Just grid. I add a small number to avoid problems with the origin point
            self.tg.grid_val_T.set_value(np.cast[self.dtype](_grid_rescaled.grid + 10e-6))
            # Universal grid
            self.tg.universal_grid_matrix_T.set_value(np.cast[self.dtype](_universal_matrix + 1e-10))

            # Initialization of the block model
            self.tg.final_block.set_value(np.zeros((_grid_rescaled.grid.shape[0]), dtype='float32'))

            # Initialization of the boolean array that represent the areas of the block model to be computed in the
            # following series
            self.tg.yet_simulated.set_value(np.ones((_grid_rescaled.grid.shape[0]), dtype='int'))

            # Unique number assigned to each lithology
            #self.tg.n_formation.set_value(np.insert(_data_rescaled.interfaces['formation number'].unique(),
            #                                        0, 0)[::-1])

            self.tg.n_formation.set_value(_data_rescaled.interfaces['formation number'].unique())

            # Number of formations per series. The function is not pretty but the result is quite clear
            self.tg.n_formations_per_serie.set_value(
                np.insert(_data_rescaled.interfaces.groupby('order_series').formation.nunique().values.cumsum(), 0, 0))

"""
Module with classes and methods to perform implicit regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 10/10 /2016

@author: Miguel de la Varga

"""
from __future__ import division
# import theano
# import theano.tensor as T
import numpy as _np
# import sys, os
import pandas as _pn
import copy
from .Visualization import PlotData
from .DataManagement import DataManagement
from IPython.core.debugger import Tracer


def rescale_data(geo_data, rescaling_factor=None):
    """
    Rescale the data of a DataManagement object between 0 and 1 due to stability problem of the float32.
    Args:
        geo_data: DataManagement object with the real scale data
        rescaling_factor(float): factor of the rescaling. Default to maximum distance in one the axis

    Returns:

    """
    max_coord = _pn.concat(
        [geo_data.foliations, geo_data.interfaces]).max()[['X', 'Y', 'Z']]
    min_coord = _pn.concat(
        [geo_data.foliations, geo_data.interfaces]).min()[['X', 'Y', 'Z']]

    if not rescaling_factor:
        rescaling_factor = 2*_np.max(max_coord - min_coord)

    centers = (max_coord+min_coord)/2

    new_coord_interfaces = (geo_data.interfaces[['X', 'Y', 'Z']] -
                           centers) / rescaling_factor + 0.5001

    new_coord_foliations = (geo_data.foliations[['X', 'Y', 'Z']] -
                           centers) / rescaling_factor + 0.5001

    new_coord_extent = (geo_data.extent - _np.repeat(centers, 2)) / rescaling_factor + 0.5001

    geo_data_rescaled = copy.deepcopy(geo_data)
    geo_data_rescaled.interfaces[['X', 'Y', 'Z']] = new_coord_interfaces
    geo_data_rescaled.foliations[['X', 'Y', 'Z']] = new_coord_foliations
    geo_data_rescaled.extent = new_coord_extent.as_matrix()

    geo_data_rescaled.grid.grid = (geo_data.grid.grid - centers.as_matrix()) /rescaling_factor + 0.5001

    geo_data_rescaled.rescaling_factor = rescaling_factor

    return geo_data_rescaled

# TODO needs to be updated
# def compute_block_model(geo_data, series_number="all",
#                         series_distribution=None, order_series=None,
#                         extent=None, resolution=None, grid_type="regular_3D",
#                         verbose=0, **kwargs):
#
#     if extent or resolution:
#         set_grid(geo_data, extent=extent, resolution=resolution, grid_type=grid_type, **kwargs)
#
#     if series_distribution:
#         set_data_series(geo_data, series_distribution=series_distribution, order_series=order_series, verbose=0)
#
#     if not getattr(geo_data, 'interpolator', None):
#         import warnings
#
#         warnings.warn('Using default interpolation values')
#         set_interpolator(geo_data)
#
#     geo_data.interpolator.tg.final_block.set_value(_np.zeros_like(geo_data.grid.grid[:, 0]))
#
#     geo_data.interpolator.compute_block_model(series_number=series_number, verbose=verbose)
#
#     return geo_data.interpolator.tg.final_block


def get_grid(geo_data):
    return geo_data.grid.grid


def get_raw_data(geo_data, dtype='all'):
    return geo_data.get_raw_data(itype=dtype)


def import_data(extent, resolution=[50, 50, 50], **kwargs):
    """
    Method to initialize the class data. Calling this function some of the data has to be provided (TODO give to
    everything a default).

    Args:
        extent (list or array):  [x_min, x_max, y_min, y_max, z_min, z_max]. Extent for the visualization of data
         and default of for the grid class.
        resolution (list or array): [nx, ny, nz]. Resolution for the visualization of data
         and default of for the grid class.
        **kwargs: Arbitrary keyword arguments.

    Keyword Args:
        Resolution ((Optional[list])): [nx, ny, nz]. Defaults to 50
        path_i: Path to the data bases of interfaces. Default os.getcwd(),
        path_f: Path to the data bases of foliations. Default os.getcwd()

    Returns:
        GeMpy.DataManagement: Object that encapsulate all raw data of the project


        dep: self.Plot(GeMpy_core.PlotData): Object to visualize data and results
    """

    return DataManagement(extent, resolution, **kwargs)


def i_set_data(geo_data, dtype="foliations", action="Open"):

    if action == 'Close':
        geo_data.i_close_set_data()

    if action == 'Open':
        geo_data.i_open_set_data(itype=dtype)


def select_series(geo_data, series):
    """
    Return the formations of a given serie in string
    :param series: list of int or list of str
    :return: formations of a given serie in string separeted by |
    """
    new_geo_data = copy.deepcopy(geo_data)

    if type(series) == int or type(series[0]) == int:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['order_series'].isin(series)]
        new_geo_data.foliations = geo_data.foliations[geo_data.foliations['order_series'].isin(series)]
    elif type(series[0]) == str:
        new_geo_data.interfaces = geo_data.interfaces[geo_data.interfaces['series'].isin(series)]
        new_geo_data.foliations = geo_data.foliations[geo_data.foliations['series'].isin(series)]
    return new_geo_data

def set_data_series(geo_data, series_distribution=None, order_series=None,
                        update_p_field=True, verbose=0):

    geo_data.set_series(series_distribution=series_distribution, order=order_series)
    try:
        if update_p_field:
            geo_data.interpolator.compute_potential_fields()
    except AttributeError:
        pass

    if verbose > 0:
        return get_raw_data(geo_data)


def set_interfaces(geo_data, interf_Dataframe, append=False, update_p_field=True):
    geo_data.set_interfaces(interf_Dataframe, append=append)
    # To update the interpolator parameters without calling a new object
    try:
        geo_data.interpolator._data = geo_data
        geo_data.interpolator._grid = geo_data.grid
       # geo_data.interpolator._set_constant_parameteres(geo_data, geo_data.interpolator._grid)
        if update_p_field:
            geo_data.interpolator.compute_potential_fields()
    except AttributeError:
        pass


def set_foliations(geo_data, foliat_Dataframe, append=False, update_p_field=True):
    geo_data.set_foliations(foliat_Dataframe, append=append)
    # To update the interpolator parameters without calling a new object
    try:
        geo_data.interpolator._data = geo_data
        geo_data.interpolator._grid = geo_data.grid
      #  geo_data.interpolator._set_constant_parameteres(geo_data, geo_data.interpolator._grid)
        if update_p_field:
            geo_data.interpolator.compute_potential_fields()
    except AttributeError:
        pass


def set_grid(geo_data, new_grid=None, extent=None, resolution=None, grid_type="regular_3D", **kwargs):
    """
    Method to initialize the class new_grid. So far is really simple and only has the regular new_grid type

    Args:
        grid_type (str): regular_3D or regular_2D (I am not even sure if regular 2D still working)
        **kwargs: Arbitrary keyword arguments.

    Returns:
        self.new_grid(GeMpy_core.new_grid): Object that contain different grids
    """
    if new_grid is not None:
        assert new_grid.shape[1] is 3 and len(new_grid.shape) is 2, 'The shape of new grid must be (n,3) where n is' \
                                                                    'the number of points of the grid'
        geo_data.grid.grid = new_grid
    else:
        if not extent:
            extent = geo_data.extent
        if not resolution:
            resolution = geo_data.resolution

        geo_data.grid = geo_data.GridClass(extent, resolution, grid_type=grid_type, **kwargs)


def set_interpolator(geo_data,  *args, **kwargs):
    """
    Method to initialize the class interpolator. All the constant parameters for the interpolation can be passed
    as args, otherwise they will take the default value (TODO: documentation of the dafault values)

    Args:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments.

    Keyword Args:
        range_var: Range of the variogram. Default None
        c_o: Covariance at 0. Default None
        nugget_effect: Nugget effect of the gradients. Default 0.01
        u_grade: Grade of the polynomial used in the universal part of the Kriging. Default 2
        rescaling_factor: Magic factor that multiplies the covariances). Default 2

    Returns:
        self.Interpolator (GeMpy_core.Interpolator): Object to perform the potential field method
        self.Plot(GeMpy_core.PlotData): Object to visualize data and results. It gets updated.
    """

    rescaling_factor = kwargs.get('rescaling_factor', None)

    if 'u_grade' in kwargs:
        compile_theano = True

    if not getattr(geo_data, 'grid', None):
        set_grid(geo_data)

    geo_data_int = rescale_data(geo_data, rescaling_factor=rescaling_factor)

    if not getattr(geo_data_int, 'interpolator', None) or compile_theano:
        geo_data_int.interpolator = geo_data_int.InterpolatorClass(geo_data_int, geo_data_int.grid,
                                                                   *args, **kwargs)
    else:
        geo_data_int.interpolator._data = geo_data_int
        geo_data_int.interpolator._grid = geo_data_int.grid
        geo_data_int.interpolator.set_theano_shared_parameteres(geo_data_int, geo_data_int.interpolator._grid, **kwargs)

    return geo_data_int


def plot_data(geo_data, direction="y", series="all", **kwargs):
    plot = PlotData(geo_data)
    plot.plot_data(direction=direction, series=series, **kwargs)
    # TODO saving options
    return plot


def plot_section(geo_data, cell_number, block=None, direction="y", **kwargs):
    plot = PlotData(geo_data)
    plot.plot_block_section(cell_number, block=block, direction=direction, **kwargs)
    # TODO saving options
    return plot


def plot_potential_field(geo_data, potential_field, cell_number, n_pf=0,
                         direction="y", plot_data=True, series="all", *args, **kwargs):

    plot = PlotData(geo_data)
    plot.plot_potential_field(potential_field, cell_number, n_pf=n_pf,
                              direction=direction,  plot_data=plot_data, series=series,
                              *args, **kwargs)


def compute_potential_fields(geo_data, verbose=0):
    geo_data.interpolator.compute_potential_fields(verbose=verbose)
"""
Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 23/09/2016
somehting
@author: Miguel de la Varga
"""


import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# TODO: inherit pygeomod classes
#import sys, os

class PlotData(object):
    """
    Class to make the different plot related with GeMpy

    Args:
        _data(GeMpy_core.DataManagement): All values of a DataManagement object
        block(theano shared): 3D array containing the lithology block
        **kwargs: Arbitrary keyword arguments.

    Keyword Args:
        potential_field(numpy.ndarray): 3D array containing a individual potential field
        verbose(int): Level of verbosity during the execution of the functions (up to 5). Default 0
    """

    def __init__(self, _data, block=None, **kwargs):

        self._data = _data
        if block:
            self._block = block

        if 'potential_field' in kwargs:
            self._potential_field_p = kwargs['potential_field']

    # TODO planning the whole visualization scheme. Only data, potential field and block. 2D 3D? Improving the iteration
    # with pandas framework
        self._set_style()

    def _set_style(self):
        """
        Private function to set some plotting options

        """

        plt.style.use(['seaborn-white', 'seaborn-paper'])
       # sns.set_context("paper")
       # matplotlib.rc("font", family="Helvetica")

    def plot_data(self, direction="y", series="all", *args, **kwargs):
        """
        Plot the projecton of the raw data (interfaces and foliations) in 2D following a specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            series(str): series to plot
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Data plot

        """

        x, y, Gx, Gy = self._slice(direction)[4:]

        if series == "all":
            series_to_plot_i = self._data.Interfaces[self._data.Interfaces["series"].
                                                     isin(self._data.series.columns.values)]
            series_to_plot_f = self._data.Foliations[self._data.Foliations["series"].
                                                     isin(self._data.series.columns.values)]

        else:
            series_to_plot_i = self._data.Interfaces[self._data.Interfaces["series"] == series]
            series_to_plot_f = self._data.Foliations[self._data.Foliations["series"] == series]
        sns.lmplot(x, y,
                   data=series_to_plot_i,
                   fit_reg=False,
                   hue="formation",
                   scatter_kws={"marker": "D",
                                "s": 100},
                   legend=True,
                   legend_out=True,
                   **kwargs)

        # Plotting orientations
        plt.quiver(series_to_plot_f[x], series_to_plot_f[y],
                   series_to_plot_f[Gx], series_to_plot_f[Gy],
                   pivot="tail")

        plt.xlabel(x)
        plt.ylabel(y)

    def _slice(self, direction, cell_number=25):
        """
        Slice the 3D array (blocks or potential field) in the specific direction selected in the plotting functions

        """
        _a, _b, _c = slice(0, self._data.nx), slice(0, self._data.ny), slice(0, self._data.nz)
        if direction == "x":
            _a = cell_number
            x = "Y"
            y = "Z"
            Gx = "G_y"
            Gy = "G_z"
            extent_val = self._data.ymin, self._data.ymax, self._data.zmin, self._data.zmax
        elif direction == "y":
            _b = cell_number
            x = "X"
            y = "Z"
            Gx = "G_x"
            Gy = "G_z"
            extent_val = self._data.xmin, self._data.xmax, self._data.zmin, self._data.zmax
        elif direction == "z":
            _c = cell_number
            x = "X"
            y = "Y"
            Gx = "G_x"
            Gy = "G_y"
            extent_val = self._data.xmin, self._data.xmax, self._data.ymin, self._data.ymax
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def plot_block_section(self, cell_number=13, direction="y", **kwargs):
        """
        Plot a section of the block model

        Args:
            cell_number(int): position of the array to plot
            direction(str): xyz. Caartesian direction to be plotted
            **kwargs: imshow keywargs

        Returns:
            Block plot
        """
        plot_block = self._block.get_value().reshape(self._data.nx, self._data.ny, self._data.nz)
        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]

        plt.imshow(plot_block[_a, _b, _c].T, origin="bottom", cmap="viridis",
                   extent=extent_val,
                   interpolation="none", **kwargs)
        plt.xlabel(x)
        plt.ylabel(y)

    def plot_potential_field(self, cell_number, potential_field=None, n_pf=0,
                             direction="y", plot_data=True, serie="all", *args, **kwargs):
        """
        Plot a potential field in a given direction.

        Args:
            cell_number(int): position of the array to plot
            potential_field(str): name of the potential field (or series) to plot
            n_pf(int): number of the  potential field (or series) to plot
            direction(str): xyz. Caartesian direction to be plotted
            serie: *Deprecated*
            **kwargs: plt.contour kwargs

        Returns:
            Potential field plot
        """
        if not potential_field:
            potential_field = self._potential_field_p[n_pf]

        if plot_data:
            self.plot_data(direction, self._data.series.columns.values[n_pf])

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
        plt.contour(potential_field[_a, _b, _c].T, 12,
                    extent=extent_val, *args,
                    **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.title(self._data.series.columns[n_pf])
        plt.xlabel(x)
        plt.ylabel(y)

    def export_vtk(self):
        """
        export vtk
        :return:
        """
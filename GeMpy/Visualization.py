"""
Module with classes and methods to visualized structural geology data and potential fields of the regional modelling based on
the potential field method.
Tested on Ubuntu 14

Created on 23/09/2016

@author: Miguel de la Varga
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# TODO: inherit pygeomod classes
# import sys, os

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

    def __init__(self, _data, **kwargs):

        self._data = _data

        if 'potential_field' in kwargs:
            self._potential_field_p = kwargs['potential_field']

            # TODO planning the whole visualization scheme. Only data, potential field
            # and block. 2D 3D? Improving the iteration
            # with pandas framework
        self._set_style()

    def _set_style(self):
        """
        Private function to set some plotting options

        """

        plt.style.use(['seaborn-white', 'seaborn-paper'])
        # sns.set_context("paper")
        # matplotlib.rc("font", family="Helvetica")

    def plot_data(self, direction="y", series="all", **kwargs):
        """
        Plot the projecton of the raw data (interfaces and foliations) in 2D following a
        specific directions

        Args:
            direction(str): xyz. Caartesian direction to be plotted
            series(str): series to plot
            **kwargs: seaborn lmplot key arguments. (TODO: adding the link to them)

        Returns:
            Data plot

        """

        x, y, Gx, Gy = self._slice(direction)[4:]

        if series == "all":
            series_to_plot_i = self._data.interfaces[self._data.interfaces["series"].
                isin(self._data.series.columns.values)]
            series_to_plot_f = self._data.foliations[self._data.foliations["series"].
                isin(self._data.series.columns.values)]

        else:
            series_to_plot_i = self._data.interfaces[self._data.interfaces["series"] == series]
            series_to_plot_f = self._data.foliations[self._data.foliations["series"] == series]

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
        _a, _b, _c = (slice(0, self._data.resolution[0]),
                      slice(0, self._data.resolution[1]),
                      slice(0, self._data.resolution[2]))
        if direction == "x":
            _a = cell_number
            x = "Y"
            y = "Z"
            Gx = "G_y"
            Gy = "G_z"
            extent_val = self._data.extent[3], self._data.extent[2], self._data.extent[5], self._data.extent[4]
        elif direction == "y":
            _b = cell_number
            x = "X"
            y = "Z"
            Gx = "G_x"
            Gy = "G_z"
            extent_val = self._data.extent[0], self._data.extent[1], self._data.extent[4], self._data.extent[5]
        elif direction == "z":
            _c = cell_number
            x = "X"
            y = "Y"
            Gx = "G_x"
            Gy = "G_y"
            extent_val = self._data.extent[1], self._data.extent[0], self._data.extent[3], self._data.extent[2]
        else:
            raise AttributeError(str(direction) + "must be a cartesian direction, i.e. xyz")
        return _a, _b, _c, extent_val, x, y, Gx, Gy

    def plot_block_section(self, cell_number=13, block=None, direction="y", interpolation='none',
                           plot_data = False, **kwargs):
        """
        Plot a section of the block model

        Args:
            cell_number(int): position of the array to plot
            direction(str): xyz. Caartesian direction to be plotted
                interpolation(str): Type of interpolation of plt.imshow. Default 'none'.  Acceptable values are 'none'
                ,'nearest', 'bilinear', 'bicubic',
                'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
                'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
                'lanczos'
            **kwargs: imshow keywargs

        Returns:
            Block plot
        """
        if block is not None:
            import theano
            import numpy
            assert (type(block) is theano.tensor.sharedvar.TensorSharedVariable or
                    type(block) is numpy.ndarray), \
                'Block has to be a theano shared object or numpy array.'
            if type(block) is numpy.ndarray:
                _block = block
            else:
                _block = block.get_value()
        else:
            try:
                _block = self._data.interpolator.tg.final_block.get_value()
            except AttributeError:
                raise AttributeError('There is no block to plot')

        plot_block = _block.reshape(self._data.resolution[0], self._data.resolution[1], self._data.resolution[2])
        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]


        if plot_data:
            self.plot_data(direction, 'all')

        plt.imshow(plot_block[_a, _b, _c].T, origin="bottom", cmap="viridis",
                   extent=extent_val,
                   interpolation=interpolation, **kwargs)

        plt.xlabel(x)
        plt.ylabel(y)

    def plot_potential_field(self, potential_field, cell_number, n_pf=0,
                             direction="y", plot_data=True, series="all", *args, **kwargs):
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

        if plot_data:
            self.plot_data(direction, self._data.series.columns.values[n_pf])

        _a, _b, _c, extent_val, x, y = self._slice(direction, cell_number)[:-2]
        plt.contour(potential_field[_a, _b, _c].T, cell_number,
                    extent=extent_val, *args,
                    **kwargs)

        if 'colorbar' in kwargs:
            plt.colorbar()

        plt.title(self._data.series.columns[n_pf])
        plt.xlabel(x)
        plt.ylabel(y)

    @staticmethod
    def annotate_plot(frame, label_col, x, y, **kwargs):
        """
        Annotate the plot of a given DataFrame using one of its columns

        Should be called right after a DataFrame or series plot method,
        before telling matplotlib to show the plot.

        Parameters
        ----------
        frame : pandas.DataFrame

        plot_col : str
            The string identifying the column of frame that was plotted

        label_col : str
            The string identifying the column of frame to be used as label

        kwargs:
            Other key-word args that should be passed to plt.annotate

        Returns
        -------
        None

        Notes
        -----
        After calling this function you should call plt.show() to get the
        results. This function only adds the annotations, it doesn't show
        them.
        """
        import matplotlib.pyplot as plt  # Make sure we have pyplot as plt

        for label, x, y in zip(frame[label_col], frame[x], frame[y]):
            plt.annotate(label, xy=(x + 0.2, y + 0.15), **kwargs)

    def export_vtk(self):
        """
        export vtk
        :return:
        """

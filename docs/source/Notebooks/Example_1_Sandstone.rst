
Example 1: Sandstone Model
==========================

.. code:: python

    # Importing
    import theano.tensor as T
    import sys, os
    sys.path.append("../GeMpy")
    
    # Importing GeMpy modules
    import GeMpy
    
    # Reloading (only for development purposes)
    import importlib
    importlib.reload(GeMpy)
    
    # Usuful packages
    import numpy as np
    import pandas as pn
    
    import matplotlib.pyplot as plt
    
    # This was to choose the gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Default options of printin
    np.set_printoptions(precision = 6, linewidth= 130, suppress =  True)
    
    %matplotlib inline
    #%matplotlib notebook

First we make a GeMpy instance with most of the parameters default
(except range that is given by the project). Then we also fix the
extension and the resolution of the domain we want to interpolate.
Finally we compile the function, only needed once every time we open the
project (the guys of theano they are working on letting loading compiled
files, even though in our case it is not a big deal).

*General note. So far the reescaling factor is calculated for all series
at the same time. GeoModeller does it individually for every potential
field. I have to look better what this parameter exactly means*

.. code:: python

    geo_data = GeMpy.import_data([696000,747000,6863000,6950000,-20000, 2000],[ 40, 40, 80],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")

All input data is stored in pandas dataframes under,
``self.Data.Interances`` and ``self.Data.Foliations``:

.. code:: python

    GeMpy.i_set_data(geo_data)

In case of disconformities, we can define which formation belong to
which series using a dictionary. Until python 3.6 is important to
specify the order of the series otherwise is random

.. code:: python

    GeMpy.set_data_series(geo_data, {"EarlyGranite_Series":geo_data.formations[-1], 
                          "BIF_Series":(geo_data.formations[0], geo_data.formations[1]),
                          "SimpleMafic_Series":geo_data.formations[2]}, 
                           order_series = ["EarlyGranite_Series",
                                  "BIF_Series",
                                  "SimpleMafic_Series"], verbose=0)

Now in the data frame we should have the series column too

.. code:: python

    GeMpy.get_raw_data(geo_data).head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th></th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>formation</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th rowspan="5" valign="top">interfaces</th>
          <th>0</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>735484.817806</td>
          <td>6.891936e+06</td>
          <td>-1819.319309</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SimpleMafic2</td>
          <td>NaN</td>
          <td>BIF_Series</td>
        </tr>
        <tr>
          <th>1</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>729854.915982</td>
          <td>6.891938e+06</td>
          <td>-1432.263309</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SimpleMafic2</td>
          <td>NaN</td>
          <td>BIF_Series</td>
        </tr>
        <tr>
          <th>2</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>724084.267161</td>
          <td>6.891939e+06</td>
          <td>-4739.830309</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SimpleMafic2</td>
          <td>NaN</td>
          <td>BIF_Series</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>733521.625000</td>
          <td>6.895282e+06</td>
          <td>521.555240</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SimpleMafic2</td>
          <td>NaN</td>
          <td>BIF_Series</td>
        </tr>
        <tr>
          <th>4</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>721933.375000</td>
          <td>6.884592e+06</td>
          <td>496.669295</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>SimpleMafic2</td>
          <td>NaN</td>
          <td>BIF_Series</td>
        </tr>
      </tbody>
    </table>
    </div>



Next step is the creating of a grid. So far only regular. By default it
takes the extent and the resolution given in the ``import_data`` method.

.. code:: python

    # Create a class Grid so far just regular grid
    GeMpy.set_grid(geo_data)
    GeMpy.get_grid(geo_data)




.. parsed-literal::

    array([[  696000.      ,  6863000.      ,   -20000.      ],
           [  696000.      ,  6863000.      ,   -19721.519531],
           [  696000.      ,  6863000.      ,   -19443.037109],
           ..., 
           [  747000.      ,  6950000.      ,     1443.037964],
           [  747000.      ,  6950000.      ,     1721.519043],
           [  747000.      ,  6950000.      ,     2000.      ]], dtype=float32)



Plotting raw data
-----------------

The object Plot is created automatically as we call the methods above.
This object contains some methods to plot the data and the results.

It is possible to plot a 2D projection of the data in a specific
direction using the following method. Also is possible to choose the
series you want to plot. Additionally all the key arguments of seaborn
lmplot can be used.

.. code:: python

    GeMpy.plot_data(geo_data, 'y', geo_data.series.columns.values[1])




.. parsed-literal::

    <Visualization.PlotData at 0x7fd6725dad30>




.. image:: Example_1_Sandstone_files/Example_1_Sandstone_14_1.png


Class Interpolator
------------------

This class will take the data from the class Data and calculate
potential fields and block. We can pass as key arguments all the
variables of the interpolation. I recommend not to touch them if you do
not know what are you doing. The default values should be good enough.
Also the first time we execute the method, we will compile the theano
function so it can take a bit of time.

.. code:: python

    GeMpy.set_interpolator(geo_data)

Now we could visualize the individual potential fields as follow:

Early granite
~~~~~~~~~~~~~

.. code:: python

    GeMpy.plot_potential_field(geo_data,10, n_pf=0)



.. image:: Example_1_Sandstone_files/Example_1_Sandstone_20_0.png


BIF Series
~~~~~~~~~~

.. code:: python

    GeMpy.plot_potential_field(geo_data,13, n_pf=1, cmap = "magma",  plot_data = True,
                                            verbose = 5)



.. image:: Example_1_Sandstone_files/Example_1_Sandstone_22_0.png


SImple mafic
~~~~~~~~~~~~

.. code:: python

    GeMpy.plot_potential_field(geo_data, 10, n_pf=2)



.. image:: Example_1_Sandstone_files/Example_1_Sandstone_24_0.png


Optimizing the export of lithologies
------------------------------------

But usually the final result we want to get is the final block. The
method ``compute_block_model`` will compute the block model, updating
the attribute ``block``. This attribute is a theano shared function that
can return a 3D array (raveled) using the method ``get_value()``.

.. code:: python

    GeMpy.compute_block_model(geo_data)




.. parsed-literal::

    Final block computed



And again after computing the model in the Plot object we can use the
method ``plot_block_section`` to see a 2D section of the model

.. code:: python

    GeMpy.plot_section(geo_data, 13, direction='y')




.. parsed-literal::

    <Visualization.PlotData at 0x7fd65d7e6d30>




.. image:: Example_1_Sandstone_files/Example_1_Sandstone_28_1.png


Export to vtk. (*Under development*)
------------------------------------

.. code:: python

    """Export model to VTK
    
    Export the geology blocks to VTK for visualisation of the entire 3-D model in an
    external VTK viewer, e.g. Paraview.
    
    ..Note:: Requires pyevtk, available for free on: https://github.com/firedrakeproject/firedrake/tree/master/python/evtk
    
    **Optional keywords**:
        - *vtk_filename* = string : filename of VTK file (default: output_name)
        - *data* = np.array : data array to export to VKT (default: entire block model)
    """
    vtk_filename = "noddyFunct2"
    
    extent_x = 10
    extent_y = 10
    extent_z = 10
    
    delx = 0.2
    dely = 0.2
    delz = 0.2
    from pyevtk.hl import gridToVTK
    # Coordinates
    x = np.arange(0, extent_x + 0.1*delx, delx, dtype='float64')
    y = np.arange(0, extent_y + 0.1*dely, dely, dtype='float64')
    z = np.arange(0, extent_z + 0.1*delz, delz, dtype='float64')
    
    # self.block = np.swapaxes(self.block, 0, 2)
    gridToVTK(vtk_filename, x, y, z, cellData = {"geology" : sol})


::


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-16-ff637538da86> in <module>()
         26 
         27 # self.block = np.swapaxes(self.block, 0, 2)
    ---> 28 gridToVTK(vtk_filename, x, y, z, cellData = {"geology" : sol})
    

    NameError: name 'sol' is not defined


Performance Analysis
--------------------

One of the advantages of theano is the posibility to create a full
profile of the function. This has to be included in at the time of the
creation of the function. At the moment it should be active (the
downside is larger compilation time and I think also a bit in the
computation so be careful if you need a fast call)

CPU
~~~

The following profile is with a 2 core laptop. Nothing spectacular.

.. code:: python

    %%timeit
    # Compute the block
    GeMpy.compute_block_model([0,1,2], verbose = 0)

Looking at the profile we can see that most of time is in pow operation
(exponential). This probably is that the extent is huge and we are doing
it with too much precision. I am working on it

.. code:: python

    geo_data.interpolator._interpolate.profile.summary()

GPU
~~~

.. code:: python

    %%timeit
    
    # Compute the block
    GeMpy.compute_block_model([0,1,2], verbose = 0)

.. code:: python

    geo_data.interpolator._interpolate.profile.summary()


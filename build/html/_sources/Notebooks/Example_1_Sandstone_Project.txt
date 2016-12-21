
Example 1: Sandstone Model
==========================

.. code:: python

    # Importing
    import theano.tensor as T
    import sys, os
    sys.path.append("../GeMpy")
    
    # Importing GeMpy modules
    import GeMpy_core
    import Visualization
    
    # Reloading (only for development purposes)
    import importlib
    importlib.reload(GeMpy_core)
    importlib.reload(Visualization)
    
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

    # Setting extend, grid and compile
    # Setting the extent
    sandstone = GeMpy_core.GeMpy()
    
    # Create Data class with raw data
    sandstone.import_data( [696000,747000,6863000,6950000,-20000, 2000],[ 40, 40, 80],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")

All input data is stored in pandas dataframes under,
``self.Data.Interances`` and ``self.Data.Foliations``:

.. code:: python

    sandstone.Data.Foliations.head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>polarity</th>
          <th>formation</th>
          <th>series</th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>739426.627684</td>
          <td>6.891935e+06</td>
          <td>75.422691</td>
          <td>220.000</td>
          <td>70.0</td>
          <td>1</td>
          <td>SimpleBIF</td>
          <td>Default serie</td>
          <td>-0.604023</td>
          <td>-7.198463e-01</td>
          <td>0.342020</td>
        </tr>
        <tr>
          <th>1</th>
          <td>717311.112372</td>
          <td>6.891941e+06</td>
          <td>-1497.488309</td>
          <td>90.000</td>
          <td>60.0</td>
          <td>1</td>
          <td>SimpleBIF</td>
          <td>Default serie</td>
          <td>0.866025</td>
          <td>5.302876e-17</td>
          <td>0.500000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>723415.321182</td>
          <td>6.891939e+06</td>
          <td>-5298.154309</td>
          <td>10.000</td>
          <td>20.0</td>
          <td>1</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>0.059391</td>
          <td>3.368241e-01</td>
          <td>0.939693</td>
        </tr>
        <tr>
          <th>3</th>
          <td>742907.686575</td>
          <td>6.891935e+06</td>
          <td>-2786.869309</td>
          <td>250.000</td>
          <td>60.0</td>
          <td>1</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>-0.813798</td>
          <td>-2.961981e-01</td>
          <td>0.500000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>712584.536312</td>
          <td>6.891942e+06</td>
          <td>-582.769334</td>
          <td>90.014</td>
          <td>60.0</td>
          <td>1</td>
          <td>SimpleMafic1</td>
          <td>Default serie</td>
          <td>0.866025</td>
          <td>-2.116099e-04</td>
          <td>0.500000</td>
        </tr>
      </tbody>
    </table>
    </div>



In case of disconformities, we can define which formation belong to
which series using a dictionary. Until python 3.6 is important to
specify the order of the series otherwise is random

.. code:: python

    sandstone.Data.set_series({"EarlyGranite_Series":sandstone.Data.formations[-1], 
                          "BIF_Series":(sandstone.Data.formations[0], sandstone.Data.formations[1]),
                          "SimpleMafic_Series":sandstone.Data.formations[2]}, 
                           order = ["EarlyGranite_Series",
                                  "BIF_Series",
                                  "SimpleMafic_Series"]) 




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>EarlyGranite_Series</th>
          <th>BIF_Series</th>
          <th>SimpleMafic_Series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>EarlyGranite</td>
          <td>SimpleMafic2</td>
          <td>SimpleMafic1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>EarlyGranite</td>
          <td>SimpleBIF</td>
          <td>SimpleMafic1</td>
        </tr>
      </tbody>
    </table>
    </div>



Now in the data frame we should have the series column too

.. code:: python

    sandstone.Data.Foliations.head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>azimuth</th>
          <th>dip</th>
          <th>polarity</th>
          <th>formation</th>
          <th>series</th>
          <th>G_x</th>
          <th>G_y</th>
          <th>G_z</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>739426.627684</td>
          <td>6.891935e+06</td>
          <td>75.422691</td>
          <td>220.000</td>
          <td>70.0</td>
          <td>1</td>
          <td>SimpleBIF</td>
          <td>BIF_Series</td>
          <td>-0.604023</td>
          <td>-7.198463e-01</td>
          <td>0.342020</td>
        </tr>
        <tr>
          <th>1</th>
          <td>717311.112372</td>
          <td>6.891941e+06</td>
          <td>-1497.488309</td>
          <td>90.000</td>
          <td>60.0</td>
          <td>1</td>
          <td>SimpleBIF</td>
          <td>BIF_Series</td>
          <td>0.866025</td>
          <td>5.302876e-17</td>
          <td>0.500000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>723415.321182</td>
          <td>6.891939e+06</td>
          <td>-5298.154309</td>
          <td>10.000</td>
          <td>20.0</td>
          <td>1</td>
          <td>SimpleMafic1</td>
          <td>SimpleMafic_Series</td>
          <td>0.059391</td>
          <td>3.368241e-01</td>
          <td>0.939693</td>
        </tr>
        <tr>
          <th>3</th>
          <td>742907.686575</td>
          <td>6.891935e+06</td>
          <td>-2786.869309</td>
          <td>250.000</td>
          <td>60.0</td>
          <td>1</td>
          <td>SimpleMafic1</td>
          <td>SimpleMafic_Series</td>
          <td>-0.813798</td>
          <td>-2.961981e-01</td>
          <td>0.500000</td>
        </tr>
        <tr>
          <th>4</th>
          <td>712584.536312</td>
          <td>6.891942e+06</td>
          <td>-582.769334</td>
          <td>90.014</td>
          <td>60.0</td>
          <td>1</td>
          <td>SimpleMafic1</td>
          <td>SimpleMafic_Series</td>
          <td>0.866025</td>
          <td>-2.116099e-04</td>
          <td>0.500000</td>
        </tr>
      </tbody>
    </table>
    </div>



Next step is the creating of a grid. So far only regular. By default it
takes the extent and the resolution given in the ``import_data`` method.

.. code:: python

    # Create a class Grid so far just regular grid
    sandstone.create_grid()
    sandstone.Grid.grid




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

    sandstone.Plot.plot_data(series = sandstone.Data.series.columns.values[1])



.. image:: Example_1_Sandstone_Project_files/Example_1_Sandstone_Project_14_0.png


Class Interpolator
------------------

This class will take the data from the class Data and calculate
potential fields and block. We can pass as key arguments all the
variables of the interpolation. I recommend not to touch them if you do
not know what are you doing. The default values should be good enough.
Also the first time we execute the method, we will compile the theano
function so it can take a bit of time.

.. code:: python

    sandstone.set_interpolator()

Now we could visualize the individual potential fields as follow:

Early granite
~~~~~~~~~~~~~

.. code:: python

    sandstone.Plot.plot_potential_field(10, n_pf=0)



.. image:: Example_1_Sandstone_Project_files/Example_1_Sandstone_Project_20_0.png


BIF Series
~~~~~~~~~~

.. code:: python

    sandstone.Plot.plot_potential_field(13, n_pf=1, cmap = "magma",  plot_data = True,
                                            verbose = 5 )



.. image:: Example_1_Sandstone_Project_files/Example_1_Sandstone_Project_22_0.png


SImple mafic
~~~~~~~~~~~~

.. code:: python

    sandstone.Plot.plot_potential_field(10, n_pf=2)



.. image:: Example_1_Sandstone_Project_files/Example_1_Sandstone_Project_24_0.png


Optimizing the export of lithologies
------------------------------------

But usually the final result we want to get is the final block. The
method ``compute_block_model`` will compute the block model, updating
the attribute ``block``. This attribute is a theano shared function that
can return a 3D array (raveled) using the method ``get_value()``.

.. code:: python

    # Reset the block
    sandstone.Interpolator.block.set_value(np.zeros_like(sandstone.Grid.grid[:,0]))
    
    # Compute the block
    sandstone.Interpolator.compute_block_model([0,1,2], verbose = 0)

.. code:: python

    sandstone.Interpolator.block.get_value(), np.unique(sandstone.Interpolator.block.get_value())




.. parsed-literal::

    (array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32),
     array([ 0.,  1.,  2.,  3.,  4.], dtype=float32))



And again after computing the model in the Plot object we can use the
method ``plot_block_section`` to see a 2D section of the model

.. code:: python

    sandstone.Plot.plot_block_section(13, interpolation = 'nearest',  direction='y')
    plt.savefig("sandstone_example.png")



.. image:: Example_1_Sandstone_Project_files/Example_1_Sandstone_Project_29_0.png


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
    # Reset the block
    sandstone.Interpolator.block.set_value(np.zeros_like(sandstone.Grid.grid[:,0]))
    
    # Compute the block
    sandstone.Interpolator.compute_block_model([0,1,2], verbose = 0)

Looking at the profile we can see that most of time is in pow operation
(exponential). This probably is that the extent is huge and we are doing
it with too much precision. I am working on it

.. code:: python

    esandstone.Interpolator._interpolate.profile.summary()

GPU
~~~

.. code:: python

    %%timeit
    # Reset the block
    sandstone.block.set_value(np.zeros_like(sandstone.grid[:,0]))
    
    # Compute the block
    sandstone.compute_block_model([0,1,2], verbose = 0)

.. code:: python

    sandstone.block_export.profile.summary()


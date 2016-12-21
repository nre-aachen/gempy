
Simple model
============

This notebook is a series of independent cells showing how to create a
simple model from the beginning to the end using GeMpy

Importing dependencies
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Importing
    import importlib
    import theano.tensor as T
    import sys, os
    sys.path.append("/home/bl3/PycharmProjects/GeMpy/GeMpy")
    
    import pandas as pn
    
    import GeMpy_core
    import Visualization
    import numpy as np
    importlib.reload(GeMpy_core)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.set_printoptions(precision = 4, linewidth= 300, suppress =  True)
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('pdf', 'png')
    plt.rcParams['savefig.dpi'] = 75
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.figsize'] = 10, 6
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 14
    
    import seaborn as sns
    
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    
    # Set the font to be serif, rather than sans
    sns.set(font='Arial')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white")
    
    import qgrid
    qgrid.nbinstall(overwrite=True)  # copies javascript dependencies to your /nbextensions folder
    
    %matplotlib inline

Visualize data
--------------

.. code:: python

    # Setting extend, grid and compile
    # Setting the extent
    test = GeMpy_core.GeMpy()
    test.import_data([0,10,0,10,0,10])
    
    # =========================
    # DATA GENERATION IN PYTHON
    # =========================
    # Layers coordinates
    layer_1 = np.array([[0.5,4,7], [2,4,6.5], [4,4,7], [5,4,6]])#-np.array([5,5,4]))/8+0.5
    layer_2 = np.array([[3,4,5], [6,4,4],[8,4,4], [7,4,3], [1,4,6]])
    layers = np.asarray([layer_1,layer_2])
    
    # Foliations coordinates
    dip_pos_1 = np.array([7,4,7])#- np.array([5,5,4]))/8+0.5
    dip_pos_2 = np.array([2.,4,4])
    
    # Dips
    dip_angle_1 = float(15)
    dip_angle_2 = float(340)
    dips_angles = np.asarray([dip_angle_1, dip_angle_2], dtype="float64")
    
    # Azimuths
    azimuths = np.asarray([90,90], dtype="float64")
    
    # Polarity
    polarity = np.asarray([1,1], dtype="float64")
    
    # Pandas Dataframe with the interfaces data
    test.Data.Interfaces = pn.DataFrame(
        data = {"X" :np.append(layer_1[:, 0],layer_2[:,0]),
                "Y" :np.append(layer_1[:, 1],layer_2[:,1]),
                "Z" :np.append(layer_1[:, 2],layer_2[:,2]),
                "formation" : np.append(
                   np.tile("Layer 1", len(layer_1)), 
                   np.tile("Layer 2", len(layer_2)))})
    
    # Pandas Dataframe with the Foliations data
    test.Data.Foliations =   pn.DataFrame(
        data = {"X" :np.append(dip_pos_1[0],dip_pos_2[0]),
                "Y" :np.append(dip_pos_1[ 1],dip_pos_2[1]),
                "Z" :np.append(dip_pos_1[ 2],dip_pos_2[2]),
                "azimuth" : azimuths,
                "dip" : dips_angles,
                "polarity" : polarity,
                "formation" : ["Layer 1", "Layer 2"]})
    
    # Creation of the formations (to be deprecated)
    test.Data.formations = test.Data.Interfaces["formation"].unique()
    
    # Calculation of the pola gradient from dip azimuth and polarity
    test.Data.calculate_gradient()
    
    # Set defautl series
    test.Data.set_series()
    
    # Method to be sure all objects get updated
    test.update_data()
    
    
    # The following code is for the visualization of labels of the input data (yet to be implemented)↔
    # ----------------------------------------------------------------------------------------------
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
            plt.annotate(label, xy=(x+0.2, y+0.15), **kwargs)
            
    inter_labels =[r'${\bf{x}}_{\alpha \, 0}^1$',
                   r'${\bf{x}}_{\alpha \, 1}^1$',
                   r'${\bf{x}}_{\alpha \, 2}^1$',
                   r'${\bf{x}}_{\alpha \, 3}^1$',
                   r'${\bf{x}}_{\alpha \, 0}^2$',
                   r'${\bf{x}}_{\alpha \, 1}^2$',
                   r'${\bf{x}}_{\alpha \, 2}^2$',
                   r'${\bf{x}}_{\alpha \, 3}^2$',
                   r'${\bf{x}}_{\alpha \, 4}^2$']              
            
    foli_labels =[r'${\bf{x}}_{\beta \,{0}}$',
                  r'${\bf{x}}_{\beta \,{1}}$']        
    
    test.Data.Interfaces['labels'] = pn.Series(inter_labels)
    test.Data.Foliations['labels'] = pn.Series(foli_labels)
    
    # Plot and table
    test.Plot.plot_data()
    annotate_plot(test.Data.Interfaces, 'labels','X', 'Z', size = 'x-large')
    annotate_plot(test.Data.Foliations, 'labels','X', 'Z', size = 'x-large')
    
    
    test.Data.Interfaces




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>X</th>
          <th>Y</th>
          <th>Z</th>
          <th>formation</th>
          <th>series</th>
          <th>labels</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.5</td>
          <td>4.0</td>
          <td>7.0</td>
          <td>Layer 1</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 0}^1$</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.0</td>
          <td>4.0</td>
          <td>6.5</td>
          <td>Layer 1</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 1}^1$</td>
        </tr>
        <tr>
          <th>2</th>
          <td>4.0</td>
          <td>4.0</td>
          <td>7.0</td>
          <td>Layer 1</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 2}^1$</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
          <td>4.0</td>
          <td>6.0</td>
          <td>Layer 1</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 3}^1$</td>
        </tr>
        <tr>
          <th>4</th>
          <td>3.0</td>
          <td>4.0</td>
          <td>5.0</td>
          <td>Layer 2</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 0}^2$</td>
        </tr>
        <tr>
          <th>5</th>
          <td>6.0</td>
          <td>4.0</td>
          <td>4.0</td>
          <td>Layer 2</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 1}^2$</td>
        </tr>
        <tr>
          <th>6</th>
          <td>8.0</td>
          <td>4.0</td>
          <td>4.0</td>
          <td>Layer 2</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 2}^2$</td>
        </tr>
        <tr>
          <th>7</th>
          <td>7.0</td>
          <td>4.0</td>
          <td>3.0</td>
          <td>Layer 2</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 3}^2$</td>
        </tr>
        <tr>
          <th>8</th>
          <td>1.0</td>
          <td>4.0</td>
          <td>6.0</td>
          <td>Layer 2</td>
          <td>Default serie</td>
          <td>${\bf{x}}_{\alpha \, 4}^2$</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: Example_2_Simple_files/Example_2_Simple_5_1.png


Interactive pandas Dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using qgrid it is possible to modify the tables in place as following:

.. code:: python

    import qgrid
    qgrid.show_grid(test.Data.Interfaces)


.. parsed-literal::

    Widget Javascript not detected.  It may not be installed properly. Did you enable the widgetsnbextension? If not, then run "jupyter nbextension enable --py --sys-prefix widgetsnbextension"


Grid and potential field
------------------------

We can see the potential field generated out of the data above

.. code:: python

    test.create_grid()
    test.set_interpolator(u_grade=0, verbose = 0)
    test.Plot.plot_potential_field(4, direction='y', colorbar = True, cmap = 'magma')
    #test.Interpolator.DK;



.. image:: Example_2_Simple_files/Example_2_Simple_9_0.png


.. code:: python

    from ipywidgets import widgets
    from ipywidgets import interact
    def cov_cubic_f(r,a = 6, c_o = 1):↔
    def cov_cubic_d1_f(r,a = 6., c_o = 1):↔
    
    def cov_cubic_d2_f(r, a = 6, c_o = 1):↔
    
    def plot_potential_var(a = 10, c_o = 1, nugget_effect = 0):↔
    
    def plot_potential_direction_var( a = 10, c_o = 1, nugget_effect = 0):↔
    
    def plot_directionU_directionU_var(a = 10, c_o = 1, nugget_effect = 0):↔
    
        plt.title("Variogram")
        plt.margins(0,0.1)
        
        ax2 = fig.add_subplot(122)
        ax2.plot(x,np.asarray(y))
        ax2.scatter(0,nugget_effect+y[0], s = 20)
    
        plt.title("Covariance Function")
        plt.tight_layout()
        plt.margins(0,0.1)
        plt.suptitle('$C_{\partial {Z}/ \partial x, \, \partial {Z}/ \partial x}(h_x)$'
                     , y = 1.08, fontsize=15)
        
    def plot_directionU_directionV_var(a = 10, c_o = 1, nugget_effect = 0):↔
    
        plt.title("Variogram")
        plt.margins(0,0.1)
        
        ax2 = fig.add_subplot(122)
        ax2.plot(x,np.asarray(y))
        ax2.scatter(0,nugget_effect+y[0], s = 20)
        plt.title("Covariance Function")
        plt.tight_layout()
        plt.margins(0,0.1)
        plt.suptitle('$C_{\partial {Z}/ \partial x, \, \partial {Z}/ \partial y}(h_x,h_y)$'
                     , y = 1.08, fontsize=15)
        
    
    def plot_all(a = 10, c_o = 1, nugget_effect = 0):↔


From potential field to block
-----------------------------

The potential field describe the deposition form and direction of a
basin. However, in most scenarios the real goal of structural modeling
is the segmentation in layers of areas with significant change of
properties (e.g. shales and carbonates). Since we need to provide at
least one point per interface, we can easily compute the value of the
potential field at the intersections between two layers. Therefore, by
simple comparison between a concrete value of the potential field and
the values of the interfaces it is possible to segment the domain into
layers Fig X.

.. code:: python

    test.Interpolator.compute_block_model()
    test.Plot.plot_block_section()



.. image:: Example_2_Simple_files/Example_2_Simple_13_0.png


Combining potential fields: Depositional series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In reality, most geological settings are formed by a concatenation of
depositional phases separated clearly by unconformity bounderies. Each
of these phases can be model by a potential field. In order to capture
this behavior, we can classify the formations that belong to individual
depositional phase into categories or series. The potential field
computed for each of these series could be seen as a sort of evolution
of the basin if an unconformity did not occur. Finally, sorting the
temporal relation between series allow to superpose the corresponding
potential field at an specific location.

In the next example, we add a new serie consisting in a layer---'Layer
3'--- Fig X, which generate the potential field of Fig X and
subsequently the block Figure X.

.. code:: python

    # Setting extend, grid and compile
    # Setting the extent
    two_pot = GeMpy_core.GeMpy()
    two_pot.import_data([0,10,0,10,0,10])
    
    
    # Data Generation
    layer_1 = np.array([[0.5,4,7], [2,4,6.5], [4,4,7], [5,4,6]])#-np.array([5,5,4]))/8+0.5
    layer_2 = np.array([[3,4,5], [6,4,4],[8,4,4], [7,4,3], [1,4,6]])
    layer_3 = np.array([[2,4,3], [8,4,2], [9,4,3]])
    
    dip_pos_1 = np.array([7,4,7])#- np.array([5,5,4]))/8+0.5
    dip_pos_2 = np.array([2.,4,4])
    dip_pos_3 = np.array([1,4,1])
    
    dip_angle_1 = float(15)
    dip_angle_2 = float(340)
    dip_angle_3 = float(80)
    
    dips_angles = np.asarray([dip_angle_1, dip_angle_2, dip_angle_3], dtype="float64")
    layers = np.asarray([layer_1,layer_2, layer_3])
    azimuths = np.asarray([90,90, 90], dtype="float64")
    polarity = np.asarray([1,1, 1], dtype="float64")
    
    two_pot.Data.Interfaces = pn.DataFrame(data = {"X" :np.hstack((layer_1[:, 0],layer_2[:,0], layer_3[:,0])),
                                                   "Y" :np.hstack((layer_1[:, 1],layer_2[:,1], layer_3[:,1])),
                                                   "Z" :np.hstack((layer_1[:, 2],layer_2[:,2], layer_3[:,2])),
    "formation" : np.hstack((np.tile("Layer 1", len(layer_1)), np.tile("Layer 2", len(layer_2)),
                            np.tile("Layer 3", len(layer_3))))})
    two_pot.Data.Foliations =   pn.DataFrame(data = {"X" :np.hstack((dip_pos_1[0], dip_pos_2[0], dip_pos_3[0])),
                                                     "Y" :np.hstack((dip_pos_1[ 1],dip_pos_2[1], dip_pos_3[1])),
                                                     "Z" :np.hstack((dip_pos_1[ 2],dip_pos_2[2], dip_pos_3[2])),
                                             "azimuth" : azimuths,
                                             "dip" : dips_angles,
                                             "polarity" : polarity,
    "formation" : ["Layer 1", "Layer 2", 'Layer 3']})
    
    inter_labels =[r'${\bf{x}}_{\alpha \, 0}^1$',
                   r'${\bf{x}}_{\alpha \, 1}^1$',
                   r'${\bf{x}}_{\alpha \, 2}^1$',
                   r'${\bf{x}}_{\alpha \, 3}^1$',
                   r'${\bf{x}}_{\alpha \, 0}^2$',
                   r'${\bf{x}}_{\alpha \, 1}^2$',
                   r'${\bf{x}}_{\alpha \, 2}^2$',
                   r'${\bf{x}}_{\alpha \, 3}^2$',
                   r'${\bf{x}}_{\alpha \, 4}^2$',
                   r'${\bf{x}}_{\alpha \, 0}^1$',
                   r'${\bf{x}}_{\alpha \, 1}^1$',
                   r'${\bf{x}}_{\alpha \, 2}^1$']              
            
    foli_labels =[r'${\bf{x}}_{\beta \,{0}}$',
                  r'${\bf{x}}_{\beta \,{1}}$',
                  r'${\bf{x}}_{\beta \,{0}}$']
    
    two_pot.Data.Interfaces['labels'] = pn.Series(inter_labels)
    two_pot.Data.Foliations['labels'] = pn.Series(foli_labels)
    
    two_pot.Data.formations = two_pot.Data.Interfaces["formation"].unique()
    two_pot.Data.calculate_gradient()
    two_pot.Data.set_series({'younger': ('Layer 1', 'Layer 2'),
                             'older': 'Layer 3'}, order = ['younger', 'older'])
    two_pot.update_data()
    
    
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
            plt.annotate(label, xy=(x+0.2, y+0.15), **kwargs)
    
            
    serie_to_plot = 'older'        
    two_pot.Plot.plot_data(series = serie_to_plot)
    annotate_plot(two_pot.Data.Interfaces[two_pot.Data.Interfaces['series'] == serie_to_plot]
                  , 'labels','X', 'Z', size = 'x-large')
    annotate_plot(two_pot.Data.Foliations[two_pot.Data.Foliations['series'] == serie_to_plot],
                  'labels','X', 'Z', size = 'x-large')
    
    two_pot.create_grid()
    two_pot.set_interpolator(u_grade=0, verbose = 0)



.. image:: Example_2_Simple_files/Example_2_Simple_16_0.png


This potential field gives the following block

.. code:: python

    two_pot.Plot.plot_potential_field(4, n_pf=1, direction='y', colorbar = True, cmap = 'magma')



.. image:: Example_2_Simple_files/Example_2_Simple_18_0.png


.. code:: python

    # Reset the block
    two_pot.Interpolator.block.set_value(np.zeros_like(two_pot.Grid.grid[:,0]))
    
    two_pot.Interpolator.compute_block_model(series_number=[1])
    two_pot.Plot.plot_block_section()



.. image:: Example_2_Simple_files/Example_2_Simple_19_0.png


Combining both potential field where the first potential field is
younger than the second we can obtain the following structure.

.. code:: python

    # Reset the block
    two_pot.Interpolator.block.set_value(np.zeros_like(two_pot.Grid.grid[:,0]))
    
    two_pot.Interpolator.compute_block_model(series_number=[0,1])
    two_pot.Plot.plot_block_section()



.. image:: Example_2_Simple_files/Example_2_Simple_21_0.png


Side note: Example of covariances involved in the cokriging system
------------------------------------------------------------------

.. code:: python

    plot_potential_var(10,10**2 / 14 / 3  , 0.01)
    
    plot_all(10,10**2 / 14 / 3  , 0.01) # 0**2 /14/3



.. image:: Example_2_Simple_files/Example_2_Simple_23_0.png



.. image:: Example_2_Simple_files/Example_2_Simple_23_1.png



.. image:: Example_2_Simple_files/Example_2_Simple_23_2.png



.. image:: Example_2_Simple_files/Example_2_Simple_23_3.png


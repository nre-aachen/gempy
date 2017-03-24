
Example 2: Implementing GeMpy into PyMC3
========================================

Generating data
~~~~~~~~~~~~~~~

.. code:: ipython3

    # Importing and data
    import theano.tensor as T
    import theano
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
    
    #%matplotlib inline
    %matplotlib inline
    
    
    
    # Setting the extent
    geo_data = GeMpy.import_data([0,10,0,10,0,10], [50,50,50])
    
    
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
    
    # Setting foliations and interfaces values
    GeMpy.set_interfaces(geo_data, pn.DataFrame(
        data = {"X" :np.append(layer_1[:, 0],layer_2[:,0]),
                "Y" :np.append(layer_1[:, 1],layer_2[:,1]),
                "Z" :np.append(layer_1[:, 2],layer_2[:,2]),
                "formation" : np.append(
                   np.tile("Layer 1", len(layer_1)), 
                   np.tile("Layer 2", len(layer_2))),
                "labels" : [r'${\bf{x}}_{\alpha \, 0}^1$',
                   r'${\bf{x}}_{\alpha \, 1}^1$',
                   r'${\bf{x}}_{\alpha \, 2}^1$',
                   r'${\bf{x}}_{\alpha \, 3}^1$',
                   r'${\bf{x}}_{\alpha \, 0}^2$',
                   r'${\bf{x}}_{\alpha \, 1}^2$',
                   r'${\bf{x}}_{\alpha \, 2}^2$',
                   r'${\bf{x}}_{\alpha \, 3}^2$',
                   r'${\bf{x}}_{\alpha \, 4}^2$'] }))
    
    GeMpy.set_foliations(geo_data,  pn.DataFrame(
        data = {"X" :np.append(dip_pos_1[0],dip_pos_2[0]),
                "Y" :np.append(dip_pos_1[ 1],dip_pos_2[1]),
                "Z" :np.append(dip_pos_1[ 2],dip_pos_2[2]),
                "azimuth" : azimuths,
                "dip" : dips_angles,
                "polarity" : polarity,
                "formation" : ["Layer 1", "Layer 2"],
                "labels" : [r'${\bf{x}}_{\beta \,{0}}$',
                  r'${\bf{x}}_{\beta \,{1}}$'] })) 
    
    
    
    layer_3 = np.array([[2,4,3], [8,4,2], [9,4,3]])
    dip_pos_3 = np.array([1,4,1])
    dip_angle_3 = float(80)
    azimuth_3 = 90
    polarity_3 = 1
    
    
    
    GeMpy.set_interfaces(geo_data, pn.DataFrame(
        data = {"X" :layer_3[:, 0],
                "Y" :layer_3[:, 1],
                "Z" :layer_3[:, 2],
                "formation" : np.tile("Layer 3", len(layer_3)), 
                "labels" : [  r'${\bf{x}}_{\alpha \, 0}^3$',
                               r'${\bf{x}}_{\alpha \, 1}^3$',
                               r'${\bf{x}}_{\alpha \, 2}^3$'] }), append = True)
    GeMpy.get_raw_data(geo_data,"interfaces")
    
    
    GeMpy.set_foliations(geo_data, pn.DataFrame(data = {
                         "X" : dip_pos_3[0],
                         "Y" : dip_pos_3[1],
                         "Z" : dip_pos_3[2],
                
                         "azimuth" : azimuth_3,
                         "dip" : dip_angle_3,
                         "polarity" : polarity_3,
                         "formation" : [ 'Layer 3'],
                         "labels" : r'${\bf{x}}_{\beta \,{2}}$'}), append = True)
    
    
    GeMpy.set_data_series(geo_data, {'younger': ('Layer 1', 'Layer 2'),
                          'older': 'Layer 3'}, order_series = ['younger', 'older'])
    
    


.. code:: ipython3

    # Select series to interpolate (if you do not want to interpolate all)
    new_series = GeMpy.select_series(geo_data, ['younger'])
    data_interp = GeMpy.set_interpolator(new_series, u_grade = 0)

.. code:: ipython3

    geo_data




.. parsed-literal::

    <DataManagement.DataManagement at 0x7ff2c4035780>



.. code:: ipython3

    # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
    input_data_T = data_interp.interpolator.tg.input_parameters_list()
    debugging = theano.function(input_data_T, data_interp.interpolator.tg.potential_field_at_all(), on_unused_input='ignore',
                                allow_input_downcast=True, profile=True)

.. code:: ipython3

    # This prepares the user data to the theano function
    input_data_P = data_interp.interpolator.data_prep() 
    
    # Solution of theano
    sol = debugging(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3],input_data_P[4], input_data_P[5])

.. code:: ipython3

    sol.shape




.. parsed-literal::

    (125014,)



.. code:: ipython3

    GeMpy.plot_potential_field(new_series, sol[:-14].reshape(50,50,50),13, plot_data = True)



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_8_0.png


.. code:: ipython3

    # If you change the values here. Here changes the plot as well
    geo_data.foliations.set_value(0, 'dip', 40)




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
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
          <th>labels</th>
          <th>order_series</th>
          <th>polarity</th>
          <th>series</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.258819</td>
          <td>1.584810e-17</td>
          <td>0.965926</td>
          <td>7.0</td>
          <td>4.0</td>
          <td>7.0</td>
          <td>90.0</td>
          <td>40.0</td>
          <td>Layer 1</td>
          <td>${\bf{x}}_{\beta \,{0}}$</td>
          <td>1</td>
          <td>1.0</td>
          <td>younger</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.342020</td>
          <td>-2.094269e-17</td>
          <td>0.939693</td>
          <td>2.0</td>
          <td>4.0</td>
          <td>4.0</td>
          <td>90.0</td>
          <td>340.0</td>
          <td>Layer 2</td>
          <td>${\bf{x}}_{\beta \,{1}}$</td>
          <td>1</td>
          <td>1.0</td>
          <td>younger</td>
        </tr>
        <tr>
          <th>0</th>
          <td>0.984808</td>
          <td>6.030208e-17</td>
          <td>0.173648</td>
          <td>1.0</td>
          <td>4.0</td>
          <td>1.0</td>
          <td>90.0</td>
          <td>40.0</td>
          <td>Layer 3</td>
          <td>${\bf{x}}_{\beta \,{2}}$</td>
          <td>2</td>
          <td>1.0</td>
          <td>older</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    # You need to set the interpolator again
    new_series = GeMpy.select_series(geo_data, ['younger'])
    data_interp = GeMpy.set_interpolator(new_series, u_grade = 0, verbose= ['cov_function'])


.. code:: ipython3

    # If you change it here is not necesary. Maybe some function in GeMpy with an attribute to choose would be good
    data_interp.interpolator._data_scaled.foliations.set_value(0, 'dip', 40)
    # In any case, data prep has to be called to convert the data to pure arrays. This function should be hidden I guess
    input_data_P = data_interp.interpolator.data_prep()

.. code:: ipython3

    sol = debugging(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3],input_data_P[4], input_data_P[5])

.. code:: ipython3

    GeMpy.plot_potential_field(new_series, sol[:-14].reshape(50,50,50),13, plot_data = True)



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_13_0.png


PyMC3
-----

.. code:: ipython3

    data_interp = GeMpy.set_interpolator(geo_data, u_grade = 0)
    
    # This are the shared parameters and the compilation of the function. This will be hidden as well at some point
    input_data_T = data_interp.interpolator.tg.input_parameters_list()
    # This prepares the user data to the theano function
    input_data_P = data_interp.interpolator.data_prep() 

.. code:: ipython3

    # We create the op. Because is an op we cannot call it with python variables anymore. Thats why we have to make them shared
    # Before
    op2 = theano.OpFromGraph(input_data_T, [data_interp.interpolator.tg.whole_block_model()], on_unused_input='ignore')

.. code:: ipython3

    import pymc3 as pm
    theano.config.compute_test_value = 'ignore'
    model = pm.Model()
    with model:
        # Stochastic value
        foliation = pm.Normal('foliation', 40, sd=10)
        
        # We convert a python variable to theano.shared
        dips = theano.shared(input_data_P[1])
        
        # We add the stochastic value to the correspondant array
        dips = T.set_subtensor(dips[0], foliation)
    
        geo_model = pm.Deterministic('GeMpy', op2(theano.shared(input_data_P[0]), dips, 
                                         theano.shared(input_data_P[2]), theano.shared(input_data_P[3]),
                                         theano.shared(input_data_P[4]), theano.shared(input_data_P[5])))
    
        trace = pm.sample(6)


.. parsed-literal::

    Auto-assigning NUTS sampler...
    Initializing NUTS using advi...
    Average ELBO = -0.012037: 100%|██████████| 200000/200000 [00:07<00:00, 25793.75it/s] 
    Finished [100%]: Average ELBO = -0.0012071
    100%|██████████| 6/6 [00:00<00:00, 18.53it/s]


.. code:: ipython3

    trace.varnames, trace.get_values("GeMpy")




.. parsed-literal::

    (['foliation', 'GeMpy'], array([[0, 0, 0, ..., 1, 1, 1],
            [0, 0, 0, ..., 1, 1, 1],
            [0, 0, 0, ..., 1, 1, 1],
            [0, 0, 0, ..., 1, 1, 1],
            [0, 0, 0, ..., 1, 1, 1],
            [0, 0, 0, ..., 1, 1, 1]]))



.. code:: ipython3

    for i in trace.get_values('GeMpy'):
        GeMpy.plot_section(new_series, 13, block = i, plot_data = False)
        plt.show()



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_19_0.png



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_19_1.png



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_19_2.png



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_19_3.png



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_19_4.png



.. image:: Example_2_Simple_PyMC3_files/Example_2_Simple_PyMC3_19_5.png


.. code:: ipython3

    import ipyvolume.pylab as p3
    import ipyvolume.serialize
    ipyvolume.serialize.performance = 1 # 1 for binary, 0 for JSON
    #p3 = ipyvolume.pylab.figure(width=200,height=600)

.. code:: ipython3

    lith0 = trace['GeMpy'][0] == 0
    lith1 = trace['GeMpy'][0] == 1
    lith2 = trace['GeMpy'][0] == 2
    lith3 = trace['GeMpy'][0] == 3
    p3.figure(width=800)
    
    p3.scatter(geo_data.grid.grid[:,0][lith0],
               geo_data.grid.grid[:,1][lith0],
               geo_data.grid.grid[:,2][lith0], marker='box', color = 'blue' )
    
    p3.scatter(geo_data.grid.grid[:,0][lith1],
               geo_data.grid.grid[:,1][lith1],
               geo_data.grid.grid[:,2][lith1], marker='box', color = 'yellow', size = 1 )
    
    p3.scatter(geo_data.grid.grid[:,0][lith2],
               geo_data.grid.grid[:,1][lith2],
               geo_data.grid.grid[:,2][lith2], marker='box', color = 'green' )
    
    p3.scatter(geo_data.grid.grid[:,0][lith3],
               geo_data.grid.grid[:,1][lith3],
               geo_data.grid.grid[:,2][lith3], marker='box', color = 'red' )
    
    p3.show()




Cholesky (Under development)
----------------------------

.. code:: ipython3

    # Cholesky solution
    L = np.linalg.cholesky(C)
    U = sc.linalg.cholesky(C)
    Y = sc.linalg.solve_triangular(L,b, lower=True)
    x = sc.linalg.solve_triangular(L.conj().T, Y)


::


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-18-c22109665cca> in <module>()
          1 # Cholesky solution
    ----> 2 L = np.linalg.cholesky(C)
          3 U = sc.linalg.cholesky(C)
          4 Y = sc.linalg.solve_triangular(L,b, lower=True)
          5 x = sc.linalg.solve_triangular(L.conj().T, Y)


    NameError: name 'C' is not defined


.. code:: ipython3

    import scipy as sc
    Y = sc.linalg.solve_triangular?


.. code:: ipython3

    debugging.profile.summary()


.. code:: ipython3

    data_interp.interpolator.tg.dips_position_all.set_value(input_data_P[0])
    data_interp.interpolator.tg.dip_angles_all.set_value(input_data_P[1])
    data_interp.interpolator.tg.azimuth_all.set_value(input_data_P[2])
    data_interp.interpolator.tg.polarity_all.set_value(input_data_P[3])
    data_interp.interpolator.tg.ref_layer_points_all.set_value(input_data_P[4])
    data_interp.interpolator.tg.rest_layer_points_all.set_value(input_data_P[5])





Example 1: Sandstone Model
==========================

.. code:: ipython3

    # Importing
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

.. code:: ipython3

    # Importing the data from csv files and settign extent and resolution
    geo_data = GeMpy.import_data([696000,747000,6863000,6950000,-20000, 2000],[ 40, 40, 40],
                             path_f = os.pardir+"/input_data/a_Foliations.csv",
                             path_i = os.pardir+"/input_data/a_Points.csv")
    
    # Assigning series to formations as well as their order (timewise)
    GeMpy.set_data_series(geo_data, {"EarlyGranite_Series":geo_data.formations[-1], 
                          "BIF_Series":(geo_data.formations[0], geo_data.formations[1]),
                          "SimpleMafic_Series":geo_data.formations[2]}, 
                           order_series = ["EarlyGranite_Series",
                                           "BIF_Series",
                                           "SimpleMafic_Series"], verbose=0)

.. code:: ipython3

    # Preprocessing data to interpolate: This rescales the coordinates between 0 and 1 for stability issues. Here we can choose also the drift degree
    # (in new updates I will change it to be possible to change the grade after compilation). From here we can set also the data type of the operations in case
    # you want to use the GPU. Verbose is huge. There is a large list of strings that select what you want to print while the computation.
    data_interp = GeMpy.set_interpolator(geo_data, u_grade = 2, dtype="float32", verbose=[])

.. code:: ipython3

    # This cell will go to the backend
    
    # Set all the theano shared parameters and return the symbolic variables (the input of the theano function)
    input_data_T = data_interp.interpolator.tg.input_parameters_list()
    
    # Prepare the input data (interfaces, foliations data) to call the theano function. Also set a few theano shared variables with the len
    # of formations series and so on
    input_data_P = data_interp.interpolator.data_prep() 
    
    # Compile the theano function.
    debugging = theano.function(input_data_T, data_interp.interpolator.tg.whole_block_model(), on_unused_input='ignore',
                                allow_input_downcast=True, profile=True)

.. code:: ipython3

    # Solve model calling the theano function
    sol = debugging(input_data_P[0], input_data_P[1], input_data_P[2], input_data_P[3],input_data_P[4], input_data_P[5])


.. code:: ipython3

    # Plot the block model. 
    GeMpy.plot_section(geo_data, 13,  block = sol, direction='y', plot_data = False)




.. parsed-literal::

    <Visualization.PlotData at 0x7f0a685fe198>




.. image:: Example_1_Sandstone_files/Example_1_Sandstone_6_1.png


.. code:: ipython3

    p3.figure

.. code:: ipython3

    # So far this is a simple 3D visualization. I have to adapt it into GeMpy 
    
    lith0 = sol == 0
    lith1 = sol == 1
    lith2 = sol == 2
    lith3 = sol == 3
    lith4 = sol == 4
    np.unique(sol)
    
    import ipyvolume.pylab as p3
    
    p3.figure(width=800)
    
    p3.scatter(geo_data.grid.grid[:,0][lith0],
               geo_data.grid.grid[:,1][lith0],
               geo_data.grid.grid[:,2][lith0], marker='box', color = 'blue', size = 0.1 )
    
    p3.scatter(geo_data.grid.grid[:,0][lith1],
               geo_data.grid.grid[:,1][lith1],
               geo_data.grid.grid[:,2][lith1], marker='box', color = 'yellow', size = 1 )
    
    p3.scatter(geo_data.grid.grid[:,0][lith2],
               geo_data.grid.grid[:,1][lith2],
               geo_data.grid.grid[:,2][lith2], marker='box', color = 'green', size = 1 )
    
    p3.scatter(geo_data.grid.grid[:,0][lith3],
               geo_data.grid.grid[:,1][lith3],
               geo_data.grid.grid[:,2][lith3], marker='box', color = 'pink', size = 1 )
    
    p3.scatter(geo_data.grid.grid[:,0][lith4],
               geo_data.grid.grid[:,1][lith4],
               geo_data.grid.grid[:,2][lith4], marker='box', color = 'red', size = 1 )
    
    p3.xlim(np.min(geo_data.grid.grid[:,0]),np.min(geo_data.grid.grid[:,0])+2175.0*40)
    p3.ylim(np.min(geo_data.grid.grid[:,1]),np.max(geo_data.grid.grid[:,1]))
    p3.zlim(np.min(geo_data.grid.grid[:,2]),np.min(geo_data.grid.grid[:,2])+2175.0*40)#np.max(geo_data.grid.grid[:,2]))
    
    p3.show()




.. code:: ipython3

    # The profile at the moment sucks because all what is whithin a scan is not subdivided
    debugging.profile.summary()


.. parsed-literal::

    Function profiling
    ==================
      Message: <ipython-input-6-22dcf15bad61>:3
      Time in 5 calls to Function.__call__: 1.357155e+01s
      Time in Function.fn.__call__: 1.357096e+01s (99.996%)
      Time in thunks: 1.357014e+01s (99.990%)
      Total compile time: 2.592983e+01s
        Number of Apply nodes: 95
        Theano Optimizer time: 1.642699e+01s
           Theano validate time: 3.617525e-02s
        Theano Linker time (includes C, CUDA code generation/compiling): 9.462233e+00s
           Import time 1.913705e-01s
           Node make_thunk time 9.450990e+00s
               Node forall_inplace,cpu,scan_fn}(Elemwise{Maximum}[(0, 0)].0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, IncSubtensor{InplaceSet;:int64:}.0, grade of the universal drift, <TensorType(float64, matrix)>, <TensorType(float64, vector)>, Value of the formation, Position of the dips, Rest of the points of the layers, Reference points for every layer, Angle of every dip, Azimuth, Polarity, InplaceDimShuffle{x,x}.0, InplaceDimShuffle{x,x}.0, Elemwise{Composite{((sqr(sqr(i0)) * sqr(i0)) * i0)}}.0, Elemwise{Composite{(sqr(sqr(i0)) * i0)}}.0, Elemwise{Composite{(sqr(i0) * i0)}}.0, Elemwise{mul,no_inplace}.0, Elemwise{neg,no_inplace}.0, Elemwise{mul,no_inplace}.0, Elemwise{true_div,no_inplace}.0, Elemwise{Mul}[(0, 1)].0, Elemwise{mul,no_inplace}.0, Elemwise{Composite{(i0 * Composite{sqr(sqr(i0))}(i1))}}.0, Elemwise{Composite{(((i0 * i1) / sqr(i2)) + i3)}}.0, Reshape{2}.0) time 9.275085e+00s
               Node Elemwise{Composite{Switch(LT((i0 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), i3), (i4 - i0), Switch(GE((i0 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), (i2 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2))), (i5 + i0), Switch(LE((i2 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), i3), (i5 + i0), i0)))}}(Elemwise{Composite{minimum(minimum(minimum(minimum(minimum(i0, i1), i2), i3), i4), i5)}}.0, TensorConstant{1}, Elemwise{add,no_inplace}.0, TensorConstant{0}, TensorConstant{-2}, TensorConstant{2}) time 3.851414e-03s
               Node Elemwise{Composite{Switch(i0, Switch(LT((i1 + i2), i3), i3, (i1 + i2)), Switch(LT(i1, i2), i1, i2))}}(Elemwise{lt,no_inplace}.0, Elemwise{Composite{minimum(minimum(minimum(minimum(minimum(i0, i1), i2), i3), i4), i5)}}.0, Elemwise{Composite{Switch(LT((i0 + i1), i2), i2, (i0 + i1))}}.0, TensorConstant{0}) time 3.796577e-03s
               Node Elemwise{Composite{minimum(minimum(minimum(minimum(minimum(i0, i1), i2), i3), i4), i5)}}(Elemwise{Composite{Switch(LT((i0 + i1), i2), i2, (i0 + i1))}}.0, Elemwise{sub,no_inplace}.0, Elemwise{Composite{Switch(LT((i0 + i1), i2), i2, (i0 + i1))}}.0, Elemwise{sub,no_inplace}.0, Elemwise{Composite{Switch(LT((i0 + i1), i2), i2, (i0 + i1))}}.0, Elemwise{sub,no_inplace}.0) time 3.589630e-03s
               Node Elemwise{Composite{(((i0 - maximum(i1, i2)) - i3) + maximum(i4, i5))}}[(0, 0)](Elemwise{Composite{Switch(LT(i0, i1), (i0 + i2), i0)}}.0, Elemwise{Composite{minimum(((i0 + i1) - i2), i3)}}.0, TensorConstant{1}, TensorConstant{1}, Elemwise{Composite{((maximum(i0, i1) - Switch(LT(i2, i3), (i2 + i4), i2)) + i1)}}[(0, 2)].0, TensorConstant{2}) time 3.567696e-03s
    
    Time in all call to theano.grad() 0.000000e+00s
    Time since theano import 74.908s
    Class
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
      99.6%    99.6%      13.517s       2.70e+00s     Py       5       1   theano.scan_module.scan_op.Scan
       0.2%    99.9%       0.034s       6.76e-03s     Py       5       1   theano.tensor.basic.Nonzero
       0.1%   100.0%       0.012s       4.31e-05s     C      285      57   theano.tensor.elemwise.Elemwise
       0.0%   100.0%       0.005s       9.77e-04s     C        5       1   theano.tensor.subtensor.AdvancedSubtensor1
       0.0%   100.0%       0.001s       2.78e-04s     C        5       1   theano.tensor.subtensor.IncSubtensor
       0.0%   100.0%       0.000s       2.13e-06s     C       40       8   theano.tensor.subtensor.Subtensor
       0.0%   100.0%       0.000s       3.59e-06s     C       15       3   theano.tensor.basic.Reshape
       0.0%   100.0%       0.000s       7.26e-07s     C       65      13   theano.tensor.basic.ScalarFromTensor
       0.0%   100.0%       0.000s       9.44e-06s     C        5       1   theano.tensor.basic.AllocEmpty
       0.0%   100.0%       0.000s       1.88e-06s     C       20       4   theano.compile.ops.Shape_i
       0.0%   100.0%       0.000s       1.86e-06s     C       20       4   theano.tensor.elemwise.DimShuffle
       0.0%   100.0%       0.000s       8.58e-07s     C        5       1   theano.compile.ops.Rebroadcast
       ... (remaining 0 Classes account for   0.00%(0.00s) of the runtime)
    
    Ops
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
      99.6%    99.6%      13.517s       2.70e+00s     Py       5        1   forall_inplace,cpu,scan_fn}
       0.2%    99.9%       0.034s       6.76e-03s     Py       5        1   Nonzero
       0.1%    99.9%       0.011s       5.50e-04s     C       20        4   Elemwise{mul,no_inplace}
       0.0%   100.0%       0.005s       9.77e-04s     C        5        1   AdvancedSubtensor1
       0.0%   100.0%       0.001s       2.78e-04s     C        5        1   IncSubtensor{InplaceSet;:int64:}
       0.0%   100.0%       0.001s       1.77e-04s     C        5        1   Elemwise{eq,no_inplace}
       0.0%   100.0%       0.000s       7.26e-07s     C       65       13   ScalarFromTensor
       0.0%   100.0%       0.000s       9.44e-06s     C        5        1   AllocEmpty{dtype='float64'}
       0.0%   100.0%       0.000s       4.32e-06s     C       10        2   Subtensor{int64}
       0.0%   100.0%       0.000s       1.40e-06s     C       30        6   Subtensor{int64:int64:int8}
       0.0%   100.0%       0.000s       1.88e-06s     C       20        4   Shape_i{0}
       0.0%   100.0%       0.000s       3.39e-06s     C       10        2   Reshape{1}
       0.0%   100.0%       0.000s       2.31e-06s     C       10        2   InplaceDimShuffle{x,x}
       0.0%   100.0%       0.000s       7.63e-07s     C       30        6   Elemwise{le,no_inplace}
       0.0%   100.0%       0.000s       1.45e-06s     C       15        3   Elemwise{Composite{Switch(LT((i0 + i1), i2), i2, (i0 + i1))}}
       0.0%   100.0%       0.000s       4.01e-06s     C        5        1   Elemwise{true_div,no_inplace}
       0.0%   100.0%       0.000s       4.01e-06s     C        5        1   Reshape{2}
       0.0%   100.0%       0.000s       3.81e-06s     C        5        1   Elemwise{Composite{Switch(LT((i0 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), i3), (i4 - i0), Switch(GE((i0 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), (i2 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2))), (i5 + i0), Switch(LE((i2 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), i3), (i5 + i0), i0)))}}
       0.0%   100.0%       0.000s       1.21e-06s     C       15        3   Elemwise{sub,no_inplace}
       0.0%   100.0%       0.000s       3.62e-06s     C        5        1   Elemwise{Composite{(i0 * Composite{sqr(sqr(i0))}(i1))}}
       ... (remaining 27 Ops account for   0.00%(0.00s) of the runtime)
    
    Apply
    ------
    <% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
      99.6%    99.6%      13.517s       2.70e+00s      5    93   forall_inplace,cpu,scan_fn}(Elemwise{Maximum}[(0, 0)].0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, Subtensor{int64:int64:int8}.0, IncSubtensor{InplaceSet;:int64:}.0, grade of the universal drift, <TensorType(float64, matrix)>, <TensorType(float64, vector)>, Value of the formation, Position of the dips, Rest of the points of the layers, Re
        input 0: dtype=int64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        input 2: dtype=int64, shape=no shape, strides=no strides 
        input 3: dtype=int64, shape=no shape, strides=no strides 
        input 4: dtype=int64, shape=no shape, strides=no strides 
        input 5: dtype=int64, shape=no shape, strides=no strides 
        input 6: dtype=int64, shape=no shape, strides=no strides 
        input 7: dtype=float64, shape=no shape, strides=no strides 
        input 8: dtype=int64, shape=no shape, strides=no strides 
        input 9: dtype=float64, shape=no shape, strides=no strides 
        input 10: dtype=float64, shape=no shape, strides=no strides 
        input 11: dtype=float64, shape=no shape, strides=no strides 
        input 12: dtype=float32, shape=no shape, strides=no strides 
        input 13: dtype=float32, shape=no shape, strides=no strides 
        input 14: dtype=float32, shape=no shape, strides=no strides 
        input 15: dtype=float32, shape=no shape, strides=no strides 
        input 16: dtype=float32, shape=no shape, strides=no strides 
        input 17: dtype=float32, shape=no shape, strides=no strides 
        input 18: dtype=float64, shape=no shape, strides=no strides 
        input 19: dtype=float64, shape=no shape, strides=no strides 
        input 20: dtype=float64, shape=no shape, strides=no strides 
        input 21: dtype=float64, shape=no shape, strides=no strides 
        input 22: dtype=float64, shape=no shape, strides=no strides 
        input 23: dtype=float64, shape=no shape, strides=no strides 
        input 24: dtype=float64, shape=no shape, strides=no strides 
        input 25: dtype=float64, shape=no shape, strides=no strides 
        input 26: dtype=float64, shape=no shape, strides=no strides 
        input 27: dtype=float64, shape=no shape, strides=no strides 
        input 28: dtype=float64, shape=no shape, strides=no strides 
        input 29: dtype=float64, shape=no shape, strides=no strides 
        input 30: dtype=float64, shape=no shape, strides=no strides 
        input 31: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.2%    99.9%       0.034s       6.76e-03s      5    37   Nonzero(Reshape{1}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=int64, shape=no shape, strides=no strides 
       0.1%    99.9%       0.011s       2.19e-03s      5    27   Elemwise{mul,no_inplace}(<TensorType(float64, matrix)>, Elemwise{eq,no_inplace}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=bool, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.005s       9.77e-04s      5    51   AdvancedSubtensor1(Reshape{1}.0, Subtensor{int64}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.001s       2.78e-04s      5    91   IncSubtensor{InplaceSet;:int64:}(AllocEmpty{dtype='float64'}.0, Rebroadcast{0}.0, Constant{1})
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=float64, shape=no shape, strides=no strides 
        input 2: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.001s       1.77e-04s      5    15   Elemwise{eq,no_inplace}(InplaceDimShuffle{0,x}.0, TensorConstant{(1, 1) of 0})
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=int8, shape=no shape, strides=no strides 
        output 0: dtype=bool, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       9.44e-06s      5    88   AllocEmpty{dtype='float64'}(Elemwise{Composite{(Switch(LT(maximum(i0, i1), i2), (maximum(i0, i1) + i3), (maximum(i0, i1) - i2)) + i4)}}[(0, 0)].0, Shape_i{0}.0)
        input 0: dtype=int64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       5.25e-06s      5    34   Reshape{1}(Elemwise{mul,no_inplace}.0, TensorConstant{(1,) of -1})
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       4.63e-06s      5    44   Subtensor{int64}(Nonzero.0, Constant{0})
        input 0: dtype=int64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=int64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       4.29e-06s      5    23   Elemwise{mul,no_inplace}(TensorConstant{(1, 1) of 4.0}, InplaceDimShuffle{x,x}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       4.01e-06s      5    94   Subtensor{int64}(forall_inplace,cpu,scan_fn}.0, ScalarFromTensor.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       4.01e-06s      5    58   Reshape{2}(AdvancedSubtensor1.0, TensorConstant{[-1  3]})
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       4.01e-06s      5    29   Elemwise{true_div,no_inplace}(TensorConstant{(1, 1) of -14.0}, Elemwise{sqr,no_inplace}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.81e-06s      5    38   Elemwise{Composite{Switch(LT((i0 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), i3), (i4 - i0), Switch(GE((i0 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), (i2 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2))), (i5 + i0), Switch(LE((i2 - Composite{Switch(LT(i0, i1), i0, i1)}(i1, i2)), i3), (i5 + i0), i0)))}}(Elemwise{Composite{minimum(minimum(minimum(minimum(minimum(i0, i1), i2), i3), i4), i5)}}.0, TensorConstant{1}, Elemwise{add,no_inplace}.
        input 0: dtype=int64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        input 2: dtype=int64, shape=no shape, strides=no strides 
        input 3: dtype=int8, shape=no shape, strides=no strides 
        input 4: dtype=int64, shape=no shape, strides=no strides 
        input 5: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=int64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.62e-06s      5    17   Elemwise{Composite{(i0 * Composite{sqr(sqr(i0))}(i1))}}(TensorConstant{(1, 1) of 15.0}, InplaceDimShuffle{x,x}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.62e-06s      5     2   Shape_i{0}(Length of interfaces in every series)
        input 0: dtype=int64, shape=no shape, strides=no strides 
        output 0: dtype=int64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.58e-06s      5    20   Elemwise{Composite{((sqr(sqr(i0)) * sqr(i0)) * i0)}}(InplaceDimShuffle{x,x}.0)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.39e-06s      5    81   Subtensor{int64:int64:int8}(Length of interfaces in every series, ScalarFromTensor.0, ScalarFromTensor.0, Constant{1})
        input 0: dtype=int64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        input 2: dtype=int64, shape=no shape, strides=no strides 
        input 3: dtype=int8, shape=no shape, strides=no strides 
        output 0: dtype=int64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.19e-06s      5     8   Elemwise{Composite{(((i0 * i1) / sqr(i2)) + i3)}}(TensorConstant{14.0}, <TensorType(float64, scalar)>, <TensorType(float64, scalar)>, <TensorType(float64, scalar)>)
        input 0: dtype=float64, shape=no shape, strides=no strides 
        input 1: dtype=float64, shape=no shape, strides=no strides 
        input 2: dtype=float64, shape=no shape, strides=no strides 
        input 3: dtype=float64, shape=no shape, strides=no strides 
        output 0: dtype=float64, shape=no shape, strides=no strides 
       0.0%   100.0%       0.000s       3.00e-06s      5    77   Elemwise{Composite{(((i0 - maximum(i1, i2)) - i3) + maximum(i4, i5))}}[(0, 0)](Elemwise{Composite{Switch(LT(i0, i1), (i0 + i2), i0)}}.0, Elemwise{Composite{minimum(((i0 + i1) - i2), i3)}}.0, TensorConstant{1}, TensorConstant{1}, Elemwise{Composite{((maximum(i0, i1) - Switch(LT(i2, i3), (i2 + i4), i2)) + i1)}}[(0, 2)].0, TensorConstant{2})
        input 0: dtype=int64, shape=no shape, strides=no strides 
        input 1: dtype=int64, shape=no shape, strides=no strides 
        input 2: dtype=int8, shape=no shape, strides=no strides 
        input 3: dtype=int8, shape=no shape, strides=no strides 
        input 4: dtype=int64, shape=no shape, strides=no strides 
        input 5: dtype=int8, shape=no shape, strides=no strides 
        output 0: dtype=int64, shape=no shape, strides=no strides 
       ... (remaining 75 Apply instances account for 0.00%(0.00s) of the runtime)
    
    Here are tips to potentially make your code run faster
                     (if you think of new ones, suggest them on the mailing list).
                     Test them first, as they are not guaranteed to always provide a speedup.
      Sorry, no tip for today.


Below here so far is deprecated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First we make a GeMpy instance with most of the parameters default
(except range that is given by the project). Then we also fix the
extension and the resolution of the domain we want to interpolate.
Finally we compile the function, only needed once every time we open the
project (the guys of theano they are working on letting loading compiled
files, even though in our case it is not a big deal).

*General note. So far the reescaling factor is calculated for all series
at the same time. GeoModeller does it individually for every potential
field. I have to look better what this parameter exactly means*

All input data is stored in pandas dataframes under,
``self.Data.Interances`` and ``self.Data.Foliations``:

In case of disconformities, we can define which formation belong to
which series using a dictionary. Until python 3.6 is important to
specify the order of the series otherwise is random

Now in the data frame we should have the series column too

Next step is the creating of a grid. So far only regular. By default it
takes the extent and the resolution given in the ``import_data`` method.

.. code:: ipython3

    # Create a class Grid so far just regular grid
    #GeMpy.set_grid(geo_data)
    #GeMpy.get_grid(geo_data)

Plotting raw data
-----------------

The object Plot is created automatically as we call the methods above.
This object contains some methods to plot the data and the results.

It is possible to plot a 2D projection of the data in a specific
direction using the following method. Also is possible to choose the
series you want to plot. Additionally all the key arguments of seaborn
lmplot can be used.

.. code:: ipython3

    #GeMpy.plot_data(geo_data, 'y', geo_data.series.columns.values[1])

Class Interpolator
------------------

This class will take the data from the class Data and calculate
potential fields and block. We can pass as key arguments all the
variables of the interpolation. I recommend not to touch them if you do
not know what are you doing. The default values should be good enough.
Also the first time we execute the method, we will compile the theano
function so it can take a bit of time.

.. code:: ipython3

    %debug


.. parsed-literal::

    > [1;32m/home/bl3/PycharmProjects/GeMpy/GeMpy/GeMpy.py[0m(46)[0;36mrescale_data[1;34m()[0m
    [1;32m     44 [1;33m[1;33m[0m[0m
    [0m[1;32m     45 [1;33m    [0mnew_coord_extent[0m [1;33m=[0m [0m_np[0m[1;33m.[0m[0mzeros_like[0m[1;33m([0m[0mgeo_data[0m[1;33m.[0m[0mextent[0m[1;33m)[0m[1;33m[0m[0m
    [0m[1;32m---> 46 [1;33m    [0mnew_coord_extent[0m[1;33m[[0m[1;33m:[0m[1;36m2[0m[1;33m][0m [1;33m=[0m [0mgeo_data[0m[1;33m.[0m[0mextent[0m[1;33m[[0m[1;33m:[0m[1;36m2[0m[1;33m][0m [1;33m-[0m [0mcenters[0m[1;33m[[0m[1;36m0[0m[1;33m][0m [1;33m/[0m [0mrescaling_factor[0m [1;33m+[0m [1;36m0.5001[0m[1;33m[0m[0m
    [0m[1;32m     47 [1;33m    [0mnew_coord_extent[0m[1;33m[[0m[1;36m2[0m[1;33m:[0m[1;36m4[0m[1;33m][0m [1;33m=[0m [0mgeo_data[0m[1;33m.[0m[0mextent[0m[1;33m[[0m[1;36m2[0m[1;33m:[0m[1;36m4[0m[1;33m][0m [1;33m-[0m [0mcenters[0m[1;33m[[0m[1;36m1[0m[1;33m][0m [1;33m/[0m [0mrescaling_factor[0m [1;33m+[0m [1;36m0.5001[0m[1;33m[0m[0m
    [0m[1;32m     48 [1;33m    [0mnew_coord_extent[0m[1;33m[[0m[1;36m4[0m[1;33m:[0m[1;36m6[0m[1;33m][0m [1;33m=[0m [0mgeo_data[0m[1;33m.[0m[0mextent[0m[1;33m[[0m[1;36m4[0m[1;33m:[0m[1;36m6[0m[1;33m][0m [1;33m-[0m [0mcenters[0m[1;33m[[0m[1;36m2[0m[1;33m][0m [1;33m/[0m [0mrescaling_factor[0m [1;33m+[0m [1;36m0.5001[0m[1;33m[0m[0m
    [0m
    ipdb> geo_data.extent[:2]
    [696000, 747000]
    ipdb> centers[0]
    396789.0625
    ipdb> new_coord_extent[:2]
    array([0, 0])
    ipdb> geo_data.extent[:2] - centers[0] / rescaling_factor + 0.5001
    *** TypeError: unsupported operand type(s) for -: 'list' and 'float'
    ipdb> geo_data.extent[:2] - centers[0] / rescaling_factor
    *** TypeError: unsupported operand type(s) for -: 'list' and 'float'
    ipdb> geo_data.extent[:2] - centers[0]
    *** TypeError: unsupported operand type(s) for -: 'list' and 'float'
    ipdb> centers[0]
    396789.0625
    ipdb> geo_data.extent[:2] - centers[0].as_matrix(
    *** SyntaxError: unexpected EOF while parsing
    ipdb> geo_data.extent[:2] - centers[0].as_matrix()
    *** AttributeError: 'float' object has no attribute 'as_matrix'
    ipdb> geo_data.extent[:2] - centers.as_matrix()[0]
    *** TypeError: unsupported operand type(s) for -: 'list' and 'float'
    ipdb> exit


.. code:: ipython3

    geo_data.interpolator.results




.. parsed-literal::

    [array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]),
     array([[ 0.343335,       inf,       inf, ..., -0.000003, -0.000002, -0.000002],
            [      inf,  0.343335,       inf, ..., -0.000003, -0.000002, -0.000002],
            [      inf,       inf,  0.343335, ..., -0.000003, -0.000002, -0.000002],
            ..., 
            [-0.000003, -0.000003, -0.000003, ...,  0.      ,  0.      ,  0.      ],
            [-0.000002, -0.000002, -0.000002, ...,  0.      ,  0.      ,  0.      ],
            [-0.000002, -0.000002, -0.000002, ...,  0.      ,  0.      ,  0.      ]]),
     array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
             nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
             nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]),
     array([[ 0.      ,  0.407196,  0.244746, ...,  0.494572,  0.551419,  0.469685],
            [ 0.407196,  0.      ,  0.629916, ...,  0.603107,  0.642273,  0.554478],
            [ 0.244746,  0.629916,       nan, ...,  0.547687,  0.6045  ,  0.542099],
            ..., 
            [ 0.494572,  0.603107,  0.547687, ...,  0.      ,  0.058977,  0.050814],
            [ 0.551419,  0.642273,  0.6045  , ...,  0.058977,  0.      ,  0.092778],
            [ 0.469685,  0.554478,  0.542099, ...,  0.050814,  0.092778,  0.      ]], dtype=float32)]



.. code:: ipython3

    geo_data.interpolator.tg.c_o_T.get_value(), geo_data.interpolator.tg.a_T.get_value()




.. parsed-literal::

    (array(253666666), array(103218))



.. code:: ipython3

    geo_data.interpolator.compile_potential_field_function()




.. parsed-literal::

    <theano.compile.function_module.Function at 0x7f30696b9780>



.. code:: ipython3

    geo_data.interpolator.compute_potential_fields('BIF_Series',verbose = 3)


.. parsed-literal::

    The serie formations are SimpleMafic2|SimpleBIF
    The formations are: 
    Layers 
                     X             Y            Z     formation      series
    0   735484.817806  6.891936e+06 -1819.319309  SimpleMafic2  BIF_Series
    1   729854.915982  6.891938e+06 -1432.263309  SimpleMafic2  BIF_Series
    2   724084.267161  6.891939e+06 -4739.830309  SimpleMafic2  BIF_Series
    3   733521.625000  6.895282e+06   521.555240  SimpleMafic2  BIF_Series
    4   721933.375000  6.884592e+06   496.669295  SimpleMafic2  BIF_Series
    5   724251.000000  6.886909e+06   484.550926  SimpleMafic2  BIF_Series
    6   727316.313000  6.886460e+06   478.254423  SimpleMafic2  BIF_Series
    7   729858.250000  6.887134e+06   484.259574  SimpleMafic2  BIF_Series
    8   732699.250000  6.885040e+06   494.526481  SimpleMafic2  BIF_Series
    9   716849.500000  6.887358e+06   508.981894  SimpleMafic2  BIF_Series
    10  719017.625000  6.892218e+06   508.179387  SimpleMafic2  BIF_Series
    11  739179.440691  6.891936e+06  -552.591309     SimpleBIF  BIF_Series
    12  735564.599804  6.891936e+06 -2652.196309     SimpleBIF  BIF_Series
    13  730009.009977  6.891938e+06 -2088.409309     SimpleBIF  BIF_Series
    14  718795.791326  6.891941e+06 -2773.169309     SimpleBIF  BIF_Series
    15  724143.386160  6.891939e+06 -5569.907309     SimpleBIF  BIF_Series
    16  723877.188000  6.899768e+06   529.152169     SimpleBIF  BIF_Series
    17  732998.313000  6.898049e+06   521.619609     SimpleBIF  BIF_Series
    18  743689.438000  6.891769e+06   512.811278     SimpleBIF  BIF_Series
    19  712961.813000  6.882722e+06   547.826016     SimpleBIF  BIF_Series
    20  716284.875000  6.891346e+06   515.586860     SimpleBIF  BIF_Series
    21  718942.875000  6.897600e+06   538.490136     SimpleBIF  BIF_Series
    22  722157.625000  6.882947e+06   481.747055     SimpleBIF  BIF_Series
    23  723952.000000  6.885488e+06   480.122832     SimpleBIF  BIF_Series
    24  728736.813000  6.885488e+06   477.929009     SimpleBIF  BIF_Series
    25  738829.813000  6.878087e+06   470.081431     SimpleBIF  BIF_Series 
     foliations 
                    X             Y            Z  azimuth   dip  polarity  \
    0  739426.627684  6.891935e+06    75.422691    220.0  70.0         1   
    1  717311.112372  6.891941e+06 -1497.488309     90.0  60.0         1   
    
       formation      series       G_x           G_y      G_z  
    0  SimpleBIF  BIF_Series -0.604023 -7.198463e-01  0.34202  
    1  SimpleBIF  BIF_Series  0.866025  5.302876e-17  0.50000  
    Dual Kriging weights:  [ nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan  nan
      nan  nan  nan  nan]




.. parsed-literal::

    array([[[ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            ..., 
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
    
           [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            ..., 
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
    
           [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            ..., 
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
    
           ..., 
           [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            ..., 
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
    
           [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            ..., 
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
    
           [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            ..., 
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan],
            [ nan,  nan,  nan, ...,  nan,  nan,  nan]]])



.. code:: ipython3

    geo_data.interpolator.potential_fields




.. parsed-literal::

    [array([[[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            ..., 
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]]]),
     array([[[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            ..., 
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]]]),
     array([[[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            ..., 
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]],
     
            [[ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             ..., 
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan],
             [ nan,  nan,  nan, ...,  nan,  nan,  nan]]])]



.. code:: ipython3

    geo_data.interpolator.results




.. parsed-literal::

    [array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]),
     array([[ 0.343335,       inf,       inf, ..., -0.000003, -0.000002, -0.000002],
            [      inf,  0.343335,       inf, ..., -0.000003, -0.000002, -0.000002],
            [      inf,       inf,  0.343335, ..., -0.000003, -0.000002, -0.000002],
            ..., 
            [-0.000003, -0.000003, -0.000003, ...,  0.      ,  0.      ,  0.      ],
            [-0.000002, -0.000002, -0.000002, ...,  0.      ,  0.      ,  0.      ],
            [-0.000002, -0.000002, -0.000002, ...,  0.      ,  0.      ,  0.      ]]),
     array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
             nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
             nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan]),
     array([[ 0.      ,  0.407196,  0.244746, ...,  0.494572,  0.551419,  0.469685],
            [ 0.407196,  0.      ,  0.629916, ...,  0.603107,  0.642273,  0.554478],
            [ 0.244746,  0.629916,       nan, ...,  0.547687,  0.6045  ,  0.542099],
            ..., 
            [ 0.494572,  0.603107,  0.547687, ...,  0.      ,  0.058977,  0.050814],
            [ 0.551419,  0.642273,  0.6045  , ...,  0.058977,  0.      ,  0.092778],
            [ 0.469685,  0.554478,  0.542099, ...,  0.050814,  0.092778,  0.      ]], dtype=float32)]



.. code:: ipython3

    geo_data.interpolator.tg.c_resc.get_value()




.. parsed-literal::

    array(48561)



Now we could visualize the individual potential fields as follow:

Early granite
~~~~~~~~~~~~~

.. code:: ipython3

    GeMpy.plot_potential_field(geo_data,10, n_pf=0)

BIF Series
~~~~~~~~~~

.. code:: ipython3

    GeMpy.plot_potential_field(geo_data,13, n_pf=1, cmap = "magma",  plot_data = True,
                                            verbose = 5)



.. image:: Example_1_Sandstone_files/Example_1_Sandstone_34_0.png


SImple mafic
~~~~~~~~~~~~

.. code:: ipython3

    GeMpy.plot_potential_field(geo_data, 10, n_pf=2)



.. image:: Example_1_Sandstone_files/Example_1_Sandstone_36_0.png


Optimizing the export of lithologies
------------------------------------

But usually the final result we want to get is the final block. The
method ``compute_block_model`` will compute the block model, updating
the attribute ``block``. This attribute is a theano shared function that
can return a 3D array (raveled) using the method ``get_value()``.

.. code:: ipython3

    GeMpy.compute_block_model(geo_data)


.. parsed-literal::

    ../GeMpy/GeMpy.py:38: UserWarning: Using default interpolation values
      warnings.warn('Using default interpolation values')




.. parsed-literal::

    Final block computed



.. code:: ipython3

    #GeMpy.set_interpolator(geo_data, u_grade = 0, compute_potential_field=True)

And again after computing the model in the Plot object we can use the
method ``plot_block_section`` to see a 2D section of the model

.. code:: ipython3

    GeMpy.plot_section(geo_data, 13, direction='y')




.. parsed-literal::

    <Visualization.PlotData at 0x7f780fd9ecc0>




.. image:: Example_1_Sandstone_files/Example_1_Sandstone_41_1.png


Export to vtk. (*Under development*)
------------------------------------

.. code:: ipython3

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

    ImportError                               Traceback (most recent call last)

    <ipython-input-14-ff637538da86> in <module>()
         19 dely = 0.2
         20 delz = 0.2
    ---> 21 from pyevtk.hl import gridToVTK
         22 # Coordinates
         23 x = np.arange(0, extent_x + 0.1*delx, delx, dtype='float64')


    ImportError: No module named 'pyevtk'


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

Looking at the profile we can see that most of time is in pow operation
(exponential). This probably is that the extent is huge and we are doing
it with too much precision. I am working on it

GPU
~~~

.. code:: ipython3

    %%timeit
    
    # Compute the block
    GeMpy.compute_block_model(geo_data, [0,1,2], verbose = 0)


.. parsed-literal::

    1 loop, best of 3: 1.74 s per loop


.. code:: ipython3

    geo_data.interpolator._interpolate.profile.summary()


.. parsed-literal::

    Function profiling
    ==================
      Message: ../GeMpy/DataManagement.py:994
      Time in 3 calls to Function.__call__: 8.400567e-01s
      Time in Function.fn.__call__: 8.395956e-01s (99.945%)
      Time in thunks: 8.275988e-01s (98.517%)
      Total compile time: 3.540267e+00s
        Number of Apply nodes: 342
        Theano Optimizer time: 2.592782e+00s
           Theano validate time: 1.640296e-01s
        Theano Linker time (includes C, CUDA code generation/compiling): 8.665011e-01s
           Import time 1.915064e-01s
    
    Time in all call to theano.grad() 0.000000e+00s
    Time since theano import 72.847s
    Class
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Class name>
      57.3%    57.3%       0.474s       2.87e-03s     C      165      55   theano.tensor.elemwise.Elemwise
      10.1%    67.4%       0.084s       2.79e-03s     C       30      10   theano.tensor.blas.Dot22Scalar
       9.6%    77.0%       0.079s       9.81e-04s     C       81      27   theano.sandbox.cuda.basic_ops.HostFromGpu
       6.4%    83.4%       0.053s       8.89e-03s     Py       6       2   theano.tensor.basic.Nonzero
       6.4%    89.8%       0.053s       1.77e-02s     Py       3       1   theano.tensor.nlinalg.MatrixInverse
       5.1%    95.0%       0.042s       2.01e-03s     C       21       7   theano.tensor.elemwise.Sum
       2.3%    97.2%       0.019s       3.13e-03s     C        6       2   theano.sandbox.cuda.basic_ops.GpuAdvancedSubtensor1
       0.9%    98.1%       0.007s       5.00e-04s     C       15       5   theano.tensor.basic.Alloc
       0.5%    98.6%       0.004s       2.34e-04s     C       18       6   theano.sandbox.cuda.basic_ops.GpuAlloc
       0.5%    99.1%       0.004s       1.43e-04s     C       27       9   theano.sandbox.cuda.basic_ops.GpuJoin
       0.4%    99.5%       0.004s       3.59e-05s     C      102      34   theano.sandbox.cuda.basic_ops.GpuElemwise
       0.2%    99.7%       0.001s       5.48e-05s     C       27       9   theano.sandbox.cuda.basic_ops.GpuFromHost
       0.1%    99.8%       0.001s       1.49e-05s     C       66      22   theano.sandbox.cuda.basic_ops.GpuReshape
       0.0%    99.9%       0.000s       4.41e-05s     C        6       2   theano.compile.ops.DeepCopyOp
       0.0%    99.9%       0.000s       2.63e-06s     C       72      24   theano.tensor.subtensor.IncSubtensor
       0.0%    99.9%       0.000s       2.80e-06s     C       48      16   theano.sandbox.cuda.basic_ops.GpuSubtensor
       0.0%    99.9%       0.000s       1.13e-06s     C      114      38   theano.sandbox.cuda.basic_ops.GpuDimShuffle
       0.0%    99.9%       0.000s       3.96e-05s     C        3       1   theano.sandbox.cuda.basic_ops.GpuAllocEmpty
       0.0%   100.0%       0.000s       3.20e-05s     Py       3       1   theano.tensor.extra_ops.FillDiagonal
       0.0%   100.0%       0.000s       1.23e-06s     C       69      23   theano.tensor.elemwise.DimShuffle
       ... (remaining 9 Classes account for   0.03%(0.00s) of the runtime)
    
    Ops
    ---
    <% time> <sum %> <apply time> <time per call> <type> <#call> <#apply> <Op name>
      36.2%    36.2%       0.300s       9.98e-02s     C        3        1   Elemwise{Composite{(i0 * i1 * LT(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4), i5) * (((i6 + ((i7 * Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4)) / i8)) - ((i9 * Composite{(sqr(i0) * i0)}(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4))) / i10)) + ((i11 * Composite{(sqr(sqr(i0)) * i0)}(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4))) / i12)))}}[(0, 4)]
      19.2%    55.4%       0.159s       5.30e-02s     C        3        1   Elemwise{Composite{(i0 * ((LT(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3), i4) * ((i5 + (i6 * Composite{(sqr(i0) * i0)}((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4))) + (i7 * Composite{((sqr(sqr(i0)) * sqr(i0)) * i0)}((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4)))) - ((i8 * sqr((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4))) + (i9 * Composite{(sqr(sqr(i0)
      10.1%    65.5%       0.084s       2.79e-03s     C       30       10   Dot22Scalar
       9.6%    75.1%       0.079s       9.81e-04s     C       81       27   HostFromGpu
       6.4%    81.5%       0.053s       8.89e-03s     Py       6        2   Nonzero
       6.4%    88.0%       0.053s       1.77e-02s     Py       3        1   MatrixInverse
       5.0%    92.9%       0.041s       4.58e-03s     C        9        3   Sum{axis=[0], acc_dtype=float64}
       2.3%    95.2%       0.019s       3.13e-03s     C        6        2   GpuAdvancedSubtensor1
       0.9%    96.1%       0.008s       1.27e-03s     C        6        2   Elemwise{Mul}[(0, 1)]
       0.9%    97.0%       0.007s       5.00e-04s     C       15        5   Alloc
       0.6%    97.6%       0.005s       1.72e-03s     C        3        1   Elemwise{Composite{((i0 / i1) + ((i2 * i3) / i1) + ((i4 * i2 * i5) / i6))}}[(0, 0)]
       0.5%    98.1%       0.004s       2.76e-04s     C       15        5   GpuAlloc
       0.5%    98.6%       0.004s       1.43e-04s     C       27        9   GpuJoin
       0.3%    98.9%       0.002s       4.95e-05s     C       45       15   GpuElemwise{sub,no_inplace}
       0.2%    99.1%       0.001s       5.48e-05s     C       27        9   GpuFromHost
       0.2%    99.2%       0.001s       1.08e-04s     C       12        4   Elemwise{Cast{float64}}
       0.1%    99.3%       0.001s       9.13e-05s     C       12        4   Sum{axis=[1], acc_dtype=float64}
       0.1%    99.5%       0.001s       6.88e-05s     C       15        5   GpuElemwise{mul,no_inplace}
       0.1%    99.6%       0.001s       6.47e-05s     C       15        5   Elemwise{Sqr}[(0, 0)]
       0.1%    99.7%       0.001s       1.61e-05s     C       60       20   GpuReshape{2}
       ... (remaining 75 Ops account for   0.29%(0.00s) of the runtime)
    
    Apply
    ------
    <% time> <sum %> <apply time> <time per call> <#call> <id> <Apply name>
      36.2%    36.2%       0.300s       9.98e-02s      3   332   Elemwise{Composite{(i0 * i1 * LT(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4), i5) * (((i6 + ((i7 * Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4)) / i8)) - ((i9 * Composite{(sqr(i0) * i0)}(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4))) / i10)) + ((i11 * Composite{(sqr(sqr(i0)) * i0)}(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4))) / i12)))}}[(0, 4)](HostFromGpu.0, HostFromGpu.0, Reshape{2
      19.2%    55.4%       0.159s       5.30e-02s      3   331   Elemwise{Composite{(i0 * ((LT(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3), i4) * ((i5 + (i6 * Composite{(sqr(i0) * i0)}((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4))) + (i7 * Composite{((sqr(sqr(i0)) * sqr(i0)) * i0)}((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4)))) - ((i8 * sqr((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4))) + (i9 * Composite{(sqr(sqr(i0)) * i0)}((C
       6.4%    61.8%       0.053s       1.77e-02s      3   318   MatrixInverse(IncSubtensor{InplaceSet;int64::, int64:int64:}.0)
       5.3%    67.1%       0.044s       1.46e-02s      3   180   Nonzero(HostFromGpu.0)
       5.0%    72.1%       0.042s       1.39e-02s      3   269   Dot22Scalar(Elemwise{Cast{float64}}.0, InplaceDimShuffle{1,0}.0, TensorConstant{2.0})
       3.8%    75.9%       0.031s       1.04e-02s      3   335   Sum{axis=[0], acc_dtype=float64}(Elemwise{Composite{(i0 * i1 * LT(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4), i5) * (((i6 + ((i7 * Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4)) / i8)) - ((i9 * Composite{(sqr(i0) * i0)}(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4))) / i10)) + ((i11 * Composite{(sqr(sqr(i0)) * i0)}(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i2, i3, i4))) / i12)))}}[(0, 4)].0)
       3.5%    79.4%       0.029s       9.55e-03s      3   286   HostFromGpu(GpuJoin.0)
       3.0%    82.3%       0.025s       8.18e-03s      3   329   HostFromGpu(GpuSubtensor{:int64:}.0)
       2.5%    84.8%       0.021s       6.96e-03s      3   268   Dot22Scalar(Elemwise{Cast{float64}}.0, InplaceDimShuffle{1,0}.0, TensorConstant{2.0})
       2.5%    87.4%       0.021s       6.93e-03s      3   267   Dot22Scalar(Elemwise{Cast{float64}}.0, InplaceDimShuffle{1,0}.0, TensorConstant{2.0})
       1.6%    89.0%       0.013s       4.46e-03s      3   216   GpuAdvancedSubtensor1(GpuReshape{1}.0, Subtensor{int64}.0)
       1.3%    90.3%       0.011s       3.56e-03s      3   328   HostFromGpu(GpuSubtensor{int64:int64:}.0)
       1.2%    91.4%       0.010s       3.22e-03s      3   200   Nonzero(HostFromGpu.0)
       0.9%    92.4%       0.008s       2.53e-03s      3   333   Elemwise{Mul}[(0, 1)](HostFromGpu.0, InplaceDimShuffle{1,0}.0, HostFromGpu.0)
       0.9%    93.3%       0.007s       2.47e-03s      3   235   Alloc(Subtensor{:int64:}.0, Elemwise{Composite{((i0 // i1) + i2)}}[(0, 0)].0, TensorConstant{1}, TensorConstant{1}, Elemwise{Composite{Switch(LT(i0, i1), Switch(LT((i2 + i0), i1), i1, (i2 + i0)), Switch(LT(i0, i2), i0, i2))}}.0)
       0.9%    94.1%       0.007s       2.36e-03s      3   334   Sum{axis=[0], acc_dtype=float64}(Elemwise{Composite{(i0 * ((LT(Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3), i4) * ((i5 + (i6 * Composite{(sqr(i0) * i0)}((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4))) + (i7 * Composite{((sqr(sqr(i0)) * sqr(i0)) * i0)}((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4)))) - ((i8 * sqr((Composite{Cast{float32}(sqrt(((i0 + i1) - i2)))}(i1, i2, i3) / i4))) + (i9 * 
       0.6%    94.8%       0.005s       1.79e-03s      3   226   GpuAdvancedSubtensor1(GpuReshape{1}.0, Subtensor{int64}.0)
       0.6%    95.4%       0.005s       1.72e-03s      3   337   Elemwise{Composite{((i0 / i1) + ((i2 * i3) / i1) + ((i4 * i2 * i5) / i6))}}[(0, 0)](Sum{axis=[0], acc_dtype=float64}.0, InplaceDimShuffle{x}.0, InplaceDimShuffle{x}.0, Sum{axis=[0], acc_dtype=float64}.0, TensorConstant{(1,) of -1.0}, Sum{axis=[0], acc_dtype=float64}.0, InplaceDimShuffle{x}.0)
       0.6%    96.0%       0.005s       1.69e-03s      3   330   HostFromGpu(GpuSubtensor{int64::}.0)
       0.5%    96.5%       0.004s       1.37e-03s      3   153   HostFromGpu(GpuReshape{1}.0)
       ... (remaining 322 Apply instances account for 3.51%(0.03s) of the runtime)
    
    Here are tips to potentially make your code run faster
                     (if you think of new ones, suggest them on the mailing list).
                     Test them first, as they are not guaranteed to always provide a speedup.
      - Try installing amdlibm and set the Theano flag lib.amdlibm=True. This speeds up only some Elemwise operation.



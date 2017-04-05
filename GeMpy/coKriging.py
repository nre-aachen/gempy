"""
Module with classes and methods to perform kriging of elements and at some point exploit the potential field to
choose the directions of the variograms

Tested on Ubuntu 16

Created on 1/5/2017

@author: Miguel de la Varga
"""

import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np
import pandas as pn

from bokeh.io import show
import bokeh.layouts as bl
import bokeh.plotting as bp


def choose_lithology_elements(df, litho, elem=None, coord = True):
    """
    litho(str): Name of the lithology-domain
    elem(list): List of strings with elements you want to analyze
    """
    # Choosing just the opx litology
    if elem is not None:
        if coord:
            domain = df[df['Lithology'] == litho][np.append(['X', 'Y', 'Z'], elem)]
        else:
            domain = df[df['Lithology'] == litho][elem]
        # Drop negative values
        domain = domain[(domain[elem] > 0).all(1)]
    else:
        domain = df[df['Lithology'] == litho][['X', 'Y', 'Z']]

    return domain


def select_segmented_grid(df, litho, grid, block):

    block = np.squeeze(block)
    assert grid.shape[0] == block.shape[0], 'The grid you want to use for kriging and the grid used for the layers ' \
                                            'segmentation are not the same'

    litho_num = df['Lithology Number'][df["Lithology"] == litho].iloc[0]
    segm_grid = grid[block == litho_num]
    return segm_grid

def transform_data(df, n_comp=1, log10=False):
    # Take log to try to aproximate better the normal distributions
    # chem_opx = np.log10(chem_opx)

    # Finding two modes in the data
    if log10:
        df[df.columns.difference(['X', 'Y', 'Z'])] = np.log10(df[df.columns.difference(['X', 'Y', 'Z'])])

    if n_comp > 1:
        from sklearn import mixture
        gmm = mixture.GaussianMixture(n_components=n_comp,
                                      covariance_type='full').fit(df[df.columns.difference(['X', 'Y', 'Z'])])

        # Adding the categories to the pandas frame
        labels_all = gmm.predict(df[df.columns.difference(['X', 'Y', 'Z'])])
        df['cluster'] = labels_all
    return df


def theano_sed():
    theano.config.compute_test_value = "ignore"

    # Set symbolic variable as matrix (with the XYZ coords)
    coord_T_x1 = T.dmatrix()
    coord_T_x2 = T.dmatrix()
    # Euclidian distances function
    def squared_euclidean_distances(x_1, x_2):
        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        return sqd

    # Compiling function
    f = theano.function([coord_T_x1, coord_T_x2],
                        squared_euclidean_distances(coord_T_x1, coord_T_x2),
                        allow_input_downcast=False)
    return f


def coord_rescale(df):
    """
    Rescale coordinates to avoid float32 errors in larga coordinates to be able to use the gpu
    """

    max_coord = df.max()[['X', 'Y', 'Z']]
    min_coord = df.min()[['X', 'Y', 'Z']]

    rescaling_factor = 2 * np.max(max_coord - min_coord)

    centers = (max_coord + min_coord) / 2

    df_res = (df[['X', 'Y', 'Z']] -
              centers) / rescaling_factor + 0.5001

    return df_res


# This is extremily ineficient. Try to vectorize it in theano, it is possible to gain X100

def compute_variogram(df, element_couple, euclidian_distances, tol=10, lags=np.logspace(0, 2.5, 100), plot=[]):
    """
    Compute the experimental variogram for a par of elements
    """
    # Tiling the element concentation to a square matrix
    element = (df[element_couple[0]].as_matrix().reshape(-1, 1) -
               np.tile(df[element_couple[1]], (df[element_couple[1]].shape[0], 1))) ** 2

    # Semivariance computation
    semivariance_lag = []
    for i in lags:
        points_at_lag = ((euclidian_distances > i - tol) * (euclidian_distances < i + tol))
        var_points = element[points_at_lag]
        semivariance_lag = np.append(semivariance_lag, np.mean(var_points) / 2)

    if "experimental" in plot:
        # Visualizetion of the experimental variogram
        plt.plot(lags, semivariance_lag, 'o')

    return semivariance_lag


def exp_lags(max_range, exp=2, n_lags=100):
    lags = np.empty(0)
    for i in range(n_lags):
        lags = np.append(lags, i ** exp)
    lags = lags / lags.max() * max_range
    return lags


def compute_crossvariogram(df, element_names, euclidian_distances=None, **kwargs):
    lag_exp = kwargs.get('lag_exp', 2)
    lag_range = kwargs.get('lag_range', 500)
    n_lags = kwargs.get('n_lags', 100)

    lags = exp_lags(lag_range, lag_exp, n_lags)
    lags = np.logspace(0, 2.5, 100)
    if not euclidian_distances:
        # coords = coord_rescale(df)
        euclidian_distances = theano_sed()(df[['X', 'Y', 'Z']])

    # Init dataframe
    experimental_variograms_frame = pn.DataFrame()

    # Nested loop. DEPRECATED enumerate
    for e_i, i in enumerate(element_names):
        for e_j, j in enumerate(element_names):
            col_name = i + '-' + j
            values = compute_variogram(df, [i, j], euclidian_distances, lags=lags)
            experimental_variograms_frame[col_name] = values

    # Add lags column

    experimental_variograms_frame['lags'] = lags
    return experimental_variograms_frame


def fit_cross_cov(df, lags, n_exp=2, n_gaus=2, range_mu=None):
    n_var = df.columns.shape[0]
    n_basis_f = n_var * (n_exp + n_gaus)
    prior_std_reg = df.std(0).max() * 10
    #
    if not range_mu:
        range_mu = lags.mean()

    # Because is a experimental variogram I am not going to have outliers
    nugget_max = df.values.max()
    # print(n_basis_f, n_var*n_exp, nugget_max, range_mu, prior_std_reg)
    # pymc3 Model
    with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        # Define priors
        sigma = pm.HalfCauchy('sigma', beta=prior_std_reg, testval=1., shape=n_var)

        psill = pm.Normal('sill', prior_std_reg, sd=.5 * prior_std_reg, shape=(n_exp + n_gaus))
        range_ = pm.Normal('range', range_mu, sd=range_mu * .3, shape=(n_exp + n_gaus))
        #  nugget = pm.Uniform('nugget', 0, nugget_max, shape=n_var)

        lambda_ = pm.Uniform('weights', 0, 1, shape=(n_var * (n_exp + n_gaus)))

        # Exponential covariance
        exp = pm.Deterministic('exp',
                               # (lambda_[:n_exp*n_var]*
                               psill[:n_exp] *
                               (1. - T.exp(T.dot(-lags.as_matrix().reshape((len(lags), 1)),
                                                 (range_[:n_exp].reshape((1, n_exp)) / 3.) ** -1))))

        gaus = pm.Deterministic('gaus',
                                psill[n_exp:] *
                                (1. - T.exp(T.dot(-lags.as_matrix().reshape((len(lags), 1)) ** 2,
                                                  (range_[n_exp:].reshape((1, n_gaus)) * 4 / 7.) ** -2))))

        func = pm.Deterministic('func', T.tile(T.horizontal_stack(exp, gaus), (n_var, 1, 1)))

        func_w = pm.Deterministic("func_w", T.sum(func * lambda_.reshape((n_var, 1, (n_exp + n_gaus))), axis=2))
        #           nugget.reshape((n_var,1)))

        for e, cross in enumerate(df.columns):
            # Likelihoods
            pm.Normal(cross + "_like", mu=func_w[e], sd=sigma[e], observed=df[cross].as_matrix())
    return model


def exp_vario(lags, sill, range_):
    return sill * (1 - np.exp(np.dot(-lags.reshape(-1, 1) * 3, range_.reshape(1, -1) ** -1)))


def gaus_vario(lags, sill, range_):
    return sill * (1 - np.exp(np.dot(-lags.reshape(-1, 1) ** 2, (range_.reshape(1, -1) * 4 / 7) ** -2)))


def plot_cross_variograms(trace, lags, df, n_exp=2, n_gaus=2, iter_plot=200, experimental=False):
    n_equations = trace['weights'].shape[1]
    n_iter = trace['weights'].shape[0]
    lags_tiled = np.tile(lags, (iter_plot, 1))
    b_var = []
    for i in range(0, df.shape[1]):  # n_equations, (n_exp+n_gaus)):
        # Init tensor
        b = np.zeros((len(lags), n_iter, 0))
        for i_exp in range(0, n_exp):
            # print(i_exp, "exp")
            b = np.dstack((b, trace['weights'][:, i_exp + i * (n_exp + n_gaus)] *
                           exp_vario(lags, trace['sill'][:, i_exp], trace['range'][:, i_exp])))
        for i_gaus in range(n_exp, n_gaus + n_exp):
            # print(i_gaus)
            b = np.dstack((b, trace['weights'][:, i_gaus + i * (n_exp + n_gaus)] *
                           gaus_vario(lags, trace['sill'][:, i_gaus], trace['range'][:, i_gaus])))
        # Sum the contributins of each function
        b_all = b.sum(axis=2)
        # Append each variable
        b_var.append(b_all[:, -iter_plot:].T)

    p_all = []
    for e, el in enumerate(df.columns):
        p = bp.figure(x_axis_type="log")
        p.multi_line(list(lags_tiled), list(b_var[e]), color='olive', alpha=0.08)
        if experimental:
            p.scatter(experimental['lags'], y=experimental[el], color='navy', size=2)
        p.title.text = el
        p.xaxis.axis_label = "lags"
        p.yaxis.axis_label = "Semivariance"

        p_all = np.append(p_all, p)

    grid = bl.gridplot(list(p_all), ncols=5, plot_width=250, plot_height=150)

    show(grid)


def plot_cross_covariance(trace, lags, df, n_exp=2, n_gaus=2, nuggets=None, iter_plot=200):
    n_equations = trace['weights'].shape[1]
    n_iter = trace['weights'].shape[0]
    lags_tiled = np.tile(lags, (iter_plot, 1))
    b_var = []
    for i in range(0, df.shape[1]):  # n_equations, (n_exp+n_gaus)):
        # Init tensor
        b = np.zeros((len(lags), n_iter, 0))
        for i_exp in range(0, n_exp):
            # print(i_exp, "exp")
            b = np.dstack((b, trace['weights'][:, i_exp + i * (n_exp + n_gaus)] *
                           exp_vario(lags, trace['sill'][:, i_exp], trace['range'][:, i_exp])))
        for i_gaus in range(n_exp, n_gaus + n_exp):
            # print(i_gaus)
            b = np.dstack((b, trace['weights'][:, i_gaus + i * (n_exp + n_gaus)] *
                           gaus_vario(lags, trace['sill'][:, i_gaus], trace['range'][:, i_gaus])))
        # Sum the contributins of each function
        if any(nuggets):
            b_all = 1 - (b.sum(axis=2) + nuggets[i])
        else:
            b_all = 1 - (b.sum(axis=2))
        # Append each variable
        b_var.append(b_all[:, -iter_plot:].T)

    p_all = []
    for e, el in enumerate(df.columns):
        p = bp.figure(x_axis_type="log")
        p.multi_line(list(lags_tiled), list(b_var[e]), color='olive', alpha=0.08)

        p.title.text = el
        p.xaxis.axis_label = "lags"
        p.yaxis.axis_label = "Semivariance"

        p_all = np.append(p_all, p)

    grid = bl.gridplot(list(p_all), ncols=5, plot_width=250, plot_height=150)

    show(grid)


def cross_covariance(trace, sed, nuggets=None, n_var=1, n_exp=2, n_gaus=2, ordinary=True):

    h = np.ravel(sed)
    n_points = len(h)
    n_points_r = sed.shape[0]
    n_points_c = sed.shape[1]
    sample = np.random.randint(400, trace['weights'].shape[0])
    n_eq = trace['weights'].shape[1]
    # Exp contribution
    exp_cont = (np.tile(
        exp_vario(h, trace['sill'][sample][:n_exp], trace['range'][sample][:n_exp]),
        n_var**2
    ) * trace['weights'][sample][np.linspace(0, n_eq-1, n_eq) % (n_exp + n_gaus) < n_exp]).reshape(n_points, n_exp, n_var**2, order="F")

    # Gauss contribution
    gaus_cont = (np.tile(
        gaus_vario(h, trace['sill'][sample][n_exp:], trace['range'][sample][n_exp:]),
        n_var**2
    ) * trace['weights'][sample][np.linspace(0, n_eq-1, n_eq) % (n_exp + n_gaus) >= n_exp]).reshape(n_points, n_gaus, n_var**2, order="F")

    # Stacking and summing
    conts = np.hstack((exp_cont, gaus_cont)).sum(axis=1)

    if nuggets is not None:
        conts += nuggets

    cov = 1 - conts

    cov_str = np.zeros((0, n_points_c*n_var))

    cov_aux = cov.reshape(n_points_r, n_points_c, n_var**2, order='F')

    # This is a incredibly convoluted way to reshape the cross covariance but I did not find anything better
    for i in range(n_var):
        cov_str = np.vstack((cov_str,
                             cov_aux[:, :, i*n_var:(i+1)*n_var].reshape(n_points_r, n_points_c*n_var, order='F')))


    if ordinary:
        ord = np.zeros((n_var, n_var*n_points_c+n_var))
        for i in range(n_var):
            ord[i, n_points_c * i:n_points_c * (i + 1)] = np.ones(n_points_c)
        cov_str = np.vstack((cov_str, ord[:, :-n_var]))

        # Stack the ordinary values to the C(h)
        if n_points_r == n_points_c:
             cov_str = np.hstack((cov_str, ord.T))

    return cov_str


def clustering_grid(grid_to_inter, n_clusters=50, plot=False):
    from sklearn.cluster import KMeans
    clust = KMeans(n_clusters=n_clusters).fit(grid_to_inter)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(clust.cluster_centers_[:, 0], clust.cluster_centers_[:, 1], clust.cluster_centers_[:, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    return clust


def select_points(df, grid_to_inter, cluster, n_rep=10):

    points_cluster = np.bincount(cluster.labels_)
    SED_f = theano_sed()

    for i in range(n_rep):
        for i_clust in range(cluster.n_clusters):
            cluster_bool = cluster.labels_ == i_clust
            cluster_grid = grid_to_inter[cluster_bool]
            # Mix the values of each cluster
            if i is 0:
                np.random.shuffle(cluster_grid)

            size_range = int(points_cluster[i_clust]/n_rep)
            selected_cluster_grid = cluster_grid[i * size_range:(i + 1) * size_range]
            dist = SED_f(df, selected_cluster_grid)

            # Checking the radio of the simulation
            for r in range(100, 1000, 100):
                select = (dist < r).any(axis=1)
                if select.sum() > 50:
                    break

            h_x0 = dist
            yield (h_x0, select, selected_cluster_grid)





"""
Function that generates the symbolic code to perform the interpolation. Calling this function creates
 both the theano functions for the potential field and the block.

Returns:
    theano function for the potential field
    theano function for the block
"""
import theano
import theano.tensor as T
import numpy as np
import sys

theano.config.optimizer = 'fast_run'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'ignore'
theano.config.floatX = 'float32'
theano.config.profile_memory = True


class TheanoGraph_pro(object):
    def __init__(self, u_grade=0, verbose=[0], dtype='float32', **kwargs):
        # Debugging options

        # self.dips_position_all = kwargs.get('dips_position',
        #                                     theano.shared(np.cast[dtype](np.zeros((2, 3))) ,"Position of the dips"))

        self.verbose = verbose

        # Creation of symbolic parameters

        # =============
        # Constants
        # =============
        self.i_reescale = theano.shared(np.cast[dtype](4))
        self.gi_reescale = theano.shared(np.cast[dtype](2))
        self.n_dimensions = 3

        # ======================
        # INITIALIZE SHARED
        # ==================
        self.u_grade_T = theano.shared(u_grade, "grade of the universal drift")
        self.c_resc = theano.shared(np.cast[dtype](1), "Rescaling factor")
        self.grid_val_T = theano.shared(np.cast[dtype](np.zeros((2, 3))))
        self.a_T = theano.shared(np.cast[dtype](1.))
        self.c_o_T = theano.shared(np.cast[dtype](1.))
        self.final_block = theano.shared(np.zeros(3, dtype='int'), "Final block computed")
        self.yet_simulated = theano.shared(np.ones(3, dtype='int'), "Points to be computed yet")

        self.nugget_effect_grad_T = theano.shared(np.cast[dtype](0.01))

        # Shape is 9x2, 9 drift funcitons and 2 points
        self.universal_grid_matrix_T = theano.shared(np.cast[dtype](np.zeros((9, 2))))

        self.len_series_i = theano.shared(np.zeros(3, dtype='int'), 'Length of interfaces in every series')
        self.len_series_f = theano.shared(np.zeros(3, dtype='int'), 'Length of foliations in every series')
        self.n_formations_per_serie = theano.shared(np.zeros(3, dtype='int'), 'List with the number of formations')
        self.n_formation = theano.shared(np.zeros(3, dtype='int'), "Value of the formation")
        self.number_of_points_per_formation_T = theano.shared(np.zeros(3, dtype='int'))

        # ======================
        # VAR
        #======================
        self.dips_position_all = T.matrix("Position of the dips")
        self.dip_angles_all = T.vector("Angle of every dip")
        self.azimuth_all = T.vector("Azimuth")
        self.polarity_all = T.vector("Polarity")
        self.ref_layer_points_all = T.matrix("Reference points for every layer")
        self.rest_layer_points_all = T.matrix("Rest of the points of the layers")

        #self.dips_position_all = theano.shared(np.cast[dtype](np.zeros((2, 3))) ,"Position of the dips")
        # self.dip_angles_all = theano.shared(np.cast[dtype](np.zeros(2)), "Angle of every dip")
        # self.azimuth_all = theano.shared(np.cast[dtype](np.zeros(2)), "Azimuth")
        # self.polarity_all = theano.shared(np.cast[dtype](np.zeros(2)), "Polarity")
        # self.ref_layer_points_all = theano.shared(np.cast[dtype](np.zeros((2, 3))) ,
        #                                            "Reference points for every layer")
        # self.rest_layer_points_all = theano.shared(np.cast[dtype](np.zeros((2, 3))) ,
        #                                             "Rest of the points of the layers")

        self.dips_position = self.dips_position_all
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # These are subsets of the data for each series
        self.dip_angles = self.dip_angles_all
        self.azimuth = self.azimuth_all
        self.polarity = self.polarity_all
        self.ref_layer_points = self.ref_layer_points_all
        self.rest_layer_points = self.rest_layer_points_all

      #  self.yet_simulated = self.yet_simulated_func()

    def testing(self):
        return self.rest_layer_points, self.ref_layer_points

    def yet_simulated_func(self, block=None):
        if not block:
            self.yet_simulated = T.eq(self.final_block, 0)
        else:
            self.yet_simulated = T.eq(block, 0)
        return self.yet_simulated

    def input_parameters_list(self):
        ipl = [self.dips_position_all, self.dip_angles_all, self.azimuth_all, self.polarity_all,
               self.ref_layer_points_all, self.rest_layer_points_all]
        return ipl

    @staticmethod
    def squared_euclidean_distances(x_1, x_2):
        sqd = T.sqrt(T.maximum(
            (x_1**2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2**2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))

        return sqd

    def matrices_shapes(self):

        # Calculating the dimensions of the
        length_of_CG = self.dips_position_tiled.shape[0]
        length_of_CGI = self.rest_layer_points.shape[0]
        if self.u_grade_T.get_value() == 0:
            length_of_U_I = 0
        else:
            length_of_U_I = 3**self.u_grade_T
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I

        if 'matrices_shapes' in self.verbose:
            length_of_CG = theano.printing.Print("length_of_CG")(length_of_CG)
            length_of_CGI = theano.printing.Print("length_of_CGI")(length_of_CGI)
            length_of_U_I = theano.printing.Print("length_of_U_I")(length_of_U_I)
            length_of_C = theano.printing.Print("length_of_C")(length_of_C)

        return length_of_CG, length_of_CGI, length_of_U_I, length_of_C

    def cov_interfaces(self):

        sed_rest_rest = self.squared_euclidean_distances(self.rest_layer_points, self.rest_layer_points)
        sed_ref_rest = self.squared_euclidean_distances(self.ref_layer_points, self.rest_layer_points)
        sed_rest_ref = self.squared_euclidean_distances(self.rest_layer_points, self.ref_layer_points)
        sed_ref_ref = self.squared_euclidean_distances(self.ref_layer_points, self.ref_layer_points)


        # Covariance matrix for interfaces
        C_I = (self.c_o_T * self.i_reescale * (
            (sed_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (sed_rest_rest / self.a_T) ** 2 +
             35 / 4 * (sed_rest_rest / self.a_T) ** 3 -
             7 / 2 * (sed_rest_rest / self.a_T) ** 5 +
             3 / 4 * (sed_rest_rest / self.a_T) ** 7) -
            ((sed_ref_rest < self.a_T) *  # Reference - Rest
             (1 - 7 * (sed_ref_rest / self.a_T) ** 2 +
              35 / 4 * (sed_ref_rest / self.a_T) ** 3 -
              7 / 2 * (sed_ref_rest / self.a_T) ** 5 +
              3 / 4 * (sed_ref_rest / self.a_T) ** 7)) -
            ((sed_rest_ref < self.a_T) *  # Rest - Reference
             (1 - 7 * (sed_rest_ref / self.a_T) ** 2 +
              35 / 4 * (sed_rest_ref / self.a_T) ** 3 -
              7 / 2 * (sed_rest_ref / self.a_T) ** 5 +
              3 / 4 * (sed_rest_ref / self.a_T) ** 7)) +
            ((sed_ref_ref < self.a_T) *  # Reference - References
             (1 - 7 * (sed_ref_ref / self.a_T) ** 2 +
              35 / 4 * (sed_ref_ref / self.a_T) ** 3 -
              7 / 2 * (sed_ref_ref / self.a_T) ** 5 +
              3 / 4 * (sed_ref_ref / self.a_T) ** 7))))  # '+ 10e-6

        C_I.name = 'Covariance Interfaces'

        return C_I

    def cov_gradients(self, verbose=0):

        sed_dips_dips = self.squared_euclidean_distances(self.dips_position_tiled, self.dips_position_tiled)

        # Cartesian distances between dips positions
        h_u = T.vertical_stack(
            T.tile(self.dips_position[:, 0] - self.dips_position[:, 0].reshape((self.dips_position[:, 0].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 1] - self.dips_position[:, 1].reshape((self.dips_position[:, 1].shape[0], 1)),
                   self.n_dimensions),
            T.tile(self.dips_position[:, 2] - self.dips_position[:, 2].reshape((self.dips_position[:, 2].shape[0], 1)),
                   self.n_dimensions))

        h_v = h_u.T

        # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        # every gradient direction covariance
        perpendicularity_matrix = T.zeros_like(sed_dips_dips)

        # Cross-covariances of x
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[0:self.dips_position.shape[0], 0:self.dips_position.shape[0]], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips_position.shape[0]:self.dips_position.shape[0] * 2,
            self.dips_position.shape[0]:self.dips_position.shape[0] * 2], 1)

        # Cross-covariances of y
        perpendicularity_matrix = T.set_subtensor(
            perpendicularity_matrix[self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3,
            self.dips_position.shape[0] * 2:self.dips_position.shape[0] * 3], 1)

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(sed_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                (h_u * h_v / sed_dips_dips ** 2) *
                ((
                     (sed_dips_dips < self.a_T) *  # first derivative
                     (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                                     35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                                     21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7))) +
                 (sed_dips_dips < self.a_T) *  # Second derivative
                 self.c_o_T * 7 * (9 * sed_dips_dips ** 5 - 20 * self.a_T ** 2 * sed_dips_dips ** 3 +
                                   15 * self.a_T ** 4 * sed_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
                (perpendicularity_matrix *
                 (sed_dips_dips < self.a_T) *  # first derivative
                 self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_dips / self.a_T ** 3 -
                               35 / 2 * sed_dips_dips ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_dips ** 5 / self.a_T ** 7)))
        )

        # Setting nugget effect of the gradients
        # TODO: This function can be substitued by simply adding the nugget effect to the diag
        C_G = T.fill_diagonal(C_G, -self.c_o_T * (-14 / self.a_T ** 2) + self.nugget_effect_grad_T)

        C_G.name = 'Covariance Gradient'

        if verbose > 1:
            theano.printing.pydotprint(C_G, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        return C_G

    def cov_interface_gradients(self):

        sed_dips_rest = self.squared_euclidean_distances(self.dips_position_tiled, self.rest_layer_points)
        sed_dips_ref  = self.squared_euclidean_distances(self.dips_position_tiled, self.ref_layer_points)

        # Cartesian distances between dips and interface points
        # Rest
        hu_rest = T.vertical_stack(
            (self.dips_position[:, 0] - self.rest_layer_points[:, 0].reshape(
                (self.rest_layer_points[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - self.rest_layer_points[:, 1].reshape(
                (self.rest_layer_points[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - self.rest_layer_points[:, 2].reshape(
                (self.rest_layer_points[:, 2].shape[0], 1))).T
        )

        # Reference point
        hu_ref = T.vertical_stack(
            (self.dips_position[:, 0] - self.ref_layer_points[:, 0].reshape(
                (self.ref_layer_points[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - self.ref_layer_points[:, 1].reshape(
                (self.ref_layer_points[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - self.ref_layer_points[:, 2].reshape(
                (self.ref_layer_points[:, 2].shape[0], 1))).T
        )

        # Cross-Covariance gradients-interfaces
        C_GI = self.gi_reescale * (
            (hu_rest *
             (sed_dips_rest < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_rest / self.a_T ** 3 -
                              35 / 2 * sed_dips_rest ** 3 / self.a_T ** 5 +
                              21 / 4 * sed_dips_rest ** 5 / self.a_T ** 7))) -
            (hu_ref *
             (sed_dips_ref < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_ref / self.a_T ** 3 -
                              35 / 2 * sed_dips_ref ** 3 / self.a_T ** 5 +
                              21 / 4 * sed_dips_ref ** 5 / self.a_T ** 7)))
        ).T

        C_GI.name = 'Covariance gradient interface'

        if str(sys._getframe().f_code.co_name)+'_g' in self.verbose:
            theano.printing.pydotprint(C_GI, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)
        return C_GI

    def cartesian_dist_reference_to_rest(self):
        # Cartesian distances between reference points and rest
        hx = T.stack(
            (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
            (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
            (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2])
        ).T

        return hx

    def universal_matrix(self):

        U_I = None
        U_G = None

        if self.u_grade_T.get_value() == 1:
            # ==========================
            # Condition of universality 1 degree

            # Gradients
            n = self.dips_position.shape[0]
            U_G = T.zeros((n * self.n_dimensions, self.n_dimensions))
            # x
            U_G = T.set_subtensor(
                U_G[:n, 0], 1)
            # y
            U_G = T.set_subtensor(
                U_G[n:n * 2, 1], 1
            )
            # z
            U_G = T.set_subtensor(
                U_G[n * 2: n * 3, 2], 1
            )

            #  Faults:  U_G = T.set_subtensor(U_G[:, -1], [0, 0, 0, 0, 0, 0])

            # Interface
            U_I = - self.cartesian_dist_reference_to_rest() * self.gi_reescale

        # Faults
        #    hxf = (T.lt(rest_layer_points[:, 0], 5) - T.lt(ref_layer_points[:, 0], 5))*2 + 1

        #    U_I = T.horizontal_stack(U_I, T.stack(hxf).T)

        elif self.u_grade_T.get_value() == 2:
            # ==========================
            # Condition of universality 2 degree
            # Gradients

            n = self.dips_position.shape[0]
            U_G = T.zeros((n * self.n_dimensions, 3 * self.n_dimensions))
            # x
            U_G = T.set_subtensor(U_G[:n, 0], 1)
            # y
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 1], 1)
            # z
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 2], 1)
            # x**2
            U_G = T.set_subtensor(U_G[:n, 3], 2 * self.gi_reescale * self.dips_position[:, 0])
            # y**2
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 4], 2 * self.gi_reescale * self.dips_position[:, 1])
            # z**2
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 5], 2 * self.gi_reescale * self.dips_position[:, 2])
            # xy
            U_G = T.set_subtensor(U_G[:n, 6], self.gi_reescale * self.dips_position[:, 1])  # This is y
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 6], self.gi_reescale * self.dips_position[:, 0])  # This is x
            # xz
            U_G = T.set_subtensor(U_G[:n, 7], self.gi_reescale * self.dips_position[:, 2])  # This is z
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 7], self.gi_reescale * self.dips_position[:, 0])  # This is x
            # yz
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 8], self.gi_reescale * self.dips_position[:, 2])  # This is z
            U_G = T.set_subtensor(U_G[n * 2:n * 3, 8], self.gi_reescale * self.dips_position[:, 1])  # This is y

            # Interface
            U_I = - T.stack(
                (self.gi_reescale * (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
                 self.gi_reescale * (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
                 self.gi_reescale * (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2]),
                 self.gi_reescale ** 2 * (self.rest_layer_points[:, 0] ** 2 - self.ref_layer_points[:, 0] ** 2),
                 self.gi_reescale ** 2 * (self.rest_layer_points[:, 1] ** 2 - self.ref_layer_points[:, 1] ** 2),
                 self.gi_reescale ** 2 * (self.rest_layer_points[:, 2] ** 2 - self.ref_layer_points[:, 2] ** 2),
                 self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1] - self.ref_layer_points[:, 0] *
                     self.ref_layer_points[:, 1]),
                 self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 0] *
                     self.ref_layer_points[:, 2]),
                 self.gi_reescale ** 2 * (
                     self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 1] *
                     self.ref_layer_points[:, 2]),
                 )).T

        if 'U_I' in self.verbose:
            U_I = theano.printing.Print('U_I')(U_I)

        if 'U_G' in self.verbose:
            U_G = theano.printing.Print('U_G')(U_G)

        if str(sys._getframe().f_code.co_name)+'_g' in self.verbose:
            theano.printing.pydotprint(U_I, outfile="graphs/" + sys._getframe().f_code.co_name + "_i.png",
                                       var_with_name_simple=True)

            theano.printing.pydotprint(U_G, outfile="graphs/" + sys._getframe().f_code.co_name + "_g.png",
                                       var_with_name_simple=True)
        if U_I:
            U_I.name = 'Drift interfaces'
            U_G.name = 'Drift foliations'

        return U_I, U_G

    def covariance_matrix(self):

        length_of_CG, length_of_CGI, length_of_U_I, length_of_C = self.matrices_shapes()
        C_G = self.cov_gradients()
        C_I = self.cov_interfaces()
        C_GI = self.cov_interface_gradients()
        U_I, U_G = self.universal_matrix()

        # =================================
        # Creation of the Covariance Matrix
        # =================================
        C_matrix = T.zeros((length_of_C, length_of_C))

        # First row of matrices
        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, 0:length_of_CG], C_G)

        C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, length_of_CG:length_of_CG + length_of_CGI], C_GI.T)

        if not self.u_grade_T.get_value() == 0:
            C_matrix = T.set_subtensor(C_matrix[0:length_of_CG, -length_of_U_I:], U_G)

        # Second row of matrices
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, 0:length_of_CG], C_GI)
        C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI,
                                   length_of_CG:length_of_CG + length_of_CGI], C_I)

        if not self.u_grade_T.get_value() == 0:
            C_matrix = T.set_subtensor(C_matrix[length_of_CG:length_of_CG + length_of_CGI, -length_of_U_I:], U_I)

            # Third row of matrices
            C_matrix = T.set_subtensor(C_matrix[-length_of_U_I:, 0:length_of_CG], U_G.T)
            C_matrix = T.set_subtensor(C_matrix[-length_of_U_I:, length_of_CG:length_of_CG + length_of_CGI], U_I.T)

        # TODO: deprecate
        self.C_matrix = C_matrix

        if str(sys._getframe().f_code.co_name) in self.verbose:
            C_matrix = theano.printing.Print('cov_function')(C_matrix)
        return C_matrix

    def b_vector(self, verbose = 0):

        length_of_C = self.matrices_shapes()[-1]
        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module
        G_x = T.sin(T.deg2rad(self.dip_angles)) * T.sin(T.deg2rad(self.azimuth)) * self.polarity
        G_y = T.sin(T.deg2rad(self.dip_angles)) * T.cos(T.deg2rad(self.azimuth)) * self.polarity
        G_z = T.cos(T.deg2rad(self.dip_angles)) * self.polarity

        G = T.concatenate((G_x, G_y, G_z))

        # Creation of the Dual Kriging vector
        b = T.zeros((length_of_C,))
        b = T.set_subtensor(b[0:G.shape[0]], G)

        if verbose > 1:
            theano.printing.pydotprint(b, outfile="graphs/" + sys._getframe().f_code.co_name + "_i.png",
                                       var_with_name_simple=True)
        b.name = 'b vector'
        return b

    def solve_kriging(self):

        C_matrix = self.covariance_matrix()
        b = self.b_vector()
        # Solving the kriging system
        # TODO: look for an eficient way to substitute nlianlg by a theano operation
        import theano.tensor.slinalg
        DK_parameters = theano.tensor.slinalg.solve(C_matrix,b)
        #T.dot(T.nlinalg.matrix_inverse(C_matrix), b)
        DK_parameters.name = 'Dual Kriging parameters'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            DK_parameters = theano.printing.Print(DK_parameters.name )(DK_parameters)
        return DK_parameters

    def x_to_interpolate(self, verbose=0):
        """
        here I add to the grid points also the references points(to check the value of the potential field at the
        interfaces). Also here I will check what parts of the grid have been already computed in a previous series
        to avoid to recompute.
        Returns:

        """
        #yet_simulated = self.yet_simulated_func()

        # Removing points no simulated
        pns = (self.grid_val_T * self.yet_simulated.reshape((self.yet_simulated.shape[0], 1))).nonzero_values()
      #  pns = theano.printing.Print('this is a very important value')(pns)

        # Adding the interface points
        grid_val = T.vertical_stack(pns.reshape((-1, 3)), self.rest_layer_points)

        if verbose > 1:
            theano.printing.pydotprint(grid_val, outfile="graphs/" + sys._getframe().f_code.co_name + ".png",
                                       var_with_name_simple=True)

        return grid_val

    def extend_dual_kriging(self):
        """
        So far I just make a matrix with the dimensions len(DK)x(grid) but in the future maybe I have to try to loop
        all this part so consume less memory
        Returns:

        """

        grid_val = self.x_to_interpolate()
        DK_parameters = self.solve_kriging()

        # ==============
        # Interpolator
        # ==============

        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)

        DK_weights = T.tile(DK_parameters, (grid_val.shape[0], 1)).T

        return DK_weights

    def gradient_contribution(self):

        weights = self.extend_dual_kriging()
        length_of_CG = self.matrices_shapes()[0]
        grid_val = self.x_to_interpolate()

        # Cartesian distances between the point to simulate and the dips
        hu_SimPoint = T.vertical_stack(
            (self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        )

        # Calculating euclidian distances between the point to simulate and the avalible data
        # SED_dips_SimPoint = (T.sqrt(
        #     (self.dips_position_tiled ** 2).sum(1).reshape((self.dips_position_tiled.shape[0], 1)) +
        #     (grid_val ** 2).sum(1).reshape((1, grid_val.shape[0])) -
        #     2 * self.dips_position_tiled.dot(grid_val.T))).astype("float32")

        sed_dips_SimPoint = self.squared_euclidean_distances(self.dips_position_tiled, grid_val)

        # Gradient contribution
        sigma_0_grad = T.sum(
            (weights[:length_of_CG, :] *
             self.gi_reescale *
             (-hu_SimPoint *
              (sed_dips_SimPoint < self.a_T) *  # first derivative
              (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * sed_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * sed_dips_SimPoint ** 5 / self.a_T ** 7)))),
            axis=0)

        sigma_0_grad.name = 'Contribution of the foliations to the potential field at every point of the grid'
        return sigma_0_grad

    def interface_contribution(self):

        weights = self.extend_dual_kriging()
        length_of_CG, length_of_CGI = self.matrices_shapes()[:2]
        grid_val = self.x_to_interpolate()

        sed_rest_SimPoint = self.squared_euclidean_distances(self.rest_layer_points, grid_val)
        sed_ref_SimPoint = self.squared_euclidean_distances(self.ref_layer_points, grid_val)

        # Interface contribution
        sigma_0_interf = (T.sum(
            -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
            (self.c_o_T * self.i_reescale * (
                (sed_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                (1 - 7 * (sed_rest_SimPoint / self.a_T) ** 2 +
                 35 / 4 * (sed_rest_SimPoint / self.a_T) ** 3 -
                 7 / 2 * (sed_rest_SimPoint / self.a_T) ** 5 +
                 3 / 4 * (sed_rest_SimPoint / self.a_T) ** 7) -
                ((sed_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                 (1 - 7 * (sed_ref_SimPoint / self.a_T) ** 2 +
                  35 / 4 * (sed_ref_SimPoint / self.a_T) ** 3 -
                  7 / 2 * (sed_ref_SimPoint / self.a_T) ** 5 +
                  3 / 4 * (sed_ref_SimPoint / self.a_T) ** 7)))), axis=0))

        sigma_0_interf.name = 'Contribution of the interfaces to the potential field at every point of the grid'
        return sigma_0_interf

    def universal_drift_contribution(self):

        weights = self.extend_dual_kriging()
        length_of_U_I = self.matrices_shapes()[2]
        grid_val = self.x_to_interpolate()
       # yet_simulated = self.yet_simulated_func()

        # Universal drift contribution
        # Universal terms used to calculate f0
        _universal_terms_layers = T.horizontal_stack(
            self.rest_layer_points,
            (self.rest_layer_points ** 2),
            T.stack((self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1],
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2],
                     self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2]), axis=1)).T

      #  universal_matrix = T.horizontal_stack(
      #      (self.universal_grid_matrix_T * self.yet_simulated).nonzero_values().reshape((9, -1)),
      #      _universal_terms_layers)
        # TODO: Fix this!!!
        if self.u_grade_T.get_value() == 0:
            f_0 = 0
        else:
            _universal_terms_interfaces = T.horizontal_stack(
                self.rest_layer_points,
                (self.rest_layer_points ** 2),
                T.stack((self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1],
                         self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2],
                         self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2]), axis=1)).T

            universal_grid_interfaces_matrix = T.horizontal_stack(
                (self.universal_grid_matrix_T * self.yet_simulated).nonzero_values().reshape((9, -1)),
                _universal_terms_interfaces)

            #universal_grid_interfaces_matrix = T.vertical_stack(self.universal_grid_matrix_T, self.rest_layer_points)

            gi_rescale_aux = T.repeat(self.gi_reescale, 9)
            gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
            _aux_magic_term = T.tile(gi_rescale_aux[:3**self.u_grade_T], (grid_val.shape[0], 1)).T
            f_0 = (T.sum(
                weights[-length_of_U_I:, :] * self.gi_reescale * _aux_magic_term *
                universal_grid_interfaces_matrix[:3**self.u_grade_T]
                , axis=0))

        if not type(f_0) == int:
            f_0.name = 'Contribution of the universal drift to the potential field at every point of the grid'
        return f_0

    #  def contribution_faults(self):
        # Contribution faults
        # f_1 = weights[-1, :] * T.lt(universal_matrix[0, :], 5) * 2 - 1
        #   f_1 = 0
        # Potential field
        # Value of the potential field

    def potential_field_at_grid(self):
      #  self.yet_simulated_func()
        sigma_0_grad = self.gradient_contribution()
        sigma_0_interf = self.interface_contribution()
        f_0 = self.universal_drift_contribution()
        length_of_CGI = self.matrices_shapes()[1]

        Z_x = (sigma_0_grad + sigma_0_interf + f_0)[:-length_of_CGI]

        Z_x.name = 'Value of the potential field at every point of the grid'
        return Z_x

    def potential_field_at_interfaces(self):

        sigma_0_grad = self.gradient_contribution()
        sigma_0_interf = self.interface_contribution()
        f_0 = self.universal_drift_contribution()
        length_of_CGI = self.matrices_shapes()[1]

        potential_field_interfaces = (sigma_0_grad + sigma_0_interf + f_0)[-length_of_CGI:]

        npf = T.cumsum(T.concatenate((T.stack(0), self.number_of_points_per_formation_T)))

        # Loop to obtain the average Zx for every intertace
        def average_potential(dim_a, dim_b, pfi):
            """

            :param dim: size of the rest values vector per formation
            :param pfi: the values of all the rest values potentials
            :return: average of the potential per formation
            """
            average = pfi[T.cast(dim_a, "int32"): T.cast(dim_b, "int32")].sum() / (dim_b - dim_a)
            return average

        potential_field_interfaces_unique, updates1 = theano.scan(
            fn=average_potential,
            outputs_info=None,
            sequences=dict(input=npf,
                           taps=[0, 1]),
            non_sequences=potential_field_interfaces)

        potential_field_interfaces_unique.name = 'Value of the potential field at the interfaces'

        if str(sys._getframe().f_code.co_name) in self.verbose:
            potential_field_interfaces_unique = theano.printing.Print(potential_field_interfaces_unique.name)\
                                                                      (potential_field_interfaces_unique)
        return potential_field_interfaces_unique

    def block_series(self):
        """
        Returns a list with the chunks of block models per series
        Returns:

        """

        Z_x = self.potential_field_at_grid()

        max_pot = T.max(Z_x)  #T.max(potential_field_unique) + 1
        min_pot = T.min(Z_x)   #T.min(potential_field_unique) - 1

        potential_field_at_interfaces = self.potential_field_at_interfaces()

        # A tensor with the values to segment
        potential_field_iter = T.concatenate((T.stack([max_pot]),
                                              T.sort(potential_field_at_interfaces)[::-1],
                                              T.stack([min_pot])))

        if "potential_field_iter" in self.verbose:
            potential_field_iter = theano.printing.Print("potential_field_iter")(potential_field_iter)

        # Loop to segment the distinct lithologies
        def compare(a, b, n_formation, Zx):
            return T.le(Zx, a) * T.ge(Zx, b) * n_formation

        partial_block, updates2 = theano.scan(
            fn=compare,
            outputs_info=None,
            sequences=[dict(input=potential_field_iter, taps=[0, 1]), self.n_formation],
            non_sequences=Z_x)

        partial_block = partial_block.sum(axis=0)

        partial_block.name = 'The chunk of block model of a specific series'
        if str(sys._getframe().f_code.co_name) in self.verbose:
            partial_block = theano.printing.Print(partial_block.name)(partial_block)

        return partial_block

    def compute_a_series(self,
                         len_i_0, len_i_1,
                         len_f_0, len_f_1,
                         n_form_per_serie_0, n_form_per_serie_1,
                         final_block):

        # ==================
        # Preparing the data
        # ==================
        self.yet_simulated = T.eq(final_block, 0)
        #yet_simulated = self.yet_simulated_func(final_block)
        self.yet_simulated.name = 'Yet simulated node'

        # Theano shared
        self.number_of_points_per_formation_T = self.number_of_points_per_formation_T[n_form_per_serie_0: n_form_per_serie_1]
        self.n_formation = self.n_formation[n_form_per_serie_0: n_form_per_serie_1]

        self.dips_position = self.dips_position_all[len_f_0: len_f_1, :]
        self.dips_position_tiled = T.tile(self.dips_position, (self.n_dimensions, 1))

        # Theano Var
        self.dip_angles = self.dip_angles_all[len_f_0: len_f_1]
        self.azimuth = self.azimuth_all[len_f_0: len_f_1]
        self.polarity = self.polarity_all[len_f_0: len_f_1]

        self.ref_layer_points = self.ref_layer_points_all[len_i_0: len_i_1, :]
        self.rest_layer_points = self.rest_layer_points_all[len_i_0: len_i_1, :]

        # Printing
        if 'yet_simulated' in self.verbose:
            self.yet_simulated = theano.printing.Print(self.yet_simulated.name)(self.yet_simulated)
        if 'n_formation' in self.verbose:
            self.n_formation = theano.printing.Print('n_formation')(self.n_formation)

        # ====================
        # Computing the series
        # ====================
        potential_field_contribution = self.block_series()
        final_block = T.set_subtensor(
            final_block[T.nonzero(T.cast(self.yet_simulated, "int8"))[0]],
            potential_field_contribution)

        return final_block

    def whole_block_model(self):

        final_block_init = self.final_block
        final_block_init.name = 'final block init'
        all_series_blocks, updates2 = theano.scan(
            fn=self.compute_a_series,
            outputs_info=[final_block_init],
            sequences=[dict(input=self.len_series_i, taps=[0, 1]),
                       dict(input=self.len_series_f, taps=[0, 1]),
                       dict(input=self.n_formations_per_serie, taps=[0, 1])]
           )

        return all_series_blocks[-1]

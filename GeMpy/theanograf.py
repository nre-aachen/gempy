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
# from .DataManagement import DataManagement.Interpolator

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'ignore'

class TheanoGraph:
    theano.config.optimizer = 'None'
    theano.config.exception_verbosity = 'high'
    theano.config.compute_test_value = 'ignore'
    def __init__(self, u_grade):

        from IPython.core.debugger import Tracer
       # this one triggers the debugger

        theano.config.optimizer = 'None'
        theano.config.exception_verbosity = 'high'
        theano.config.compute_test_value = 'ignore'

        # Creation of symbolic variables
        self.dips_position = T.matrix("Position of the dips")
        self.dip_angles = T.vector("Angle of every dip")
        self.azimuth = T.vector("Azimuth")
        self.polarity = T.vector("Polarity")
        self.ref_layer_points = T.matrix("Reference points for every layer")
        self.rest_layer_points = T.matrix("Rest of the points of the layers")

        self.u_grade_T = theano.shared(u_grade, "grade of the universal drift", allow_downcast=True)

        self.c_resc = theano.shared(1, "Rescaling factor", allow_downcast=True)
        self.grid_val_T = theano.shared(np.array([[0, 0., 0.],
                                                  [0, 0., 0.204082]]), allow_downcast=True)

        self.final_block = theano.shared(np.zeros_like(np.array([0, 0., 0.])),
                                         "Final block computed", allow_downcast=True)

        self.nugget_effect_grad_T = theano.shared(0.01, allow_downcast=True)
        self.number_of_points_per_formation_T = theano.shared(np.zeros(2), allow_downcast=True)
        self.a_T = theano.shared(1, allow_downcast=True)
        self.c_o_T = theano.shared(1, allow_downcast=True)
        self.universal_matrix_T = theano.shared(np.array([[0., 0.],
                                                     [0., 0.],
                                                     [0., 0.204082],
                                                     [0., 0.],
                                                     [0., 0.],
                                                     [0., 0.041649],
                                                     [0., 0.],
                                                     [0., 0.],
                                                     [0., 0.]]), allow_downcast=True)
        # Init values
        n_dimensions = 3
        grade_universal = self.u_grade_T

        # Calculating the dimensions of the
        length_of_CG = self.dips_position.shape[0] * n_dimensions
        length_of_CGI = self.rest_layer_points.shape[0]
        length_of_U_I = grade_universal
        length_of_C = length_of_CG + length_of_CGI + length_of_U_I

        # Extra parameters
        i_reescale = 1 / (self.c_resc ** 2)
        gi_reescale = 1 / self.c_resc

        # TODO: Check that the distances does not go nuts when I use too large numbers

        # Here we create the array with the points to simulate:
        #   grid points except those who have been simulated in a younger serie
        #   interfaces points to segment the lithologies
        self.yet_simulated = T.vector("boolean function that avoid to simulate twice a point of a different serie")
        grid_val = T.vertical_stack((self.grid_val_T * self.yet_simulated.reshape(
                                    (self.yet_simulated.shape[0], 1))).nonzero_values().reshape((-1, 3)),
                                    self.rest_layer_points)

        # ==========================================
        # Calculation of Cartesian and Euclidian distances
        # ===========================================
        # Auxiliary tile for dips and transformation to float 64 of variables in order to calculate
        #  precise euclidian
        # distances
        _aux_dips_pos = T.tile(self.dips_position, (n_dimensions, 1))#.astype("float64")
        _aux_rest_layer_points = self.rest_layer_points#.astype("float64")
        _aux_ref_layer_points = self.ref_layer_points#.astype("float64")
        _aux_grid_val = grid_val#.astype("float64")

        # Calculation of euclidian distances giving back float32
        SED_rest_rest = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_rest_layer_points.T))).astype("float32")

        SED_ref_rest = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_rest_layer_points.T))).astype("float32")

        SED_rest_ref = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_ref_layer_points.T))).astype("float32")

        SED_ref_ref = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_ref_layer_points.T))).astype("float32")

        SED_dips_dips = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_dips_pos ** 2).sum(1).reshape((1, _aux_dips_pos.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_dips_pos.T))).astype("float32")

        SED_dips_rest = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_rest_layer_points ** 2).sum(1).reshape((1, _aux_rest_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_rest_layer_points.T))).astype("float32")

        SED_dips_ref = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_ref_layer_points ** 2).sum(1).reshape((1, _aux_ref_layer_points.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_ref_layer_points.T))).astype("float32")

        # Calculating euclidian distances between the point to simulate and the avalible data
        SED_dips_SimPoint = (T.sqrt(
            (_aux_dips_pos ** 2).sum(1).reshape((_aux_dips_pos.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_dips_pos.dot(_aux_grid_val.T))).astype("float32")

        SED_rest_SimPoint = (T.sqrt(
            (_aux_rest_layer_points ** 2).sum(1).reshape((_aux_rest_layer_points.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_rest_layer_points.dot(_aux_grid_val.T))).astype("float32")

        SED_ref_SimPoint = (T.sqrt(
            (_aux_ref_layer_points ** 2).sum(1).reshape((_aux_ref_layer_points.shape[0], 1)) +
            (_aux_grid_val ** 2).sum(1).reshape((1, _aux_grid_val.shape[0])) -
            2 * _aux_ref_layer_points.dot(_aux_grid_val.T))).astype("float32")

        self.printing = SED_dips_dips

        # Cartesian distances between dips positions
        h_u = T.vertical_stack(
            T.tile(self.dips_position[:, 0] - self.dips_position[:, 0].reshape((self.dips_position[:, 0].shape[0], 1)),
                   n_dimensions),
            T.tile(self.dips_position[:, 1] - self.dips_position[:, 1].reshape((self.dips_position[:, 1].shape[0], 1)),
                   n_dimensions),
            T.tile(self.dips_position[:, 2] - self.dips_position[:, 2].reshape((self.dips_position[:, 2].shape[0], 1)),
                   n_dimensions))

        h_v = h_u.T

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

        # Cartesian distances between reference points and rest
        hx = T.stack(
            (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
            (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
            (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2])
        ).T

        # Cartesian distances between the point to simulate and the dips
        hu_SimPoint = T.vertical_stack(
            (self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
            (self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
            (self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
        )

        # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
        # every gradient direction covariance
        perpendicularity_matrix = T.zeros_like(SED_dips_dips)

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


        # printing = (self.c_o_T * i_reescale * (
        #     (SED_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
        #printing = (1 - 7 * (SED_rest_rest / self.a_T) ** 2 +
        #     35 / 4 * (SED_rest_rest / self.a_T) ** 3 -
        #     7 / 2 * (SED_rest_rest / self.a_T) ** 5 +
        #     3 / 4 * (SED_rest_rest / self.a_T) ** 7)
        printing = SED_rest_rest*self.a_T
        # ==========================
        # Creating covariance Matrix
        # ==========================
        # Covariance matrix for interfaces
        C_I = (self.c_o_T * i_reescale * (
            (SED_rest_rest < self.a_T) *  # Rest - Rest Covariances Matrix
            (1 - 7 * (SED_rest_rest / self.a_T) ** 2 +
             35 / 4 * (SED_rest_rest / self.a_T) ** 3 -
             7 / 2 * (SED_rest_rest / self.a_T) ** 5 +
             3 / 4 * (SED_rest_rest / self.a_T) ** 7) -
            ((SED_ref_rest < self.a_T) *  # Reference - Rest
             (1 - 7 * (SED_ref_rest / self.a_T) ** 2 +
              35 / 4 * (SED_ref_rest / self.a_T) ** 3 -
              7 / 2 * (SED_ref_rest / self.a_T) ** 5 +
              3 / 4 * (SED_ref_rest / self.a_T) ** 7)) -
            ((SED_rest_ref < self.a_T) *  # Rest - Reference
             (1 - 7 * (SED_rest_ref / self.a_T) ** 2 +
              35 / 4 * (SED_rest_ref / self.a_T) ** 3 -
              7 / 2 * (SED_rest_ref / self.a_T) ** 5 +
              3 / 4 * (SED_rest_ref / self.a_T) ** 7)) +
            ((SED_ref_ref < self.a_T) *  # Reference - References
             (1 - 7 * (SED_ref_ref / self.a_T) ** 2 +
              35 / 4 * (SED_ref_ref / self.a_T) ** 3 -
              7 / 2 * (SED_ref_ref / self.a_T) ** 5 +
              3 / 4 * (SED_ref_ref / self.a_T) ** 7)))) #'+ 10e-6

        SED_dips_dips = T.switch(T.eq(SED_dips_dips, 0), 1, SED_dips_dips)

        # Covariance matrix for gradients at every xyz direction and their cross-covariances
        C_G = T.switch(
            T.eq(SED_dips_dips, 0),  # This is the condition
            0,  # If true it is equal to 0. This is how a direction affect another
            (  # else, following Chiles book
                (h_u * h_v / SED_dips_dips ** 2) *
                ((
                     (SED_dips_dips < self.a_T) *  # first derivative
                     (-self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_dips / self.a_T ** 3 -
                                     35 / 2 * SED_dips_dips ** 3 / self.a_T ** 5 +
                                     21 / 4 * SED_dips_dips ** 5 / self.a_T ** 7))) +
                 (SED_dips_dips < self.a_T) *  # Second derivative
                 self.c_o_T * 7 * (9 * SED_dips_dips ** 5 - 20 * self.a_T ** 2 * SED_dips_dips ** 3 +
                                   15 * self.a_T ** 4 * SED_dips_dips - 4 * self.a_T ** 5) / (2 * self.a_T ** 7)) -
                (perpendicularity_matrix *
                 (SED_dips_dips < self.a_T) *  # first derivative
                 self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_dips / self.a_T ** 3 -
                               35 / 2 * SED_dips_dips ** 3 / self.a_T ** 5 +
                               21 / 4 * SED_dips_dips ** 5 / self.a_T ** 7)))
        )

        # Setting nugget effect of the gradients
        # TODO: This function can be substitued by simply adding the nugget effect to the diag
        C_G = T.fill_diagonal(C_G, -self.c_o_T * (-14 / self.a_T ** 2) + self.nugget_effect_grad_T)

        # Cross-Covariance gradients-interfaces
        C_GI = gi_reescale * (
            (hu_rest *
             (SED_dips_rest < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_rest / self.a_T ** 3 -
                              35 / 2 * SED_dips_rest ** 3 / self.a_T ** 5 +
                              21 / 4 * SED_dips_rest ** 5 / self.a_T ** 7))) -
            (hu_ref *
             (SED_dips_ref < self.a_T) *  # first derivative
             (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_ref / self.a_T ** 3 -
                              35 / 2 * SED_dips_ref ** 3 / self.a_T ** 5 +
                              21 / 4 * SED_dips_ref ** 5 / self.a_T ** 7)))
        ).T

        if self.u_grade_T.get_value() == 1:
            # ==========================
            # Condition of universality 1 degree

            # Gradients
            n = self.dips_position.shape[0]
            U_G = T.zeros((n * n_dimensions, n_dimensions))
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
            U_I = -hx * gi_reescale

        # Faults
        #    hxf = (T.lt(rest_layer_points[:, 0], 5) - T.lt(ref_layer_points[:, 0], 5))*2 + 1

        #    U_I = T.horizontal_stack(U_I, T.stack(hxf).T)

        elif self.u_grade_T.get_value() == 2:
            # ==========================
            # Condition of universality 2 degree
            # Gradients

            n = self.dips_position.shape[0]
            U_G = T.zeros((n * n_dimensions, 3 * n_dimensions))
            # x
            U_G = T.set_subtensor(U_G[:n, 0], 1)
            # y
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 1], 1)
            # z
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 2], 1)
            # x**2
            U_G = T.set_subtensor(U_G[:n, 3], 2 * gi_reescale * self.dips_position[:, 0])
            # y**2
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 4], 2 * gi_reescale * self.dips_position[:, 1])
            # z**2
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 5], 2 * gi_reescale * self.dips_position[:, 2])
            # xy
            U_G = T.set_subtensor(U_G[:n, 6], gi_reescale * self.dips_position[:, 1])  # This is y
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 6], gi_reescale * self.dips_position[:, 0])  # This is x
            # xz
            U_G = T.set_subtensor(U_G[:n, 7], gi_reescale * self.dips_position[:, 2])  # This is z
            U_G = T.set_subtensor(U_G[n * 2: n * 3, 7], gi_reescale * self.dips_position[:, 0])  # This is x
            # yz
            U_G = T.set_subtensor(U_G[n * 1:n * 2, 8], gi_reescale * self.dips_position[:, 2])  # This is z
            U_G = T.set_subtensor(U_G[n * 2:n * 3, 8], gi_reescale * self.dips_position[:, 1])  # This is y

            # Interface
            U_I = - T.stack(
                (gi_reescale * (self.rest_layer_points[:, 0] - self.ref_layer_points[:, 0]),
                 gi_reescale * (self.rest_layer_points[:, 1] - self.ref_layer_points[:, 1]),
                 gi_reescale * (self.rest_layer_points[:, 2] - self.ref_layer_points[:, 2]),
                 gi_reescale ** 2 * (self.rest_layer_points[:, 0] ** 2 - self.ref_layer_points[:, 0] ** 2),
                 gi_reescale ** 2 * (self.rest_layer_points[:, 1] ** 2 - self.ref_layer_points[:, 1] ** 2),
                 gi_reescale ** 2 * (self.rest_layer_points[:, 2] ** 2 - self.ref_layer_points[:, 2] ** 2),
                 gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1] - self.ref_layer_points[:, 0] *
                     self.ref_layer_points[:, 1]),
                 gi_reescale ** 2 * (
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 0] *
                     self.ref_layer_points[:, 2]),
                 gi_reescale ** 2 * (
                     self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2] - self.ref_layer_points[:, 1] *
                     self.ref_layer_points[:, 2]),
                 )).T



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


        self.C_matrix = C_matrix
        # =====================
        # Creation of the gradients G vector
        # Calculation of the cartesian components of the dips assuming the unit module
        G_x = T.sin(T.deg2rad(self.dip_angles)) * T.sin(T.deg2rad(self.azimuth)) * self.polarity
        G_y = T.sin(T.deg2rad(self.dip_angles)) * T.cos(T.deg2rad(self.azimuth)) * self.polarity
        G_z = T.cos(T.deg2rad(self.dip_angles)) * self.polarity

        self.G = T.concatenate((G_x, G_y, G_z))

        # Creation of the Dual Kriging vector
        b = T.zeros_like(C_matrix[:, 0])
        b = T.set_subtensor(b[0:self.G.shape[0]], self.G)

        # Solving the kriging system
        # TODO: look for an eficient way to substitute nlianlg by a theano operation
        self.DK_parameters = T.dot(T.nlinalg.matrix_inverse(C_matrix), b)

        # ==============
        # Interpolator
        # ==============

        # Creation of a matrix of dimensions equal to the grid with the weights for every point (big 4D matrix in
        # ravel form)
        weights = T.tile(self.DK_parameters, (grid_val.shape[0], 1)).T

        # Gradient contribution
        sigma_0_grad = T.sum(
            (weights[:length_of_CG, :] *
             gi_reescale *
             (-hu_SimPoint *
              (SED_dips_SimPoint < self.a_T) *  # first derivative
              (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * SED_dips_SimPoint / self.a_T ** 3 -
                               35 / 2 * SED_dips_SimPoint ** 3 / self.a_T ** 5 +
                               21 / 4 * SED_dips_SimPoint ** 5 / self.a_T ** 7)))),
            axis=0)

        # Interface contribution
        sigma_0_interf = (T.sum(
            -weights[length_of_CG:length_of_CG + length_of_CGI, :] *
            (self.c_o_T * i_reescale * (
                (SED_rest_SimPoint < self.a_T) *  # SimPoint - Rest Covariances Matrix
                (1 - 7 * (SED_rest_SimPoint / self.a_T) ** 2 +
                 35 / 4 * (SED_rest_SimPoint / self.a_T) ** 3 -
                 7 / 2 * (SED_rest_SimPoint / self.a_T) ** 5 +
                 3 / 4 * (SED_rest_SimPoint / self.a_T) ** 7) -
                ((SED_ref_SimPoint < self.a_T) *  # SimPoint- Ref
                 (1 - 7 * (SED_ref_SimPoint / self.a_T) ** 2 +
                  35 / 4 * (SED_ref_SimPoint / self.a_T) ** 3 -
                  7 / 2 * (SED_ref_SimPoint / self.a_T) ** 5 +
                  3 / 4 * (SED_ref_SimPoint / self.a_T) ** 7)))), axis=0))

        # Universal drift contribution
        # Universal terms used to calculate f0
        _universal_terms_layers = T.horizontal_stack(
            self.rest_layer_points,
            (self.rest_layer_points ** 2),
            T.stack((self.rest_layer_points[:, 0] * self.rest_layer_points[:, 1],
                     self.rest_layer_points[:, 0] * self.rest_layer_points[:, 2],
                     self.rest_layer_points[:, 1] * self.rest_layer_points[:, 2]), axis=1)).T

        universal_matrix = T.horizontal_stack(
            (self.universal_matrix_T * self.yet_simulated).nonzero_values().reshape((9, -1)),
            _universal_terms_layers)

        if self.u_grade_T.get_value() == 0:
            f_0 = 0
        else:
            gi_rescale_aux = T.repeat(gi_reescale, 9)
            gi_rescale_aux = T.set_subtensor(gi_rescale_aux[:3], 1)
            _aux_magic_term = T.tile(gi_rescale_aux[:grade_universal], (grid_val.shape[0], 1)).T
            f_0 = (T.sum(
                weights[-length_of_U_I:, :] * gi_reescale * _aux_magic_term *
                universal_matrix[:grade_universal]
                , axis=0))

        # Contribution faults
        # f_1 = weights[-1, :] * T.lt(universal_matrix[0, :], 5) * 2 - 1
        #   f_1 = 0
        # Potential field
        # Value of the potential field

        self.Z_x = (sigma_0_grad + sigma_0_interf + f_0)[:-self.rest_layer_points.shape[0]]
        self.potential_field_interfaces = (sigma_0_grad + sigma_0_interf + f_0)[-self.rest_layer_points.shape[0]:]

        # =======================================================================
        #               CODE TO EXPORT THE BLOCK DIRECTLY
        # ========================================================================


        # Value of the lithology-segment
        self.n_formation = T.vector("The assigned number of the lithologies in this serie")

        # Loop to obtain the average Zx for every intertace
        def average_potential(dim_a, dim_b, pfi):
            """

            :param dim: size of the rest values vector per formation
            :param pfi: the values of all the rest values potentials
            :return: average of the potential per formation
            """
            average = pfi[T.cast(dim_a, "int32"): T.cast(dim_b, "int32")].sum() / (dim_b - dim_a)
            return average

        potential_field_unique, updates1 = theano.scan(fn=average_potential,
                                                       outputs_info=None,
                                                       sequences=dict(
                                                           input=T.concatenate(
                                                               (T.stack(0),
                                                                self.number_of_points_per_formation_T)),
                                                           taps=[0, 1]),
                                                       non_sequences=self.potential_field_interfaces)

        infinite_pos = T.max(potential_field_unique) + 10
        infinite_neg = T.min(potential_field_unique) - 10

        # Loop to segment the distinct lithologies
        potential_field_iter = T.concatenate((T.stack([infinite_pos]),
                                              potential_field_unique,
                                              T.stack([infinite_neg])))

        def compare(a, b, n_formation, Zx):
            return T.le(Zx, a) * T.ge(Zx, b) * n_formation

        partial_block, updates2 = theano.scan(fn=compare,
                                              outputs_info=None,
                                              sequences=[dict(input=potential_field_iter, taps=[0, 1]),
                                                         self.n_formation],
                                              non_sequences=self.Z_x)

        # Adding to the block the contribution of the potential field
        self.potential_field_contribution = T.set_subtensor(
            self.final_block[T.nonzero(T.cast(self.yet_simulated, "int8"))[0]],
            partial_block.sum(axis=0))

        # Some gradient testing
        # grad = T.jacobian(T.flatten(printing), rest_layer_points)
        grad = T.grad(T.sum(self.Z_x), self.a_T)
        #from theano.compile.nanguardmode import NanGuardMode







import numpy as np
import theano
import theano.tensor as T


class GeoPhysicsPreprocessing():

    # TODO geophysics grid different that modeling grid
    def __init__(self, interp_data, z=2100, res_grav=[5, 5], n_cells=1000, grid=None):


        self.interp_data = interp_data
        self.res_grav = res_grav
        self.z = z
        self.closest_cells_index = None
        self.n_cells = n_cells
        self.vox_size = self.set_vox_size()

        if not grid:
            self.grid = interp_data.grid.gridastype(self.interp_data.dtype)
        else:
            self.grid = grid.astype(self.interp_data.dtype)

        self.airborne_plane = self.airborne_plane(z, res_grav)

    def set_airborne_plane(self, z, res_grav):

        # Rescale z
        z_res = (z-self.interp_data.centers[2])/self.interp_data.rescaling_factor + 0.5001

        # Create xy meshgrid
        xy = np.meshgrid(np.linspace(self.interp_data.extent_rescaled.iloc[0],
                                     self.interp_data.extent_rescaled.iloc[1], res_grav[0]),
                         np.linspace(self.interp_data.extent_rescaled.iloc[2],
                                     self.interp_data.extent_rescaled.iloc[3], res_grav[1]))
        z = np.ones(res_grav[0]*res_grav[1])*z_res

        # Transformation
        xy_ravel = np.vstack(map(np.ravel, xy))
        airborne_plane = np.vstack((xy_ravel, z)).T.astype(self.interp_data.dtype)

        return airborne_plane

    def compute_distance(self):
        # if the resolution is too high is going to consume too much memory

        # Theano function
        x_1 = T.matrix()
        x_2 = T.matrix()

        sqd = T.sqrt(T.maximum(
            (x_1 ** 2).sum(1).reshape((x_1.shape[0], 1)) +
            (x_2 ** 2).sum(1).reshape((1, x_2.shape[0])) -
            2 * x_1.dot(x_2.T), 0
        ))
        eu = theano.function([x_1, x_2], sqd)

        # Distance
        r = eu(self.grid, self.airborne_plane)

        return r

    def closest_cells(self):

        r = self.compute_distance()

        # This is a integer matrix at least
        self.closest_cells_index = np.argsort(r, axis=0)[:self.n_cells, :]

        # I need to make an auxiliary index for axis 1
        self._axis_1 = np.indices((self.n_cells, self.res_grav[0] * self.res_grav[1]))[1]

        # I think it is better to save it in memory since recompute distance can be too heavy
        self.selected_dist = r[self.closest_cells_index, self._axis_1]

    def select_grid(self):

        selected_grid = np.zeros((self.n_cells, 0))

        # I am going to loop it in order to keep low memory (first loop in gempy?)
        for i in range(self.res_grav[0] * self.res_grav[1]):

            selected_grid = np.hstack((selected_grid, self.grid[self.closest_cells_index, self._axis_1]))

        return selected_grid


    def set_vox_size(self):

        x_extent = self.interp_data.extent_rescaled.iloc[1] - self.interp_data.extent_rescaled.iloc[0]
        y_extent = self.interp_data.extent_rescaled.iloc[3] - self.interp_data.extent_rescaled.iloc[2]
        z_extent = self.interp_data.extent_rescaled.iloc[5] - self.interp_data.extent_rescaled.iloc[4]
        vox_size = np.array([x_extent, y_extent, z_extent]) / self.interp_data.resolution
        return vox_size

    def z_decomposition(self):
        from scipy.constants import G

        s_gr = self.select_grid()
        s_r = self.selected_dist

        x_cor = np.vstack((g[:, 0] - self.vox_size[0], g[:, 0] + self.vox_size[0])).T
        y_cor = np.vstack((g[:, 1] - self.vox_size[1], g[:, 1] + self.vox_size[1])).T
        z_cor = np.vstack((g[:, 2] - self.vox_size[2], g[:, 2] + self.vox_size[2])).T

        # Now we expand them in the 8 combinations. Equivalent to 3 nested loops
        #  see #TODO add paper
        x_matrix = np.repeat(x_cor, 4, axis=1)
        y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
        z_matrix = np.tile(z_cor, (1, 4))

        mu = np.array([1,-1,-1,1,-1,1,1,-1])

        tz = np.sum(- G * mu * (
                x_matrix * np.log(y_matrix + s_r) +
                y_matrix * np.log(x_matrix + s_r) -
                z_matrix * np.arctan(x_matrix * y_matrix /
                                    (z_matrix * s_r))), axis=1)




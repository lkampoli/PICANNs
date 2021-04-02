'''
Copyright 2020 Amanpreet Singh,
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''


import numpy as np

'''
scale - function to scale the input to the given range

Parameters ----- x ----- input
                 l ----- lower limit
                 u ----- upper limit
'''
def scale(x, l, u):
    return (x * (u - l)) + l


'''
Rectangle -------- Create a geometry class for 2d with the specified limits

Parameters ----- x_lower ---- lower limit in x dimension
                 x_upper ---- upper limit in x dimension
                 y_lower ---- lower limit in y dimension
                 y_upper ---- upper limit in y dimension
'''
class Rectangle():
    def __init__(self, x_lower, x_upper, y_lower, y_upper):
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_lower = y_lower
        self.y_upper = y_upper

    '''
    generate_points_on_boundary --- Generates specified number of random points on the boundary.
    
    Parameters ------- num_points -- number of points to generate
    
    Output -- Array of random points on the boundary.
    '''
    def generate_points_on_boundary(self, num_points):
        x_pts = np.random.rand(num_points, 1)
        y_pts = np.random.rand(num_points, 1)

        x_pts = scale(x_pts, self.x_lower, self.x_upper)
        y_pts = scale(y_pts, self.y_lower, self.y_upper)

        for it in range(num_points):
            push_to_axis = np.random.rand(1, 1)
            if push_to_axis > 0.5:
                push_to_upper = np.random.rand(1, 1)
                if push_to_upper > 0.5:
                    x_pts[it, 0] = self.x_lower
                else:
                    x_pts[it, 0] = self.x_upper
            else:
                push_to_upper = np.random.rand(1, 1)
                if push_to_upper > 0.5:
                    y_pts[it, 0] = self.y_lower
                else:
                    y_pts[it, 0] = self.y_upper

        return np.concatenate((x_pts, y_pts), axis=-1)

    '''
    generate_random_points --- Generates specified number of random points.

    Parameters ------- num_points -- number of points to generate

    Output -- Array of random points.
    '''
    def generate_random_points(self, num_points):
        x_pts = np.random.rand(num_points, 1)
        y_pts = np.random.rand(num_points, 1)

        x_pts = scale(x_pts, self.x_lower, self.x_upper)
        y_pts = scale(y_pts, self.y_lower, self.y_upper)

        return np.concatenate((x_pts, y_pts), axis=-1)

    '''
    generate_uniform_points --- Generates grid of size as specified.

    Parameters ------- num_points -- size of grid is num_points x num_points.

    Output -- Grid of the specified size vectorized.
    '''
    def generate_uniform_points(self, num_points):
        x_pts = np.linspace(self.x_lower, self.x_upper, num_points)
        y_pts = np.linspace(self.y_lower, self.y_upper, num_points)

        complete_set = np.zeros((x_pts.shape[0] * y_pts.shape[0], 2))

        idx = 0
        for x in range(x_pts.shape[0]):
            for y in range(y_pts.shape[0]):
                complete_set[idx, 0] = x_pts[x]
                complete_set[idx, 1] = y_pts[y]

                idx += 1
        return complete_set

    '''
    generate_corner_points --- Generates the 4 corner points

    Parameters ------- None

    Output -- Array of 4 corner points.
    '''
    def generate_corner_points(self):
        x_pts = np.random.rand(4, 1)
        y_pts = np.random.rand(4, 1)

        x_pts[0:2, 0] = self.x_lower
        x_pts[2:, 0] = self.x_upper

        y_pts[0, 0] = self.y_lower
        y_pts[1:, 0] = self.y_upper
        y_pts[2, 0] = self.y_lower
        y_pts[3:, 0] = self.y_upper

        return np.concatenate((x_pts, y_pts), axis=-1)


'''
Cube -------- Create a geometry class for 3d with the specified limits

Parameters ----- x_lower ---- lower limit in x dimension
                 x_upper ---- upper limit in x dimension
                 y_lower ---- lower limit in y dimension
                 y_upper ---- upper limit in y dimension
                 z_lower ---- lower limit in z dimension
                 z_upper ---- upper limit in z dimension
'''
class Cube():
    def __init__(self, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.y_lower = y_lower
        self.y_upper = y_upper
        self.z_lower = z_lower
        self.z_upper = z_upper

    '''
       generate_points_on_boundary --- Generates specified number of random points on the boundary.
    
       Parameters ------- num_points -- number of points to generate
    
       Output -- Array of random points on the boundary.
    '''
    def generate_points_on_boundary(self, num_points):
        x_pts = np.random.rand(num_points, 1)
        y_pts = np.random.rand(num_points, 1)
        z_pts = np.random.rand(num_points, 1)

        x_pts = scale(x_pts, self.x_lower, self.x_upper)
        y_pts = scale(y_pts, self.y_lower, self.y_upper)
        z_pts = scale(z_pts, self.z_lower, self.z_upper)

        for it in range(num_points):
            push_to_axis = np.random.rand(1, 1)
            if (push_to_axis > 1/3) and (push_to_axis < 2/3):
                push_to_upper = np.random.rand(1, 1)
                if push_to_upper > 0.5:
                    x_pts[it, 0] = self.x_lower
                else:
                    x_pts[it, 0] = self.x_upper
            elif push_to_axis < 1/3 :
                push_to_upper = np.random.rand(1, 1)
                if push_to_upper > 0.5:
                    y_pts[it, 0] = self.y_lower
                else:
                    y_pts[it, 0] = self.y_upper
            else:
                push_to_upper = np.random.rand(1, 1)
                if push_to_upper > 0.5:
                    z_pts[it, 0] = self.z_lower
                else:
                    z_pts[it, 0] = self.z_upper

        return np.concatenate((x_pts, y_pts, z_pts), axis=-1)

    '''
    generate_random_points --- Generates specified number of random points.

    Parameters ------- num_points -- number of points to generate

    Output -- Array of random points.
    '''
    def generate_random_points(self, num_points):
        x_pts = np.random.rand(num_points, 1)
        y_pts = np.random.rand(num_points, 1)
        z_pts = np.random.rand(num_points, 1)

        x_pts = scale(x_pts, self.x_lower, self.x_upper)
        y_pts = scale(y_pts, self.y_lower, self.y_upper)
        z_pts = scale(z_pts, self.z_lower, self.z_upper)

        return np.concatenate((x_pts, y_pts, z_pts), axis=-1)

    '''
    generate_uniform_points --- Generates grid of size as specified.

    Parameters ------- num_points -- size of grid is num_points x num_points.

    Output -- Grid of the specified size vectorized.
    '''
    def generate_uniform_points(self, num_points):
        x_pts = np.linspace(self.x_lower, self.x_upper, num_points)
        y_pts = np.linspace(self.y_lower, self.y_upper, num_points)
        z_pts = np.linspace(self.z_lower, self.z_upper, num_points)

        complete_set = np.zeros((x_pts.shape[0] * y_pts.shape[0] * z_pts.shape[0], 3))

        idx = 0
        for x in range(x_pts.shape[0]):
            for y in range(y_pts.shape[0]):
                for z in range(z_pts.shape[0]):
                    complete_set[idx, 0] = x_pts[x]
                    complete_set[idx, 1] = y_pts[y]
                    complete_set[idx, 2] = z_pts[z]
                    idx += 1
        return complete_set

    '''
    generate_corner_points --- Generates the 8 corner points

    Parameters ------- None

    Output -- Array of 8 corner points.
    '''
    def generate_corner_points(self):
        x_pts = np.random.rand(8, 1)
        y_pts = np.random.rand(8, 1)
        z_pts = np.random.rand(8, 1)

        x_pts[0:4, 0] = self.x_lower
        x_pts[4:, 0] = self.x_upper

        y_pts[0:2, 0] = self.y_lower
        y_pts[2:4, 0] = self.y_upper
        y_pts[4:6, 0] = self.y_lower
        y_pts[6:8, 0] = self.y_upper

        z_pts[0:8:2, 0] = self.y_upper
        z_pts[1:8:2, 0] = self.y_lower

        return np.concatenate((x_pts, y_pts, z_pts), axis=-1)


'''
MultiDimensionalCube -------- Create a geometry class for n-d with the specified limits

Parameters ----- lower ---- lower limit in any dimension
                 upper ---- upper limit in any dimension
'''


class MultiDimensionalCude():
    def __init__(self, n_dim, lower, upper):
        self.lower = lower
        self.upper = upper
        self.n_dims = n_dim

    '''
       generate_points_on_boundary --- Generates specified number of random points on the boundary.

       Parameters ------- num_points -- number of points to generate

       Output -- Array of random points on the boundary.
    '''

    def generate_points_on_boundary(self, num_points):
        pts = np.random.rand(num_points, self.n_dims)

        pts = scale(pts, self.lower, self.upper)

        for it in range(num_points):
            push_to_axis = np.random.rand(1, 1)
            for c in range(self.n_dims):
                if (push_to_axis > c / self.n_dims) and (push_to_axis < (c+1) / self.n_dims):
                    push_to_upper = np.random.rand(1, 1)
                    if push_to_upper > 0.5:
                        pts[it, c] = self.lower
                    else:
                        pts[it, c] = self.upper

        return pts

    '''
    generate_random_points --- Generates specified number of random points.

    Parameters ------- num_points -- number of points to generate

    Output -- Array of random points.
    '''

    def generate_random_points(self, num_points):
        pts = np.random.rand(num_points, self.n_dims)

        pts = scale(pts, self.lower, self.upper)

        return pts

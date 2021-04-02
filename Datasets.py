from torch.utils.data import Dataset
import numpy as np
import torch
'''
This is the data set definitions of all the data sets used at some point in the training.
'''


'''
GaussianMixtureDataset - Samples randomly from the two gaussians as specified in the parameters

Parameters -- mu1 ---- Mean of the first gaussian
              sigma1 - Variance of the first gaussian
              mu2 ---- Mean of the second gaussian
              sigma2 - Variance of the second gaussian

When indexed - 
Outputs ----- 1. Random points sampled from the Gaussian Mixture as float data type 
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''
class GaussianMixtureDataset(Dataset):
    def __init__(self, mu1, sigma1, mu2, sigma2, length):
        self.mu1 = mu1
        self.sig1 = np.sqrt(sigma1)
        self.mu2 = mu2
        self.sig2 = np.sqrt(sigma2)
        self.length = length
        self.data = np.zeros((self.length, 2))
        self.data_init()

    def data_init(self):
        self.data[0:int(self.length/2), :] = np.array(self.sig1).reshape(1, 2) * np.random.randn(int(self.length/2), 2)\
                                             + np.array(self.mu1).reshape(1, 2)
        self.data[int(self.length/2):, :] = np.array(self.sig2).reshape(1, 2) * np.random.randn(int(self.length/2), 2) \
                                            + np.array(self.mu2).reshape(1, 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :], np.zeros((1, 1)), np.zeros((1, 1))


'''
GaussianDataset - Generates a data set by sampling points from the 2d gaussian as specified in the parameters

Parameters -- length --------- Number of points to sample in the dataset.
              sigma ---------- Variance of the Gaussian
              mu ------------- Mean of the Gaussian. Has to be a list.

When indexed - 
Outputs ----- 1. Random sampled points from the gaussian
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''
class GaussianDataset(Dataset):
    def __init__(self, length, sigma, mu):
        self.length = length
        self.data = np.zeros([self.length, 2])
        self.sigma = sigma
        self.mu = mu
        self.init_data()

    def init_data(self):
        self.data = np.sqrt(self.sigma) * np.random.randn(self.length, 2) + self.mu

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :], np.zeros((1, 1)), np.zeros((1, 1))


'''
ConstantDataset - Target is a constant as specified in the parameters.

Parameters -- geometry ------- Takes in the a Geometry class as input to sample points from.
              constant ------- Constant that will be target.
              length --------- Number of points to sample in the dataset.

When indexed - 
Outputs ----- 1. Random points sampled from geometry as float data type
              2. Vector of same length as the random points of the constant
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. 
'''
class ConstantDataset(Dataset):
    def __init__(self, geometry, const, length):
        self.geom = geometry
        self.const = const
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > self.__len__()/20:
            x = self.geom.generate_random_points(1)
            boundary_flag = np.zeros((1, 1))
        else:
            x = self.geom.generate_points_on_boundary(1)
            boundary_flag = np.ones((1, 1))
        return x, self.const*np.ones((1, 1)), boundary_flag



'''
MultipleGaussianMixtureDataset - Samples randomly from the n gaussians as specified in the parameters

Parameters -- mu1 ---- Mean of the first gaussian
              sigma1 - Variance of the first gaussian
              mu2 ---- Mean of the second gaussian
              sigma2 - Variance of the second gaussian

When indexed - 
Outputs ----- 1. Random points sampled from the Gaussian Mixture as float data type 
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''
class MultipleGaussianMixtureDataset(Dataset):
    def __init__(self, n_gauss, length, mu, sigma):
        self.n_gauss = n_gauss
        self.length = length
        self.mu = mu
        self.sigma = np.sqrt(sigma)
        self.per_gauss = int(self.length/self.n_gauss)
        self.data = np.zeros((self.length, 2))
        self.data_init()

    def data_init(self):
        start_pt = 0
        end_pt = self.per_gauss
        for i in range(self.n_gauss):
            if i != self.n_gauss - 1:
                self.data[start_pt:end_pt, :] = np.array(self.sigma[i]).reshape(1, 2) * np.random.randn(self.per_gauss, 2) + np.array(self.mu[i]).reshape(1, 2)
                start_pt += self.per_gauss
                end_pt += self.per_gauss
            else:
                sz = self.data.shape[0] - start_pt
                self.data[start_pt:, :] = np.array(self.sigma[i]).reshape(1, 2) * np.random.randn(sz, 2) + np.array(self.mu[i]).reshape(1, 2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :], np.zeros((1, 1)), np.zeros((1, 1))



'''
MNIST_dataset - Loads the encoded MNIST data set

When indexed - 
Outputs ----- 1. Encoded MNIST point in nd space.
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''
class MNIST_dataset(Dataset):
    def __init__(self, path, dim):
        self.data = np.load(path)
        self.dim = dim

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :self.dim], np.zeros((1, 1)), np.zeros((1, 1))


'''
AnnulusDataset - Generates a data set by annulusly deforming points as sampled by the parameters.

Parameters -- length --------- Number of points to sample in the dataset.

When indexed - 
Outputs ----- 1. Random sampled points deformed annulusly
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''
class AnnulusDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.data = np.zeros([self.length, 2])
        self.init_data()

    def init_data(self):
        gauss_pts = np.random.randn(self.length, 2)
        r = gauss_pts[:, 0]**2 + gauss_pts[:, 1]**2
        self.data = gauss_pts/np.expand_dims((r**(1/3)), -1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :], np.zeros((1, 1)), np.zeros((1, 1))


'''
FunnyDistDataset - Generates a data set using the path provided. In our case we used points created using the 
                   algorithm mentioned in paper.

Parameters -- length --------- Number of points to sample in the dataset.

When indexed - 
Outputs ----- 1. Random sampled points deformed annulusly
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''
class FunnyDistDataset(Dataset):
    def __init__(self, path):
        self.data = np.loadtxt(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], np.zeros((1, 1)), np.zeros((1, 1))


'''
mdGaussianDataset - Generates a data set by sampling points from the n-d gaussian as specified in the parameters

Parameters -- length --------- Number of points to sample in the dataset.
              sigma ---------- Variance of the Gaussian
              mu ------------- Mean of the Gaussian. Has to be a list.
              cov ------------ Covariance Matrix
              dim ------------ dimensionality of data

When indexed - 
Outputs ----- 1. Random sampled points from the gaussian
              2. Vector of 0s which is to be used as target in the PINNs training
              3. Boundary Flag to denote whether the point is at the boundary of the geometry to apply 
                 boundary constraints. This here would be zeros as we do not impose any boundary 
                 conditions when training with our discreet formulation.
'''

class mdGaussianDataset_torch(Dataset):
    def __init__(self, length, mu, cov, dim):
        self.length = length
        self.dim = dim
        self.data = torch.zeros([self.length, self.dim])
        self.cov = torch.from_numpy(cov)
        self.mu = torch.from_numpy(mu.squeeze())
        self.dist = torch.distributions.MultivariateNormal(self.mu, self.cov)
        self.init_data()

    def init_data(self):
        self.data = self.dist.sample([self.length])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :], np.zeros((1, 1)), np.zeros((1, 1))
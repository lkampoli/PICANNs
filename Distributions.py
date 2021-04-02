import torch
import numpy as np
import math


'''
gauss - Gaussian function -with default parameters this is a unit gaussian.

Parametrs -    x -------- Input points (should come from the geometry class in our implementation).
               c -------- Constant never used.
               mu ------- Mean of the gaussian.
               sigma ---- Variance of the gaussian.

Outputs ---    1. Value of the Gaussian at the given points 

Note ------    Was used in the testing of our formulation in continuous domain.
'''
def gauss(x, c, mu=[0, 0], sigma=1):
    ret = torch.exp(-(torch.pow(x[:, 1:] - mu[0], 2) + torch.pow(x[:, 0:1] - mu[1], 2)) /(2 * (sigma**2))).squeeze()
    ret = ret/((sigma**2)*(2*torch.from_numpy(np.array(np.pi))))
    return ret


'''
gass_md ------ A multi dimensional gaussian function.

Parameters  -- x -------- Input points (should come from the geometry class in our implementation).
               c -------- Constant never used.
               mu ------- Mean of the gaussian.
               sigma ---- Variance of the gaussian.

Outputs ---    1. Value of the Gaussian at the given points 

'''
def gauss_md(x, c):

    dim = x.shape[1]

    mu = torch.zeros(dim, 1)
    sigma = torch.eye(dim)

    x = x.unsqueeze(-1)

    mu = mu.reshape(1, dim, 1).repeat(x.shape[0], 1, 1).cuda()
    sigma_r = sigma.reshape(1, dim, dim).repeat(x.shape[0], 1, 1).cuda()
    ret = torch.matmul(torch.matmul(torch.transpose(x - mu, 1, 2), torch.inverse(sigma_r)), x - mu)

    ret = torch.exp(-0.5 * ret)

    ret = ret/torch.sqrt((2 * math.pi)**dim * torch.det(sigma))

    return ret.squeeze()


'''
gass_md ------ A multi-dimensional unit gaussian function. 

Parameters  -- x -------- Input points (should come from the geometry class in our implementation).
               c -------- Constant never used.

Outputs ---    1. Log probability at the given point

'''
def gauss_md_2(x, c):

    dim = x.shape[1]

    mu = torch.zeros(dim).cuda()
    sigma = torch.eye(dim).cuda()

    m = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)

    ret = m.log_prob(x)

    return ret


'''
gass_md ------ A multi gaussian function. With default parameters this is the unit gaussian.

Parameters  -- x -------- Input points (should come from the geometry class in our implementation).
               c -------- Constant never used.
               mu ------- Mean of the gaussian.
               sigma ---- Should be named covariance. Takes in the Covariance Matrix.

Outputs ---    1. Log probability at the given point

'''
def gauss_md_2ms(x, mu=None, sigma=None):

    dim = x.shape[1]

    if mu is None:
        mu = torch.zeros(dim).cuda()
    else:
        mu = torch.from_numpy(mu).cuda()
    if sigma is None:
        sigma = torch.eye(dim).cuda()
    else:
        sigma = torch.from_numpy(sigma).cuda()

    m = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma)

    ret = m.log_prob(x)

    return ret


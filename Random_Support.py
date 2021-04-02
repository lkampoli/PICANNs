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
from Datasets import mdGaussianDataset_torch
import torch
from torch.autograd import grad
from sqrtm import sqrtm


'''
get_random_matrices --- Returns randomly sampled means and covariance matrix for the input dim

Parameters ------------ dim ---- dimensionality of data
'''
def get_random_matrices(dim):
    mu = np.random.uniform(-1, 1, [dim, 1])
    A = np.random.uniform(0, 0.75, (3*dim, dim))
    cov = np.matmul(A.transpose(), A)
    return mu, cov


'''
write_matrix_to_file --- Saves mean, covariance and determinant of the covariance Matrix 
                         provided to the specified path as txt files.

Parameters ------------- mu -------- Mean
                         cov ------- Covariance Matrix
                         path ------ Path where to save files.
'''
def write_matrix_to_file(mu, cov, path):
    file = open(path + 'mean.txt', "w")
    np.savetxt(file, mu)
    file.close()
    file = open(path + 'cov.txt', "w")
    np.savetxt(file, cov)
    file.close()
    det = np.linalg.det(cov)
    file = open(path + 'det.txt', "w")
    file.write(str(det))
    file.close()


'''
grad_in_batches  -------------- Divides the input in batches and computes the gradient of the network on it.

Parameters  ---- x ------------ Input, which is the array of points
                 num_batches -- Number of batches you want to divide the input in.
                 model -------- Network on GPU.
'''
def grad_in_batches(x, num_batches, model):
    x = x.detach().cpu().numpy()
    start_pt = 0
    jump = int(x.shape[0] / num_batches)
    end_pt = jump
    ret = np.zeros_like(x)

    for i in range(num_batches):
        torch.cuda.empty_cache()
        temp = torch.from_numpy(x[start_pt:end_pt, :]).cuda()
        temp = temp.requires_grad_()
        results_temp = model(temp)
        ret[start_pt:end_pt, :] = grad(results_temp.sum(), temp, create_graph=True, retain_graph=True)[0].detach().cpu().numpy()

        start_pt += jump
        end_pt += jump

        if end_pt > x.shape[0]:
            end_pt = x.shape[0]

    return ret


'''
write_w2 --------------- Computes the theortical w2 metric between unit gaussian and the gaussian specified using mu and 
                         cov inputs. Also approximates the w2 using the PICANN approach. Saves the true and approximated
                         w2 metrics in the path provided as a txt file.

Parameters ------------- dim ------- dimensionality of data
                         path ------ Path where to save files.
                         model ----- Network on GPU.
                         epoch ----- Epoch number that will be used to write in the file.
                         mu -------- Mean
                         cov ------- Covariance Matrix
'''
def write_w2(dim, path, model, epochs, mu, cov):
    ## Compute the Theoretical w2
    m1 = torch.zeros(dim, 1)
    sig1 = torch.eye(dim)

    m2 = torch.from_numpy(mu).float()
    sig2 = torch.from_numpy(cov).float()

    p1 = torch.pow(m1 - m2, 2).sum()
    p2 = torch.trace(sig1 + sig2 - 2*sqrtm(torch.matmul(torch.matmul(sqrtm(sig1), sig2), sqrtm(sig1))))
    t_w2 = torch.sqrt(p1 + p2)

    ## Approximated w2

    n_batches = 5
    x = mdGaussianDataset_torch(100000, mu, cov, dim)[:][0]
    y = x.detach().requires_grad_().float()
    dy = grad_in_batches(y, n_batches, model)
    x = x.detach().cpu().numpy()
    m = x.sum(axis=0)/x.shape[0]
    sig1 = np.cov(np.transpose(x))

    w = np.sqrt(((dy - x)**2).sum(axis=1).mean())

    w1 = np.sqrt(((dy - x)**2).sum()/x.shape[0])

    file = open(path + 'W2_e_' + str(epochs) + '.txt', "w")
    file.write("Network trained for : " + str(epochs) + " Epochs\n \n")
    file.write("Theoretical W2 : " + str(t_w2) + "\n")
    file.write("Approximated W2 : " + str(w1) + "\n")
    file.write("Percentage Error : " + str((t_w2 - w1)/t_w2 * 100)  + "\n")
    file.close()

'''
Copyright 2020 Amanpreet Singh,
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''



import torch
from torch.autograd import grad

'''
KL_loss ------------- Computes the KL divergence loss for the given input. (This is for the discrete formulation)
                      Refer to Eq 7 in paper.

Parameters -------- x --------------------------------- input points
                    y --------------------------------- output of the network
                    flag ------------------------------ boundary flag for points 
                    BC_error_func --------------------- Function to compute loss at the boundaries
                    func_f ---------------------------- Function f as specified in our formulation
                    func_g ---------------------------- Function g as specified in our formulation
                    constant -------------------------- normalisation constant for functions f and g
                    k --------------------------------- Not used here. 
                    periodic_Boundary_Flag ------------ Flag to specify the if we want to impose boundary conditions on 
                                                        input, i.e. to wrap around the points that go out of domain.
                                                        Note that this does not work well and should be kept as False.
                    add_boundary_loss_flag ------------ Flag if we want to add boundary loss in our loss function.
                                                        As pf now no boundary conditions are imposed. 
                                                        Not even implemented. 
                    deformation flag ------------------ True -------- diffeo given by x + grad(network)(Not mentioned in 
                                                        Paper. Difficult to impose convexity with such a formulation)
                                                        False ------- diffeo given by grad(network)
'''
def KLLoss(x, y, flag, BC_error_func, func_f, func_g, constant, k, periodic_Boundary_Flag=False,
           add_boundary_loss_flag=1, deformation_flag=False):
    dy, = grad(y.sum(), x, create_graph=True, retain_graph=True)

    # print("Gradient is NaN : ", (dy != dy).sum())

    if deformation_flag:
        dy_2 = x + dy
    else:
        dy_2 = dy

    x_grad_u = dy_2

    g_grad_u = func_g(x_grad_u, constant).squeeze()

    der = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).cuda()

    for dim in range(x.shape[1]):
        der[:, dim, :] = grad(dy[:, dim].sum(), x, create_graph=True, retain_graph=True)[0]

    if deformation_flag:
        I = torch.eye(x.shape[1])
        I = I.reshape((1, x.shape[1], x.shape[1]))
        I = I.repeat(x.shape[0], 1, 1).cuda()
        det_Hu = torch.det(I + der)
    else:
        det_Hu = torch.det(der)

    kl_loss = -(torch.log(det_Hu) + g_grad_u)#torch.log(g_grad_u))

    if deformation_flag:
        f_norm = torch.pow(der, 2).reshape(der.shape[0], der.shape[1]**2).sum(dim=-1)
    else:
        I = torch.eye(x.shape[1])
        I = I.reshape((1, x.shape[1], x.shape[1]))
        I = I.repeat(x.shape[0], 1, 1).cuda()

        f_norm = torch.pow(I - der, 2).reshape(der.shape[0], der.shape[1]**2).sum(dim=-1)

    kl_loss = kl_loss + k*f_norm

    return kl_loss.unsqueeze(-1)

'''
Hess_loss ------------- Loss function to train a network such that grad(u) = x

Parameters -------- x --------------------------------- input points
                    y --------------------------------- output of the network
                    flag ------------------------------ boundary flag for points 
                    BC_error_func --------------------- Function to compute loss at the boundaries
                    func_f ---------------------------- Function f as specified in our formulation
                    func_g ---------------------------- Function g as specified in our formulation
                    constant -------------------------- normalisation constant for functions f and g
                    k --------------------------------- Not used here. 
                    periodic_Boundary_Flag ------------ Flag to specify the if we want to impose boundary conditions on 
                                                        input, i.e. to wrap around the points that go out of domain.
                                                        Note that this does not work well and should be kept as False.
                    add_boundary_loss_flag ------------ Flag if we want to add boundary loss in our loss function.
                                                        As pf now no boundary conditions are imposed. 
                                                        Not even implemented. 
                    deformation flag ------------------ True -------- diffeo given by x + grad(network)(Not mentioned in 
                                                        Paper. Difficult to impose convexity with such a formulation)
                                                        False ------- diffeo given by grad(network)
                                                        
                    Most parameters are ignored. Are provided just to ensure we can use the same training code across 
                    all experiments.
'''
def HessLoss(x, y, flag, BC_error_func, func_f, func_g, constant, k, periodic_Boundary_Flag, add_boundary_loss_flag=1,
             deformation_flag=False):
    dy, = grad(y.sum(), x, create_graph=True, retain_graph=True)

    loss = torch.zeros_like(x[:, 0])

    for dim in range(x.shape[1]):
        loss = loss + torch.abs(dy[:, dim] - x[:, dim])

    return loss.unsqueeze(-1)


'''
IdentityLoss --- returns the mean of outputs as loss

Parameters -------- outputs -- output of network
                    targets -- not used here
'''
def IdentityLoss(outputs, targets):
    return outputs.sum()/outputs.shape[0]
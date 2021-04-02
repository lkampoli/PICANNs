import torch
from torch.autograd import grad

'''
Jac_Det ------------- Computes the Jacobian Determinant for the given input.

Parameters -------- x --------------------------------- input points
                    y --------------------------------- output of the network
                    func_g ---------------------------- Function g as specified in our formulation
                    constant -------------------------- normalisation constant for functions f and g
                    periodic_Boundary_Flag ------------ Flag to specify the if we want to impose boundary conditions on 
                                                        input, i.e. to wrap around the points that go out of domain.
                    flag ------------------------------ True -- diffeo given by gradient of network
                                                        False - diffeo given by x + grad(Network)
'''
def Jac_Det(y, x, func_g, constant, periodic_Boundary_Flag, flag=True):
    dy, = grad(y.sum(), x, create_graph=True, retain_graph=True)

    if periodic_Boundary_Flag:
        if flag:
            dy_2 = dy.clone()
            dy_2[:, 0] = Periodic_Cond(dy[:, 0], -3, 3)
            dy_2[:, 1] = Periodic_Cond(dy[:, 1], -3, 3)

            grad_u = dy_2
        else:
            dy_2 = dy.clone()
            dy_2[:, 0] = Periodic_Cond(x[:, 0] + dy[:, 0], -3, 3)
            dy_2[:, 1] = Periodic_Cond(x[:, 1] + dy[:, 1], -3, 3)

            x_grad_u = dy_2
    else:
        if flag:
            grad_u = dy
        else:
            x_grad_u = x + dy

    du_x, du_y = dy[:, 0], dy[:, 1]

    dy, = grad(du_x.sum(), x, create_graph=True, retain_graph=True)
    d2u_x, du_xy = dy[:, 0], dy[:, 1]

    dy, = grad(du_y.sum(), x, create_graph=True, retain_graph=True)
    d2u_y, du_yx = dy[:, 1], dy[:, 0]

    if flag:
        det_u = (d2u_x * d2u_y) - (du_xy * du_yx)
        g = func_g(grad_u, constant).squeeze()
    else:
        det_u = ((1 + d2u_x) * (1 + d2u_y)) - (du_xy * du_yx)
        g = func_g(x_grad_u, constant).squeeze()

    return g * det_u


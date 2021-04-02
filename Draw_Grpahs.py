'''
Copyright 2020 Amanpreet Singh,
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''


from PDE import *
import matplotlib.pyplot as plt
import numpy as np

'''
run_in_batches  ------ Divides the input in batches and computes the Jacobian Determinant on that.

Parameters  ---- x ------------ Input, which is the array of points
                 f ------------ Function g, which is the unit gaussian in our formulation
                 num_batches -- Number of batches you want to divide the input in.
                 p_flag ------- Periodic boundary flag
                 norm_const --- In case you want to scale function g.
                 flag --------- Diffeo formulation flag
                 model -------- Network on GPU. Can be CINNs or simple fully connected network.
'''
def run_in_batches(x, f, num_batches, p_flag, norm_const, flag, model):
    torch.cuda.empty_cache()

    x = x.detach().cpu().numpy()
    start_pt = 0
    jump = int(x.shape[0]/num_batches)
    end_pt = jump
    ret = np.zeros_like(x[:, 0])

    for i in range(num_batches):
        torch.cuda.empty_cache()
        temp = torch.from_numpy(x[start_pt:end_pt, :]).cuda()
        temp = temp.requires_grad_()
        results_temp = model(temp)
        ret[start_pt:end_pt] = Jac_Det(results_temp, temp, f, norm_const, p_flag, flag).detach().cpu().numpy()

        start_pt += jump
        end_pt += jump

        if end_pt > x.shape[0]:
            end_pt = x.shape[0]

    return ret


'''
grad_in_batches  ------ Divides the input in batches and computes the gradient of the network on it.

Parameters  ---- x ------------ Input, which is the array of points
                 num_batches -- Number of batches you want to divide the input in.
                 model -------- Network on GPU. Can be CINNs or simple fully connected network.
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

    torch.cuda.empty_cache()
    return ret[:, 0], ret[:, 1]


'''
draw_graphs ------ Draws all the graphs for the given input, for continuous formulation

Parameters  ---- y ------------ Input, which is the array of points
                 model -------- Network on GPU. Can be CINNs or simple fully connected network.
                 graph_path --- Folder location where to draw all graphs.
                 func_f ------- Function f, which is only used in the continuous formulation             
                 func_g ------- Function g, which is the unit gaussian in our formulation                 
                 norm_const --- In case you want to scale function g.
                 period_Boundary_Flag ------- Periodic boundary flag
'''
def draw_graphs(y, model, graph_path, func_f, func_g, norm_const, period_Boundary_Flag):
    num_batches = 5
    y1 = y.detach().cpu().numpy()

    hx = y1[:, 0].reshape(500, 500)
    hy = y1[:, 1].reshape(500, 500)

    hx1 = hx[0:-1:10, 0:-1:10]
    hy1 = hy[0:-1:10, 0:-1:10]

    plt.figure()
    plt.plot(hx1, hy1)
    plt.plot(np.transpose(hx1), np.transpose(hy1))
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.savefig(graph_path + 'testing_data.png')
    plt.close()

    # f = func_f(y, norm_const).cpu().numpy()
    # im = f.reshape(500, 500)
    #
    # # plt.figure(figsize=(10, 10))
    # ax = plt.gca()
    # x = plt.imshow(im)
    # # divider = make_axes_locatable(ax)
    # # cax = divider.append_axes("right", size="10%", pad=0.1)
    # #
    # plt.colorbar(x)
    # # plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.savefig(graph_path + 'f.png')
    # plt.close()

    g = func_g(y, norm_const).cpu().numpy()
    im = g.reshape(500, 500)

    plt.figure()
    plt.imshow(np.flipud(np.rot90(im)), origin='lower')
    plt.colorbar()
    # plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(graph_path + 'g.png')
    plt.close()

    y = y.detach().requires_grad_()
    results = model(y)

    results1 = results.detach().cpu().numpy()

    im = results1.reshape(500, 500)

    plt.figure()
    plt.imshow(im, origin='lower')
    plt.colorbar()
    # plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(graph_path + 'u.png')
    plt.close()


    # dy, = grad(results.sum(), y, create_graph=True, retain_graph=True)
    du_x, du_y = grad_in_batches(y, num_batches, model)

    # du_x = du_x.detach().cpu().numpy()
    # du_y = du_y.detach().cpu().numpy()

    du_x = du_x.reshape(500, 500)
    du_y = du_y.reshape(500, 500)

    grad_x = du_x[0:-1:10, 0:-1:10]
    grad_y = du_y[0:-1:10, 0:-1:10]

    # plt.figure()
    # plt.plot(grad_x, grad_y)
    # plt.plot(np.transpose(grad_x), np.transpose(grad_y))
    # plt.ylim(-2, 2)
    # plt.xlim(-2, 2)
    # plt.savefig(graph_path + 'deformation.png')
    # plt.close()

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.plot(grad_x, grad_y)
    ax.plot(np.transpose(grad_x), np.transpose(grad_y))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    fig.savefig(graph_path + 'Approximated_Deformation.png')
    plt.close(fig)


    g_jac_det = run_in_batches(y, func_g, num_batches, period_Boundary_Flag, norm_const, True, model)

    g_jac_det = g_jac_det.reshape(500, 500)

    # plt.figure()
    # plt.imshow(np.flipud(np.rot90(g_jac_det)), origin='lower')
    # plt.colorbar()
    # # plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.savefig(graph_path + 'g_jac_det.png')
    # plt.close()

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(np.flipud(np.rot90(g_jac_det)), origin='lower')
    fig.savefig(graph_path + 'Density_Approximation.png')
    plt.close(fig)

    del results, du_x, du_y, g, grad_x, grad_y, hx, hx1, hy, hy1, y, g_jac_det, im, results1, y1, model
    torch.cuda.empty_cache()


'''
draw_graphs ------ Draws all the graphs for the given input, for discrete formulation

Parameters  ---- y ------------ Input, which is the array of points
                 model -------- Network on GPU. Can be CINNs or simple fully connected network.
                 graph_path --- Folder location where to draw all graphs.
                 func_f ------- Function f, which is only used in the continuous formulation             
                 func_g ------- Function g, which is the unit gaussian in our formulation                 
                 norm_const --- In case you want to scale function g.
                 period_Boundary_Flag ------- Periodic boundary flag
'''
def draw_graphs_2(y, model, graph_path, func_f, func_g, norm_const, period_Boundary_Flag):

    y1 = y.detach().cpu().numpy()

    hx = y1[:, 1].reshape(500, 500)
    hy = y1[:, 0].reshape(500, 500)

    hx1 = hx[0:-1:10, 0:-1:10]
    hy1 = hy[0:-1:10, 0:-1:10]

    plt.figure()
    plt.plot(hx1, hy1)
    plt.plot(np.transpose(hx1), np.transpose(hy1))
    plt.savefig(graph_path + 'testing_data.png')
    plt.close()

    g = func_g(y, norm_const).cpu().numpy()
    im = g.reshape(500, 500)

    plt.figure()
    plt.imshow(im, origin='lower')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(graph_path + 'g.png')
    plt.close()

    y = y.detach().requires_grad_()
    results = model(y)

    results1 = results.detach().cpu().numpy()

    im = results1.reshape(500, 500)

    plt.figure()
    plt.imshow(im, origin='lower')
    plt.colorbar()
    plt.savefig(graph_path + 'u.png')
    plt.close()

    dy, = grad(results.sum(), y, create_graph=True, retain_graph=True)
    du_x, du_y = dy[:, 1], dy[:, 0]

    du_x = du_x.detach().cpu().numpy()
    du_y = du_y.detach().cpu().numpy()

    du_x = du_x.reshape(500, 500)
    du_y = du_y.reshape(500, 500)

    du_x = du_x + hx
    du_y = du_y + hy

    grad_x = du_x[0:-1:10, 0:-1:10]
    grad_y = du_y[0:-1:10, 0:-1:10]

    plt.figure()
    plt.plot(grad_x, grad_y)
    plt.plot(np.transpose(grad_x), np.transpose(grad_y))
    plt.ylim(0, 250)
    plt.xlim(0, 250)
    plt.savefig(graph_path + 'deformation.png')
    plt.close()

    # plt.figure()
    # plt.quiver(grad_x, grad_y)
    # plt.savefig(graph_path + 'quiver.png')
    # plt.close()

    del results, dy, du_x, du_y, g, grad_x, grad_y, hx, hx1, hy, hy1, y1
    torch.cuda.empty_cache()

    g_jac_det = run_in_batches(y, func_g, 5, period_Boundary_Flag, norm_const, False, model)

    g_jac_det = g_jac_det.reshape(500, 500)
    plt.figure()
    plt.imshow(g_jac_det, origin='lower')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(graph_path + 'g_jac_det.png')
    plt.close()

'''
Copyright 2020 Amanpreet Singh,
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''



import os
from Geometry import *
from Draw_Grpahs import *

'''
make_graphs_all_epochs ------- Function to generate graphs for the given set of parameters as explained in detail below.

Parameters ---- graph_path --- Folder location where to draw all graphs.
                model_path --- Folder location where all the models are stored.
                model -------- Model with the same parameters as the trained model. Should be on GPU.
                sv_e --------- An integer. This will be the epoch where you want to start drawing the graphs. 
                               The next graph will be at epoch 2*sv_e. Steps in sv_e will be taken until we reach the 
                               total epochs as specified by the next parameter.
                epochs ------- The last epoch till which we want the graphs to be drawn. If this is not a multiple of 
                               of sv_e. The largest multiple of sv_e before 'epochs' will the end point.
                func_g ------- Function g. In our formulation this should be the unit gaussian.
                norm_const --- If you need to scale the function g. In discrete formulation this will be unused. Was 
                               useful in the continuous formulation.
                
                periodic_boundary_flag -- Flag to indicate periodic boundary conditions. This is used only in continuous
                                          formulation.
                deformation flag -------- True -- Formulation used - diffeomorphism = x + grad(Network)
                                          False - Formulation used - diffeomorphism = grad(Network)
'''
def make_graphs_all_epochs(graph_path, model_path, model, sv_e, epochs, func_g, norm_const, periodic_boundary_flag,
                           deformation_flag):
    model_extension = '.pth'

    # Create a directory if it does not exist.
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)

    # Define the geometry in which to plot the graphs. This should be made a parameter.
    Geom = Rectangle(-2, 2, -2, 2)

    # Sample uniform grid from the above geometry and put them on gpu.
    y = Geom.generate_uniform_points(500)
    y_new = torch.from_numpy(y).type(torch.FloatTensor).cuda()

    # Draw for all epochs
    for i in range(sv_e, epochs+1, sv_e):
        # Load the corresponding model
        model.load(model_path + 'e_' + str(i) + model_extension)

        # Make the path specific to the epoch inside the main graph directory
        e_path = graph_path + 'e_' + str(i) + '/'
        if not os.path.exists(e_path):
            os.mkdir(e_path)

        # Use the correct method to draw graphs depending on the diffeo formulation
        if deformation_flag:
            draw_graphs_2(y_new, model, e_path, None, func_g, norm_const, periodic_boundary_flag)
        else:
            draw_graphs(y_new, model, e_path, None, func_g, norm_const, periodic_boundary_flag)

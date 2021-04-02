'''
Copyright 2020 Amanpreet Singh,
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''



from Geometry import *
from torch.autograd import grad
from Train import *
from Datasets import *
from ICNN import *
import os
from Loss_Function import *

'''
Use this file to train a network with desired initial conditions
'''

# Specify the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# PARAMETERS for the network
hidden_layers = 5
npl = 256
epochs = 100

dim = 8

# Variable to save model in multiples of this number of epochs
save_e = 10

# FILE PATHS
model_path = './Models/Softplus_11/Initial/{0}d/'.format(dim)
graph_path = './Graphs/Softplus_11/Initial/{0}d/'.format(dim)
model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_' + str(epochs) + '/'

os.makedirs(model_path + model_name, exist_ok=True)
os.makedirs(graph_path + model_name, exist_ok=True)

##################################################

# Define the model and put that on the GPU
model = ICNN(dim, 1, hidden_layers, npl)
model = model.cuda()

# Define the geometry you want to train your network on and create a data set from that.
Geom = MultiDimensionalCude(dim, -10, 10)
dataset_length = 20000

const_value = 0
train_data = ConstantDataset(Geom, const_value, dataset_length)

# Define the Loss function you want to impose
criterion = torch.nn.MSELoss(reduction='mean')

# Train the network
model = train_network(train_data, model, HessLoss, criterion, epochs, None, None, None, None, save_e,
                      model_path + model_name, cinn_flag=True)

# Save the network
model.save(model_path, model_name)

# For debugging
print("All Done")

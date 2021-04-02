from ICNN import *
import torch
from Train import *
from Datasets import *
from Distributions import *
import os
from Loss_Function import *
from Geometry import Rectangle
import matplotlib.pyplot as plt

'''
Use this file to train the forward part of the PICANN Network.
'''

# Specify the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# PARAMETERS for Network
hidden_layers = 5
npl = 128
epochs = 10000

dim = 2

# Regularization constant
k = 0

# Variable to save model in multiples of this
save_after_e = 100

Period_Boundary_Flag = F
CINN_Flag = True
deformation_flag = False

# Function g
func_g = gauss_md_2

# FILE PATHS
model_path = './Models/4c/'.format(dim)
graph_path = './Graphs/4c/'.format(dim)


model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_k_' + str(k) + '/'

os.makedirs(model_path + model_name, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)


# Initial model to start the training from.
init_model_path = './Models/Softplus_11/Initial/{0}d/'.format(dim)
init_model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_10000/e_10000.pth'

init_model_path = init_model_path + init_model_name


# Specify the data set that you want to train

# train_data = AnnulusDataset(20000)
# train_data = GaussianDataset(10000, 0.25, 1)
train_data = MultipleGaussianMixtureDataset(4, 10000, [[-0.5, -0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]],
                                            [[0.25**2, 0.25**2], [0.25**2, 0.25**2], [0.25**2, 0.25**2],
                                             [0.25**2, 0.25**2]])
# train_data = FunnyDistDataset('./funnydist_samples.txt')


# Define the model and put that on the GPU
model = ICNN(dim, 1, hidden_layers, npl)
model = model.cuda()

# Define the Loss function you want to impose
criterion = IdentityLoss

# Train the network
model = train_network(train_data, model, KLLoss, criterion, epochs, None, func_g, None, k, save_after_e,
                      model_path + model_name, init_weights_flag=False, initial_path=init_model_path,
                      periodic_Flag=Period_Boundary_Flag, cinn_flag=CINN_Flag, deformation_flag=deformation_flag)

# For debugging
print("All done")

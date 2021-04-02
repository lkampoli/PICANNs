from ICNN import *
import torch
from Train_Dual import *
from Datasets import *
from Geometry import *
import os

'Use this file to train the inverse part of the PICANN network'

# Specify the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# PARAMETERS for Network
hidden_layers = 5
npl = 128
epochs = 10000

dim = 2
density_epoch = 3000

# Regularization constant
k = 0

# FILE PATHS
model_path = './Models/Annulus/'
graph_path = './Graphs/Annulus/Dual_{}/'.format(density_epoch)

model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_k_' + str(k) + '/'

os.makedirs(model_path + model_name, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)


# Initial model to start the training from.
init_model_path = './Models/Softplus_11/Initial/{0}d/'.format(dim)
init_model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_10000/e_10000.pth'

init_model_path = init_model_path + init_model_name


u_path = model_path + model_name + 'e_{}.pth'.format(density_epoch)
v_path = model_path + 'Dual_{}/'.format(density_epoch) + model_name

os.makedirs(v_path, exist_ok=True)

# Variable to save model in multiples of this
save_after_e = 100

u = ICNN(dim, 1, hidden_layers, npl)
v = ICNN(dim, 1, hidden_layers, npl)

u.load(u_path)
v.load(init_model_path)

u = u.cuda()
v = v.cuda()


# Define the geometry you want to train your network on and create a data set from that.
# train_data = MultipleGaussianMixtureDataset(4, 10000, [[-0.5, -0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]], [[0.25**2, 0.25**2], [0.25**2, 0.25**2], [0.25**2, 0.25**2], [0.25**2, 0.25**2]])
train_data = AnnulusDataset(10000)

Train_Dual(u, v, train_data, save_after_e, epochs, v_path)
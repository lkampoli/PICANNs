from ICNN import *
import torch
from Train import *
from Datasets import *
from Distributions import *
import os
from Loss_Function import *
from Random_Support import *
from sqrtm import sqrtm
import argparse

parser = argparse.ArgumentParser(description='Random_Covariance_Experiments')

parser.add_argument('--Num_Samples', type=int, default=10000)
parser.add_argument('--GPU', type=str, default="0")
parser.add_argument('--Run', type=int, default=1)
parser.add_argument('--Dim', type=int, default=2)

args = parser.parse_args()

# Specify the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

# PARAMETERS for Network
hidden_layers = 5
npl = 128
epochs = 100

dim = args.Dim
run = args.Run

samples = args.Num_Samples

# Regularization constant
k = 0

# Variable to save model in multiples of this
save_after_e = 10

Period_Boundary_Flag = False
CINN_Flag = True
deformation_flag = False

# Function g
func_g = gauss_md_2

matrix_file_path = './Models/Random_Cov/{}d/Samples_{}k/Run_{}/'.format(dim, samples/1000, run)
model_path = './Models/Random_Cov/{}d/Samples_{}k/Run_{}/'.format(dim, samples/1000, run)
model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_k_' + str(k) + '/'

os.makedirs(model_path + model_name, exist_ok=True)

# Initial model to start the training from.
init_model_path = './Models/Softplus_11/Initial/{0}d/'.format(dim)
init_model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_10000/e_10000.pth'

init_model_path = init_model_path + init_model_name

mu, cov = get_random_matrices(dim)
write_matrix_to_file(mu, cov, matrix_file_path)


# Specify the data set that you want to train
train_data = mdGaussianDataset_torch(samples, mu, cov, dim)

# Define the model and put that on the GPU
model = ICNN(dim, 1, hidden_layers, npl)
model = model.cuda()

# Define the Loss function you want to impose
criterion = IdentityLoss

# Train the network
model = train_network(train_data, model, KLLoss, criterion, epochs, None, func_g, None, k, save_after_e,
                      model_path + model_name, init_weights_flag=False, initial_path=init_model_path,
                      periodic_Flag=Period_Boundary_Flag, cinn_flag=CINN_Flag, deformation_flag=deformation_flag, mu=mu,
                      cov=cov, w2_flag=True)

# For debugging
print("All done")

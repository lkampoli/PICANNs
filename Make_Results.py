from Geometry import *
from Distributions import *
from ICNN import *
from Draw_Graphs_Supp import *
from Datasets import *
import matplotlib.pyplot as plt

# Specify the GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dim = 2

# PARAMETERS for the network
hidden_layers = 5
npl = 128
epochs = 10000

# Regularisation constant
k = 0

# Starting from this epoch till the end with step size same as this variable
save_after_e = 10000

# Flags as used in the training of network.
Period_Boundary_Flag = False
deformation_flag = False

# Functions g and f. Note in discrete formulation we don't have any function f. function g is a unit gaussian.
func_g = gauss

# train_data = FunnyDistDataset('/hdscratch/Monge_Ampere/FunnyDist/oit-random-master/funnydist_samples.txt')
train_data = MultipleGaussianMixtureDataset(4, 10000, [[-0.5, -0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]],
                                            [[0.25**2, 0.25**2], [0.25**2, 0.25**2], [0.25**2, 0.25**2],
                                             [0.25**2, 0.25**2]])
data = train_data[:][0]


# mu = data[:, 0:dim].mean(axis=0)
# cov = np.cov(data[:, 0:dim].T)


# def func_g(x, c):
#     sum = torch.exp(gauss_md_2ms(x))
#     return sum

# func_g = gauss_md_2ms

# FILE PATHS
# FILE PATHS
model_path = './Models/4c/'.format(dim)
graph_path = './Graphs/4c/'.format(dim)
model_name = 'model_hl_' + str(hidden_layers) + '_npl_' + str(npl) + '_k_' + str(k) + '/'

model_path = model_path + model_name
graph_path = graph_path + model_name

os.makedirs(graph_path, exist_ok=True)


fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0, 0, 1, 1])
ax.set_axis_off()
fig.add_axes(ax)
ax.scatter(data[:, 0], data[:, 1], c='k', s=0.1)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
fig.savefig(graph_path + 'Training_data.png')
plt.close(fig)


# Define the model and put that on the GPU
model = ICNN(2, 1, hidden_layers, npl)
model = model.cuda()

# Call the function to generate graphs
make_graphs_all_epochs(graph_path, model_path, model, save_after_e, epochs, func_g,
                       norm_const=None, periodic_boundary_flag=Period_Boundary_Flag, deformation_flag=deformation_flag)

# For debugging
print("All Done")

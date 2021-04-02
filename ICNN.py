from torch import nn
import torch

'''
Custom Activation function used in the CINNs
Right now this is :     0  for x <= 0
                        x^(2.5) for x > 0
'''
class customActivation(nn.Module):
    def __init(self):
        super.__init__()

    def forward(self, x):
        # out = torch.zeros_like(x)
        # out[x <= 0] = torch.exp(x[x <= 0])
        # out[x > 0] = x[x > 0]**3 + x[x > 0]**2 + x[x > 0]

        # No Clone
        x[x <= 0] = 0

        return x**2.5

'''
Custom Activation function used in the CINNs
Right now this is :    Softplus ^ (1.1)
'''
class softplus_power(nn.Module):
    def __init(self):
        super.__init__()

    def forward(self, x):
        m = nn.Softplus()
        x = m(x)
        return x**1.1


'''
Simple Fully connected network with some special functions as defined below.
Parameters :        n_inputs --- number of inputs
                    n_outputs -- number of outputs
                    n_layers --- number of hidden layers
                    n_npl ------ number of nodes per hidden layer
'''
class ICNN(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_layers, n_npl):
        super().__init__()
        self.n_inps = n_inputs
        if n_layers == 0:
            self.layers = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.layers.append(nn.Linear(n_inputs, n_outputs))
        else:
            self.layers = nn.ModuleList()
            self.activations = nn.ModuleList()
            self.layers.append(nn.Linear(n_inputs, n_npl))
            self.activations.append(softplus_power())
            for x in range(n_layers-1):
                self.layers.append(nn.Linear(n_npl + n_inputs, n_npl))
                self.activations.append(softplus_power())
            # self.layers.append(nn.Linear(n_npl + n_inputs, n_npl))
            # self.activations.append(softplus_squared())
            self.layers.append(nn.Linear(n_npl + n_inputs, n_outputs))

    '''
    Forward pass - Simple fully connected network with given activations
    '''
    def forward(self, x):
        inps = x
        out = x
        for i in range(len(self.layers)-1):
            if i > 0:
                out = torch.cat((out, inps), dim=1)
            out = self.layers[i](out)
            out = self.activations[i](out)

        if len(self.layers) != 1:
            out = torch.cat((out, inps), dim=1)

        out = self.layers[-1](out)
        return out

    '''
    Makes the all the weights positive. 
    Refer to : https://arxiv.org/pdf/1609.07152.pdf
    '''
    def make_convex(self):
        with torch.no_grad():
            for i in range(1, len(self.layers)):
                if i == 0:
                    self.layers[i].weight[:, :][self.layers[i].weight[:, :] < 0] = \
                        torch.abs(self.layers[i].weight[:, :][self.layers[i].weight[:, :] < 0])
                self.layers[i].weight[:, :-self.n_inps][self.layers[i].weight[:, :-self.n_inps] < 0] = \
                    torch.abs(self.layers[i].weight[:, :-self.n_inps][self.layers[i].weight[:, :-self.n_inps] < 0])

    '''
    Sets all negative weights (W^z) to 0
    '''
    def project(self):
        with torch.no_grad():
            for i in range(1, len(self.layers)):
                if i == 0:
                    self.layers[i].weight[:, :][self.layers[i].weight[:, :] < 0] = 0
                self.layers[i].weight[:, :-self.n_inps][self.layers[i].weight[:, :-self.n_inps] < 0] = 0

    '''
    Saves the model in the specified path and name with .pth extension
    '''
    def save(self, path, model_name):
        torch.save(self.state_dict(), path + model_name + '.pth')

    '''
    Loads the model given its path.
    Different Architecture or activation function (with parameters) in the saved model will throw an error
    '''
    def load(self, path):
        self.load_state_dict(torch.load(path))



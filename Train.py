import numpy as np
import torch
from torch.utils.data import Dataset
from statistics import mean
from Loss_Function import *
import os
from torch import autograd
from Random_Support import write_w2

'''
zeroboundaryloss --- Returns a vector of 0s of the same length as input

Parameters ---------- x ---- input points
'''
def zeroboundaryloss(x):
    return torch.zeros((x.shape[0], 1)).type(torch.FloatTensor).cuda()


'''
init_weights ---------  Initializes the weights of the neural network. Weights and Biases of ICNN are initialized using 
                        a very small gaussian centered around 0.

Parameters ----------- m ------ modules in the network

Use as --------------- model.apply(init_weights)
'''
def init_weights(m):
    with torch.no_grad():
        if type(m) == torch.nn.Linear:
            m.weight.normal_(0, 0.01)
            m.bias.normal_(0, 0.01)


'''
train_network -------- Trains and returns a network with the specified conditions


Parameters ----------- data ----------------------- data set to be used for training.
                       model ---------------------- model that needs to be trained
                       loss_func ------------------ Function to compute the loss.
                       criterion ------------------ Function to compare targets with the loss.ex, MSE, Identity
                       n_epoch -------------------- number of epochs to train for
                       func_f --------------------- function f as stated in our formulation
                       func_g --------------------- function g as stated in our formulation
                       constant ------------------- normalisation constant if you want to normalise functions f and g
                       k -------------------------- Regularisation constant
                       save_e --------------------- Save the model in multiples of this variable
                       path ----------------------- Location to save the models
                       init_weights_flag ---------- If you want to initialize the weights or not
                       initial_path --------------- If we do not initialize weights then the path of the model that has
                                                    to be taken as the starting point of the training.
                       periodic_Flag -------------- Flag to wrap points that go out of domain
                       cinn_flag ------------------ Flag to indicate whether the model is a CINN to impose convexity 
                                                    constraints,
                       deformation_flag ----------- True -------- diffeo given by x + grad(network)
                                                    False ------- diffeo given by grad(network)
'''
def train_network(data, model, loss_func, criterion, n_epoch, func_f, func_g, constant, k, save_e, path,
                  init_weights_flag=True, initial_path=None, periodic_Flag=False, cinn_flag=False,
                  deformation_flag=False, Barrier_Flag=False, mu=None, cov=None, w2_flag=False):
    if not os.path.exists(path):
        os.mkdir(path)

    train_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1024, num_workers=8)
    if init_weights_flag:
        model.apply(init_weights)
    else:
        model.load(initial_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # if cinn_flag:
    model.make_convex()
    for epoch in range(1, n_epoch+1):
        epoch_loss = []
        for i, (x, y, flag) in enumerate(train_loader):

            x = torch.squeeze(x.type(torch.FloatTensor), 1).cuda()
            x = x.detach().requires_grad_()
            y = torch.squeeze(y.type(torch.FloatTensor), 1).cuda()
            flag = torch.squeeze(flag.type(torch.FloatTensor), 1).cuda()

            outputs = model(x)

            error = loss_func(x, outputs, flag, zeroboundaryloss, func_f, func_g, constant, k, periodic_Flag, 0, deformation_flag)

            if Barrier_Flag:
                error = Barrier_loss(error, model)


            loss = criterion(error, y)

            optimizer.zero_grad()
            loss.backward()

            epoch_loss.append(loss.item())
            optimizer.step()

            if cinn_flag:
                model.project()


        print('Loss after Epoch {} is : {}'.format(epoch, mean(epoch_loss)))

        if epoch % save_e == 0:
            model.save(path, 'e_' + str(epoch))
            if w2_flag:
                write_w2(x.shape[1], path, model, epoch, mu, cov)


    return model

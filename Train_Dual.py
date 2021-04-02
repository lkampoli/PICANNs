'''
Copyright 2020 Amanpreet Singh,
               Martin Bauer,
               Sarang Joshi

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''



import torch
from torch import autograd
from torch.autograd import grad
from statistics import mean

def Dual_Loss(u, v, x):
    y_true = u(x)
    Y, = grad(y_true.sum(), x, create_graph=True, retain_graph=True)

    v_dy = v(Y)

    x_recon, = grad(v_dy.sum(), Y, create_graph=True, retain_graph=True)

    y_hat = u(x_recon)

    y = v(x)
    Y, = grad(y.sum(), x, create_graph=True, retain_graph=True)

    v_dy = u(Y)

    x_recon1, = grad(v_dy.sum(), Y, create_graph=True, retain_graph=True)

    loss = torch.pow(x_recon - x, 2).sum()

    return loss


def Train_Dual(u, v, data, save_e, n_epoch, path):
    train_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=1024, num_workers=8)
    v.make_convex()
    optimizer = torch.optim.Adam(v.parameters(), lr=0.001)
    for epoch in range(1, n_epoch+1):
        epoch_loss = []
        for i, (x, y, flag) in enumerate(train_loader):

            x = torch.squeeze(x.type(torch.FloatTensor), 1).cuda()
            x = x.detach().requires_grad_()
            y = torch.squeeze(y.type(torch.FloatTensor), 1).cuda()
            flag = torch.squeeze(flag.type(torch.FloatTensor), 1).cuda()

            loss = Dual_Loss(u, v, x)

            optimizer.zero_grad()
            loss.backward()

            epoch_loss.append(loss.item())
            optimizer.step()

            v.project()


        print('Loss after Epoch {} is : {}'.format(epoch, mean(epoch_loss)))

        if epoch % save_e == 0:
            v.save(path, 'e_' + str(epoch))
    return

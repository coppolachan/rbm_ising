###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
#
#
# 2017 Guido Cossu <gcossu.work@gmail.com>
#
##############################################################


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from tqdm import *

import argparse
import torch
import torch.utils.data
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp

import rbm_pytorch

#####################################################
MNIST_SIZE = 784  # 28x28
#####################################################

def imgshow(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    ##Wmin = img.min
    ##Wmax = img.max
    ##plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)
    plt.imsave(f, npimg)


# Parse command line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', dest='model', default='mnist', help='choose the model',
                    type=str, choices=['mnist', 'ising'])
parser.add_argument('--ckpoint', dest='ckpoint', help='pass a saved state',
                    type=str)
parser.add_argument('--epochs', dest='epochs', default=10, help='number of epochs',
                    type=int)
parser.add_argument('--batch', dest='batches', default=128, help='batch size',
                    type=int)
parser.add_argument('--hidden', dest='hidden_size', default=500, help='hidden feature size',
                    type=int)
parser.add_argument('--ising_size', dest='ising_size', default=32, help='lattice size for this Ising 2D model',
                    type=int)
parser.add_argument('--k', dest='kCD', default=2, help='number of Contrastive Divergence steps',
                    type=int)
parser.add_argument('--imgout', dest='image_output_dir', default='./', help='directory in which to save output images',
                    type=str)
parser.add_argument('--train', dest='training_data', default='state0.data', help='path to training input data',
                    type=str)
parser.add_argument('--txtout', dest='text_output_dir', default='./', help='directory in which to save text output data',
                    type=str)

args = parser.parse_args()
print(args)
print("Using library:", torch.__file__)
hidden_layers = args.hidden_size * args.hidden_size

# For the MNIST data set
if args.model == 'mnist':
    model_size = MNIST_SIZE
    image_size = 28
    dataset = datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batches, shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./DATA/MNIST_data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=args.batches)
#############################
elif args.model == 'ising':
    # For the Ising Model data set
    model_size = args.ising_size * args.ising_size
    image_size = args.ising_size
    train_loader = torch.utils.data.DataLoader(rbm_pytorch.CSV_Ising_dataset(args.training_data, size=args.ising_size), shuffle=True,
                                               batch_size=args.batches, drop_last=True)

# Read the model, example
rbm = rbm_pytorch.RBM(k=args.kCD, n_vis=model_size, n_hid=hidden_layers)

# load the model, if the file is present
if args.ckpoint is not None:
    print("Loading saved network state from file", args.ckpoint)
    rbm.load_state_dict(torch.load(args.ckpoint))

##############################
# Training parameters

learning_rate = 0.01   # Mnist 0.3
mom = 0.0  # momentum
damp = 0.0  # dampening factor
wd = 0.0  # weight decay  ## 2.27 0.0002    Mnist 0.0001

train_op = optim.SGD(rbm.parameters(), lr=learning_rate,
                     momentum=mom, dampening=damp, weight_decay=wd)

# progress bar
pbar = tqdm(range(args.epochs))

loss_file = open(args.text_output_dir + "Loss_timeline.data_" + str(args.model) + "_lr" + str(learning_rate) + "_wd" + str(wd) + "_mom" + str(
    mom) + "_epochs" + str(args.epochs), "w")

# Run the RBM training
for epoch in pbar:
    loss_ = []
    full_reconstruction_error = []

    for i, (data, target) in enumerate(train_loader):
        data_input = Variable(data.view(-1, model_size))
        # how to randomize?
        new_visible, hidden, h_prob, v_prob = rbm(data_input)

        # loss function: see Fisher eq 28 (Training RBM: an Introduction)
        # the average on the training set of the gradients is
        # the sum of the derivative averaged over the training set minus the average on the model
        # still possible instabilities here, so I am computing the gradient myself
        data_free_energy = rbm.free_energy(
            data_input)  # note: it does not include Z
        loss = data_free_energy - rbm.free_energy(new_visible)
        loss_.append(loss.data[0])

        reconstruction_error = rbm.loss(data_input, new_visible)
        full_reconstruction_error.append(reconstruction_error.data[0])
        loss_file.write(str(i) + "\t" + str(epoch) + "\t" + str(loss.data[0]) + "\t" + str(
            data_free_energy.data[0]) + "\t" + str(reconstruction_error.data[0]) + "\n")

        # Update gradients
        train_op.zero_grad()
        # manually update the gradients, do not use autograd
        rbm.backward(data_input, new_visible, h_prob)
        train_op.step()

       
    re_mean = np.mean(full_reconstruction_error)
    loss_mean = np.mean(loss_)
    pbar.set_description("Epoch %3d - Loss %8.5f - RE %5.3g " % (epoch, loss_mean, re_mean))

    # confirm output
    #imgshow(args.image_output_dir + "real" + str(epoch),
    #        make_grid(data_input.view(-1, 1, image_size, image_size).data))
    #imgshow(args.image_output_dir + "generate" + str(epoch),
    #        make_grid(new_visible.view(-1, 1, image_size, image_size).data))

    # Normalize the 
    imgshow(args.image_output_dir + "parameter" + str(epoch),
            make_grid(rbm.W.view(hidden_layers, 1, image_size, image_size).data))
    #imgshow(args.image_output_dir + "hidden" + str(epoch),
    #        make_grid(hidden.view(-1, 1, args.hidden_size, args.hidden_size).data))

    #np.savetxt(args.text_output_dir + "W.data" +
    #           str(epoch), rbm.W.data.numpy())
    # .data is used to retrieve the tensor held by the Parameter(Variable) W, then we can get the numpy representation

    if epoch % 10  == 0 :
        torch.save(rbm.state_dict(), "trained_rbm.pytorch."+str(epoch))

# Save the model
torch.save(rbm.state_dict(), "trained_rbm.pytorch.last")
loss_file.close()

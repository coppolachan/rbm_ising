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
import sys

from tqdm import *

import argparse
import torch
import torch.utils.data
# import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from math import exp, sqrt

import rbm_pytorch

#####################################################
MNIST_SIZE = 784  # 28x28
#####################################################


def imgshow(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    Wmin = img.min
    Wmax = img.max
    plt.imsave(f, npimg, vmin=Wmin, vmax=Wmax)


def sample_probability(prob, random):
    """Get samples from a tensor of probabilities.

        :param probs: tensor of probabilities
        :param rand: tensor (of the same shape as probs) of random values
        :return: binary sample of probabilities
    """
    torchReLu = nn.ReLU()
    return torchReLu(torch.sign(prob - random)).data


def hidden_from_visible(visible, W, h_bias):
    # Enable or disable neurons depending on probabilities
    probability = torch.sigmoid(F.linear(visible, W, h_bias))
    random_field = torch.rand(probability.size())
    new_states = sample_probability(probability, random_field)
    return new_states, probability


def visible_from_hidden(hid, W, v_bias):
    # Enable or disable neurons depending on probabilities
    probability=torch.sigmoid(F.linear(hid, W.t(), v_bias))
    random_field=torch.rand(probability.size())
    new_states=sample_probability(probability, random_field)
    return new_states, probability


def sample_from_rbm(steps, nstates, v_in, W, v_bias, h_bias, image_size):
    """ Samples from the RBM distribution function """

    # Random initial visible state
    #v = F.relu(torch.sign(torch.rand(nstates,v_bias.shape[0])-0.5)).data
    v=torch.zeros(nstates, v_bias.shape[0])
    #v = v_in
    
    v_prob = v
    # Run the Gibbs sampling for a number of steps
    for s in xrange(steps):
        if (s%2 == 0):
            imgshow(args.image_output_dir + "dream" + str(s), make_grid(v_prob.view(-1, 1, image_size, image_size)))
            print(s,"OK")
        h, h_prob=hidden_from_visible(v, W, h_bias)
        v, v_prob=visible_from_hidden(h, W, v_bias)
    return v


# Parse command line arguments
parser=argparse.ArgumentParser(description = 'Process some integers.')
parser.add_argument('--model', dest = 'model', default = 'mnist', help = 'choose the model',
                    type = str, choices = ['mnist', 'ising'])
parser.add_argument('--ckpoint', dest = 'ckpoint', help = 'pass a saved state',
                    type = str)
parser.add_argument('--imgout', dest = 'image_output_dir', default = './', help = 'directory in which to save output images',
                    type = str)
parser.add_argument('--train', dest = 'training_data', default = 'state0.data', help = 'path to training input data',
                    type = str)
parser.add_argument('--txtout', dest = 'text_output_dir', default = './', help = 'directory in which to save text output data',
                    type = str)
parser.add_argument('--ising_size', dest = 'ising_size', default = 32, help = 'lattice size for this Ising 2D model',
                    type = int)
parser.add_argument('--batch', dest = 'batches', default = 128, help = 'batch size',
                    type = int)
parser.add_argument('--hidden', dest = 'hidden_size', default = 500, help = 'hidden feature size',
                    type = int)

args=parser.parse_args()
print(args)
hidden_layers=args.hidden_size * args.hidden_size


# For the MNIST data set
if args.model == 'mnist':
    model_size=MNIST_SIZE
    image_size=28
    train_loader=datasets.MNIST('./DATA/MNIST_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
#############################
elif args.model == 'ising':
    # For the Ising Model data set
    model_size = args.ising_size * args.ising_size
    image_size = args.ising_size
    train_loader = torch.utils.data.DataLoader(rbm_pytorch.CSV_Ising_dataset(args.training_data, size=args.ising_size), shuffle=True,
                                               batch_size=args.batches)


# Read the model, example
rbm = rbm_pytorch.RBM(k=1, n_vis=model_size, n_hid=hidden_layers)

# load the model, if the file is present
try:
    print("Loading saved network state from file", args.ckpoint)
    rbm.load_state_dict(torch.load(args.ckpoint))
except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

print('Model succesfully loaded')


data = torch.zeros(30, model_size)
for i in xrange(30):
    data[i] = train_loader[i+100][0].view(-1, model_size)

#print(data)
#data_input = data.view(-1, model_size)
sample_from_rbm(500, 30, data, rbm.W.data, rbm.v_bias.data, rbm.h_bias.data, image_size)


###############################################################
#
# Restricted Binary Boltzmann machine in Pytorch
# Possible input sets: MNIST, ISING model configurations
#
# RBM module
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

class CSV_Ising_dataset(Dataset):
    """ Defines a CSV reader """
    def __init__(self, csv_file, size=32, transform=None):
        self.csv_file = csv_file
        self.size = size
        csvdata = np.loadtxt(csv_file, delimiter=",",
                             skiprows=1, dtype="float32")
        self.imgs = torch.from_numpy(csvdata.reshape(-1, size * size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)

class Numpy_Ising_dataset(Dataset):
    """ A reader for npy files 
        still debugging, do not use
    """
    def __init__(self, npy_file, size=32, transform=None):
        self.npy_file = npy_file
        self.size = size
        isingdata = np.load(npy_file)
        print(np.asarray(isingdata))
        self.imgs = torch.from_numpy(isingdata.reshape(-1, size * size))
        self.datasize, sizesq = self.imgs.shape
        self.transform = transform
        print("Loaded training set of %d states" % self.datasize)

    def __getitem__(self, index):
        return self.imgs[index], index

    def __len__(self):
        return len(self.imgs)


class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500, k=5):
        super(RBM, self).__init__()
        # definition of the constructor
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.W = nn.Parameter(torch.rand(n_hid, n_vis) * 1e-2 )
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))

        self.CDiter = k

    def sample_probability(self, prob, random):
        """Get samples from a tensor of probabilities.

            :param probs: tensor of probabilities
            :param rand: tensor (of the same shape as probs) of random values
            :return: binary sample of probabilities
        """
        return F.relu(torch.sign(prob - random))

    def hidden_from_visible(self, visible, temperature = 1):
        # Enable or disable neurons depending on probabilities
        probability = torch.sigmoid(F.linear(visible, self.W, self.h_bias))
        random_field = Variable(torch.rand(probability.size()))
        new_states = self.sample_probability(probability.div(temperature), random_field)
        return new_states, probability

    def visible_from_hidden(self, hid, temperature = 1):
        # Enable or disable neurons depending on probabilities
        probability = torch.sigmoid(F.linear(hid, self.W.t(), self.v_bias))
        random_field = Variable(torch.rand(probability.size()))
        new_states = self.sample_probability(probability.div(temperature), random_field)

        return new_states, probability

    def new_state(self, visible):
        hidden, probhid = self.hidden_from_visible(visible)
        new_visible, probvis = self.visible_from_hidden(probhid)
        # new_hidden, probhid_new = self.hidden_from_visible(probvis)

        return hidden, probhid, new_visible, probvis

    def forward(self, input):
        """
        Necessary function for Module classes
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.

        """
        #hidden, h_prob, new_vis, v_prob = self.new_state(input)

        # Contrastive divergence
        new_vis = input
        
        for _ in range(self.CDiter):
            hidden, h_prob, vis, v_prob = self.new_state(new_vis)
            new_vis = vis

        return new_vis, hidden, h_prob, v_prob

    def loss(self, ref, test):
        # mseloss = torch.nn.MSELoss()# not ok, it will check for gradients
        # difference in prediction
        return F.mse_loss(test, ref, size_average=True)

    def free_energy(self, v):
        # computes -log( p(v) )
        # eq 2.20 Asja Fischer
        # double precision
        vbias_term = v.mv(self.v_bias).double()  # = v*v_bias
        wx_b = F.linear(v, self.W, self.h_bias).double()  # = vW^T + h_bias
        
        # sum over the elements of the vector
        hidden_term = wx_b.exp().add(1).log().sum(1)  

        # notice that for batches of data the result is still a vector of size num_batches
        return (-hidden_term - vbias_term).mean()  # mean along the batches

    def backward(self, target, v, h_prob):
        # vbias_term = v.mv(self.v_bias)  # = v*v_bias
        # pv = (F.linear(v, self.W, self.h_bias)).exp().add(1).prod(1)*vbias_term

        # p(H_i | v) where v is the input data
        probability = torch.sigmoid(F.linear(target, self.W, self.h_bias))

        # p(H_i | v) where v is the negative visible layer
        h_prob_negative = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        # Update the W
        training_set_avg = probability.t().mm(target)
        # The minus sign comes from the implementation of the SGD in pytorch 
        # see http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
        # the learning rate has a negative sign in front
        self.W.grad = -(training_set_avg - h_prob_negative.t().mm(v)) / probability.size(0)

        # Update the v_bias
        # pv_v = v.t().mv(pv)
        self.v_bias.grad = -(target - v).mean(0)

        # Update the h_bias
        # pv_hp = h_prob.t().mv(pv) # p(H_i | v) * v  where the probability is given v as a machine state
        self.h_bias.grad = -(probability - h_prob_negative).mean(0)

    def sample_from_rbm(self, steps, nstates, temperature = 0.0, decay = 0.0):
        """ Samples from the RBM distribution function 
            Uses annealing 
        """

        # Random initial visible state
        #v = F.relu(torch.sign(Variable(torch.rand(nstates,self.n_vis))-0.5))
        v = Variable(torch.zeros(nstates,self.n_vis))
        #v = v_in

        # Run the Gibbs sampling for a number of steps
        for s in xrange(steps):
            t = exp(-decay*s)*temperature + 1.0
            h, h_prob = self.hidden_from_visible(v, t)
            v, v_prob = self.visible_from_hidden(h, t)
            ## Compute the free energy of this state
            #print("Free energy: ",self.free_energy(v))
        return v